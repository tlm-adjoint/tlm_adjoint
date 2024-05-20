from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake import manager as _manager
from tlm_adjoint.firedrake.backend import (
    backend_Cofunction, backend_Constant, backend_Function, complex_mode)
from tlm_adjoint.firedrake.interpolation import interpolate_expression
from tlm_adjoint.alias import gc_disabled
from tlm_adjoint.patch import patch_method

from ..test_base import chdir_tmp_path, jax_tlm_config, seed_test, tmp_path
from ..test_base import (
    run_example as _run_example, run_example_notebook as _run_example_notebook)

import functools
import gc
import logging
import numbers
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
import os
import pytest
import sys
import weakref

__all__ = \
    [
        "complex_mode",
        "interpolate_expression",

        "chdir_tmp_path",
        "jax_tlm_config",
        "run_example",
        "run_example_notebook",
        "seed_test",
        "setup_test",
        "test_configurations",
        "test_leaks",
        "tmp_path",

        "assemble_action",
        "assemble_rhs",
        "interpolate_expr",
        "solve_eq",
        "test_rhs",

        "ls_parameters_cg",
        "ls_parameters_gmres",
        "ns_parameters_newton_cg",
        "ns_parameters_newton_gmres"
    ]

_logger = logging.getLogger("firedrake")
_logger.removeHandler(_logger.handlers[0])
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
_logger.addHandler(_handler)


@pytest.fixture
def setup_test():
    parameters["tlm_adjoint"]["Assembly"]["match_quadrature"] = False
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = True
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] = True
    parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"] = False
    # parameters["tlm_adjoint"]["assembly_verification"]["jacobian_tolerance"] = 1.0e-15  # noqa: E501
    # parameters["tlm_adjoint"]["assembly_verification"]["rhs_tolerance"] \
    #     = 1.0e-12

    set_default_float_dtype(backend_ScalarType)
    set_default_jax_dtype(backend_ScalarType)

    logging.getLogger("firedrake").setLevel(logging.INFO)
    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()


@pytest.fixture(params=[{"enable_caching": True},
                        {"enable_caching": False}])
def test_configurations(request):
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] \
        = request.param["enable_caching"]


_var_ids = weakref.WeakValueDictionary()


def clear_var_references():
    _var_ids.clear()


@gc_disabled
def referenced_vars():
    return tuple(F_ref for F_ref in map(call, _var_ids.valuerefs())
                 if F_ref is not None)


@patch_method(Vector, "__init__")
def Vector__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@patch_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@patch_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@patch_method(backend_Cofunction, "__init__")
def Cofunction__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@pytest.fixture
def test_leaks():
    clear_var_references()

    yield

    gc.collect()
    garbage_cleanup(DEFAULT_COMM)

    # Clear some internal storage that is allowed to keep references
    clear_caches()
    manager = _manager()
    manager.drop_references()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    manager._tlm.clear()
    for tlm_map in manager._tlm_map.values():
        del tlm_map._M, tlm_map._dM
    manager._adj_cache.clear()
    for block in list(manager._blocks) + [manager._block]:
        for eq in block:
            if isinstance(eq, PointInterpolation):
                del eq._interp

    gc.collect()
    garbage_cleanup(DEFAULT_COMM)

    refs = 0
    for F in referenced_vars():
        if var_name(F) != f"{DEFAULT_MESH_NAME:s}_coordinates":
            info(f"{var_name(F):s} referenced")
            refs += 1
    if refs == 0:
        info("No references")

    clear_var_references()
    assert refs == 0

    manager.reset("memory", {"drop_references": False})


def run_example(example, *,
                add_example_path=True, clear_forward_globals=True):
    if add_example_path:
        filename = os.path.join(os.path.dirname(__file__),
                                os.path.pardir, os.path.pardir,
                                "examples", "firedrake", example)
    else:
        filename = example
    _run_example(filename, clear_forward_globals=clear_forward_globals)


def run_example_notebook(example, tmp_path, *,
                         add_example_path=True):
    if add_example_path:
        filename = os.path.join(os.path.dirname(__file__),
                                os.path.pardir, os.path.pardir,
                                "docs", "source", "examples", example)
    else:
        filename = example
    _run_example_notebook(filename, tmp_path)


def rhs_Form(m, test):
    return inner(m, test) * dx


def rhs_Cofunction(m, test):
    return assemble(rhs_Form(m, test))


def rhs_FormSum(alpha, m, test):
    if isinstance(alpha, numbers.Complex) \
            and not isinstance(alpha, numbers.Real) \
            and not complex_mode:
        pytest.skip()
    return alpha * rhs_Form(m, test) + (1.0 - alpha) * rhs_Cofunction(m, test)


@pytest.fixture(params=[{"test_rhs": rhs_Form},
                        {"test_rhs": rhs_Cofunction},
                        {"test_rhs": functools.partial(rhs_FormSum, 1.5)},
                        {"test_rhs": functools.partial(rhs_FormSum, 0.5)},
                        {"test_rhs": functools.partial(rhs_FormSum, -0.5)},
                        {"test_rhs": functools.partial(rhs_FormSum, 0.5 + 0.5j)}])  # noqa: E501
def test_rhs(request):
    return request.param["test_rhs"]


def assemble_Assembly(b, rhs):
    Assembly(b, rhs).solve()
    return b


def assemble_assemble_assign(b, rhs):
    return b.assign(assemble(rhs))


def assemble_assemble_tensor(b, rhs):
    return assemble(rhs, tensor=b)


@pytest.fixture(params=[{"assemble_rhs": assemble_Assembly},
                        {"assemble_rhs": assemble_assemble_assign},
                        {"assemble_rhs": assemble_assemble_tensor}])
def assemble_rhs(request):
    return request.param["assemble_rhs"]


def assemble_action_InnerProduct(J, b, u):
    InnerProduct(J, b, u).solve()


def assemble_action_Action(J, b, u):
    return Assembly(J, b(u)).solve()


@pytest.fixture(params=[{"assemble_action": assemble_action_InnerProduct},
                        {"assemble_action": assemble_action_Action}])
def assemble_action(request):
    return request.param["assemble_action"]


def solve_eq_EquationSolver(eq, u, *, solver_parameters=None):
    EquationSolver(eq, u, solver_parameters=solver_parameters).solve()


def solve_eq_solve(eq, u, *, solver_parameters=None):
    solve(eq, u, solver_parameters=solver_parameters)


@pytest.fixture(params=[{"solve_eq": solve_eq_EquationSolver},
                        {"solve_eq": solve_eq_solve}])
def solve_eq(request):
    return request.param["solve_eq"]


def interpolate_interpolate(v, V):
    return interpolate(v, V)


def interpolate_Function_interpolate(v, V):
    return space_new(V).interpolate(v)


def interpolate_Interpolator_Function(v, V):
    interp = Interpolator(v, V)
    x = space_new(V)
    interp.interpolate(output=x)
    return x


def interpolate_Interpolator_test(v, V):
    interp = Interpolator(TestFunction(v.function_space()), V)
    x = space_new(V)
    interp.interpolate(v, output=x)
    return x


def interpolate_Interpolator_assemble(v, V):
    return assemble(Interpolate(v, V))


def interpolate_Interpolator_assemble_tensor(v, V):
    x = space_new(V)
    assemble(Interpolate(v, V), tensor=x)
    return x


@pytest.fixture(params=[{"interpolate_expr": interpolate_interpolate},
                        {"interpolate_expr": interpolate_Function_interpolate},
                        {"interpolate_expr": interpolate_Interpolator_Function},  # noqa: E501
                        {"interpolate_expr": interpolate_Interpolator_test},
                        {"interpolate_expr": interpolate_Interpolator_assemble},  # noqa: E501
                        {"interpolate_expr": interpolate_Interpolator_assemble_tensor}])  # noqa: E501
def interpolate_expr(request):
    return request.param["interpolate_expr"]


ls_parameters_cg = {"ksp_type": "cg",
                    "pc_type": "sor",
                    "ksp_rtol": 1.0e-14,
                    "ksp_atol": 1.0e-16}


ls_parameters_gmres = {"ksp_type": "gmres",
                       "pc_type": "sor",
                       "ksp_rtol": 1.0e-14,
                       "ksp_atol": 1.0e-16}


ns_parameters_newton_cg = {"snes_type": "newtonls",
                           "ksp_type": "cg",
                           "pc_type": "sor",
                           "ksp_rtol": 1.0e-14,
                           "ksp_atol": 1.0e-16,
                           "snes_rtol": 1.0e-13,
                           "snes_atol": 1.0e-15,
                           "snes_stol": 0.0}

ns_parameters_newton_gmres = {"snes_type": "newtonls",
                              "ksp_type": "gmres",
                              "pc_type": "sor",
                              "ksp_rtol": 1.0e-14,
                              "ksp_atol": 1.0e-16,
                              "snes_rtol": 1.0e-13,
                              "snes_atol": 1.0e-15,
                              "snes_stol": 0.0}
