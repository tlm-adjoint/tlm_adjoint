from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics import manager as _manager
from tlm_adjoint.fenics.backend import backend_Constant, backend_Function
from tlm_adjoint.fenics.backend_code_generator_interface import (
    complex_mode, interpolate_expression)
from tlm_adjoint.alias import gc_disabled
from tlm_adjoint.override import override_method

from ..test_base import chdir_tmp_path, jax_tlm_config, seed_test, tmp_path
from ..test_base import run_example as _run_example

import gc
import logging
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
import os
import petsc4py.PETSc as PETSc
import pytest
import weakref

__all__ = \
    [
        "complex_mode",
        "interpolate_expression",

        "chdir_tmp_path",
        "jax_tlm_config",
        "run_example",
        "seed_test",
        "setup_test",
        "test_configurations",
        "test_ghost_modes",
        "test_leaks",
        "tmp_path",

        "ls_parameters_cg",
        "ls_parameters_gmres",
        "ns_parameters_newton_cg",
        "ns_parameters_newton_gmres"
    ]


@pytest.fixture
def setup_test():
    if DEFAULT_COMM.size > 1 and not hasattr(PETSc, "garbage_cleanup"):
        gc_enabled = gc.isenabled()
        gc.disable()

    parameters["ghost_mode"] = "none"
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

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()

    if DEFAULT_COMM.size > 1 and not hasattr(PETSc, "garbage_cleanup") \
            and gc_enabled:
        gc.enable()


@pytest.fixture(params=[{"enable_caching": True},
                        {"enable_caching": False}])
def test_configurations(request):
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] \
        = request.param["enable_caching"]


@pytest.fixture(
    params=["none",
            pytest.param("shared_facets",
                         marks=pytest.mark.skipif(DEFAULT_COMM.size == 1,
                                                  reason="parallel only")),
            pytest.param("shared_vertices",
                         marks=pytest.mark.skipif(DEFAULT_COMM.size == 1,
                                                  reason="parallel only"))])
def test_ghost_modes(request):
    parameters["ghost_mode"] = request.param


_var_ids = weakref.WeakValueDictionary()


def clear_var_references():
    _var_ids.clear()


@gc_disabled
def referenced_vars():
    return tuple(F_ref for F_ref in map(call, _var_ids.valuerefs())
                 if F_ref is not None)


@override_method(Vector, "__init__")
def Vector__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _var_ids[var_id(self)] = self


@override_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, *args, **kwargs):
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
    manager._adj_cache.clear()

    gc.collect()
    garbage_cleanup(DEFAULT_COMM)

    refs = 0
    for F in referenced_vars():
        if not isinstance(F, ZeroConstant):
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
                                "examples", "fenics", example)
    else:
        filename = example
    _run_example(filename, clear_forward_globals=clear_forward_globals)


ls_parameters_cg = {"linear_solver": "cg",
                    "preconditioner": "sor",
                    "krylov_solver": {"relative_tolerance": 1.0e-14,
                                      "absolute_tolerance": 1.0e-16},
                    "symmetric": True}

ls_parameters_gmres = {"linear_solver": "gmres",
                       "preconditioner": "sor",
                       "krylov_solver": {"relative_tolerance": 1.0e-14,
                                         "absolute_tolerance": 1.0e-16}}

ns_parameters_newton_cg = {"linear_solver": "cg",
                           "preconditioner": "sor",
                           "krylov_solver": {"relative_tolerance": 1.0e-14,
                                             "absolute_tolerance": 1.0e-16},
                           "relative_tolerance": 1.0e-13,
                           "absolute_tolerance": 1.0e-15}

ns_parameters_newton_gmres = {"linear_solver": "gmres",
                              "preconditioner": "sor",
                              "krylov_solver": {"relative_tolerance": 1.0e-14,
                                                "absolute_tolerance": 1.0e-16},
                              "relative_tolerance": 1.0e-13,
                              "absolute_tolerance": 1.0e-15}
