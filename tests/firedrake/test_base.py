#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake import manager as _manager
from tlm_adjoint.firedrake.backend import (
    backend_Cofunction, backend_Constant, backend_Function)
from tlm_adjoint.firedrake.backend_code_generator_interface import (
    complex_mode, interpolate_expression)
from tlm_adjoint.alias import gc_disabled
from tlm_adjoint.override import override_method

from ..test_base import chdir_tmp_path, seed_test, tmp_path
from ..test_base import (
    run_example as _run_example, run_example_notebook as _run_example_notebook)

import gc
import logging
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
        "run_example",
        "run_example_notebook",
        "seed_test",
        "setup_test",
        "test_configurations",
        "test_leaks",
        "tmp_path",

        "ls_parameters_cg",
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
    parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] \
        = False
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


@pytest.fixture(params=[{"enable_caching": True,
                         "defer_adjoint_assembly": True},
                        {"enable_caching": True,
                         "defer_adjoint_assembly": False},
                        {"enable_caching": False,
                         "defer_adjoint_assembly": True},
                        {"enable_caching": False,
                         "defer_adjoint_assembly": False}])
def test_configurations(request):
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] \
        = request.param["defer_adjoint_assembly"]


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


@override_method(backend_Cofunction, "__init__")
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
    manager._adj_cache.clear()
    for block in list(manager._blocks) + [manager._block]:
        for eq in block:
            if isinstance(eq, PointInterpolation):
                del eq._interp

    gc.collect()
    garbage_cleanup(DEFAULT_COMM)

    refs = 0
    for F in referenced_vars():
        if not isinstance(F, ZeroConstant) \
                and var_name(F) != f"{DEFAULT_MESH_NAME:s}_coordinates":
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


ls_parameters_cg = {"ksp_type": "cg",
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
