#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics import manager as _manager
from tlm_adjoint.fenics.backend import backend_Constant, backend_Function
from tlm_adjoint.fenics.backend_code_generator_interface import \
    complex_mode, interpolate_expression
from tlm_adjoint.alias import gc_disabled

import copy
import functools
import gc
import hashlib
import inspect
import logging
import mpi4py.MPI as MPI
import numpy as np
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
from operator import itemgetter
import os
import pytest
import runpy
import weakref

__all__ = \
    [
        "complex_mode",
        "interpolate_expression",

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
    parameters["ghost_mode"] = "none"
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

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    stop_manager()
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()


def seed_test(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        h_kwargs = copy.copy(kwargs)
        if "tmp_path" in inspect.signature(fn).parameters:
            # Raises an error if tmp_path is a positional argument
            del h_kwargs["tmp_path"]

        h = hashlib.sha256()
        h.update(fn.__name__.encode("utf-8"))
        h.update(str(args).encode("utf-8"))
        h.update(str(sorted(h_kwargs.items(), key=itemgetter(0))).encode("utf-8"))  # noqa: E501
        seed = int(h.hexdigest(), 16) + MPI.COMM_WORLD.rank
        seed %= 2 ** 32
        np.random.seed(seed)

        return fn(*args, **kwargs)
    return wrapped_fn


def params_set(names, *values):
    if len(values) > 1:
        sub_params = params_set(names[1:], *values[1:])
        params = []
        for value in values[0]:
            for sub_params_ in sub_params:
                new_params = copy.deepcopy(sub_params_)
                new_params[names[0]] = value
                params.append(new_params)
    else:
        params = [{names[0]:value} for value in values[0]]
    return params


@pytest.fixture(params=params_set(["enable_caching", "defer_adjoint_assembly"],
                                  [True, False],
                                  [True, False]))
def test_configurations(request):
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] \
        = request.param["enable_caching"]
    parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] \
        = request.param["defer_adjoint_assembly"]


@pytest.fixture(
    params=["none",
            pytest.param("shared_facets",
                         marks=pytest.mark.skipif(MPI.COMM_WORLD.size == 1,
                                                  reason="parallel only")),
            pytest.param("shared_vertices",
                         marks=pytest.mark.skipif(MPI.COMM_WORLD.size == 1,
                                                  reason="parallel only"))])
def test_ghost_modes(request):
    parameters["ghost_mode"] = request.param


_function_ids = weakref.WeakValueDictionary()


def clear_function_references():
    _function_ids.clear()


@gc_disabled
def referenced_functions():
    return tuple(F_ref for F_ref in map(call, _function_ids.valuerefs())
                 if F_ref is not None)


def _Constant__init__(self, *args, **kwargs):
    _Constant__init__orig(self, *args, **kwargs)
    _function_ids[function_id(self)] = self


_Constant__init__orig = backend_Constant.__init__
backend_Constant.__init__ = _Constant__init__


def _Function__init__(self, *args, **kwargs):
    _Function__init__orig(self, *args, **kwargs)
    _function_ids[function_id(self)] = self


_Function__init__orig = backend_Function.__init__
backend_Function.__init__ = _Function__init__


@pytest.fixture
def test_leaks():
    clear_function_references()

    yield

    gc.collect()

    # Clear some internal storage that is allowed to keep references
    clear_caches()
    manager = _manager()
    manager.drop_references()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    manager._tlm.clear()
    manager._adj_cache.clear()

    gc.collect()

    refs = 0
    for F in referenced_functions():
        info(f"{function_name(F):s} referenced")
        refs += 1
    if refs == 0:
        info("No references")

    clear_function_references()
    assert refs == 0

    manager.reset("memory", {"drop_references": False})


@pytest.fixture
def tmp_path(tmp_path):
    if MPI.COMM_WORLD.rank != 0:
        tmp_path = None
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def run_example(example, clear_forward_globals=True):
    start_manager()
    filename = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, os.path.pardir,
                            "examples", "fenics", example)
    gl = runpy.run_path(filename)
    if clear_forward_globals:
        # Clear objects created by the script. Requires the script to define a
        # 'forward' function.
        gl["forward"].__globals__.clear()


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
