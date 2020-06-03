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
from tlm_adjoint_fenics import *
from tlm_adjoint_fenics import manager as _manager
from tlm_adjoint_fenics.backend import backend_Constant, backend_Function

import copy
import gc
import mpi4py.MPI as MPI
import numpy as np
import os
import pytest
import runpy
import ufl
import weakref

__all__ = \
    [
        "interpolate_expression",

        "run_example",
        "setup_test",
        "test_configurations",
        "test_leaks",

        "ls_parameters_cg",
        "ls_parameters_gmres",
        "ns_parameters_newton_cg",
        "ns_parameters_newton_gmres"
    ]


@pytest.fixture
def setup_test():
    parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"] = False
    parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] \
        = True
    parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] = True
    parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"] = False
    parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] \
        = False
    # parameters["tlm_adjoint"]["assembly_verification"]["jacobian_tolerance"] = 1.0e-15  # noqa: E501
    # parameters["tlm_adjoint"]["assembly_verification"]["rhs_tolerance"] \
    #     = 1.0e-12

    reset_manager("memory", {"replace": True})
    clear_caches()
    stop_manager()

    np.random.seed(14012313 + MPI.COMM_WORLD.rank)


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


function_ids = {}


def _Constant__init__(self, *args, **kwargs):
    _Constant__init__orig(self, *args, **kwargs)
    function_ids[function_id(self)] = weakref.ref(self)


_Constant__init__orig = backend_Constant.__init__
backend_Constant.__init__ = _Constant__init__


def _Function__init__(self, *args, **kwargs):
    _Function__init__orig(self, *args, **kwargs)
    function_ids[function_id(self)] = weakref.ref(self)


_Function__init__orig = backend_Function.__init__
backend_Function.__init__ = _Function__init__


@pytest.fixture
def test_leaks():
    function_ids.clear()

    yield

    gc.collect()

    # Clear some internal storage that is allowed to keep references
    manager = _manager()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    tlm_values = list(manager._tlm.values())  # noqa: F841
    manager._tlm.clear()
    tlm_eqs_values = [list(eq_tlm_eqs.values()) for eq_tlm_eqs in manager._tlm_eqs.values()]  # noqa: E501,F841
    manager._tlm_eqs.clear()
    manager.drop_references()

    gc.collect()
    gc.collect()

    refs = 0
    for F in function_ids.values():
        F = F()
        if F is not None:
            info(f"{F.name():s} referenced")
            refs += 1
    if refs == 0:
        info("No references")

    function_ids.clear()
    assert refs == 0


def run_example(example, clear_forward_globals=True):
    filename = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, os.path.pardir,
                            "examples", "fenics", example)
    gl = runpy.run_path(filename)
    if clear_forward_globals:
        # Clear objects created by the script. Requires the script to define a
        # 'forward' function.
        gl["forward"].__globals__.clear()


def interpolate_expression(F, ex):
    def cpp(ex):
        if isinstance(ex, ufl.classes.Cos):
            x, = ex.ufl_operands
            return f"cos({cpp(x):s})"
        elif isinstance(ex, ufl.classes.Division):
            x, y = ex.ufl_operands
            return f"({cpp(x):s}) / ({cpp(y):s})"
        elif isinstance(ex, ufl.classes.Exp):
            x, = ex.ufl_operands
            return f"exp({cpp(x):s})"
        elif isinstance(ex, ufl.classes.FloatValue):
            return f"{float(ex):.16e}"
        elif isinstance(ex, ufl.classes.Indexed):
            x, i = ex.ufl_operands
            i, = map(int, i)
            return f"({cpp(x):s})[{i:d}]"
        elif isinstance(ex, ufl.classes.IntValue):
            return f"{int(ex):d}"
        elif isinstance(ex, ufl.classes.Power):
            x, y = ex.ufl_operands
            return f"pow({cpp(x):s}, {cpp(y):s})"
        elif isinstance(ex, ufl.classes.Product):
            return " * ".join(map(lambda op: f"({cpp(op):s})",
                                  ex.ufl_operands))
        elif isinstance(ex, ufl.classes.Sin):
            x, = ex.ufl_operands
            return f"sin({cpp(x):s})"
        elif isinstance(ex, ufl.classes.SpatialCoordinate):
            return "x"
        elif isinstance(ex, ufl.classes.Sqrt):
            x, = ex.ufl_operands
            return f"sqrt({cpp(x):s})"
        elif isinstance(ex, ufl.classes.Sum):
            return " + ".join(map(lambda op: f"({cpp(op):s})",
                                  ex.ufl_operands))
        else:
            raise TypeError(f"Unsupported type: {type(ex)}")

    F.interpolate(Expression(cpp(ex),
                             element=F.function_space().ufl_element()))


ls_parameters_cg = {"linear_solver": "cg",
                    "preconditioner": "sor",
                    "krylov_solver": {"relative_tolerance": 1.0e-14,
                                      "absolute_tolerance": 1.0e-16}}

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
