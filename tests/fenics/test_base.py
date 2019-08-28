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
from tlm_adjoint_fenics.backend import backend_Function

import copy
import gc
import numpy as np
import pytest
import ufl
import weakref

__all__ = \
    [
        "interpolate_expression",

        "setup_test",
        "test_configurations",
        "test_leaks",

        "ls_parameters_cg",
        "ls_parameters_gmres",
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

    reset_manager("memory", {"replace": True})
    clear_caches()
    stop_manager()

    np.random.seed(14012313 + default_comm().rank)


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


Function_ids = {}
_orig_Function_init = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function_init(self, *args, **kwargs)
    Function_ids[self.id()] = weakref.ref(self)


backend_Function.__init__ = _Function__init__


@pytest.fixture
def test_leaks():
    Function_ids.clear()

    yield

    # Clear some internal storage that is allowed to keep references
    manager = _manager()
    manager._cp.clear(clear_refs=True)
    tlm_values = manager._tlm.values()  # noqa: F841
    manager._tlm.clear()
    tlm_eqs_values = manager._tlm_eqs.values()  # noqa: F841
    manager._tlm_eqs.clear()

    gc.collect()

    refs = 0
    for F in Function_ids.values():
        F = F()
        if F is not None:
            info(f"{F.name():s} referenced")
            refs += 1
    if refs == 0:
        info("No references")

    Function_ids.clear()
    assert(refs == 0)


def interpolate_expression(F, ex):
    def cpp(ex):
        if isinstance(ex, ufl.classes.Cos):
            x, = ex.ufl_operands
            return f"cos({cpp(x):s})"
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

ns_parameters_newton_gmres = {"linear_solver": "gmres",
                              "preconditioner": "sor",
                              "krylov_solver": {"relative_tolerance": 1.0e-14,
                                                "absolute_tolerance": 1.0e-16},
                              "relative_tolerance": 1.0e-13,
                              "absolute_tolerance": 1.0e-15}
