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

from test_base import *

import pytest


@pytest.mark.fenics
def test_minimize_project(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(alpha, x_ref=None):
        x = Function(space, name="x")
        solve(inner(test, trial) * dx == inner(test, alpha) * dx,
              x, solver_parameters=ls_parameters_cg)

        if x_ref is None:
            x_ref = Function(space, name="x_ref", static=True)
            function_assign(x_ref, x)

        J = Functional(name="J")
        J.assign(inner(x - x_ref, x - x_ref) * dx)
        return x_ref, J

    alpha_ref = Function(space, name="alpha_ref", static=True)
    interpolate_expression(alpha_ref, exp(X[0] + X[1]))
    x_ref, _ = forward(alpha_ref)

    alpha0 = Function(space, name="alpha0", static=True)
    start_manager()
    _, J = forward(alpha0, x_ref=x_ref)
    stop_manager()

    def forward_J(alpha):
        return forward(alpha, x_ref=x_ref)[1]

    alpha, result = minimize_scipy(forward_J, alpha0, J0=J,
                                   method="L-BFGS-B",
                                   options={"ftol": 0.0, "gtol": 1.0e-10})
    assert(result.success)

    error = Function(space, name="error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    assert(function_linf_norm(error) < 1.0e-7)
