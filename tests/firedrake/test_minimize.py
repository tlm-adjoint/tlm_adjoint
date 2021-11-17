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

from firedrake import *
from tlm_adjoint.firedrake import *

from test_base import *

import numpy as np
import petsc4py.PETSc as PETSc
import pytest


@pytest.mark.firedrake
@pytest.mark.skipif(issubclass(PETSc.ScalarType,
                               (complex, np.complexfloating)),
                    reason="real only")
@seed_test
def test_minimize_project(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(alpha, x_ref=None):
        x = Function(space, name="x")
        solve(inner(trial, test) * dx == inner(alpha, test) * dx,
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
    assert result.success

    error = Function(space, name="error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    assert function_linf_norm(error) < 1.0e-7


@pytest.mark.firedrake
@pytest.mark.skipif(issubclass(PETSc.ScalarType,
                               (complex, np.complexfloating)),
                    reason="real only")
@seed_test
def test_minimize_project_multiple(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(alpha, beta, x_ref=None, y_ref=None):
        x = Function(space, name="x")
        solve(inner(trial, test) * dx == inner(alpha, test) * dx,
              x, solver_parameters=ls_parameters_cg)

        y = Function(space, name="y")
        solve(inner(trial, test) * dx == inner(beta, test) * dx,
              y, solver_parameters=ls_parameters_cg)

        if x_ref is None:
            x_ref = Function(space, name="x_ref", static=True)
            function_assign(x_ref, x)
        if y_ref is None:
            y_ref = Function(space, name="y_ref", static=True)
            function_assign(y_ref, y)

        J = Functional(name="J")
        J.assign(inner(x - x_ref, x - x_ref) * dx)
        J.addto(inner(y - y_ref, y - y_ref) * dx)
        return x_ref, y_ref, J

    alpha_ref = Function(space, name="alpha_ref", static=True)
    interpolate_expression(alpha_ref, exp(X[0] + X[1]))
    beta_ref = Function(space, name="beta_ref", static=True)
    interpolate_expression(beta_ref, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    x_ref, y_ref, _ = forward(alpha_ref, beta_ref)

    alpha0 = Function(space, name="alpha0", static=True)
    beta0 = Function(space, name="beta0", static=True)
    start_manager()
    _, _, J = forward(alpha0, beta0, x_ref=x_ref, y_ref=y_ref)
    stop_manager()

    def forward_J(alpha, beta):
        return forward(alpha, beta, x_ref=x_ref, y_ref=y_ref)[2]

    (alpha, beta), result = minimize_scipy(forward_J, (alpha0, beta0), J0=J,
                                           method="L-BFGS-B",
                                           options={"ftol": 0.0,
                                                    "gtol": 1.0e-11})
    assert result.success

    error = Function(space, name="error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    assert function_linf_norm(error) < 1.0e-8

    function_assign(error, beta_ref)
    function_axpy(error, -1.0, beta)
    assert function_linf_norm(error) < 1.0e-9
