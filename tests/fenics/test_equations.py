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
def test_AssignmentSolver(setup_test, test_leaks,):
    space = RealFunctionSpace()
    x = Function(space, name="x", static=True)
    function_assign(x, 16.0)

    def forward(x):
        y = [Function(space, name=f"y_{i:d}") for i in range(9)]
        z = Function(space, name="z")

        AssignmentSolver(x, y[0]).solve()
        for i in range(len(y) - 1):
            AssignmentSolver(y[i], y[i + 1]).solve()
        NormSqSolver(y[-1], z).solve()

        x_norm_sq = Function(space, name="x_norm_sq")
        NormSqSolver(x, x_norm_sq).solve()

        z_norm_sq = Function(space, name="z_norm_sq")
        NormSqSolver(z, z_norm_sq).solve()

        J = Functional(name="J", space=space)
        AxpySolver(z_norm_sq, 2.0, x_norm_sq, J.fn()).solve()

        K = Functional(name="K", space=space)
        AssignmentSolver(z_norm_sq, K.fn()).solve()

        return J, K

    start_manager()
    J, K = forward(x)
    stop_manager()

    assert(abs(J.value() - 66048.0) == 0.0)
    assert(abs(K.value() - 65536.0) == 0.0)

    dJs = compute_gradient([J, K], x)

    dm = Function(space, name="dm", static=True)
    function_assign(dm, 1.0)

    for forward_J, J_val, dJ in [(lambda x: forward(x)[0], J.value(), dJs[0]),
                                 (lambda x: forward(x)[1], K.value(), dJs[1])]:
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ, dM=dm)
        assert(min_order > 2.00)

        ddJ = Hessian(forward_J)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ, dM=dm)
        assert(min_order > 3.00)

        min_order = taylor_test_tlm(forward_J, x, tlm_order=1, dMs=(dm,))
        assert(min_order > 2.00)

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1,
                                            dMs=(dm,))
        assert(min_order > 2.00)

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2,
                                            dMs=(dm, dm))
        assert(min_order > 2.00)


@pytest.mark.fenics
def test_AxpySolver(setup_test, test_leaks):
    space = RealFunctionSpace()
    x = Function(space, name="x", static=True)
    function_assign(x, 1.0)

    def forward(x):
        y = [Function(space, name=f"y_{i:d}") for i in range(5)]
        z = [Function(space, name=f"z_{i:d}") for i in range(2)]
        function_assign(z[0], 7.0)

        AssignmentSolver(x, y[0]).solve()
        for i in range(len(y) - 1):
            AxpySolver(y[i], i + 1, z[0], y[i + 1]).solve()
        NormSqSolver(y[-1], z[1]).solve()

        J = Functional(name="J", space=space)
        NormSqSolver(z[1], J.fn()).solve()

        return J

    start_manager()
    J = forward(x)
    stop_manager()

    J_val = J.value()
    assert(abs(J_val - 25411681.0) == 0.0)

    dJ = compute_gradient(J, x)

    dm = Function(space, name="dm", static=True)
    function_assign(dm, 1.0)

    # Usage as in dolfin-adjoint tests
    min_order = taylor_test(forward, x, J_val=J_val, dJ=dJ, dM=dm)
    assert(min_order > 2.00)

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, x, J_val=J_val, ddJ=ddJ, dM=dm,
                            seed=2.0e-2)
    assert(min_order > 3.00)

    min_order = taylor_test_tlm(forward, x, tlm_order=1, dMs=(dm,))
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, x, adjoint_order=1, dMs=(dm,))
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, x, adjoint_order=2,
                                        dMs=(dm, dm))
    assert(min_order > 2.00)


@pytest.mark.fenics
def test_DirichletBCSolver(setup_test, test_leaks, test_configurations):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    def forward(bc):
        x_0 = Function(space, name="x_0")
        x_1 = Function(space, name="x_1")
        x = Function(space, name="x")

        DirichletBCSolver(bc, x_1, "on_boundary").solve()

        solve(inner(grad(test), grad(trial)) * dx
              == inner(test, F) * dx - inner(grad(test), grad(x_1)) * dx,
              x_0, DirichletBC(space, 0.0, "on_boundary",
                               static=True, homogeneous=True),
              solver_parameters=ls_parameters_cg)

        AxpySolver(x_0, 1.0, x_1, x).solve()

        J = Functional(name="J")
        J.assign(inner(x * x, x * x) * dx)
        return x, J

    bc = Function(space, name="bc", static=True)
    function_assign(bc, 1.0)

    start_manager()
    x, J = forward(bc)
    stop_manager()

    x_ref = Function(space, name="x_ref")
    solve(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx,
          x_ref,
          DirichletBC(space, 1.0, "on_boundary", static=True),
          solver_parameters=ls_parameters_cg)
    error = Function(space, name="error")
    function_assign(error, x_ref)
    function_axpy(error, -1.0, x)
    assert(function_linf_norm(error) < 1.0e-13)

    J_val = J.value()

    dJ = compute_gradient(J, bc)

    def forward_J(bc):
        return forward(bc)[1]

    # Usage as in dolfin-adjoint tests
    min_order = taylor_test(forward_J, bc, J_val=J_val, dJ=dJ)
    assert(min_order > 2.00)

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, bc, J_val=J_val, ddJ=ddJ)
    assert(min_order > 3.00)

    min_order = taylor_test_tlm(forward_J, bc, tlm_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward_J, bc, adjoint_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward_J, bc, adjoint_order=2)
    assert(min_order > 2.00)
