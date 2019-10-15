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

from test_base import *

import pytest


@pytest.mark.fenics
def test_long_range(setup_test, test_leaks):
    n_steps = 200
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 0,
                             "snaps_in_ram": 2, "verbose": True})

    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(F, x_ref=None):
        x_old = Function(space, name="x_old")
        x = Function(space, name="x")
        AssignmentSolver(F, x_old).solve()
        J = Functional(name="J")
        gather_ref = x_ref is None
        if gather_ref:
            x_ref = {}
        for n in range(n_steps):
            terms = [(1.0, x_old)]
            if n % 11 == 0:
                terms.append((1.0, F))
            LinearCombinationSolver(x, *terms).solve()
            if n % 17 == 0:
                if gather_ref:
                    x_ref[n] = function_copy(x, name=f"x_ref_{n:d}")
                J.addto(inner(x * x * x, x_ref[n]) * dx)
            AssignmentSolver(x, x_old).solve()
            if n < n_steps - 1:
                new_block()

        return x_ref, J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sin(pi * X[0]))

    start_manager()
    x_ref, J = forward(F)
    stop_manager()

    J_val = J.value()

    dJ = compute_gradient(J, F)

    def forward_J(F):
        return forward(F, x_ref=x_ref)[1]

    min_order = taylor_test(forward_J, F, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, F, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, F, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, F, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, F, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.fenics
def test_EmptySolver(setup_test, test_leaks):
    class EmptySolver(Equation):
        def __init__(self):
            Equation.__init__(self, [], [], nl_deps=[], ic_deps=[])

        def forward_solve(self, X, deps=None):
            pass

    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(F):
        EmptySolver().solve()

        F_norm_sq = Constant(name="F_norm_sq")
        NormSqSolver(F, F_norm_sq).solve()

        J = Functional(name="J")
        NormSqSolver(F_norm_sq, J.fn()).solve()
        return J

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[0]))

    start_manager()
    J = forward(F)
    stop_manager()

    manager = _manager()
    manager.finalize()
    manager.info()
    assert len(manager._blocks) == 1
    assert len(manager._blocks[0]) == 3
    assert len(manager._blocks[0][0].X()) == 0

    J_val = J.value()
    assert abs(J_val - F.vector().norm("l2") ** 4) < 1.0e-11

    dJ = compute_gradient(J, F)

    min_order = taylor_test(forward, F, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, F, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, F, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=2)
    assert min_order > 2.00


@pytest.mark.fenics
def test_empty(setup_test, test_leaks):
    def forward(m):
        return Functional(name="J")

    m = Constant(name="m", static=True)

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)
    assert float(dJ) == 0.0
