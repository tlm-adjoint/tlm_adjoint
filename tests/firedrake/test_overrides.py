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

from .test_base import *

import mpi4py.MPI as MPI
import pytest

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_overrides(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    bc = DirichletBC(space, 1.0, "on_boundary")

    def project_project(F):
        return project(F, space, bcs=bc, name="G",
                       solver_parameters=ls_parameters_cg)

    def project_assemble_LinearSolver(F):
        G = Function(space, name="G")

        A = assemble(inner(trial, test) * dx, bcs=bc)
        b = assemble(inner(F, test) * dx)

        solver = LinearSolver(A, solver_parameters=ls_parameters_cg)
        solver.solve(G, b)

        return G

    def project_LinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(trial, test) * dx == inner(F, test) * dx
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, G, bcs=bc)
        solver = LinearVariationalSolver(
            problem, solver_parameters=ls_parameters_cg)
        solver.solve()

        return G

    def project_NonlinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(G, test) * dx - inner(F, test) * dx
        problem = NonlinearVariationalProblem(eq, G,
                                              J=inner(trial, test) * dx,
                                              bcs=bc)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=ns_parameters_newton_cg)
        solver.solve()

        return G

    for project_fn in [project_project,
                       project_assemble_LinearSolver,
                       project_LinearVariationalSolver,
                       project_NonlinearVariationalSolver]:
        def forward(F):
            G = project_fn(F)

            J = Functional(name="J")
            J.assign(dot(G, G * (1 + G)) * dx)
            return G, J

        reset_manager("memory", {"drop_references": True})
        start_manager()
        G, J = forward(F)
        stop_manager()

        error = Function(space, name="error")
        solve(inner(trial, test) * dx == inner(F, test) * dx,
              error, bc, solver_parameters=ls_parameters_cg)
        function_axpy(error, -1.0, G)
        assert function_linf_norm(error) < 1.0e-14

        J_val = J.value()

        dJ = compute_gradient(J, F)

        def forward_J(F):
            return forward(F)[1]

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


@pytest.mark.firedrake
@seed_test
def test_Function_assign(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        u = Function(space, name="u")
        u.assign(m)
        u.assign(u)

        v = Function(space, name="v")
        v.assign(u)
        v.assign(u + Constant(1.0))
        v.assign(2.5 * u + 3.6 * v + 4.7 * m)

        J = Functional(name="J")
        J.assign(((v - 1.0) ** 4) * dx)
        return J

    m = Constant(2.0, name="m")

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value()
    assert abs(J_val - 342974.2096) < 1.0e-9

    dJ = compute_gradient(J, m)

    dm = Constant(1.0)

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ, dM=dm)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, m, tlm_order=1, dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                        dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                        dMs=(dm, dm))
    assert min_order > 2.00


@pytest.mark.firedrake
@seed_test
def test_Nullspace(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(F):
        psi = Function(space, name="psi")

        solve(inner(grad(trial), grad(test)) * dx
              == -inner(F * F, test) * dx, psi,
              solver_parameters=ls_parameters_cg,
              nullspace=VectorSpaceBasis(constant=True),
              transpose_nullspace=VectorSpaceBasis(constant=True))

        J = Functional(name="J")
        J.assign((dot(psi, psi) ** 2) * dx
                 + dot(grad(psi), grad(psi)) * dx)

        return psi, J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sqrt(sin(pi * X[1])))

    start_manager()
    psi, J = forward(F)
    stop_manager()

    assert abs(function_sum(psi)) < 1.0e-15

    J_val = J.value()

    dJ = compute_gradient(J, F)

    def forward_J(F):
        return forward(F)[1]

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
    assert min_order > 2.00
