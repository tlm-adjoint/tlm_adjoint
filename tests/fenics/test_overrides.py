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
def test_overrides(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    bc = DirichletBC(space, 1.0, "on_boundary")

    def project_project(F):
        G = Function(space, name="G")

        project(F, space, bcs=bc, function=G,
                solver_parameters=ls_parameters_cg)

        return G

    def project_assemble_system_KrylovSolver(F):
        G = Function(space, name="G")

        A, b = assemble_system(0.2 * inner(test, trial) * dx,
                               0.3 * inner(test, F) * dx)
        A, b = assemble_system(0.8 * inner(test, trial) * dx,
                               0.7 * inner(test, F) * dx,
                               A_tensor=A, b_tensor=b, add_values=True)
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_assemble_KrylovSolver(F):
        G = Function(space, name="G")

        A = assemble(inner(test, trial) * dx)
        b = assemble(inner(test, F) * dx)
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_assemble_mult_KrylovSolver(F):
        G = Function(space, name="G")

        A = assemble(inner(test, trial) * dx)
        b = A * F.vector()
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_LinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(test, trial) * dx == inner(test, F) * dx
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, G, bcs=bc)
        solver = LinearVariationalSolver(problem)
        solver.parameters.update(ls_parameters_cg)
        solver.solve()

        return G

    def project_NonlinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(test, G) * dx - inner(test, F) * dx
        problem = NonlinearVariationalProblem(eq, G,
                                              J=inner(test, trial) * dx,
                                              bcs=bc)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["newton_solver"].update(ns_parameters_newton_cg)
        solver.solve()

        return G

    for project_fn in [project_project,
                       project_assemble_system_KrylovSolver,
                       project_assemble_KrylovSolver,
                       project_assemble_mult_KrylovSolver,
                       project_LinearVariationalSolver,
                       project_NonlinearVariationalSolver]:
        def forward(F):
            G = project_fn(F)

            J = Functional(name="J")
            J.assign(inner(G, G * (1 + G)) * dx)
            return G, J

        reset_manager("memory", {"drop_references": True})
        start_manager()
        G, J = forward(F)
        stop_manager()

        error = Function(space, name="error")
        solve(inner(test, trial) * dx == inner(test, F) * dx,
              error, bc, solver_parameters=ls_parameters_cg)
        function_axpy(error, -1.0, G)
        assert function_linf_norm(error) < 1.0e-13

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
