#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics.backend import backend_assemble
from tlm_adjoint.fenics.backend_code_generator_interface import (
    assemble as backend_code_generator_interface_assemble)

from .test_base import *

import mpi4py.MPI as MPI
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
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
        G = Function(space, name="G")

        project(F, space, bcs=bc, function=G,
                solver_parameters=ls_parameters_cg)

        return G

    def project_assemble_system_KrylovSolver(F):
        G = Function(space, name="G")

        A, b = assemble_system(0.2 * inner(trial, test) * dx,
                               0.3 * inner(F, test) * dx)
        A, b = assemble_system(0.8 * inner(trial, test) * dx,
                               0.7 * inner(F, test) * dx,
                               A_tensor=A, b_tensor=b, add_values=True)
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_assemble_KrylovSolver(F):
        G = Function(space, name="G")

        A = assemble(inner(trial, test) * dx)
        b = assemble(inner(F, test) * dx)
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_assemble_mult_KrylovSolver(F):
        G = Function(space, name="G")

        A = assemble(inner(trial, test) * dx)
        b = A * F.vector()
        bc.apply(A, b)

        solver = KrylovSolver(A, "gmres", "sor")
        solver.parameters.update({"relative_tolerance": 1.0e-14,
                                  "absolute_tolerance": 1.0e-16})
        solver.solve(G.vector(), b)

        return G

    def project_LinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(trial, test) * dx == inner(F, test) * dx
        problem = LinearVariationalProblem(eq.lhs, eq.rhs, G, bcs=bc)
        solver = LinearVariationalSolver(problem)
        solver.parameters.update(ls_parameters_cg)
        solver.solve()

        return G

    def project_NonlinearVariationalSolver(F):
        G = Function(space, name="G")

        eq = inner(G, test) * dx - inner(F, test) * dx
        problem = NonlinearVariationalProblem(eq, G,
                                              J=inner(trial, test) * dx,
                                              bcs=bc)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["nonlinear_solver"] = "newton"
        solver.parameters["symmetric"] = True
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


@pytest.mark.fenics
@seed_test
def test_Function_assign(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        u = Constant(0.0, name="u")
        u.assign(m)
        u.assign(-2.0)
        u.assign(u + 2.0 * m)

        m_ = Function(space, name="m")
        m_.assign(m)
        m = m_
        del m_

        u_ = Function(space, name="u")
        u_.assign(u)
        u = u_
        del u_

        one = Function(space, name="one")
        one.assign(Constant(1.0))

        v = Function(space, name="v")
        v.assign(u)
        v.assign(u + one)
        v.assign(Constant(0.0))
        v.assign(u + v + one)
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


@pytest.mark.fenics
@pytest.mark.parametrize("ZeroFunction", [Function, ZeroFunction])
@pytest.mark.parametrize("assemble", [backend_assemble,
                                      assemble,
                                      backend_code_generator_interface_assemble])  # noqa: E501
@seed_test
def test_assemble_ZeroFunction(setup_test, test_leaks,
                               ZeroFunction, assemble):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)

    F = ZeroFunction(space, name="F")
    G = Function(space, name="G")

    form = (F + G) * dx

    b = assemble(form)
    assert abs(b) == 0.0

    G.assign(Constant(np.sqrt(2.0)))
    b_ref = backend_assemble(form)
    assert abs(b_ref - np.sqrt(2.0)) < 1.0e-15

    for _ in range(3):
        b = assemble(form)
        assert abs(b - b_ref) == 0.0

    G = Function(space, name="G")
    F = ZeroFunction(space, name="F")

    form = (F + G) * dx

    b = assemble(form)
    assert abs(b) == 0.0

    G.assign(Constant(np.sqrt(2.0)))
    b_ref = backend_assemble(form)
    assert abs(b_ref - np.sqrt(2.0)) < 1.0e-15

    for _ in range(3):
        b = assemble(form)
        assert abs(b - b_ref) == 0.0
