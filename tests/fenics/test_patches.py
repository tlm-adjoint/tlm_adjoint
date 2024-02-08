from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics.backend import backend_assemble, backend_Constant
from tlm_adjoint.fenics.assembly import assemble as assembly_assemble
from tlm_adjoint.fenics.parameters import copy_parameters

from .test_base import *

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
@pytest.mark.parametrize("constant_cls", [backend_Constant, Constant])
@pytest.mark.parametrize("p", [1, 2])
@seed_test
def test_Constant_init(setup_test,
                       constant_cls, p):
    def forward(m):
        x = constant_cls(m if p == 1 else m ** p)
        return to_float(x) ** (4 / p)

    if complex_mode:
        m = constant_cls(np.cbrt(2.0) + 1.0j * np.cbrt(3.0))
        dm = constant_cls(np.sqrt(0.5) + 1.0j * np.sqrt(0.5))
        dM = tuple(map(constant_cls, (complex(dm), complex(dm).conjugate())))
    else:
        m = constant_cls(np.cbrt(2.0))
        dm = constant_cls(1.0)
        dM = (dm, dm)

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value

    dJ = compute_gradient(J, m)
    assert abs(complex(dJ).conjugate() - 4.0 * (complex(m) ** 3)) < 1.0e-14

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ, dM=dm)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, m, tlm_order=1, dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1, dMs=dM)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2, dMs=dM)
    assert min_order > 2.00


def project_project(F, space, bc):
    if DEFAULT_COMM.size > 1:
        pytest.skip()
    G = Function(space, name="G")
    project(F, space, bcs=bc, function=G, solver_type="lu")
    return G


def project_project_solver_parameters(F, space, bc):
    G = Function(space, name="G")
    project(F, space, bcs=bc, function=G,
            solver_parameters=ls_parameters_cg)
    return G


def project_LUSolver(F, space, bc):
    if DEFAULT_COMM.size > 1:
        pytest.skip()
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    A = assemble(inner(trial, test) * dx)
    b = assemble(inner(F, test) * dx)
    bc.apply(A, b)

    solver = LUSolver(A)
    solver.solve(G.vector(), b)

    return G


def project_assemble_system_KrylovSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    A, b = assemble_system(-0.2 * inner(trial, test) * dx,
                           -0.3 * inner(F, test) * dx)
    A, b = assemble_system(1.2 * inner(trial, test) * dx,
                           1.3 * inner(F, test) * dx,
                           A_tensor=A, b_tensor=b, add_values=True)
    bc.apply(A, b)

    solver = KrylovSolver(A, "gmres", "sor")
    solver.parameters.update({"relative_tolerance": 1.0e-14,
                              "absolute_tolerance": 1.0e-16})
    solver.solve(G.vector(), b)

    return G


def project_assemble_KrylovSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    A = assemble(inner(trial, test) * dx)
    b = assemble(inner(F, test) * dx)
    bc.apply(A, b)

    solver = KrylovSolver(A, "gmres", "sor")
    solver.parameters.update({"relative_tolerance": 1.0e-14,
                              "absolute_tolerance": 1.0e-16})
    solver.solve(G.vector(), b)

    return G


def project_assemble_mult_KrylovSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    A = assemble(inner(trial, test) * dx)
    P = assemble(inner(TrialFunction(F.function_space()), test) * dx)
    b = P * F.vector()
    bc.apply(A, b)

    solver = KrylovSolver(A, "gmres", "sor")
    solver.parameters.update({"relative_tolerance": 1.0e-14,
                              "absolute_tolerance": 1.0e-16})
    solver.solve(G.vector(), b)

    return G


def project_LinearVariationalSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    eq = inner(trial, test) * dx == inner(F, test) * dx
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, G, bcs=bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters.update(copy_parameters(ls_parameters_cg))
    solver.solve()

    return G


def project_NonlinearVariationalSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    eq = inner(G, test) * dx - inner(F, test) * dx
    problem = NonlinearVariationalProblem(eq, G,
                                          J=inner(trial, test) * dx,
                                          bcs=bc)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["nonlinear_solver"] = "newton"
    solver.parameters["symmetric"] = True
    solver.parameters["newton_solver"].update(copy_parameters(ns_parameters_newton_cg))  # noqa: E501
    solver.solve()

    return G


def project_solve_linear(F, space, bc):
    if DEFAULT_COMM.size > 1:
        pytest.skip()
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")
    A, b = assemble_system(inner(trial, test) * dx, inner(F, test) * dx,
                           bcs=bc)
    solve(A, G.vector(), b, "lu")
    return G


def project_solve_variational_problem(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")
    solve(inner(trial, test) * dx == inner(F, test) * dx,
          G, bc, solver_parameters=ls_parameters_cg)
    return G


@pytest.mark.fenics
@pytest.mark.parametrize("project_fn", [project_project,
                                        project_project_solver_parameters,
                                        project_LUSolver,
                                        project_assemble_system_KrylovSolver,
                                        project_assemble_KrylovSolver,
                                        project_assemble_mult_KrylovSolver,
                                        project_LinearVariationalSolver,
                                        project_NonlinearVariationalSolver,
                                        project_solve_linear,
                                        project_solve_variational_problem])
@seed_test
def test_project_patches(setup_test, test_leaks,
                         project_fn):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(FunctionSpace(mesh, "Lagrange", 2), name="F", static=True)
    if complex_mode:
        interpolate_expression(
            F,
            sin(pi * X[0]) * sin(3.0 * pi * X[1])
            + 1.0j * sin(5.0 * pi * X[0]) * sin(7.0 * pi * X[1]))
    else:
        interpolate_expression(
            F,
            sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    bc = DirichletBC(space, 1.0, "on_boundary")

    def forward(F):
        G = project_fn(F, space, bc)

        J = Functional(name="J")
        J.assign(G * G * (Constant(1.0) + G) * dx)
        return G, J

    F_ref = F.copy(deepcopy=True)
    start_manager()
    G, J = forward(F)
    stop_manager()

    error = Function(space, name="error")
    solve(inner(trial, test) * dx == inner(F_ref, test) * dx,
          error, bc, solver_parameters=ls_parameters_cg)
    var_axpy(error, -1.0, G)
    assert var_linf_norm(error) < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, F)

    def forward_J(F):
        _, J = forward(F)
        return J

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
@pytest.mark.parametrize("assign_fn", [lambda x, y: x.assign(y),
                                       lambda x, y: x.interpolate(y)])
@seed_test
def test_Function_assign(setup_test, test_leaks,
                         assign_fn):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        u = Constant(0.0, name="u")
        u.assign(m)
        u.assign(-2.0)
        u.assign(u + 2.0 * m)

        m_ = Function(space, name="m")
        assign_fn(m_, m)
        m = m_
        del m_

        u_ = Function(space, name="u")
        assign_fn(u_, u)
        u = u_
        del u_

        one = Function(space, name="one")
        assign_fn(one, Constant(1.0))

        v = Function(space, name="v")
        assign_fn(v, u)
        v.assign(u + one)
        assign_fn(v, Constant(0.0))
        v.assign(u + v + one)
        v.assign(2.5 * u + 3.6 * v + 4.7 * m)

        J = Functional(name="J")
        J.assign(((v - 1.0) ** 4) * dx)
        return J

    m = Constant(2.0, name="m")

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value
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
                                      assembly_assemble])
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

    G.interpolate(Constant(np.sqrt(2.0)))
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

    G.interpolate(Constant(np.sqrt(2.0)))
    b_ref = backend_assemble(form)
    assert abs(b_ref - np.sqrt(2.0)) < 1.0e-15

    for _ in range(3):
        b = assemble(form)
        assert abs(b - b_ref) == 0.0


@pytest.mark.fenics
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_LUSolver(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    bc = DirichletBC(space, 0.0, "on_boundary")

    def forward(m):
        b = assemble(inner(m, test) * dx)
        bc.apply(b)
        K = assemble(inner(grad(trial), grad(test)) * dx)
        bc.apply(K)

        u = Function(space, name="u")
        K_solver = LUSolver(K)
        K_solver.solve(u, b)

        J = Functional(name="J")
        J.assign(((u - Constant(1.0)) ** 4) * dx)
        return u, J

    m = Function(space, name="m")
    interpolate_expression(m, X[0] * sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    u_ref = Function(space, name="u_ref")
    solve(inner(grad(trial), grad(test)) * dx == inner(m, test) * dx,
          u_ref, bc, solver_parameters=ls_parameters_cg)

    start_manager()
    u, J = forward(m)
    stop_manager()

    u_error_norm = np.sqrt(abs(assemble(inner(u - u_ref, u - u_ref) * dx)))
    info(f"{u_error_norm=}")
    assert u_error_norm < 1.0e-17

    def forward_J(m):
        _, J = forward(m)
        return J

    J_val = J.value

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2)
    assert min_order > 1.99
