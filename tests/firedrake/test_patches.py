from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend import backend_assemble, backend_Constant
from tlm_adjoint.firedrake.assembly import assemble as assembly_assemble

from .test_base import *

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
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
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2, dMs=dM)
    assert min_order > 1.99


def project_project(F, space, bc):
    return project(F, space, bcs=bc, name="G",
                   solver_parameters=ls_parameters_cg)


def project_assemble_LinearSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    A = assemble(inner(trial, test) * dx, bcs=bc)
    b = assemble(inner(F, test) * dx)

    solver = LinearSolver(A, solver_parameters=ls_parameters_cg)
    solver.solve(G, b)

    return G


def project_LinearVariationalSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    eq = inner(trial, test) * dx == inner(F, test) * dx
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, G, bcs=bc)
    solver = LinearVariationalSolver(
        problem, solver_parameters=ls_parameters_cg)
    solver.solve()

    return G


def project_NonlinearVariationalSolver(F, space, bc):
    test, trial = TestFunction(space), TrialFunction(space)
    G = Function(space, name="G")

    eq = inner(G, test) * dx - inner(F, test) * dx
    problem = NonlinearVariationalProblem(eq, G,
                                          J=inner(trial, test) * dx,
                                          bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=ns_parameters_newton_cg)
    solver.solve()

    return G


@pytest.mark.firedrake
@pytest.mark.parametrize("project_fn", [project_project,
                                        project_assemble_LinearSolver,
                                        project_LinearVariationalSolver,
                                        project_NonlinearVariationalSolver])
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


@pytest.mark.firedrake
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

        v = Function(space, name="v")
        assign_fn(v, u)
        assign_fn(v, u + Constant(1.0))
        v.assign(0.0)
        assign_fn(v, u + v + Constant(1.0))
        assign_fn(v, 2.5 * u + 3.6 * v + 4.7 * m)

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


def subset_assign_assign(x, bc):
    x.assign(bc.function_arg, subset=bc.node_set)


def subset_assign_apply(x, bc):
    bc.apply(x)


@pytest.mark.firedrake
@pytest.mark.parametrize("subset_assign", [subset_assign_assign,
                                           subset_assign_apply])
@seed_test
def test_Function_assign_subset(setup_test, test_leaks,
                                subset_assign):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        # The following is a questionable forward calculation (we should not
        # usually set just the boundary dofs and then compute domain integrals)
        # but for this test we need the functional to depend on the interior
        # degrees of freedom
        m_0 = Function(space, name="m_0")
        subset_assign(m_0, DirichletBC(space, m, "on_boundary"))
        J = Functional(name="J")
        J.assign(((m_0 - Constant(1.0)) ** 4) * dx)
        return J

    m = Function(space, name="m")
    if complex_mode:
        interpolate_expression(m, sin(pi * X[0]) + 1.0j * cos(pi * X[0]))
    else:
        interpolate_expression(m, sin(pi * X[0]))

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, seed=1.0e-2)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, m, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Function_in_place(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m_0, m_1, c_0, c_1=None):
        u = Function(space, name="u")
        u += m_0
        u -= 0.5 * m_1
        u *= c_0
        if c_1 is not None:
            # Complex mode verification failure due to Firedrake issue #2376
            u /= c_1

        J = Functional(name="J")
        J.assign(((u - Constant(1.0)) ** 4) * dx)
        return u, J

    c_0 = Constant(np.sqrt(2.0))
    c_1 = Constant(np.sqrt(5.0))
    m_0 = Function(space, name="m_0")
    m_1 = Function(space, name="m_1")
    if complex_mode:
        interpolate_expression(m_0, cos(pi * X[0]) + 1.0j * cos(2 * pi * X[0]))
        interpolate_expression(m_1, -exp(X[0]) + 1.0j * exp(2 * X[0]))
    else:
        interpolate_expression(m_0, cos(pi * X[0]))
        interpolate_expression(m_1, -exp(X[0]))

    u_ref = Function(space, name="u")
    if complex_mode:
        M = (m_0, m_1, c_0)
        interpolate_expression(u_ref, c_0 * (m_0 - 0.5 * m_1))
    else:
        M = (m_0, m_1, c_0, c_1)
        interpolate_expression(u_ref, (c_0 / c_1) * (m_0 - 0.5 * m_1))

    start_manager()
    u, J = forward(*M)
    stop_manager()

    error_norm = np.sqrt(abs(assemble(inner(u - u_ref, u - u_ref) * dx)))
    assert error_norm < 1.0e-16

    J_val = J.value

    dJ = compute_gradient(J, M)

    def forward_J(*M):
        _, J = forward(*M)
        return J

    min_order = taylor_test(forward_J, M, J_val=J_val, dJ=dJ, seed=1.0e-4)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, M, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, M, tlm_order=1, seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=1,
                                        seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=2,
                                        seed=1.0e-4)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("riesz_map, riesz_map_ref",
                         [("L2", lambda u, test: inner(u, test) * dx),
                          ("H1", lambda u, test: inner(u, test) * dx + inner(grad(u), grad(test)) * dx)])  # noqa: E501
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_Function_riesz_representation(setup_test, test_leaks,
                                       riesz_map, riesz_map_ref):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        u = m.riesz_representation(riesz_map)
        v = Function(space).interpolate(m - Constant(1.0))

        J = Functional(name="J")
        J.assign(u(v))
        return u, J ** 2

    m = Function(space, name="m")
    interpolate_expression(m, Constant(1.5) - exp(X[0] * X[1]))
    m_ref = var_copy(m)

    start_manager()
    u, J = forward(m)
    stop_manager()

    m_u = Function(space, name="m_u")
    solve(riesz_map_ref(trial, test) == u, m_u,
          solver_parameters=ls_parameters_cg)
    m_error = var_copy(m_ref)
    var_axpy(m_error, -1.0, m_u)
    assert var_linf_norm(m_error) < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, m)

    def forward_J(m):
        _, J = forward(m)
        return J

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, seed=1.0e-4)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, seed=1.0e-4)
    assert min_order > 2.97

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1, seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1,
                                        seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2,
                                        seed=1.0e-4)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Cofunction_in_place(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    if complex_mode:
        c = 1.5 - 0.5j
    else:
        c = 1.5

    def forward(m_0, m_1):
        b = Cofunction(space.dual(), name="b")
        b += assemble(inner(m_0, test) * dx)
        b -= assemble(inner(m_1, test) * dx)
        b *= c

        u = Function(space, name="u")
        solve(inner(trial, test) * dx == b, u,
              solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        J.assign(((u - Constant(1.0)) ** 4) * dx)
        return u, J

    m_0 = Function(space, name="m_0")
    m_1 = Function(space, name="m_1")
    if complex_mode:
        interpolate_expression(m_0, cos(pi * X[0]) + 1.0j * cos(2 * pi * X[0]))
        interpolate_expression(m_1, -exp(X[0]) + 1.0j * exp(2 * X[0]))
    else:
        interpolate_expression(m_0, cos(pi * X[0]))
        interpolate_expression(m_1, -exp(X[0]))
    M = (m_0, m_1)

    u_ref = Function(space, name="u")
    interpolate_expression(u_ref, c * (m_0 - m_1))

    start_manager()
    u, J = forward(*M)
    stop_manager()

    error_norm = np.sqrt(abs(assemble(inner(u - u_ref, u - u_ref) * dx)))
    assert error_norm < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, M)

    def forward_J(*M):
        _, J = forward(*M)
        return J

    min_order = taylor_test(forward_J, M, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, M, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, M, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Cofunction_assign(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        b = assemble(inner(m, test) * dx)

        l0 = Cofunction(space.dual(), name="l0")
        l0.assign(b)
        l0_error = var_copy(l0)
        var_axpy(l0_error, -1.0, b)
        assert var_linf_norm(l0_error) == 0.0
        l0.assign(0)
        assert var_linf_norm(l0) == 0.0

        l1 = Cofunction(space.dual(), name="l1")
        l1.assign(b)
        l1_error = var_copy(l1)
        var_axpy(l1_error, -1.0, b)
        assert var_linf_norm(l1_error) == 0.0

        l2 = Cofunction(space.dual(), name="l2")
        l2.assign(l1)
        l2_error = var_copy(l2)
        var_axpy(l2_error, -1.0, b)
        assert var_linf_norm(l2_error) == 0.0

        l3 = Cofunction(space.dual(), name="l3")
        l3.assign(0.7 * l1 - 0.2 * l2)
        l3_error = var_copy(l3)
        var_axpy(l3_error, -0.5, b)
        assert var_linf_norm(l3_error) < 1.0e-16

        l4 = Cofunction(space.dual(), name="l4")
        l4.assign(b)
        l4.assign(l0 + 2.0 * l3 - 0.5 * l4)
        l4_error = var_copy(l4)
        var_axpy(l4_error, -0.5, b)
        assert var_linf_norm(l4_error) < 1.0e-16

        v = Function(space, name="v")
        solve(inner(trial, test) * dx == l4, v,
              solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        J.assign(((v - Constant(1.0)) ** 3) * dx)
        return J

    m = Function(space, name="m")
    m.interpolate(Constant(-1.5) + exp(X[0]))
    J_ref = assemble(((0.5 * m - Constant(1.0)) ** 3) * dx)

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value
    assert abs(J_val - J_ref) < 1.0e-15

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, m, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("riesz_map, riesz_map_ref",
                         [("L2", lambda u, test: inner(u, test) * dx),
                          ("H1", lambda u, test: inner(u, test) * dx + inner(grad(u), grad(test)) * dx)])  # noqa: E501
@seed_test
def test_Cofunction_riesz_representation(setup_test, test_leaks,
                                         riesz_map, riesz_map_ref):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    def forward(m):
        u = m.riesz_representation(riesz_map,
                                   solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        J.assign(((u - Constant(1.0)) ** 4) * dx)
        return u, J

    u_ref = Function(space, name="u_ref")
    interpolate_expression(
        u_ref, Constant(1.5 + (1.0j if complex_mode else 0.0)) - exp(X[0] * X[1]))  # noqa: E501
    m = assemble(riesz_map_ref(u_ref, test))

    start_manager()
    u, J = forward(m)
    stop_manager()

    u_error = var_copy(u_ref)
    var_axpy(u_error, -1.0, u)
    assert var_linf_norm(u_error) < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, m)

    def forward_J(m):
        _, J = forward(m)
        return J

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, seed=1.0e-4)
    assert min_order > 1.98

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, seed=1.0e-4)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1, seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1,
                                        seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2,
                                        seed=1.0e-4)
    assert min_order > 1.99


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
              nullspace=VectorSpaceBasis(constant=True, comm=psi.comm),
              transpose_nullspace=VectorSpaceBasis(constant=True, comm=psi.comm))  # noqa: E501

        J = Functional(name="J")
        J.assign((dot(psi, psi) ** 2) * dx
                 + dot(grad(psi), grad(psi)) * dx)

        return psi, J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sqrt(sin(pi * X[1])))

    start_manager()
    psi, J = forward(F)
    stop_manager()

    with psi.dat.vec_ro as psi_v:
        psi_sum = psi_v.sum()
    assert abs(psi_sum) < 1.0e-15

    J_val = J.value

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


@pytest.mark.firedrake
@pytest.mark.parametrize("degree", [1, 2, 3])
@seed_test
def test_interpolate(setup_test, test_leaks,
                     interpolate_expr, degree):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", degree)

    y_2 = Function(space_2, name="y_2")
    if complex_mode:
        interpolate_expression(y_2,
                               cos(3.0 * pi * X[0])
                               + 1.0j * sin(5.0 * pi * X[0]))
    else:
        interpolate_expression(y_2,
                               cos(3.0 * pi * X[0]))
    y_1_ref = Function(space_1, name="y_1_ref")
    y_1_ref.interpolate(y_2)

    def forward(y_2):
        y_1 = interpolate_expr(y_2, space_1)

        J = Functional(name="J")
        J.assign(((y_1 - Constant(1.0)) ** 4) * dx)
        return y_1, J

    reset_manager("memory", {"drop_references": True})
    start_manager()
    y_1, J = forward(y_2)
    stop_manager()

    y_1_error = var_copy(y_1)
    var_axpy(y_1_error, -1.0, y_1_ref)
    assert var_linf_norm(y_1_error) < 1.0e-14

    J_val = J.value

    dJ = compute_gradient(J, y_2)

    def forward_J(y_2):
        _, J = forward(y_2)
        return J

    min_order = taylor_test(forward_J, y_2, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, y_2, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, y_2, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y_2, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y_2, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_assemble_arity_1(setup_test, test_leaks,
                          assemble_rhs, test_rhs, assemble_action):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    def forward(F):
        x = Cofunction(space.dual(), name="x")
        assemble_rhs(x, test_rhs(F ** 3, test))

        J = Functional(name="J")
        assemble_action(J, x, F)
        return J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, X[0] * sin(pi * X[1]))

    start_manager()
    J = forward(F)
    stop_manager()

    J_val = J.value
    assert abs(J_val - assemble((F ** 4) * dx)) < 1.0e-16

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


@pytest.mark.firedrake
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
        assert abs(b - b_ref) < 1.0e-15

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
        assert abs(b - b_ref) < 1.0e-15


@pytest.mark.firedrake
@seed_test
def test_DirichletBC_apply(setup_test, test_leaks, tmp_path):
    configure_checkpointing("periodic_disk",
                            {"period": 1,
                             "path": str(tmp_path / "checkpoints~")})

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 2)

    def forward(y):
        x = Function(space, name="x")
        x.assign(2 * y)
        DirichletBC(space, y, "on_boundary").apply(x)

        y_bc = Function(space, name="y_bc")
        DirichletBC(space, y, "on_boundary").apply(y_bc)

        assert np.sqrt(abs(assemble(inner(x + y_bc - 2 * y,
                                          x + y_bc - 2 * y) * dx))) < 1.0e-15
        assert np.sqrt(abs(assemble(inner(x - y, x - y) * ds))) < 1.0e-15
        assert np.sqrt(abs(assemble(inner(y_bc - y, y_bc - y) * ds))) < 1.0e-15

        J = Functional(name="J")
        J.assign(((x - Constant(1.0)) ** 3) * ds
                 + ((x + y_bc) ** 3) * dx)
        return J

    y = Function(space, name="y")
    if complex_mode:
        interpolate_expression(
            y,
            cos(pi * X[0]) * cos(2.0 * pi * X[1])
            + 1.0j * cos(3.0 * pi * X[0]) * cos(4.0 * pi * X[1]))
    else:
        interpolate_expression(
            y,
            cos(pi * X[0]) * cos(2.0 * pi * X[1]))

    start_manager()
    J = forward(y)
    stop_manager()

    J_val = J.value

    dJ = compute_gradient(J, y)

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, y, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2)
    assert min_order > 1.99
