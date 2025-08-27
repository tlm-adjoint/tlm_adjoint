from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_interface import assemble_linear_solver

from .test_base import *

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.parametrize("N_eigenvalues", [0, 5, 16])
def test_hessian_solve(setup_test,
                       N_eigenvalues):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitSquareMesh(5, 5)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    bc = DirichletBC(space, 0.0, "on_boundary")

    alpha = Constant(1.0 / np.sqrt(2.0))
    beta = Constant(1.0 / np.sqrt(5.0))

    def B_inv(u):
        b = Cofunction(space.dual())
        assemble(beta * inner(u, test) * dx, tensor=b)
        return b

    def B(b):
        u = Function(space)
        solver, _, _ = assemble_linear_solver(
            beta * inner(trial, test) * dx, bcs=bc,
            linear_solver_parameters=ls_parameters_cg)
        solver.solve(u, var_copy(b))
        return u

    def forward(u_ref, m):
        m_1 = Function(space, name="m_1")
        DirichletBC(space, m, "on_boundary").apply(m_1)
        m_0 = Function(space, name="m_0").assign(m - m_1)
        m = m_0
        del m_0, m_1
        assert np.sqrt(abs(assemble(inner(m, m) * ds))) == 0.0

        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == inner(m + m * m, test) * dx,  # noqa: E501
              u, bc,
              solver_parameters=ls_parameters_cg)

        J_mismatch = Functional(name="J")
        J_mismatch.assign(0.5 * alpha * dot(u - u_ref, u - u_ref) * dx)

        J = Functional(name="J")
        J.assign(J_mismatch)
        J.addto(0.5 * beta * dot(m, m) * dx)

        return u, J, J_mismatch

    def forward_J(m):
        _, J, _ = forward(u_ref, m)
        return J

    u_ref = Function(space, name="u_ref")
    interpolate_expression(
        u_ref,
        sin(2.0 * pi * X[0]) * sin(3.0 * pi * X[1]) * exp(4.0 * X[0] * X[1]))

    m0 = Function(space, name="m0")
    m = minimize_tao(
        forward_J, m0,
        solver_parameters={"tao_type": "lmvm",
                           "tao_gatol": 1.0e-7,
                           "tao_grtol": 0.0,
                           "tao_gttol": 0.0},
        H_0_action=B)

    b_ref = Cofunction(space.dual(), name="b_ref")
    assemble(inner((sin(5.0 * pi * X[0]) * sin(7.0 * pi * X[1])) ** 2, test) * dx,  # noqa: E501
             tensor=b_ref)
    bc.apply(l2_riesz(b_ref, alias=True))

    start_manager()
    _, J, J_mismatch = forward(u_ref, m)
    stop_manager()
    H = CachedHessian(J)
    H_mismatch = CachedHessian(J_mismatch)
    nullspace = DirichletBCNullspace(bc)

    v = Function(space, name="v")

    if N_eigenvalues == 0:
        pc_fn = None
    else:
        try:
            import slepc4py.SLEPc as SLEPc    # noqa: F401
        except ModuleNotFoundError:
            pytest.skip(reason="SLEPc not available")

        esolver = HessianEigensolver(
            H_mismatch, m, B_inv, B, nullspace=nullspace,
            solver_parameters={"eps_type": "krylovschur",
                               "eps_gen_hermitian": None,
                               "eps_largest_magnitude": None,
                               "eps_nev": N_eigenvalues,
                               "eps_conv_rel": None,
                               "eps_tol": 1.0e-14,
                               "eps_purify": False})
        esolver.solve()
        assert len(esolver) >= N_eigenvalues
        assert esolver.B_orthonormality_test() < 1.0e-14

        Lam, V = esolver.eigenpairs()
        assert issubclass(Lam.dtype.type, np.floating)
        V = tuple(v_r for v_r, _ in V)

        assert len(Lam) == len(V)
        for lam_i, v_i in zip(Lam, V):
            _, _, v_error = H_mismatch.action(m, v_i)
            var_axpy(v_error, -lam_i, B_inv(v_i))
            bc.apply(l2_riesz(v_error, alias=True))
            assert var_linf_norm(v_error) < 1.0e-16

        pc_fn = esolver.spectral_pc_fn()

    H_solver = HessianLinearSolver(
        H, m,
        solver_parameters={"ksp_type": "cg",
                           "ksp_atol": 1.0e-12,
                           "ksp_rtol": 1.0e-12},
        pc_fn=pc_fn,
        nullspace=nullspace)
    H_solver.solve(
        v, b_ref)
    ksp_its = H_solver.ksp.getIterationNumber()

    if N_eigenvalues == 0:
        assert ksp_its <= 14
    elif N_eigenvalues == 5:
        assert ksp_its <= 6
    elif N_eigenvalues == 16:
        assert ksp_its == 1

    H = Hessian(forward_J)
    _, _, b = H.action(m, v)
    assert var_linf_norm(b) > 0.0
    b_error = var_copy(b, name="b_error")
    var_axpy(b_error, -1.0, b_ref)
    assert var_linf_norm(b_error) < 1.0e-13
