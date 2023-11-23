#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_code_generator_interface import (
    assemble_linear_solver)

from .test_base import *

import numpy as np
import pytest
import ufl

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
        assemble(ufl.conj(beta) * inner(ufl.conj(u), test) * dx, tensor=b)
        return b

    def B(b):
        u = Function(space)
        solver, _, _ = assemble_linear_solver(
            ufl.conj(beta) * inner(ufl.conj(trial), test) * dx, bcs=bc,
            linear_solver_parameters=ls_parameters_cg)
        solver.solve(u, var_copy(b))
        return u

    def forward(u_ref, m):
        m_1 = Function(space, name="m_1")
        DirichletBCApplication(m_1, m, "on_boundary").solve()
        m_0 = Function(space, name="m_0")
        LinearCombination(m_0, (1.0, m), (-1.0, m_1)).solve()
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
    m, _ = minimize_l_bfgs(
        forward_J, m0,
        s_atol=0.0, g_atol=1.0e-9,
        H_0_action=B, M_action=B_inv, M_inv_action=B)

    b_ref = Cofunction(space.dual(), name="b_ref")
    assemble(inner((sin(5.0 * pi * X[0]) * sin(7.0 * pi * X[1])) ** 2, test) * dx,  # noqa: E501
             tensor=b_ref)
    bc.apply(b_ref)

    start_manager()
    _, J, J_mismatch = forward(u_ref, m)
    stop_manager()
    H = CachedHessian(J)
    H_mismatch = CachedHessian(J_mismatch)
    nullspace = DirichletBCNullspace(bc)

    v = Function(space, name="v")
    system = HessianSystem(H, m, nullspace=nullspace)

    if N_eigenvalues == 0:
        pc_fn = None
    else:
        try:
            import slepc4py.SLEPc as SLEPc
        except ImportError:
            pytest.skip(reason="SLEPc not available")

        Lam, V = hessian_eigendecompose(
            H_mismatch, m, B_inv, B, nullspace=nullspace,
            N_eigenvalues=N_eigenvalues,
            solver_type=SLEPc.EPS.Type.KRYLOVSCHUR,
            which=SLEPc.EPS.Which.LARGEST_MAGNITUDE,
            tolerance=1.0e-14)

        assert issubclass(Lam.dtype.type, np.floating)

        assert len(Lam) == len(V)
        for lam_i, v_i in zip(Lam, V):
            _, _, v_error = H_mismatch.action(m, v_i)
            var_axpy(v_error, -lam_i, B_inv(v_i))
            bc.apply(v_error)
            assert var_linf_norm(v_error) < 1.0e-16

        diag_error_norm, off_diag_error_norm = B_inv_orthonormality_test(V, B_inv)  # noqa: E501
        assert diag_error_norm < 1.0e-14
        assert off_diag_error_norm < 1.0e-14

        pc_fn = hessian_eigendecomposition_pc(B, Lam, V)

    ksp_its = system.solve(
        v, b_ref, pc_fn=pc_fn,
        solver_parameters={"linear_solver": "cg",
                           "absolute_tolerance": 1.0e-12,
                           "relative_tolerance": 1.0e-12})

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
