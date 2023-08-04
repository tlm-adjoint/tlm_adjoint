#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_code_generator_interface import \
    function_vector, matrix_multiply

from .test_base import *

import numpy as np
import petsc4py.PETSc as PETSc
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@no_space_type_checking
@seed_test
def test_HEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    M = assemble(inner(trial, test) * dx)

    def M_action(x):
        y = function_new_conjugate_dual(x)
        assemble(inner(x, test) * dx, tensor=function_vector(y))
        return y

    import slepc4py.SLEPc as SLEPc
    lam, V = eigendecompose(
        space, M_action, action_space_type="conjugate_dual",
        problem_type=SLEPc.EPS.ProblemType.HEP)

    assert (lam > 0.0).all()

    diff = Function(space)
    assert len(lam) == len(V)
    for lam_val, v in zip(lam, V):
        matrix_multiply(M, function_vector(v),
                        tensor=function_vector(diff))
        function_axpy(diff, -lam_val, v)
        assert function_linf_norm(diff) < 1.0e-16


@pytest.mark.firedrake
@no_space_type_checking
@seed_test
def test_NHEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    N = assemble(inner(trial.dx(0), test) * dx)

    def N_action(x):
        y = function_new_conjugate_dual(x)
        assemble(inner(x.dx(0), test) * dx, tensor=function_vector(y))
        return y

    lam, V = eigendecompose(
        space, N_action, action_space_type="conjugate_dual",)

    assert abs(lam.real).max() < 1.0e-15

    diff = Function(space)
    if issubclass(PETSc.ScalarType, (complex, np.complexfloating)):
        assert len(lam) == len(V)
        for lam_val, v in zip(lam, V):
            matrix_multiply(N, function_vector(v),
                            tensor=function_vector(diff))
            function_axpy(diff, -lam_val, v)
            assert function_linf_norm(diff) < 1.0e-14
    else:
        V_r, V_i = V
        assert len(lam) == len(V_r)
        assert len(lam) == len(V_i)
        for lam_val, v_r, v_i in zip(lam, V_r, V_i):
            matrix_multiply(N, function_vector(v_r),
                            tensor=function_vector(diff))
            function_axpy(diff, -lam_val.real, v_r)
            function_axpy(diff, +lam_val.imag, v_i)
            assert function_linf_norm(diff) < 1.0e-14
            matrix_multiply(N, function_vector(v_i),
                            tensor=function_vector(diff))
            function_axpy(diff, -lam_val.real, v_i)
            function_axpy(diff, -lam_val.imag, v_r)
            assert function_linf_norm(diff) < 1.0e-14


@pytest.mark.firedrake
@seed_test
def test_CachedHessian(setup_test):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitIntervalMesh(5)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    zero = Constant(0.0, name="zero")

    def forward(F):
        y = Function(space, name="y")
        EquationSolver(
            inner(grad(trial), grad(test)) * dx
            == inner(F, test) * dx + inner(zero * sin(F), test) * dx,
            y, HomogeneousDirichletBC(space, "on_boundary"),
            solver_parameters=ls_parameters_cg).solve()

        J = Functional(name="J")
        J.addto((dot(y, y) ** 2) * dx)
        return J

    F = Function(space, name="F", static=True)
    function_assign(F, 1.0)

    start_manager()
    J = forward(F)
    stop_manager()

    H = Hessian(forward)
    H_opt = CachedHessian(J)

    # Test consistency of matrix action for static direction

    zeta = Function(space, name="zeta", static=True)
    for i in range(5):
        zeta_arr = np.random.random(function_local_size(zeta))
        if issubclass(function_dtype(zeta), (complex, np.complexfloating)):
            zeta_arr = zeta_arr \
                + 1.0j * np.random.random(function_local_size(zeta))
        function_set_values(zeta, zeta_arr)
        del zeta_arr

        # Leads to an inconsistency if the stored value is not used
        zero.assign(np.NAN)
        _, _, ddJ_opt = H_opt.action(F, zeta)
        zero.assign(0.0)
        _, _, ddJ = H.action(F, zeta)

        error = function_copy(ddJ)
        function_axpy(error, -1.0, ddJ_opt)
        assert function_linf_norm(error) == 0.0

    # Test consistency of eigenvalues

    with paused_space_type_checking():
        lam, _ = eigendecompose(space, H.action_fn(F))
    if not issubclass(space_dtype(space), (complex, np.complexfloating)):
        assert max(abs(lam.imag)) == 0.0

    with paused_space_type_checking():
        lam_opt, _ = eigendecompose(space, H_opt.action_fn(F))
    if not issubclass(space_dtype(space), (complex, np.complexfloating)):
        assert max(abs(lam_opt.imag)) == 0.0

    error = (np.array(sorted(lam.real), dtype=np.float64)
             - np.array(sorted(lam_opt.real), dtype=np.float64))
    assert abs(error).max() == 0.0

    if issubclass(space_dtype(space), (complex, np.complexfloating)):
        # Minor gap in test, as could use a different order from the real
        # components
        error = (np.array(sorted(lam.imag), dtype=np.float64)
                 - np.array(sorted(lam_opt.imag), dtype=np.float64))
        assert abs(error).max() == 0.0
