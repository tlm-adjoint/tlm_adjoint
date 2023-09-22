#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

import petsc4py.PETSc as PETSc
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


def scipy_l_bfgs_b_minimization(forward, M0):
    M, result = minimize_scipy(
        forward, M0, method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1.0e-11})
    assert result.success
    return M


def scipy_trust_ncg_minimization(forward, M0):
    M, result = minimize_scipy(
        forward, M0, method="trust-ncg",
        options={"gtol": 1.0e-11})
    assert result.success
    return M


def l_bfgs_minimization(forward, M0):
    M, _ = minimize_l_bfgs(
        forward, M0, s_atol=0.0, g_atol=1.0e-11)
    return M


def tao_lmvm_minimization(forward, m0):
    return minimize_tao(forward, m0,
                        method=PETSc.TAO.Type.LMVM,
                        gatol=1.0e-11, grtol=0.0, gttol=0.0)


def tao_nls_minimization(forward, m0):
    return minimize_tao(forward, m0,
                        method=PETSc.TAO.Type.NLS,
                        gatol=1.0e-11, grtol=0.0, gttol=0.0)


@pytest.mark.firedrake
@pytest.mark.parametrize("minimize", [scipy_l_bfgs_b_minimization,
                                      scipy_trust_ncg_minimization,
                                      l_bfgs_minimization,
                                      tao_lmvm_minimization,
                                      tao_nls_minimization])
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_minimize_project(setup_test, test_leaks,
                          minimize):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(alpha, x_ref=None):
        x = Function(space, name="x")
        solve(inner(trial, test) * dx == inner(alpha, test) * dx,
              x, solver_parameters=ls_parameters_cg)

        if x_ref is None:
            x_ref = Function(space, name="x_ref", static=True)
            function_assign(x_ref, x)

        J = Functional(name="J")
        J.assign(inner(x - x_ref, x - x_ref) * dx)
        return x_ref, J

    alpha_ref = Function(space, name="alpha_ref", static=True)
    interpolate_expression(alpha_ref, exp(X[0] + X[1]))
    x_ref, _ = forward(alpha_ref)

    alpha0 = Function(space, name="alpha0", static=True)

    def forward_J(alpha):
        return forward(alpha, x_ref=x_ref)[1]

    alpha = minimize(forward_J, alpha0)

    error = Function(space, name="error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    assert function_linf_norm(error) < 1.0e-8


@pytest.mark.firedrake
@pytest.mark.parametrize("minimize", [scipy_l_bfgs_b_minimization,
                                      scipy_trust_ncg_minimization,
                                      l_bfgs_minimization])
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_minimize_project_multiple(setup_test, test_leaks,
                                   minimize):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(alpha, beta, x_ref=None, y_ref=None):
        x = Function(space, name="x")
        solve(inner(trial, test) * dx == inner(alpha, test) * dx,
              x, solver_parameters=ls_parameters_cg)

        y = Function(space, name="y")
        solve(inner(trial, test) * dx == inner(beta, test) * dx,
              y, solver_parameters=ls_parameters_cg)

        if x_ref is None:
            x_ref = Function(space, name="x_ref", static=True)
            function_assign(x_ref, x)
        if y_ref is None:
            y_ref = Function(space, name="y_ref", static=True)
            function_assign(y_ref, y)

        J = Functional(name="J")
        J.assign(inner(x - x_ref, x - x_ref) * dx)
        J.addto(inner(y - y_ref, y - y_ref) * dx)
        return x_ref, y_ref, J

    alpha_ref = Function(space, name="alpha_ref", static=True)
    interpolate_expression(alpha_ref, exp(X[0] + X[1]))
    beta_ref = Function(space, name="beta_ref", static=True)
    interpolate_expression(beta_ref, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    x_ref, y_ref, _ = forward(alpha_ref, beta_ref)

    alpha0 = Function(space, name="alpha0", static=True)
    beta0 = Function(space, name="beta0", static=True)

    def forward_J(alpha, beta):
        return forward(alpha, beta, x_ref=x_ref, y_ref=y_ref)[2]

    (alpha, beta) = minimize(forward_J, (alpha0, beta0))

    error = Function(space, name="error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    assert function_linf_norm(error) < 1.0e-8

    function_assign(error, beta_ref)
    function_axpy(error, -1.0, beta)
    assert function_linf_norm(error) < 1.0e-9


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_l_bfgs_single(setup_test, test_leaks):
    mesh = UnitSquareMesh(3, 3)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    M_l = Cofunction(space.dual(), name="M_l")
    assemble(test * dx, tensor=M_l)

    x_star = Function(space, name="x_star")
    interpolate_expression(x_star, sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    def F(x):
        check_space_type(x, "primal")
        return assemble(0.5 * inner(x - x_star, x - x_star) * dx)

    def Fp(x):
        check_space_type(x, "primal")
        Fp = Cofunction(space.dual(), name="Fp")
        assemble(inner(x - x_star, test) * dx, tensor=Fp)
        return Fp

    def H_0_action(x):
        check_space_type(x, "conjugate_dual")
        H_0_action = Function(space, name="H_0_action")
        function_set_values(H_0_action,
                            function_get_values(x)
                            / function_get_values(M_l))
        return H_0_action

    def B_0_action(x):
        check_space_type(x, "primal")
        B_0_action = Cofunction(space.dual(), name="B_0_action")
        function_set_values(B_0_action,
                            function_get_values(x)
                            * function_get_values(M_l))
        return B_0_action

    x0 = Function(space, name="x0")
    x, (its, F_calls, Fp_calls, _) = l_bfgs(
        F, Fp, x0, m=30, s_atol=0.0, g_atol=1.0e-12,
        H_0_action=H_0_action, M_action=B_0_action, M_inv_action=H_0_action)

    error = function_copy(x, name="error")
    function_axpy(error, -1.0, x_star)
    error_norm = function_linf_norm(error)
    info(f"{error_norm=:.6e}")
    info(f"{F_calls=:d}")
    info(f"{Fp_calls=:d}")

    assert abs(F(x)) < 1.0e-25
    assert error_norm < 1.0e-12
    assert its == 15
    assert F_calls == 17
    assert Fp_calls == 17


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_l_bfgs_multiple(setup_test, test_leaks):
    mesh = UnitSquareMesh(3, 3)
    X = SpatialCoordinate(mesh)
    space_x = FunctionSpace(mesh, "Lagrange", 1)
    space_y = FunctionSpace(mesh, "Discontinuous Lagrange", 1)
    test_x = TestFunction(space_x)
    test_y = TestFunction(space_y)
    M_l_x = Cofunction(space_x.dual(), name="M_l_x")
    M_l_y = Cofunction(space_y.dual(), name="M_l_y")
    assemble(test_x * dx, tensor=M_l_x)
    assemble(test_y * dx, tensor=M_l_y)

    x_star = Function(space_x, name="x_star")
    interpolate_expression(x_star, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    y_star = Function(space_y, name="y_star")
    interpolate_expression(y_star, exp(X[0]) * exp(X[1]))
    alpha_y = (1.0 + X[0]) * (1.0 + X[0])

    def F(x, y):
        check_space_type(x, "primal")
        check_space_type(y, "primal")
        return assemble(0.5 * inner(x - x_star, x - x_star) * dx
                        + 0.5 * inner(y - y_star, alpha_y * (y - y_star)) * dx)

    def Fp(x, y):
        check_space_type(x, "primal")
        check_space_type(y, "primal")
        Fp = (Cofunction(space_x.dual(), name="Fp_0"),
              Cofunction(space_y.dual(), name="Fp_1"))
        assemble(inner(x - x_star, test_x) * dx, tensor=Fp[0])
        assemble(inner(alpha_y * (y - y_star), test_y) * dx, tensor=Fp[1])
        return Fp

    def H_0_action(x, y):
        check_space_type(x, "conjugate_dual")
        check_space_type(y, "conjugate_dual")
        H_0_action = (Function(space_x, name="H_0_action_0"),
                      Function(space_y, name="H_0_action_1"))
        function_set_values(H_0_action[0],
                            function_get_values(x)
                            / function_get_values(M_l_x))
        function_set_values(H_0_action[1],
                            function_get_values(y)
                            / function_get_values(M_l_y))
        return H_0_action

    def B_0_action(x, y):
        check_space_type(x, "primal")
        check_space_type(y, "primal")
        B_0_action = (Cofunction(space_x.dual(), name="B_0_action_0"),
                      Cofunction(space_y.dual(), name="B_0_action_1"))
        function_set_values(B_0_action[0],
                            function_get_values(x)
                            * function_get_values(M_l_x))
        function_set_values(B_0_action[1],
                            function_get_values(y)
                            * function_get_values(M_l_y))
        return B_0_action

    x0 = Function(space_x, name="x0")
    y0 = Function(space_y, name="y0")
    (x, y), (its, F_calls, Fp_calls, _) = \
        l_bfgs(F, Fp, (x0, y0), m=30, s_atol=0.0, g_atol=1.0e-12,
               H_0_action=H_0_action,
               M_action=B_0_action, M_inv_action=H_0_action)

    x_error = function_copy(x, name="x_error")
    function_axpy(x_error, -1.0, x_star)
    x_error_norm = function_linf_norm(x_error)
    y_error = function_copy(y, name="y_error")
    function_axpy(y_error, -1.0, y_star)
    y_error_norm = function_linf_norm(y_error)
    info(f"{x_error_norm=:.6e}")
    info(f"{y_error_norm=:.6e}")
    info(f"{F_calls=:d}")
    info(f"{Fp_calls=:d}")

    assert abs(F(x, y)) < 1.0e-24
    assert x_error_norm < 1.0e-12
    assert y_error_norm < 1.0e-11
    assert its <= 38
    assert F_calls <= 42
    assert Fp_calls <= 42
