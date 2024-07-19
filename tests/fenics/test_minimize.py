from fenics import *
from tlm_adjoint.fenics import *

from .test_base import *

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


def tao_lmvm_minimization(forward, m0):
    return minimize_tao(forward, m0,
                        solver_parameters={"tao_type": "lmvm",
                                           "tao_gatol": 1.0e-11,
                                           "tao_grtol": 0.0,
                                           "tao_gttol": 0.0})


def tao_nls_minimization(forward, m0):
    return minimize_tao(forward, m0,
                        solver_parameters={"tao_type": "nls",
                                           "tao_gatol": 1.0e-11,
                                           "tao_grtol": 0.0,
                                           "tao_gttol": 0.0})


@pytest.mark.fenics
@pytest.mark.parametrize("minimize", [scipy_l_bfgs_b_minimization,
                                      scipy_trust_ncg_minimization,
                                      tao_lmvm_minimization,
                                      pytest.param(tao_nls_minimization, marks=pytest.mark.xfail)])  # noqa: E501
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
            var_assign(x_ref, x)

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
    var_assign(error, alpha_ref)
    var_axpy(error, -1.0, alpha)
    assert var_linf_norm(error) < 1.0e-8


@pytest.mark.fenics
@pytest.mark.parametrize("minimize", [scipy_l_bfgs_b_minimization,
                                      scipy_trust_ncg_minimization,
                                      tao_lmvm_minimization,
                                      pytest.param(tao_nls_minimization, marks=pytest.mark.xfail)])  # noqa: E501
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
            var_assign(x_ref, x)
        if y_ref is None:
            y_ref = Function(space, name="y_ref", static=True)
            var_assign(y_ref, y)

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
    var_assign(error, alpha_ref)
    var_axpy(error, -1.0, alpha)
    assert var_linf_norm(error) < 1.0e-8

    var_assign(error, beta_ref)
    var_axpy(error, -1.0, beta)
    assert var_linf_norm(error) < 1.0e-8
