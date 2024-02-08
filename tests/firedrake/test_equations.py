from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_code_generator_interface import (
    assemble_linear_solver)
from tlm_adjoint.firedrake.expr import extract_coefficients

from .test_base import *

import firedrake
import functools
import numpy as np
import os
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_Assignment(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(x):
        y = Function(x.function_space())
        Assignment(y, x).solve()

        J = Functional(name="J")
        J.assign(((y - Constant(1.0)) ** 4) * dx)
        return y, J

    x = Function(space, name="x")
    if complex_mode:
        interpolate_expression(
            x,
            sin(pi * X[0]) * sin(3.0 * pi * X[1])
            + 1.0j * sin(5.0 * pi * X[0]) * sin(7.0 * pi * X[1]))
    else:
        interpolate_expression(
            x,
            sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    start_manager()
    y, J = forward(x)
    stop_manager()
    assert var_state(x) == 1

    y_error = var_copy(y)
    var_axpy(y_error, -1.0, x)
    assert var_linf_norm(y_error) == 0.0

    J_val = J.value

    dJ = compute_gradient(J, x)

    def forward_J(x):
        _, J = forward(x)
        return J

    min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, x, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("c", [3.0, -3.0])
@seed_test
def test_Axpy(setup_test, test_leaks,
              c):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(x):
        y_0 = Function(x.function_space())
        Assignment(y_0, x).solve()

        y_1 = Function(x.function_space())
        y_1.interpolate(Constant(2.0))

        y_2 = Function(x.function_space())
        Axpy(y_2, y_0, c, y_1).solve()

        J = Functional(name="J")
        J.assign(((y_2 - Constant(1.0)) ** 4) * dx)
        return y_2, J

    x = Function(space, name="x")
    if complex_mode:
        interpolate_expression(
            x,
            sin(pi * X[0]) * sin(3.0 * pi * X[1])
            + 1.0j * sin(5.0 * pi * X[0]) * sin(7.0 * pi * X[1]))
    else:
        interpolate_expression(
            x,
            sin(pi * X[0]) * sin(3.0 * pi * X[1]))

    start_manager()
    y, J = forward(x)
    stop_manager()
    assert var_state(x) == 1

    y_error = var_copy(y)
    var_axpy(y_error, -1.0, x)
    y_1 = Function(space)
    y_1.interpolate(Constant(2.0))
    var_axpy(y_error, -c, y_1)
    del y_1
    assert var_linf_norm(y_error) == 0.0

    J_val = J.value

    dJ = compute_gradient(J, x)

    def forward_J(x):
        _, J = forward(x)
        return J

    min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, x, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_FixedPointSolver(setup_test, test_leaks):
    x = Constant(name="x")
    z = Constant(name="z")

    a = Constant(2.0, name="a", static=True)
    b = Constant(3.0, name="b", static=True)

    def forward(a, b):
        x.assign(0.0)
        z.assign(0.0)

        eqs = [LinearCombination(z, (1.0, x), (1.0, b)),
               ExprInterpolation(x, a / sqrt(z))]

        fp_parameters = {"absolute_tolerance": 0.0,
                         "relative_tolerance": 1.0e-14}
        FixedPointSolver(eqs, solver_parameters=fp_parameters).solve()

        J = Functional(name="J")
        J.assign(x)
        return J

    start_manager()
    J = forward(a, b)
    stop_manager()

    x_val = var_scalar_value(x)
    a_val = var_scalar_value(a)
    b_val = var_scalar_value(b)
    assert abs(x_val * np.sqrt(x_val + b_val) - a_val) < 1.0e-14

    J_val = J.value

    dJda, dJdb = compute_gradient(J, [a, b])

    dm = Constant(1.0, name="dm", static=True)

    for M, dM, forward_J, dJ in \
            [(a, dm, lambda a: forward(a, b), dJda),
             (b, dm, lambda b: forward(a, b), dJdb),
             ((a, b), (dm, dm), forward, (dJda, dJdb))]:
        min_order = taylor_test(forward_J, M, J_val=J_val, dJ=dJ, dM=dM)
        assert min_order > 1.99

        ddJ = Hessian(forward_J)
        min_order = taylor_test(forward_J, M, J_val=J_val, ddJ=ddJ, dM=dM)
        assert min_order > 2.99

        min_order = taylor_test_tlm(forward_J, M, tlm_order=1, dMs=(dM,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=1,
                                            dMs=(dM,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=2,
                                            dMs=(dM, dM))
        assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize(
    "overlap_type", [(firedrake.DistributedMeshOverlapType.NONE, 0),
                     pytest.param(
                         (firedrake.DistributedMeshOverlapType.FACET, 1),
                         marks=pytest.mark.skipif(DEFAULT_COMM.size == 1,
                                                  reason="parallel only")),
                     pytest.param(
                         (firedrake.DistributedMeshOverlapType.VERTEX, 1),
                         marks=pytest.mark.skipif(DEFAULT_COMM.size == 1,
                                                  reason="parallel only"))])
@pytest.mark.parametrize("N_x, N_y, N_z", [(2, 2, 2),
                                           (5, 5, 5)])
@pytest.mark.parametrize("c", [-1.5, 1.5])
@seed_test
def test_PointInterpolation(setup_test, test_leaks,
                            overlap_type,
                            N_x, N_y, N_z,
                            c):
    mesh = UnitCubeMesh(N_x, N_y, N_z,
                        distribution_parameters={"partition": True,
                                                 "overlap_type": overlap_type})
    X = SpatialCoordinate(mesh)
    y_space = FunctionSpace(mesh, "Lagrange", 3)
    X_coords = np.array([[0.1, 0.1, 0.1],
                         [0.2, 0.3, 0.4],
                         [0.9, 0.8, 0.7],
                         [0.4, 0.2, 0.3]], dtype=backend_RealType)

    def forward(y):
        X_vals = [Constant(name=f"x_{i:d}")
                  for i in range(X_coords.shape[0])]
        eq = PointInterpolation(X_vals, y, X_coords, tolerance=1.0e-14)
        eq.solve()

        J = Functional(name="J")
        for x in X_vals:
            term = Constant()
            ExprInterpolation(term, x ** 3).solve()
            J.addto(term)
        return X_vals, J

    y = Function(y_space, name="y", static=True)
    if complex_mode:
        interpolate_expression(y, pow(X[0], 3) - 1.5 * X[0] * X[1] + c
                               + 1.0j * pow(X[0], 2))
    else:
        interpolate_expression(y, pow(X[0], 3) - 1.5 * X[0] * X[1] + c)

    start_manager()
    X_vals, J = forward(y)
    stop_manager()

    def x_ref(x):
        if complex_mode:
            return x[0] ** 3 - 1.5 * x[0] * x[1] + c + 1.0j * x[0] ** 2
        else:
            return x[0] ** 3 - 1.5 * x[0] * x[1] + c

    x_error_norm = 0.0
    assert len(X_vals) == len(X_coords)
    for x, x_coord in zip(X_vals, X_coords):
        x_error_norm = max(x_error_norm,
                           abs(var_scalar_value(x) - x_ref(x_coord)))
    info(f"Error norm = {x_error_norm:.16e}")
    assert x_error_norm < 1.0e-13

    J_val = J.value

    dJ = compute_gradient(J, y)

    def forward_J(y):
        return forward(y)[1]

    min_order = taylor_test(forward_J, y, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, y, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, y, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("ExprAssignment_cls", [ExprAssignment,
                                                ExprInterpolation])
@seed_test
def test_ExprAssignment(setup_test, test_leaks,
                        ExprAssignment_cls):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 2)

    def test_expression(c, y, z):
        return (c ** 3) * y + np.sqrt(5.0) * z

    def forward(c, y, z):
        x = Function(space, name="x")
        ExprAssignment_cls(x, test_expression(c, y, z)).solve()

        J = Functional(name="J")
        J.assign(((x - Constant(1.0)) ** 3) * dx)
        return x, J

    y = Function(space, name="y")
    z = Function(space, name="z")
    if complex_mode:
        c = Constant(np.sqrt(2.0 + 1.0j * np.sqrt(3.0)))
        interpolate_expression(y,
                               cos(3.0 * pi * X[0])
                               + 1.0j * sin(5.0 * pi * X[0]))
        interpolate_expression(z,
                               cos(7.0 * pi * X[0])
                               + 1.0j * sin(11.0 * pi * X[0]))
    else:
        c = Constant(np.sqrt(2.0))
        interpolate_expression(y,
                               cos(3.0 * pi * X[0]))
        interpolate_expression(z,
                               cos(7.0 * pi * X[0]))
    x_ref = Function(space, name="x_ref")
    interpolate_expression(x_ref, test_expression(c, y, z))

    start_manager()
    x, J = forward(c, y, z)
    stop_manager()

    error_norm = np.sqrt(abs(assemble(inner(x - x_ref, x - x_ref) * dx)))
    info(f"Error norm = {error_norm:.16e}")
    assert error_norm < 1.0e-14

    J_val = J.value

    dJ_c, dJ_y = compute_gradient(J, (c, y))

    def forward_J_c(c):
        _, J = forward(c, y, z)
        return J

    def forward_J_y(y):
        _, J = forward(c, y, z)
        return J

    for m, dJ, forward_J in ((c, dJ_c, forward_J_c), (y, dJ_y, forward_J_y)):
        min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, seed=1.0e-3)
        assert min_order > 1.99

        ddJ = Hessian(forward_J)
        min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ,
                                seed=1.0e-3)
        assert min_order > 2.99

        min_order = taylor_test_tlm(forward_J, m, tlm_order=1,
                                    seed=1.0e-3)
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1,
                                            seed=1.0e-3)
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2,
                                            seed=1.0e-3)
        assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("c0", [lambda mesh: 2,
                                lambda mesh: Constant(2.0),
                                lambda mesh: Function(FunctionSpace(mesh, "R", 0)).assign(Constant(2.0))])  # noqa: E501
@seed_test
def test_ExprAssignment_vector(setup_test, test_leaks,
                               c0):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = VectorFunctionSpace(mesh, "Lagrange", 1)

    def forward(c, m):
        u = Function(space, name="u").assign(m)
        u.assign(c * u + m)

        J = Functional(name="J")
        J.assign(((dot(u, u) + Constant(1.0)) ** 2) * dx)
        return u, J

    c = c0(mesh)
    m0 = Function(space, name="m0")
    if complex_mode:
        interpolate_expression(
            m0, as_vector((cos(pi * X[0]) + 1.0j * cos(2.0 * pi * X[1]),
                           X[1] * exp(X[0]))))
    else:
        interpolate_expression(
            m0, as_vector((cos(pi * X[0]),
                           X[1] * exp(X[0]))))
    m = Function(space, name="m").assign(m0)

    if is_var(c):
        M = (c, m)
    else:
        M = (m,)
        forward = functools.partial(forward, c)

    start_manager()
    u, J = forward(*M)
    stop_manager()

    assert np.sqrt(abs(assemble(inner(u - (1 + c0(mesh)) * m0, u - (1 + c0(mesh)) * m0) * dx))) < 1.0e-15  # noqa: E501

    def forward_J(*M):
        _, J = forward(*M)
        return J

    J_val = J.value

    dJ = compute_gradient(J, M)

    min_order = taylor_test(forward_J, M, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, M, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward_J, M, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, M, adjoint_order=2)
    assert min_order > 2.00


@pytest.mark.firedrake
@seed_test
def test_ExprInterpolation(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def test_expression(y, y_int):
        return (y_int * y * sin(y)
                + 2.0 + (y ** 2) + y / (1.0 + (y ** 2)))

    def forward(y):
        x = Function(space, name="x")
        y_int = Constant(name="y_int")
        Assembly(y_int, y * dx).solve()
        ExprInterpolation(x, test_expression(y, y_int)).solve()

        J = Functional(name="J")
        J.assign(x * x * x * dx)
        return x, J

    y = Function(space, name="y")
    interpolate_expression(y, cos(3.0 * pi * X[0]))
    x_ref = Function(space, name="x_ref")
    interpolate_expression(x_ref,
                           test_expression(y, Constant(assemble(y * dx))))

    start_manager()
    x, J = forward(y)
    stop_manager()

    error_norm = np.sqrt(abs(assemble(inner(x - x_ref, x - x_ref) * dx)))
    info(f"Error norm = {error_norm:.16e}")
    assert error_norm == 0.0

    J_val = J.value

    dJ = compute_gradient(J, y)

    def forward_J(y):
        _, J = forward(y)
        return J

    min_order = taylor_test(forward_J, y, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, y, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward_J, y, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=2)
    assert min_order > 2.00


@pytest.mark.firedrake
@pytest.mark.parametrize("degree", [1, 2, 3])
@seed_test
def test_ExprInterpolation_transpose(setup_test, test_leaks,
                                     degree):
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
        y_1 = Function(space_1, name="y_1")
        ExprInterpolation(y_1, y_2).solve()

        J = Functional(name="J")
        J.assign(((y_1 - Constant(1.0)) ** 4) * dx)
        return y_1, J

    start_manager()
    y_1, J = forward(y_2)
    stop_manager()

    y_1_error = var_copy(y_1)
    var_axpy(y_1_error, -1.0, y_1_ref)
    assert var_linf_norm(y_1_error) == 0.0

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
@seed_test
def test_ExprInterpolation_transpose_vector(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_1 = VectorFunctionSpace(mesh, "Lagrange", 1)
    space_2 = VectorFunctionSpace(mesh, "Lagrange", 2)

    def forward(y):
        x = Function(space_1, name="x")
        ExprInterpolation(x, Constant(2.0) * y).solve()

        J = Functional(name="J")
        J.assign(((dot(x, x) - 1.0) ** 2) * dx)
        return x, J

    y = Function(space_2, name="y")
    interpolate_expression(y, as_vector([cos(2.0 * pi * X[0]) * cos(3.0 * pi * X[1]),  # noqa: E501
                                         cos(5.0 * pi * X[0]) * cos(7.0 * pi * X[1])]))  # noqa: E501
    start_manager()
    x, J = forward(y)
    stop_manager()

    x_ref = Function(space_1, name="x_ref").interpolate(2 * y)
    error_norm = np.sqrt(abs(assemble(inner(x - x_ref, x - x_ref) * dx)))
    info(f"Error norm = {error_norm:.16e}")
    assert error_norm < 1.0e-16

    J_val = J.value

    dJ = compute_gradient(J, y)

    def forward_J(y):
        _, J = forward(y)
        return J

    min_order = taylor_test(forward_J, y, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, y, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward_J, y, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, y, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_LocalProjection(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Discontinuous Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    def forward(G):
        F = Function(space_1, name="F")
        LocalProjection(F, G).solve()

        J = Functional(name="J")
        J.assign((F ** 2 + F ** 3) * dx)
        return F, J

    G = Function(space_2, name="G", static=True)
    interpolate_expression(G, sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    start_manager()
    F, J = forward(G)
    stop_manager()

    F_ref = Function(space_1, name="F_ref")
    solve(inner(trial_1, test_1) * dx == inner(G, test_1) * dx, F_ref,
          solver_parameters=ls_parameters_cg)
    F_error = Function(space_1, name="F_error")
    var_assign(F_error, F_ref)
    var_axpy(F_error, -1.0, F)

    F_error_norm = var_linf_norm(F_error)
    info(f"Error norm = {F_error_norm:.16e}")
    assert F_error_norm < 1.0e-14

    J_val = J.value

    dJ = compute_gradient(J, G)

    def forward_J(G):
        return forward(G)[1]

    min_order = taylor_test(forward_J, G, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, G, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, G, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, G, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, G, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Assembly_arity_0(setup_test, test_leaks):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(F):
        x = Constant(name="x")
        Assembly(x, (F ** 4) * dx).solve()

        J = Functional(name="J")
        J.assign(x)
        return J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, X[0] * sin(pi * X[1]))

    start_manager()
    J = forward(F)
    stop_manager()

    J_val = J.value
    assert abs(J_val - assemble((F ** 4) * dx)) == 0.0

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
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_Assembly_arity_1(setup_test, test_leaks,
                          assemble_action):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    def forward(F):
        x = Cofunction(space.dual(), name="x")
        Assembly(x, inner(F ** 3, test) * dx).solve()

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
@seed_test
def test_Assembly_arity_1_FormSum(setup_test, test_leaks,
                                  solve_eq, assemble_rhs, test_rhs):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        u = Function(space, name="u")
        b = Cofunction(space.dual(), name="b")
        assemble_rhs(b, test_rhs(m.dx(0), test))
        solve_eq(inner(trial, test) * dx == b, u,
                 solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        J.assign(((u + Constant(1.0)) ** 3) * dx)
        return u, J

    m = Function(space, name="m")
    m.interpolate(X[0])
    start_manager()
    u, J = forward(m)
    stop_manager()

    assert np.sqrt(abs(assemble(inner(u - Constant(1.0),
                                      u - Constant(1.0)) * dx))) < 1.0e-15

    J_val = J.value

    dJ = compute_gradient(J, m)

    def forward_J(m):
        _, J = forward(m)
        return J

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, size=4)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Storage(setup_test, test_leaks,
                 tmp_path):
    comm = comm_dup_cached(manager().comm(), key="test_Storage")

    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(x, d=None, h=None):
        y = Function(space, name="y")
        x_s = Function(space, name="x_s")
        y_s = Function(space, name="y_s")

        if d is None:
            var_assign(x_s, x)
            d = {}
        MemoryStorage(x_s, d, var_name(x_s), save=True).solve()

        Projection(y, x * x * x * x_s,
                   solver_parameters=ls_parameters_cg).solve()

        if h is None:
            var_assign(y_s, y)

            if comm.rank == 0:
                pid = os.getpid()
            else:
                pid = None
            root_pid = comm.bcast(pid, root=0)
            filename = f"storage_{root_pid:d}.hdf5"

            import h5py
            if comm.size > 1:
                h = h5py.File(str(tmp_path / filename),
                              "w", driver="mpio", comm=comm)
            else:
                h = h5py.File(str(tmp_path / filename),
                              "w")
        HDF5Storage(y_s, h, var_name(y_s), save=True).solve()

        J = Functional(name="J")
        J.assign(((dot(y, y_s) + 1.0) ** 2) * dx)
        return d, h, J

    x = Function(space, name="x", static=True)
    interpolate_expression(x, cos(pi * X[0]) * exp(X[1]))

    start_manager()
    d, h, J = forward(x)
    stop_manager()

    assert len(manager()._cp._refs) == 1
    assert tuple(manager()._cp._refs.keys()) == (var_id(x),)
    assert len(manager()._cp._cp) == 0
    assert len(manager()._cp._data) == 4
    assert tuple(len(nl_deps) for nl_deps in manager()._cp._data.values()) \
        == (0, 2, 0, 2)
    assert len(manager()._cp._storage) == 4

    J_val = J.value

    def forward_J(x):
        _, _, J = forward(x, d=d, h=h)
        return J

    dJ = compute_gradient(J, x)

    min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ, seed=1.0e-3,
                            size=4)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, x, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 2.00

    h.close()


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_InnerProduct(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test = TestFunction(space)

    def forward(F):
        G = Cofunction(space.dual(), name="G")
        assemble(inner(F, test) * dx, tensor=G)

        J = Float(name="J")
        InnerProduct(J, F, G).solve()
        return J

    F = Function(space, name="F")
    interpolate_expression(F, X[0] * sin(pi * X[0]))

    start_manager()
    J = forward(F)
    stop_manager()

    J_val = float(J)
    assert abs(J_val - assemble(inner(F, F) * dx)) == 0.0

    dJ = compute_gradient(J, F)
    min_order = taylor_test(forward, F, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    min_order = taylor_test_tlm(forward, F, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=1)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("test_adj_ic", [False, True])
@seed_test
def test_initial_guess(setup_test, test_leaks,
                       test_adj_ic):
    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    test_1 = TestFunction(space_1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    zero = ZeroConstant(name="zero")

    def forward(y):
        x_0 = project(y, space_1,
                      solver_parameters=ls_parameters_cg)
        x = Function(space_1, name="x")

        class CustomProjection(Projection):
            def forward_solve(self, x, deps=None):
                rhs = self._rhs
                if deps is not None:
                    rhs = self._replace(rhs, deps)
                solver, _, b = assemble_linear_solver(
                    self._J, rhs,
                    form_compiler_parameters=self._form_compiler_parameters,
                    linear_solver_parameters=self._linear_solver_parameters)
                solver.solve(x, b)
                its = solver.ksp.getIterationNumber()
                assert its == 0

            def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
                assert adj_x is not None
                solver, _, _ = assemble_linear_solver(
                    self._J,
                    form_compiler_parameters=self._form_compiler_parameters,
                    linear_solver_parameters=self._adjoint_solver_parameters)
                solver.solve(adj_x, b)
                its = solver.ksp.getIterationNumber()
                assert (its == 0) == test_adj_ic
                return adj_x

        x.assign(x_0)
        CustomProjection(
            x, y,
            solver_parameters={"ksp_type": "cg",
                               "pc_type": "sor",
                               "ksp_rtol": 1.0e-10,
                               "ksp_atol": 1.0e-16,
                               "ksp_initial_guess_nonzero": True}).solve()

        J = Functional(name="J")
        J.assign((dot(x, x) ** 2) * dx)

        adj_x_0 = Cofunction(space_1.dual(), name="adj_x_0", static=True)
        with paused_manager():
            assemble(4 * dot(ufl.conj(dot(x, x) * x), ufl.conj(test_1)) * dx,
                     tensor=adj_x_0)
        Projection(x, zero,
                   solver_parameters=ls_parameters_cg).solve()
        if not test_adj_ic:
            ZeroAssignment(x).solve()
        J_term = var_new(J)
        InnerProduct(J_term, x, adj_x_0).solve()
        J.addto(J_term)

        return x_0, x, adj_x_0, J

    y = Function(space_2, name="y", static=True)
    if issubclass(var_dtype(y), np.complexfloating):
        interpolate_expression(y, exp(X[0]) * (1.0 + 1.0j + X[1] * X[1]))
    else:
        interpolate_expression(y, exp(X[0]) * (1.0 + X[1] * X[1]))

    start_manager()
    x_0, _, adj_x_0, J = forward(y)
    stop_manager()

    assert len(manager()._cp._refs) == 3
    assert tuple(manager()._cp._refs.keys()) == (var_id(y),
                                                 var_id(zero),
                                                 var_id(adj_x_0))
    assert len(manager()._cp._cp) == 0
    if test_adj_ic:
        assert len(manager()._cp._data) == 9
        assert tuple(map(len, manager()._cp._data.values())) \
            == (0, 0, 0, 0, 1, 0, 2, 0, 0)
    else:
        assert len(manager()._cp._data) == 10
        assert tuple(map(len, manager()._cp._data.values())) \
            == (0, 0, 0, 0, 1, 0, 0, 2, 0, 0)
    assert len(manager()._cp._storage) == 5

    dJdx_0, dJdy = compute_gradient(J, [x_0, y])
    assert var_linf_norm(dJdx_0) == 0.0

    J_val = J.value

    def forward_J(y):
        _, _, _, J = forward(y)
        return J

    min_order = taylor_test(forward_J, y, J_val=J_val, dJ=dJdy)
    assert min_order > 2.00


@pytest.mark.firedrake
@pytest.mark.parametrize("cache_rhs_assembly", [True, False])
@seed_test
def test_EquationSolver_forward_solve_deps(setup_test, test_leaks,
                                           cache_rhs_assembly):
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        class CustomEquationSolver(EquationSolver):
            def forward_solve(self, x, deps=None):
                super().forward_solve(x, deps=self.dependencies())

        x = Function(space, name="x")
        CustomEquationSolver(
            inner(m * trial, test) * dx == inner(Constant(2.0), test) * dx,
            x, DirichletBC(space, 1.0, "on_boundary"),
            solver_parameters=ls_parameters_cg,
            cache_jacobian=False,
            cache_rhs_assembly=cache_rhs_assembly).solve()

        J = Functional(name="J")
        J.assign(((1 + x) ** 3) * dx)
        return J

    m = Function(space, name="m")
    m.interpolate(Constant(1.0))

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value

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
@seed_test
def test_EquationSolver_FormSum(setup_test, test_leaks, test_configurations,
                                solve_eq, test_rhs):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        u = Function(space, name="u")
        solve_eq(inner(trial, test) * dx == test_rhs(m.dx(0), test), u,
                 solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        J.assign(((u + Constant(1.0)) ** 3) * dx)
        return u, J

    m = Function(space, name="m")
    m.interpolate(X[0])
    start_manager()
    u, J = forward(m)
    stop_manager()

    assert np.sqrt(abs(assemble(inner(u - Constant(1.0),
                                      u - Constant(1.0)) * dx))) < 1.0e-15

    J_val = J.value

    dJ = compute_gradient(J, m)

    def forward_J(m):
        _, J = forward(m)
        return J

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, seed=1.0e-3,
                            size=4)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("g", [  # Bc value is a control
                                 lambda m: m,
                                 # Bc value is a forward variable
                                 # (extra copy operation)
                                 lambda m: m.copy(deepcopy=True),
                                 # Bc value is a function of a forward variable
                                 # (extra operation via bc.function_arg)
                                 lambda m: 2 * m])
@pytest.mark.parametrize("sub_domains", [(1,),
                                         (1, 3, 4),
                                         ("on_boundary",)])
@seed_test
def test_EquationSolver_DirichletBC(setup_test, test_leaks, test_configurations,  # noqa: E501
                                    g, sub_domains):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def eq(u):
        space = u.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        return (inner(trial, test) * dx + inner(trial.dx(0), test) * dx
                == inner(Constant(1.0), test) * dx)

    def bcs(space, m, sub_domains):
        return tuple(DirichletBC(space, g(m), sub_domain)
                     for sub_domain in sub_domains)

    def forward(m):
        with paused_manager():
            u_ref = Function(space, name="u_ref")
            solve(eq(u_ref), u_ref, bcs=bcs(space, m, sub_domains))

        u = Function(space, name="u")
        EquationSolver(eq(u), u, bcs=bcs(space, m, sub_domains)).solve()

        with paused_manager():
            error_norm = np.sqrt(abs(assemble(inner(u - u_ref,
                                                    u - u_ref) * dx)))
            assert error_norm < 1.0e-14

        J = Functional(name="J")
        J.assign(((u - Constant(2.0)) ** 4) * dx)
        return J

    m = Function(space, name="m").assign(Constant(1.0))

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = J.value

    dJ = compute_gradient(J, m)

    # Perturbation direction which is non-zero on the overlapping bcs
    dm = Function(space).interpolate(Constant(1.0) - X[0])

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ, seed=5.0e-3, dM=dm)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, seed=5.0e-3,
                            dM=dm)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, m, tlm_order=1, seed=5.0e-3,
                                dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                        seed=5.0e-3, dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                        seed=5.0e-3)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_eliminate_zeros(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    F = ZeroFunction(space, name="F")

    L = inner(F, test) * dx

    for i in range(3):
        L_z = eliminate_zeros(L)
        assert isinstance(L_z, ufl.classes.ZeroBaseForm)


@pytest.mark.firedrake
@seed_test
def test_ZeroFunction_expand_derivatives(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = ZeroFunction(space, name="F")

    expr = ufl.algorithms.expand_derivatives(F.dx(0))
    assert F in extract_coefficients(expr)
    assert F not in extract_coefficients(eliminate_zeros(expr))


@pytest.mark.firedrake
@pytest.mark.parametrize("Space", [FunctionSpace,
                                   VectorFunctionSpace,
                                   TensorFunctionSpace])
@seed_test
def test_eliminate_zeros_arity_1(setup_test, test_leaks,
                                 Space):
    mesh = UnitSquareMesh(10, 10)
    space = Space(mesh, "Lagrange", 1)
    test = TestFunction(space)
    F = ZeroFunction(space, name="F")

    form = (inner(F, test) * dx
            + Constant(1.0) * inner(grad(F), grad(test)) * dx)

    zero_form = eliminate_zeros(form)
    assert isinstance(zero_form, ufl.classes.ZeroBaseForm)


@pytest.mark.firedrake
@seed_test
def test_ZeroFunction(setup_test, test_leaks, test_configurations):
    mesh = UnitIntervalMesh(10)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        X = [Function(space, name=f"x_{i:d}") for i in range(4)]

        Assignment(X[0], m).solve()
        LinearCombination(X[1], (1.0, X[0])).solve()
        ExprInterpolation(X[2], m + X[1]).solve()
        Projection(X[3], m + X[2],
                   solver_parameters=ls_parameters_cg).solve()

        J = Functional(name="J")
        J.assign((dot(X[-1] + 1.0, X[-1] + 1.0) ** 2) * dx
                 + (dot(m + 2.0, m + 2.0) ** 2) * dx)
        return J

    m = ZeroFunction(space, name="m")
    for m_i in m.subfunctions:
        assert var_linf_norm(m_i) == 0.0

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)

    J_val = J.value

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, m, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2)
    assert min_order > 2.00
