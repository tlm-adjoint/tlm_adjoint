from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake import manager as _manager
from tlm_adjoint.checkpoint_schedules.binomial import optimal_steps

from .test_base import *

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_long_range(setup_test, test_leaks,
                    tmp_path):
    n_steps = 200
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 0,
                             "snaps_in_ram": 2,
                             "path": str(tmp_path / "checkpoints~")})

    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(F, x_ref=None):
        x_old = Function(space, name="x_old")
        x = Function(space, name="x")
        Assignment(x_old, F).solve()
        J = Functional(name="J")
        gather_ref = x_ref is None
        if gather_ref:
            x_ref = {}
        for n in range(n_steps):
            terms = [(1.0, x_old)]
            if n % 11 == 0:
                terms.append((1.0, F))
            LinearCombination(x, *terms).solve()
            if n % 17 == 0:
                if gather_ref:
                    x_ref[n] = var_copy(x, name=f"x_ref_{n:d}")
                J.addto(dot(x * x * x, x_ref[n]) * dx)
            Assignment(x_old, x).solve()
            if n < n_steps - 1:
                new_block()

        return x_ref, J

    F = Function(space, name="F", static=True)
    interpolate_expression(F, sin(pi * X[0]))

    start_manager()
    x_ref, J = forward(F)
    stop_manager()

    J_val = J.value

    dJ = compute_gradient(J, F)

    def forward_J(F):
        return forward(F, x_ref=x_ref)[1]

    min_order = taylor_test(forward_J, F, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, F, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward_J, F, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, F, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward_J, F, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_adjoint_graph_pruning(setup_test, test_leaks):
    mesh = UnitIntervalMesh(10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(y):
        x = Function(space, name="x")

        ZeroAssignment(x).solve()

        Assignment(x, y).solve()

        J_0 = Functional(name="J_0")
        J_0.assign((dot(x, x) ** 2) * dx)

        J_1 = Functional(name="J_1")
        J_1.assign(x * dx)

        J_0_val = J_0.value
        ZeroAssignment(x).solve()
        assert var_linf_norm(x) == 0.0
        J_0.addto(dot(x, y) * dx)
        assert J_0.value == J_0_val

        J_2 = Functional(name="J_2")
        J_2.assign(x * dx)

        return J_0

    y = Function(space, name="y", static=True)
    interpolate_expression(y, exp(X[0]))

    start_manager()
    J = forward(y)
    stop_manager()

    eqs = {(0, 0, i) for i in range(9)}
    active_eqs = {(0, 0, 1), (0, 0, 2), (0, 0, 5), (0, 0, 6), (0, 0, 7)}

    def callback(J_i, n, i, eq, adj_X):
        if n == 0:
            eqs.remove((J_i, n, i))
            assert adj_X is None or (J_i, n, i) in active_eqs

    dJ = compute_gradient(J, y, callback=callback)
    assert len(eqs) == 0

    J_val = J.value

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


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_Referrers_LinearEquation(setup_test, test_leaks):
    def forward(m, forward_run=False):
        class IdentityMatrix(Matrix):
            def __init__(self):
                super().__init__(nl_deps=[], ic=False, adj_ic=False)

            @no_space_type_checking
            def forward_action(self, nl_deps, x, b, *, method="assign"):
                if method == "assign":
                    var_assign(b, x)
                else:
                    raise ValueError(f"Unexpected method '{method:s}'")

            @no_space_type_checking
            def adjoint_action(self, nl_deps, adj_x, b, b_index=0, *,
                               method="assign"):
                if b_index != 0:
                    raise ValueError("Unexpected b_index")

                if method == "assign":
                    var_assign(b, adj_x)
                else:
                    raise ValueError(f"Unexpected method '{method:s}'")

            @no_space_type_checking
            def forward_solve(self, x, nl_deps, b):
                var_assign(x, b)

            @no_space_type_checking
            def adjoint_solve(self, adj_x, nl_deps, b):
                assert adj_x is None
                adj_x = var_new_conjugate_dual(b)
                var_assign(adj_x, b)
                return adj_x

            def tangent_linear_rhs(self, tlm_map, x):
                return None

        x = Constant(0.0, name="x")

        M = IdentityMatrix()
        b = InnerProductRHS(m, m, M=M)
        linear_eq = LinearEquation(x, [b, b], A=M)
        linear_eq.solve()

        if forward_run:
            manager = _manager()

            assert len(manager._to_drop_references) == 0
            assert not linear_eq._references_dropped
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in linear_eq.dependencies():
                assert not var_is_replacement(dep)
            for dep in b.dependencies():
                assert not var_is_replacement(dep)

            linear_eq = linear_eq._weak_alias

            assert len(manager._to_drop_references) == 1
            assert not linear_eq._references_dropped
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in linear_eq.dependencies():
                assert not var_is_replacement(dep)
            for dep in b.dependencies():
                assert not var_is_replacement(dep)

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert linear_eq._references_dropped
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in linear_eq.dependencies():
                assert var_is_replacement(dep)
            for dep in b.dependencies():
                assert not var_is_replacement(dep)

        y = Constant(0.0, name="y")
        LinearEquation(y, b, A=M).solve()

        z = Constant(0.0, name="z")
        Axpy(z, x, 1.0, y).solve()

        if forward_run:
            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in b.dependencies():
                assert not var_is_replacement(dep)

            M = M._weak_alias
            b = b._weak_alias

            assert len(manager._to_drop_references) == 1
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in b.dependencies():
                assert not var_is_replacement(dep)

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert b._references_dropped
            assert M._references_dropped
            for dep in b.dependencies():
                assert var_is_replacement(dep)

        M = IdentityMatrix()

        J = Functional(name="J")
        InnerProduct(J, z, z, M=M).solve()
        return J

    m = Constant(np.sqrt(2.0), name="m")

    start_manager()
    J = forward(m, forward_run=True)
    stop_manager()

    J_val = J.value
    info(f"J = {J_val:.16e}")
    assert abs(J_val - 36.0) < 1.0e-13

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, dJ=dJ,
                            seed=5.0e-4)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, ddJ=ddJ,
                            seed=5.0e-4)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, m, tlm_order=1, dMs=(Constant(1.0),),
                                seed=5.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                        dMs=(Constant(1.0),), seed=5.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                        dMs=(Constant(1.0), Constant(1.0)),
                                        seed=5.0e-4)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_Referrers_FixedPointEquation(setup_test, test_leaks):
    def forward(m, forward_run=False):
        class NewtonSolver(Equation):
            def __init__(self, x, m, x0):
                check_space_type(x, "primal")
                check_space_type(m, "primal")
                check_space_type(x0, "primal")

                super().__init__(x, deps=[x, x0, m], nl_deps=[x0, m],
                                 ic=False, adj_ic=False)

            def forward_solve(self, x, deps=None):
                _, x0, m = self.dependencies() if deps is None else deps
                var_assign(
                    x,
                    0.5 * (var_scalar_value(x0) ** 2
                           + var_scalar_value(m))
                    / var_scalar_value(x0))

            def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
                return b

            def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
                if dep_index == 1:
                    x0, m = nl_deps
                    F = var_new_conjugate_dual(x0)
                    var_assign(
                        F,
                        (0.5 * var_scalar_value(adj_x)
                         * (var_scalar_value(m)
                            / (var_scalar_value(x0) ** 2) - 1.0)).conjugate())
                    return F
                elif dep_index == 2:
                    x0, m = nl_deps
                    F = var_new_conjugate_dual(x0)
                    var_assign(
                        F,
                        (-0.5 * var_scalar_value(adj_x)
                         / var_scalar_value(x0)).conjugate())
                    return F
                else:
                    raise ValueError("Unexpected dep_index")

        x0 = Constant(1.0, name="x0")
        x1 = Constant(0.0, name="x1")

        eq0 = NewtonSolver(x1, m, x0)
        eq1 = Assignment(x0, x1)

        fp_eq = FixedPointSolver(
            [eq0, eq1],
            solver_parameters={"absolute_tolerance": 0.0,
                               "relative_tolerance": 1.0e-14})
        fp_eq.solve()

        if forward_run:
            manager = _manager()

            assert len(manager._to_drop_references) == 0
            for eq in (fp_eq, eq0, eq1):
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not var_is_replacement(dep)
            del eq

            fp_eq = fp_eq._weak_alias

            assert len(manager._to_drop_references) == 1
            for eq in (fp_eq, eq0, eq1):
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not var_is_replacement(dep)
            del eq

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert fp_eq._references_dropped
            for dep in fp_eq.dependencies():
                assert var_is_replacement(dep)
            for eq in (eq0, eq1):
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not var_is_replacement(dep)
            del eq

        eq0.solve()
        eq1.solve()

        if forward_run:
            assert len(manager._to_drop_references) == 0
            for eq in (eq0, eq1):
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not var_is_replacement(dep)
            del eq

            eq0 = eq0._weak_alias
            eq1 = eq1._weak_alias

            assert len(manager._to_drop_references) == 2
            for eq in (eq0, eq1):
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not var_is_replacement(dep)
            del eq

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            for eq in (eq0, eq1):
                assert eq._references_dropped
                for dep in eq.dependencies():
                    assert var_is_replacement(dep)
            del eq

        J = Functional(name="J")
        J.assign(x1)
        return J

    m = Constant(2.0, name="m")

    start_manager()
    J = forward(m, forward_run=True)
    stop_manager()

    J_val = J.value
    info(f"J = {J_val:.16e}")
    assert abs(J_val - np.sqrt(2.0)) < 1.0e-15

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, dJ=dJ)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("n_steps, snaps_in_ram", [(1, 1),
                                                   (10, 1),
                                                   (10, 2),
                                                   (10, 3),
                                                   (10, 5),
                                                   (100, 3),
                                                   (100, 5),
                                                   (100, 10),
                                                   (100, 20)])
@no_space_type_checking
@seed_test
def test_binomial_checkpointing(setup_test, test_leaks,
                                tmp_path, n_steps, snaps_in_ram):
    n_forward_solves = 0

    class Counter(Instruction):
        def forward_solve(self, X, deps=None):
            nonlocal n_forward_solves

            n_forward_solves += 1

    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 0,
                             "snaps_in_ram": snaps_in_ram,
                             "path": str(tmp_path / "checkpoints~")})

    def forward(m):
        for n in range(n_steps):
            Counter().solve()
            if n < n_steps - 1:
                new_block()

        J = Functional(name="J")
        DotProduct(J, m, m).solve()
        return J

    m = Constant(1.0, name="m", static=True)

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)

    info(f"Number of forward steps        : {n_forward_solves:d}")
    n_forward_solves_optimal = optimal_steps(n_steps, snaps_in_ram)
    info(f"Optimal number of forward steps: {n_forward_solves_optimal:d}")
    assert n_forward_solves == n_forward_solves_optimal

    min_order = taylor_test(forward, m, J_val=J.value, dJ=dJ)
    assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_adjoint_caching(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(m):
        u = Function(space, name="u")

        solve(inner(grad(trial), grad(test)) * dx
              == inner(m * cos(Constant(0.0) * m), test) * dx,
              u, DirichletBC(space, 0.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)

        J = Functional(name="J")
        v = Constant(1.0, name="v")
        J.assign((dot(u + v, u + v) ** 3) * dx)

        K = Functional(name="K")
        K.assign((dot(u + v, u + v) ** 4) * dx)

        return J, K

    m = Function(space, name="m")
    interpolate_expression(m, sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    dm_0 = Function(space, name="dm_0")
    if issubclass(var_dtype(dm_0), np.complexfloating):
        dm_0.interpolate(Constant(1.0 + 1.0j))
    else:
        dm_0.interpolate(Constant(1.0))
    dm_1 = var_copy(dm_0, name="dm_1")

    start_manager()
    J, K = forward(m)
    stop_manager()

    def forward_J(m):
        J, _ = forward(m)
        return J

    def forward_K(m):
        _, K = forward(m)
        return K

    J_val = J.value
    K_val = K.value
    dJ_0, dK_0 = compute_gradient(
        [J, K], m,
        cache_adjoint_degree=0)

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ_0, dM=dm_0)
    assert min_order > 1.99

    min_order = taylor_test(forward_K, m, J_val=K_val, dJ=dK_0, dM=dm_0)
    assert min_order > 1.99

    ddJ = Hessian(forward_J)
    J_val_0b, dJ_0b, ddJ_0 = ddJ.action(m, dm_0)
    assert abs(J_val - J_val_0b) == 0.0
    assert abs(dJ_0b - var_inner(dm_0, dJ_0)) < 1.0e-15

    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, dM=dm_0)
    assert min_order > 2.99

    for order in range(1, 4):
        min_order = taylor_test_tlm(
            forward_J, m, tlm_order=order,
            dMs=tuple(dm_0 for i in range(order)))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(
            forward_J, m, adjoint_order=order,
            dMs=tuple(dm_0 for i in range(order)))
        assert min_order > 1.99

    reset_manager()

    configure_tlm((m, dm_0))
    start_manager()
    J, K = forward(m)
    stop_manager()

    dJ_1, ddJ_1, dK_1 = manager().compute_gradient(
        [J, var_tlm(J, (m, dm_0)), K], m, cache_adjoint_degree=1)

    adj_cache = manager()._adj_cache
    assert tuple(adj_cache._keys.keys()) == ((2, 0, 4), (1, 0, 3),
                                             (1, 0, 1), (2, 0, 0))
    assert tuple(adj_cache._keys[(2, 0, 4)]) == ()
    assert tuple(adj_cache._keys[(1, 0, 3)]) == ((0, 0, 2),)
    assert tuple(adj_cache._keys[(1, 0, 1)]) == ((0, 0, 0),)
    assert tuple(adj_cache._keys[(2, 0, 0)]) == ()

    dJ_error = var_copy(dJ_0)
    var_axpy(dJ_error, -1.0, dJ_1)
    assert var_linf_norm(dJ_error) < 1.0e-17

    dK_error = var_copy(dK_0)
    var_axpy(dK_error, -1.0, dK_1)
    assert var_linf_norm(dK_error) < 1.0e-17

    ddJ_error = var_copy(ddJ_0)
    var_axpy(ddJ_error, -1.0, ddJ_1)
    assert var_linf_norm(ddJ_error) < 1.0e-18

    reset_manager()

    configure_tlm((m, dm_0), (m, dm_1))
    start_manager()
    J, K = forward(m)
    stop_manager()

    dddJ_2 = compute_gradient(
        var_tlm(J, (m, dm_0), (m, dm_1)), m,
        cache_adjoint_degree=0)

    ddJ_2a, ddJ_2b, dddJ_2b, dJ_2, dK_2 = manager().compute_gradient(
        [var_tlm(J, (m, dm_0)),
         var_tlm(J, (m, dm_1)),
         var_tlm(J, (m, dm_0), (m, dm_1)),
         J,
         K],
        m, cache_adjoint_degree=1)

    adj_cache = manager()._adj_cache
    assert tuple(adj_cache._keys.keys()) == ((4, 0, 8), (2, 0, 7),
                                             (2, 0, 3), (4, 0, 0))
    assert tuple(adj_cache._keys[(4, 0, 8)]) == ()
    assert tuple(adj_cache._keys[(2, 0, 7)]) == ((1, 0, 6), (0, 0, 5), (3, 0, 4))  # noqa: E501
    assert tuple(adj_cache._keys[(2, 0, 3)]) == ((1, 0, 2), (0, 0, 1), (3, 0, 0))  # noqa: E501
    assert tuple(adj_cache._keys[(4, 0, 0)]) == ()

    dJ_error = var_copy(dJ_0)
    var_axpy(dJ_error, -1.0, dJ_2)
    assert var_linf_norm(dJ_error) < 1.0e-17

    dK_error = var_copy(dK_0)
    var_axpy(dK_error, -1.0, dK_2)
    assert var_linf_norm(dK_error) < 1.0e-17

    ddJ_error = var_copy(ddJ_0)
    var_axpy(ddJ_error, -1.0, ddJ_2a)
    assert var_linf_norm(ddJ_error) < 1.0e-18

    ddJ_error = var_copy(ddJ_0)
    var_axpy(ddJ_error, -1.0, ddJ_2b)
    assert var_linf_norm(ddJ_error) < 1.0e-18

    dddJ_error = var_copy(dddJ_2)
    var_axpy(dddJ_error, -1.0, dddJ_2b)
    assert var_linf_norm(dddJ_error) < 1.0e-18

    reset_manager()

    configure_tlm((m, dm_0), (m, dm_0))
    start_manager()
    J, K = forward(m)
    stop_manager()

    ddJ_3, dddJ_3, dJ_3, dK_3 = manager().compute_gradient(
        [var_tlm(J, (m, dm_0)),
         var_tlm(J, (m, dm_0), (m, dm_0)),
         J,
         K],
        m, cache_adjoint_degree=1)

    adj_cache = manager()._adj_cache
    assert tuple(adj_cache._keys.keys()) == ((3, 0, 6), (1, 0, 5),
                                             (1, 0, 2), (3, 0, 0))
    assert tuple(adj_cache._keys[(3, 0, 6)]) == ()
    assert tuple(adj_cache._keys[(1, 0, 5)]) == ((0, 0, 4), (2, 0, 3))
    assert tuple(adj_cache._keys[(1, 0, 2)]) == ((0, 0, 1), (2, 0, 0))
    assert tuple(adj_cache._keys[(3, 0, 0)]) == ()

    dJ_error = var_copy(dJ_0)
    var_axpy(dJ_error, -1.0, dJ_3)
    assert var_linf_norm(dJ_error) < 1.0e-17

    dK_error = var_copy(dK_0)
    var_axpy(dK_error, -1.0, dK_3)
    assert var_linf_norm(dK_error) < 1.0e-17

    ddJ_error = var_copy(ddJ_0)
    var_axpy(ddJ_error, -1.0, ddJ_3)
    assert var_linf_norm(ddJ_error) < 1.0e-18

    dddJ_error = var_copy(dddJ_2)
    var_axpy(dddJ_error, -1.0, dddJ_3)
    assert var_linf_norm(dddJ_error) < 1.0e-18

    reset_manager()

    dm_0 = Function(space, name="dm_0")
    interpolate_expression(dm_0, sin(pi * X[0]) * sin(pi * X[1]))
    dm_1 = Function(space, name="dm_1")
    interpolate_expression(dm_1, sin(2.0 * pi * X[0]) * sin(pi * X[1]))
    dm_2 = Function(space, name="dm_2")
    interpolate_expression(dm_2, sin(3.0 * pi * X[0]) * sin(pi * X[1]))
    dm_3 = Function(space, name="dm_3")
    interpolate_expression(dm_3, sin(4.0 * pi * X[0]) * sin(pi * X[1]))

    configure_tlm((m, dm_0), (m, dm_2))
    configure_tlm((m, dm_0), (m, dm_3))
    configure_tlm((m, dm_1), (m, dm_2))
    configure_tlm((m, dm_1), (m, dm_3))
    start_manager()
    J, K = forward(m)
    stop_manager()

    dddJ_02_0 = compute_gradient(
        var_tlm(J, (m, dm_0), (m, dm_2)), m,
        cache_adjoint_degree=0)
    dddJ_03_0 = compute_gradient(
        var_tlm(J, (m, dm_0), (m, dm_3)), m,
        cache_adjoint_degree=0)
    dddJ_12_0 = compute_gradient(
        var_tlm(J, (m, dm_1), (m, dm_2)), m,
        cache_adjoint_degree=0)
    dddJ_13_0 = compute_gradient(
        var_tlm(J, (m, dm_1), (m, dm_3)), m,
        cache_adjoint_degree=0)

    dddJ_02_1, dddJ_03_1, dddJ_12_1, dddJ_13_1 = compute_gradient(
        [var_tlm(J, (m, dm_0), (m, dm_2)),
         var_tlm(J, (m, dm_0), (m, dm_3)),
         var_tlm(J, (m, dm_1), (m, dm_2)),
         var_tlm(J, (m, dm_1), (m, dm_3))],
        m, cache_adjoint_degree=2)

    adj_cache = manager()._adj_cache
    assert tuple(adj_cache._keys.keys()) == ((3, 0, 17), (3, 0, 8),
                                             (2, 0, 4), (3, 0, 4),
                                             (1, 0, 3), (3, 0, 3))
    # First order
    assert tuple(adj_cache._keys[(3, 0, 17)]) == ((2, 0, 16), (1, 0, 15), (0, 0, 14))  # noqa: E501
    assert tuple(adj_cache._keys[(3, 0, 8)]) == ((2, 0, 7), (1, 0, 6), (0, 0, 5))  # noqa: E501
    # Second order
    assert tuple(adj_cache._keys[(2, 0, 4)]) == ((0, 0, 1),)
    assert tuple(adj_cache._keys[(3, 0, 4)]) == ((1, 0, 1),)
    assert tuple(adj_cache._keys[(1, 0, 3)]) == ((0, 0, 2),)
    assert tuple(adj_cache._keys[(3, 0, 3)]) == ((2, 0, 2),)

    dddJ_error = var_copy(dddJ_02_0)
    var_axpy(dddJ_error, -1.0, dddJ_02_1)
    assert var_linf_norm(dddJ_error) < 1.0e-20

    dddJ_error = var_copy(dddJ_03_0)
    var_axpy(dddJ_error, -1.0, dddJ_03_1)
    assert var_linf_norm(dddJ_error) < 1.0e-21

    dddJ_error = var_copy(dddJ_12_0)
    var_axpy(dddJ_error, -1.0, dddJ_12_1)
    assert var_linf_norm(dddJ_error) < 1.0e-21

    dddJ_error = var_copy(dddJ_13_0)
    var_axpy(dddJ_error, -1.0, dddJ_13_1)
    assert var_linf_norm(dddJ_error) < 1.0e-21
