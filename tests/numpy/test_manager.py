#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *
from tlm_adjoint.numpy import manager as _manager
from tlm_adjoint.alias import WeakAlias
from tlm_adjoint.checkpoint_schedules.binomial import optimal_steps

from .test_base import *

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.numpy
@no_space_type_checking
@seed_test
def test_EmptyEquation(setup_test, test_leaks, test_default_dtypes):
    space = FunctionSpace(100)

    def forward(F):
        EmptyEquation().solve()

        F_dot_F = Constant(name="F_dot_F")
        DotProduct(F_dot_F, F, F).solve()

        J = Functional(name="J")
        DotProduct(J.function(), F_dot_F, F_dot_F).solve()
        return J

    F = Function(space, name="F")
    F.vector()[:] = np.arange(len(F.vector()), dtype=function_dtype(F))

    start_manager()
    J = forward(F)
    stop_manager()

    manager = _manager()
    manager.finalize()
    manager.info()
    assert len(manager._blocks) == 1
    assert len(manager._blocks[0]) == 3
    assert len(manager._blocks[0][0].X()) == 0

    J_val = J.value()
    assert abs(J_val - (F.vector() ** 2).sum() ** 2) == 0.0

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


@pytest.mark.numpy
@seed_test
def test_empty(setup_test, test_leaks, test_default_dtypes):
    def forward(m):
        return Functional(name="J")

    m = Constant(name="m", static=True)

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)
    assert function_scalar_value(dJ) == 0.0


@pytest.mark.numpy
@seed_test
def test_Referrers_LinearEquation(setup_test, test_leaks):
    def forward(m, forward_run=False):
        class IdentityMatrix(Matrix):
            def __init__(self):
                super().__init__(nl_deps=[], ic=False, adj_ic=False)

            @no_space_type_checking
            def forward_action(self, nl_deps, x, b, *, method="assign"):
                if method == "assign":
                    function_assign(b, x)
                else:
                    raise ValueError(f"Unexpected method '{method:s}'")

            @no_space_type_checking
            def adjoint_action(self, nl_deps, adj_x, b, b_index=0, *,
                               method="assign"):
                if b_index != 0:
                    raise IndexError("Invalid index")
                if method == "assign":
                    function_assign(b, adj_x)
                else:
                    raise ValueError(f"Unexpected method '{method:s}'")

            @no_space_type_checking
            def forward_solve(self, x, nl_deps, b):
                function_assign(x, b)

            @no_space_type_checking
            def adjoint_solve(self, adj_x, nl_deps, b):
                assert adj_x is None
                adj_x = function_new_conjugate_dual(b)
                function_assign(adj_x, b)
                return adj_x

            def tangent_linear_rhs(self, M, dM, tlm_map, x):
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
                assert not function_is_replacement(dep)
            for dep in b.dependencies():
                assert not function_is_replacement(dep)

            linear_eq = WeakAlias(linear_eq)

            assert len(manager._to_drop_references) == 1
            assert not linear_eq._references_dropped
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in linear_eq.dependencies():
                assert not function_is_replacement(dep)
            for dep in b.dependencies():
                assert not function_is_replacement(dep)

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert linear_eq._references_dropped
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in linear_eq.dependencies():
                assert function_is_replacement(dep)
            for dep in b.dependencies():
                assert not function_is_replacement(dep)

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
                assert not function_is_replacement(dep)

            M = WeakAlias(M)
            b = WeakAlias(b)

            assert len(manager._to_drop_references) == 1
            assert not b._references_dropped
            assert not M._references_dropped
            for dep in b.dependencies():
                assert not function_is_replacement(dep)

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert b._references_dropped
            assert M._references_dropped
            for dep in b.dependencies():
                assert function_is_replacement(dep)

        M = IdentityMatrix()

        J = Functional(name="J")
        InnerProduct(J.function(), z, z, M=M).solve()
        return J

    m = Constant(np.sqrt(2.0), name="m")

    start_manager()
    J = forward(m, forward_run=True)
    stop_manager()

    J_val = J.value()
    info(f"J = {J_val:.16e}")
    assert abs(J_val - 36.0) < 1.0e-13

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, dJ=dJ,
                            seed=5.0e-4)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, ddJ=ddJ,
                            seed=5.0e-4)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, m, tlm_order=1, dMs=(Constant(1.0),),
                                seed=5.0e-4)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                        dMs=(Constant(1.0),), seed=5.0e-4)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                        dMs=(Constant(1.0), Constant(1.0)),
                                        seed=5.0e-4)
    assert min_order > 2.00


@pytest.mark.numpy
@seed_test
def test_Referrers_FixedPointEquation(setup_test, test_leaks, test_default_dtypes):  # noqa: E501
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
                function_assign(
                    x,
                    0.5 * (function_scalar_value(x0) ** 2
                           + function_scalar_value(m))
                    / function_scalar_value(x0))

            def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
                return b

            def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
                if dep_index == 1:
                    x0, m = nl_deps
                    F = function_new_conjugate_dual(x0)
                    function_assign(
                        F,
                        (0.5 * function_scalar_value(adj_x)
                         * (function_scalar_value(m)
                            / (function_scalar_value(x0) ** 2) - 1.0)).conjugate())  # noqa: E501
                    return F
                elif dep_index == 2:
                    x0, m = nl_deps
                    F = function_new_conjugate_dual(x0)
                    function_assign(
                        F,
                        (-0.5 * function_scalar_value(adj_x)
                         / function_scalar_value(x0)).conjugate())
                    return F
                else:
                    raise IndexError("Unexpected dep_index")

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
            for eq in [fp_eq, eq0, eq1]:
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not function_is_replacement(dep)
            del eq

            fp_eq = WeakAlias(fp_eq)

            assert len(manager._to_drop_references) == 1
            for eq in [fp_eq, eq0, eq1]:
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not function_is_replacement(dep)
            del eq

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            assert fp_eq._references_dropped
            for dep in fp_eq.dependencies():
                assert function_is_replacement(dep)
            for eq in [eq0, eq1]:
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not function_is_replacement(dep)
            del eq

        eq0.solve()
        eq1.solve()

        if forward_run:
            assert len(manager._to_drop_references) == 0
            for eq in [eq0, eq1]:
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not function_is_replacement(dep)
            del eq

            eq0 = WeakAlias(eq0)
            eq1 = WeakAlias(eq1)

            assert len(manager._to_drop_references) == 2
            for eq in [eq0, eq1]:
                assert not eq._references_dropped
                for dep in eq.dependencies():
                    assert not function_is_replacement(dep)
            del eq

            manager.drop_references()

            assert len(manager._to_drop_references) == 0
            for eq in [eq0, eq1]:
                assert eq._references_dropped
                for dep in eq.dependencies():
                    assert function_is_replacement(dep)
            del eq

        J = Functional(name="J")
        J.assign(x1)
        return J

    m = Constant(2.0, name="m")

    start_manager()
    J = forward(m, forward_run=True)
    stop_manager()

    J_val = J.value()
    info(f"J = {J_val:.16e}")
    assert abs(J_val - np.sqrt(2.0)) < 1.0e-15

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, dM=Constant(1.0), J_val=J_val, dJ=dJ)
    assert min_order > 1.99


@pytest.mark.numpy
@pytest.mark.parametrize("n_steps, snaps_in_ram", [(1, 1),
                                                   (10, 1),
                                                   (10, 2),
                                                   (10, 3),
                                                   (10, 5),
                                                   (100, 3),
                                                   (100, 5),
                                                   (100, 10),
                                                   (100, 20),
                                                   (200, 5),
                                                   (200, 10),
                                                   (200, 20),
                                                   (200, 50),
                                                   (1000, 50)])
@no_space_type_checking
@seed_test
def test_binomial_checkpointing(setup_test, test_leaks, test_default_dtypes,
                                tmp_path, n_steps, snaps_in_ram):
    n_forward_solves = [0]

    class Counter(Instruction):
        def forward_solve(self, X, deps=None):
            n_forward_solves[0] += 1

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
        DotProduct(J.function(), m, m).solve()
        return J

    m = Constant(1.0, name="m", static=True)

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)

    info(f"Number of forward steps        : {n_forward_solves[0]:d}")
    n_forward_solves_optimal = optimal_steps(n_steps, snaps_in_ram)
    info(f"Optimal number of forward steps: {n_forward_solves_optimal:d}")
    assert n_forward_solves[0] == n_forward_solves_optimal

    min_order = taylor_test(forward, m, J_val=J.value(), dJ=dJ)
    assert min_order > 1.99


@pytest.mark.numpy
@pytest.mark.parametrize("max_degree", [1, 2, 3, 4, 5])
@no_space_type_checking
@seed_test
def test_TangentLinearMap_finalizes(setup_test, test_leaks, test_default_dtypes,  # noqa: E501
                                    max_degree):
    m = Constant(1.0, name="m")
    dm = Constant(1.0, name="dm")
    configure_tlm(*[(m, dm) for i in range(max_degree)])

    start_manager()
    x = Constant(0.0, name="x")
    DotProduct(x, m, m).solve()
    stop_manager()


@pytest.mark.numpy
@seed_test
def test_tlm_annotation(setup_test, test_leaks):
    F = Constant(1.0, name="F")
    zeta = Constant(1.0, name="zeta")
    G = Constant(1.0, name="G")

    reset_manager()
    configure_tlm((F, zeta))
    start_manager()
    Assignment(G, F).solve()
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 2

    reset_manager()
    configure_tlm((F, zeta))
    start_manager()
    stop_manager(tlm=False)
    Assignment(G, F).solve()
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 0

    reset_manager()
    configure_tlm((F, zeta), (F, zeta))
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    Assignment(G, F).solve()
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 3

    reset_manager()
    configure_tlm((F, zeta), (F, zeta))
    configure_tlm((F, zeta), annotate=False)
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    Assignment(G, F).solve()
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 1

    reset_manager()
    configure_tlm((F, zeta))
    configure_tlm((F, zeta), (F, zeta), annotate=False)
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    Assignment(G, F).solve()
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 2
