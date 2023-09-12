#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    CachedHessian, DEFAULT_COMM, EmptyEquation, Float, Functional, Hessian,
    Instruction, compute_gradient, configure_checkpointing, configure_tlm,
    manager, manager as _manager, new_block, paused_float_overloading,
    reset_manager, start_manager, stop_manager, taylor_test, taylor_test_tlm,
    taylor_test_tlm_adjoint)
from tlm_adjoint.checkpoint_schedules.binomial import optimal_steps

from .test_base import seed_test, setup_test  # noqa: F401

import itertools
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.base
@seed_test
def test_EmptyEquation(setup_test):  # noqa: F811
    def forward(F):
        EmptyEquation().solve()

        F_sq = F * F

        J = Functional(name="J")
        J.assign(F_sq * F_sq)
        return J

    F = Float(name="F")
    F.assign(-2.0)

    start_manager()
    J = forward(F)
    stop_manager()

    manager = _manager()
    manager.finalize()
    assert len(manager._blocks[0][0].X()) == 0

    J_val = J.value()
    assert abs(J_val - 16.0) == 0.0

    dJ = compute_gradient(J, F)

    dm = Float(1.0, name="dm")

    min_order = taylor_test(forward, F, J_val=J_val, dJ=dJ, dM=dm)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, F, J_val=J_val, ddJ=ddJ, dM=dm)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, F, tlm_order=1, dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=1, dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=2,
                                        dMs=(dm, dm))
    assert min_order > 1.99


@pytest.mark.base
@seed_test
def test_empty(setup_test):  # noqa: F811
    def forward(m):
        return Functional(name="J")

    m = Float(name="m")

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)
    assert dJ.value() == 0.0


@pytest.mark.base
@pytest.mark.parametrize("n_steps, snaps_in_ram", [(1, 1),
                                                   (10, 1),
                                                   (10, 2),
                                                   (10, 3),
                                                   (10, 5),
                                                   (100, 3),
                                                   (100, 5),
                                                   (100, 10),
                                                   (100, 20)])
@seed_test
def test_binomial_checkpointing(setup_test,  # noqa: F811
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
        J.assign(m * m)
        return J

    m = Float(1.0, name="m")

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)

    print(f"Number of forward steps        : {n_forward_solves:d}")
    n_forward_solves_optimal = optimal_steps(n_steps, snaps_in_ram)
    print(f"Optimal number of forward steps: {n_forward_solves_optimal:d}")
    assert n_forward_solves == n_forward_solves_optimal

    min_order = taylor_test(forward, m, J_val=J.value(), dJ=dJ)
    assert min_order > 1.99


@pytest.mark.base
@pytest.mark.parametrize("max_degree", [1, 2, 3, 4, 5])
@seed_test
def test_TangentLinearMap_finalizes(setup_test,  # noqa: F811
                                    max_degree):
    m = Float(1.0, name="m")
    dm = Float(1.0, name="dm")
    configure_tlm(*((m, dm) for _ in range(max_degree)))

    start_manager()
    x = Float(0.0, name="x")
    x.assign(m * m)
    stop_manager()


@pytest.mark.base
@seed_test
def test_tlm_annotation(setup_test):  # noqa: F811
    F = Float(1.0, name="F")
    zeta = Float(1.0, name="zeta")
    G = Float(1.0, name="G")

    reset_manager()
    configure_tlm((F, zeta))
    start_manager()
    G.assign(F)
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 2

    reset_manager()
    configure_tlm((F, zeta))
    start_manager()
    stop_manager(tlm=False)
    G.assign(F)
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 0

    reset_manager()
    configure_tlm((F, zeta), (F, zeta))
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    G.assign(F)
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 3

    reset_manager()
    configure_tlm((F, zeta), (F, zeta))
    configure_tlm((F, zeta), annotate=False)
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    G.assign(F)
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 1

    reset_manager()
    configure_tlm((F, zeta))
    configure_tlm((F, zeta), (F, zeta), annotate=False)
    manager().function_tlm(G, (F, zeta), (F, zeta))
    start_manager()
    G.assign(F)
    stop_manager()

    assert len(manager()._blocks) == 0 and len(manager()._block) == 2


@pytest.mark.base
@pytest.mark.parametrize("cp_method", ["memory", "multistage"])
@seed_test
def test_random_computational_graph(setup_test,  # noqa: F811
                                    tmp_path, cp_method):
    N = 20

    if cp_method == "memory":
        configure_checkpointing("memory", {"drop_references": False})
    else:
        assert cp_method == "multistage"
        configure_checkpointing("multistage",
                                {"blocks": N, "snaps_on_disk": 0,
                                 "snaps_in_ram": 5,
                                 "path": str(tmp_path / "checkpoints~")})

    for max_N_terms in [5, 8, 11] * {"memory": 2, "multistage": 1}[cp_method]:
        state = None
        Cs = []

        def forward(m):
            nonlocal state

            if state is None:
                state = np.random.get_state()
            else:
                np.random.set_state(state)

            X = [Float().assign(m)]

            for n in range(N):
                for _ in range(np.random.randint(5)):
                    Y = [X[np.random.randint(len(X))]
                         for _ in range(np.random.randint(max_N_terms))]
                    C = [cls(0.1 + np.random.random())
                         for cls in (np.random.choice([float, Float]) for _ in Y)]  # noqa: E501
                    Cs.extend(C)
                    with paused_float_overloading():
                        X.append(Float().assign(sum(c * y for c, y in zip(C, Y))))  # noqa: E501
                if n < N - 1:
                    new_block()
            if X[-1].value() == 0.0:
                return X[-1] ** 4 + (X[0] + 0.5) ** 4
            else:
                return X[-1] ** 4

        m = Float(-1.0)
        dm = Float(-1.0, name="dm")
        J = None

        for prune_forward, prune_adjoint, prune_replay in itertools.product(
                [False, True], [False, True],
                {"memory": [True], "multistage": [False, True]}[cp_method]):
            if J is None or manager()._cp_schedule.is_exhausted:
                reset_manager()
                start_manager()
                J = forward(m)
                stop_manager()

            for c in Cs:
                if isinstance(c, Float):
                    # Tests that CachedHessian uses stored values
                    c.assign(np.NAN)

            J_val = J.value()

            N_eq = sum(len(block) for block in (list(manager()._blocks)
                                                + [manager()._block]))
            N_adj = 0

            def callback(J_i, n, i, eq, adj_X):
                nonlocal N_adj

                if n >= 0 and n < N and adj_X is not None:
                    N_adj += 1

            dJ = compute_gradient(J, m, callback=callback,
                                  prune_forward=prune_forward,
                                  prune_adjoint=prune_adjoint,
                                  prune_replay=prune_replay)

            print(f"{N_eq} forward variables computed")
            # Excludes control and functional blocks
            print(f"{N_adj} adjoint variables computed")
            if prune_forward or prune_adjoint:
                assert N_adj <= N_eq
            else:
                assert N_adj == N_eq

            min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ, dM=dm,
                                    size=3)
            assert min_order > 2.00

        ddJ = Hessian(forward)
        min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm,
                                size=3)
        assert min_order > 3.00

        if cp_method == "memory":
            ddJ = CachedHessian(J, cache_adjoint=False)
            min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm,
                                    size=3)
            assert min_order > 3.00

            ddJ = CachedHessian(J, cache_adjoint=True)
            min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm,
                                    size=3)
            assert min_order > 3.00
            min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ, dM=dm,
                                    size=3)
            assert min_order > 3.00

        min_order = taylor_test_tlm(forward, m, tlm_order=1, dMs=(dm,),
                                    size=3)
        assert min_order > 2.00

        min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1,
                                            dMs=(dm,), size=3)
        assert min_order > 2.00

        min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2,
                                            dMs=(dm, dm), size=3)
        assert min_order > 2.00
