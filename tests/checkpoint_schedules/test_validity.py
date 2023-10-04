#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.checkpoint_schedules import \
    Clear, Configure, Forward, Reverse, Read, Write, EndForward, EndReverse
from tlm_adjoint.checkpoint_schedules import \
    (MemoryCheckpointSchedule,
     PeriodicDiskCheckpointSchedule,
     MultistageCheckpointSchedule,
     TwoLevelCheckpointSchedule,
     HRevolveCheckpointSchedule,
     MixedCheckpointSchedule)

import functools
import pytest

try:
    import hrevolve
except ImportError:
    hrevolve = None
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None

pytestmark = pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.size > 1,
    reason="tests must be run in serial")


def memory(n, s):
    return (MemoryCheckpointSchedule(),
            {"RAM": 0, "disk": 0}, 1 + n)


def periodic_disk(n, s, *, period):
    return (PeriodicDiskCheckpointSchedule(period),
            {"RAM": 0, "disk": 1 + (n - 1) // period}, period)


def multistage(n, s):
    return (MultistageCheckpointSchedule(n, 0, s),
            {"RAM": 0, "disk": s}, 1)


def two_level(n, s, *, period):
    return (TwoLevelCheckpointSchedule(period, s, binomial_storage="RAM"),
            {"RAM": s, "disk": 1 + (n - 1) // period}, 1)


def h_revolve(n, s):
    if s <= 1:
        return (None,
                {"RAM": 0, "disk": 0}, 0)
    else:
        return (HRevolveCheckpointSchedule(n, s // 2, s - (s // 2)),
                {"RAM": s // 2, "disk": s - (s // 2)}, 1)


def mixed(n, s):
    return (MixedCheckpointSchedule(n, s),
            {"RAM": 0, "disk": s}, 1)


@pytest.mark.checkpoint_schedules
@pytest.mark.parametrize(
    "schedule, schedule_kwargs",
    [(memory, {}),
     (periodic_disk, {"period": 1}),
     (periodic_disk, {"period": 2}),
     (periodic_disk, {"period": 7}),
     (periodic_disk, {"period": 10}),
     (multistage, {}),
     (two_level, {"period": 1}),
     (two_level, {"period": 2}),
     (two_level, {"period": 7}),
     (two_level, {"period": 10}),
     pytest.param(
         h_revolve, {},
         marks=pytest.mark.skipif(hrevolve is None,
                                  reason="H-Revolve not available")),
     (mixed, {})])
@pytest.mark.parametrize("n, S", [(1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (10, tuple(range(1, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))])
def test_validity(schedule, schedule_kwargs,
                  n, S):
    @functools.singledispatch
    def action(cp_action):
        raise TypeError("Unexpected action")

    @action.register(Clear)
    def action_clear(cp_action):
        if cp_action.clear_ics:
            ics.clear()
        if cp_action.clear_data:
            data.clear()

    @action.register(Configure)
    def action_configure(cp_action):
        nonlocal store_ics, store_data

        store_ics = cp_action.store_ics
        store_data = cp_action.store_data

    @action.register(Forward)
    def action_forward(cp_action):
        nonlocal model_n

        # Start at the current location of the forward
        assert model_n is not None and cp_action.n0 == model_n
        # If the schedule has been finalized, end at or before the end of the
        # forward
        assert cp_schedule.max_n is None or cp_action.n1 <= n

        if cp_schedule.max_n is not None:
            # Do not advance further than the current location of the adjoint
            assert cp_action.n1 <= n - model_r
        n1 = min(cp_action.n1, n)

        if store_ics:
            # No forward restart data for these steps is stored
            assert len(ics.intersection(range(cp_action.n0, n1))) == 0

        if store_data:
            # No non-linear dependency data for these steps is stored
            assert len(data.intersection(range(cp_action.n0, n1))) == 0

        model_n = n1
        if store_ics:
            ics.update(range(cp_action.n0, n1))
        if store_data:
            data.update(range(cp_action.n0, n1))
        if n1 == n:
            cp_schedule.finalize(n1)

    @action.register(Reverse)
    def action_reverse(cp_action):
        nonlocal model_r

        # Start at the current location of the adjoint
        assert cp_action.n1 == n - model_r
        # Advance at least one step
        assert cp_action.n0 < cp_action.n1
        # Non-linear dependency data for these steps is stored
        assert data.issuperset(range(cp_action.n0, cp_action.n1))

        model_r += cp_action.n1 - cp_action.n0

    @action.register(Read)
    def action_read(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots[cp_action.storage]

        cp = snapshots[cp_action.storage][cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data
        # The checkpoint contains forward restart or non-linear dependency data
        assert len(cp[0]) > 0 or len(cp[1]) > 0

        # The checkpoint data is before the current location of the adjoint
        assert cp_action.n < n - model_r

        model_n = None

        if len(cp[0]) > 0:
            ics.clear()
            ics.update(cp[0])
            model_n = cp_action.n

            # Can advance the forward to the current location of the adjoint
            assert ics.issuperset(range(model_n, n - model_r))

        if len(cp[1]) > 0:
            data.clear()
            data.update(cp[1])

        if cp_action.delete:
            del snapshots[cp_action.storage][cp_action.n]

    @action.register(Write)
    def action_write(cp_action):
        # The checkpoint contains forward restart or non-linear dependency data
        assert len(ics) > 0 or len(data) > 0

        # The checkpoint location is associated with the earliest step for
        # which data has been stored
        if len(ics) > 0:
            if len(data) > 0:
                assert cp_action.n == min(min(ics), min(data))
            else:
                assert cp_action.n == min(ics)
        elif len(data) > 0:
            assert cp_action.n == min(data)

        snapshots[cp_action.storage][cp_action.n] = (set(ics), set(data))

    @action.register(EndForward)
    def action_end_forward(cp_action):
        # The correct number of forward steps has been taken
        assert model_n is not None and model_n == n

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        nonlocal model_r

        # The correct number of adjoint steps has been taken
        assert model_r == n

        if not cp_action.exhausted:
            model_r = 0

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0

        store_ics = False
        ics = set()
        store_data = False
        data = set()

        snapshots = {"RAM": {}, "disk": {}}

        cp_schedule, storage_limits, data_limit = schedule(n, s, **schedule_kwargs)  # noqa: E501
        if cp_schedule is None:
            pytest.skip("Incompatible with schedule type")
        assert cp_schedule.n == 0
        assert cp_schedule.r == 0
        assert cp_schedule.max_n is None or cp_schedule.max_n == n

        while True:
            cp_action = next(cp_schedule)
            action(cp_action)

            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n is None or model_n == cp_schedule.n
            assert model_r == cp_schedule.r
            assert cp_schedule.max_n is None or cp_schedule.max_n == n

            # Checkpoint storage limits are not exceeded
            for storage_type, storage_limit in storage_limits.items():
                assert len(snapshots[storage_type]) <= storage_limit
            # Data storage limit is not exceeded
            assert min(1, len(ics)) + len(data) <= data_limit

            if isinstance(cp_action, EndReverse):
                break
