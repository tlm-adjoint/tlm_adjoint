from tlm_adjoint.checkpoint_schedules import (
    MultistageCheckpointSchedule, Clear, Configure, Forward, Reverse, Read,
    Write, EndForward, EndReverse)
from tlm_adjoint.checkpoint_schedules.binomial import optimal_steps

import functools
import pytest

try:
    import checkpoint_schedules
except ModuleNotFoundError:
    checkpoint_schedules = None
try:
    import mpi4py.MPI as MPI
except ModuleNotFoundError:
    MPI = None

pytestmark = pytest.mark.skipif(
    MPI is not None and MPI.COMM_WORLD.size > 1,
    reason="tests must be run in serial")


def checkpoint_schedules_multistage(
        max_n, snapshots_in_ram, snapshots_on_disk, *,
        trajectory="maximum"):
    if checkpoint_schedules is None:
        pytest.skip("checkpoint_schedules not available")

    from tlm_adjoint.checkpoint_schedules.checkpoint_schedules \
        import MultistageCheckpointSchedule
    return MultistageCheckpointSchedule(
        max_n, snapshots_in_ram, snapshots_on_disk,
        trajectory=trajectory)


@pytest.mark.checkpoint_schedules
@pytest.mark.parametrize("schedule", [MultistageCheckpointSchedule,
                                      checkpoint_schedules_multistage])
@pytest.mark.parametrize("trajectory", ["revolve",
                                        "maximum"])
@pytest.mark.parametrize("n, S", [(1, (0,)),
                                  (2, (1,)),
                                  (3, (1, 2)),
                                  (10, tuple(range(1, 10))),
                                  (100, tuple(range(1, 100))),
                                  (250, tuple(range(25, 250, 25)))])
def test_MultistageCheckpointSchedule(schedule,
                                      trajectory,
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
        nonlocal model_n, model_steps

        # Start at the current location of the forward
        assert cp_action.n0 == model_n
        # Do not advance further than the current location of the adjoint
        assert cp_action.n1 <= n - model_r

        if store_ics:
            # Advance at least one step when storing forward restart data
            assert cp_action.n1 > cp_action.n0
            # Do not advance further than one step before the current location
            # of the adjoint
            assert cp_action.n1 < n - model_r
            # No data for these steps is stored
            assert len(ics.intersection(range(cp_action.n0, cp_action.n1))) == 0  # noqa: E501

        if store_data:
            # Advance exactly one step when storing non-linear dependency data
            assert cp_action.n1 == cp_action.n0 + 1
            # Start from one step before the current location of the adjoint
            assert cp_action.n0 == n - model_r - 1
            # No data for this step is stored
            assert len(data.intersection(range(cp_action.n0, cp_action.n1))) == 0  # noqa: E501

        # The forward is able to advance over these steps
        assert replay is None or replay.issuperset(range(cp_action.n0, cp_action.n1))  # noqa: E501

        model_n = cp_action.n1
        model_steps += cp_action.n1 - cp_action.n0
        if store_ics:
            ics.update(range(cp_action.n0, cp_action.n1))
        if store_data:
            data.update(range(cp_action.n0, cp_action.n1))

    @action.register(Reverse)
    def action_reverse(cp_action):
        nonlocal model_r

        # Start at the current location of the adjoint
        assert cp_action.n1 == n - model_r
        # Advance exactly one step
        assert cp_action.n0 == cp_action.n1 - 1
        # Non-linear dependency data for the step is stored
        assert cp_action.n0 in data

        model_r += 1
        replay.clear()

    @action.register(Read)
    def action_read(cp_action):
        nonlocal model_n

        # The checkpoint exists
        assert cp_action.n in snapshots
        assert cp_action.storage == "disk"

        cp = snapshots[cp_action.n]

        # No data is currently stored for this step
        assert cp_action.n not in ics
        assert cp_action.n not in data
        # The checkpoint contains forward restart data
        assert len(cp[0]) > 0
        assert len(cp[1]) == 0

        # The checkpoint data is at least one step away from the current
        # location of the adjoint
        assert cp_action.n < n - model_r
        # The loaded data is deleted iff it is exactly one step away from the
        # current location of the adjoint
        assert cp_action.delete == (cp_action.n == n - model_r - 1)

        ics.clear()
        ics.update(cp[0])
        assert len(replay) == 0
        replay.update(cp[0])
        model_n = cp_action.n

        # Can advance the forward to the current location of the adjoint
        assert ics.issuperset(range(model_n, n - model_r))

        if cp_action.delete:
            del snapshots[cp_action.n]

    @action.register(Write)
    def action_write(cp_action):
        assert cp_action.storage == "disk"
        assert cp_schedule.uses_disk_storage

        # Written data consists of forward restart data
        assert len(ics) > 0
        assert len(data) == 0

        # The checkpoint location is associated with the earliest step for
        # which data has been stored
        assert cp_action.n == min(ics)

        assert cp_action.n not in snapshots
        snapshots[cp_action.n] = (set(ics), set(data))

    @action.register(EndForward)
    def action_end_forward(cp_action):
        nonlocal replay

        # The correct number of forward steps has been taken
        assert model_n == n

        replay = set()

    @action.register(EndReverse)
    def action_end_reverse(cp_action):
        # The correct number of adjoint steps has been taken
        assert model_r == n
        # The schedule has concluded
        assert cp_action.exhausted

    for s in S:
        print(f"{n=:d} {s=:d}")

        model_n = 0
        model_r = 0
        model_steps = 0

        store_ics = False
        ics = set()
        store_data = False
        data = set()
        replay = None

        snapshots = {}

        cp_schedule = schedule(n, 0, s, trajectory=trajectory)
        assert n == 1 or cp_schedule.uses_disk_storage
        assert cp_schedule.n == 0
        assert cp_schedule.r == 0
        assert cp_schedule.max_n == n

        while True:
            cp_action = next(cp_schedule)
            action(cp_action)

            # The schedule state is consistent with both the forward and
            # adjoint
            assert model_n == cp_schedule.n
            assert model_r == cp_schedule.r
            assert cp_schedule.max_n == n

            # Either no data is being stored, or exactly one of forward restart
            # or non-linear dependency data is being stored
            assert not store_ics or not store_data
            assert len(ics) == 0 or len(data) == 0
            # Non-linear dependency data is stored for at most one step
            assert len(data) <= 1
            # Checkpoint storage limits are not exceeded
            assert len(snapshots) <= s

            if isinstance(cp_action, EndReverse):
                break

        # The correct total number of forward steps has been taken
        assert model_steps == optimal_steps(n, s)
        # No data is stored
        assert len(ics) == 0 and len(data) == 0 and len(replay) == 0
        # No checkpoints are stored
        assert len(snapshots) == 0

        # The schedule has concluded
        assert cp_schedule.is_exhausted
        try:
            next(cp_schedule)
        except StopIteration:
            pass
        except Exception:
            raise RuntimeError("Iterator not exhausted")
