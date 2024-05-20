from .schedule import (
    CheckpointSchedule, Clear, Configure, Forward, Reverse, Read, Write,
    EndForward, EndReverse)

import enum
import functools
import numpy as np
import warnings

try:
    import numba
    from numba import njit
except ModuleNotFoundError:
    numba = None

    def njit(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped_fn


__all__ = \
    [
        "MixedCheckpointSchedule"
    ]


class StepType(enum.IntEnum):
    NONE = 0
    FORWARD = 1
    FORWARD_REVERSE = 2
    WRITE_DATA = 3
    WRITE_ICS = 4


def cache_step(fn):
    _cache = {}

    @functools.wraps(fn)
    def wrapped_fn(n, s):
        # Avoid some cache misses
        s = min(s, n - 1)
        if (n, s) not in _cache:
            _cache[(n, s)] = fn(n, s)
        return _cache[(n, s)]

    return wrapped_fn


@cache_step
def optimal_steps(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n <= s + 1:
        return n
    elif s == 1:
        return n * (n + 1) // 2 - 1
    else:
        m = 1 + optimal_steps(n - 1, s - 1)
        for i in range(2, n):
            m = min(
                m,
                i
                + optimal_steps(i, s)
                + optimal_steps(n - i, s - 1))
        return m


@cache_step
def mixed_step_memoization(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n == 1:
        return (StepType.FORWARD_REVERSE, 1, 1)
    elif n <= s + 1:
        return (StepType.WRITE_DATA, 1, n)
    elif s == 1:
        return (StepType.WRITE_ICS, n - 1, n * (n + 1) // 2 - 1)
    else:
        m = None
        for i in range(2, n):
            m1 = (
                i
                + mixed_step_memoization(i, s)[2]
                + mixed_step_memoization(n - i, s - 1)[2])
            if m is None or m1 <= m[2]:
                m = (StepType.WRITE_ICS, i, m1)
        if m is None:
            raise RuntimeError("Failed to determine total number of steps")
        m1 = 1 + mixed_step_memoization(n - 1, s - 1)[2]
        if m1 < m[2]:
            m = (StepType.WRITE_DATA, 1, m1)
        return m


_NONE = int(StepType.NONE)
_FORWARD = int(StepType.FORWARD)
_FORWARD_REVERSE = int(StepType.FORWARD_REVERSE)
_WRITE_DATA = int(StepType.WRITE_DATA)
_WRITE_ICS = int(StepType.WRITE_ICS)


@njit
def mixed_steps_tabulation(n, s):
    schedule = np.zeros((n + 1, s + 1, 3), dtype=np.int_)
    schedule[:, :, 0] = _NONE
    schedule[:, :, 1] = 0
    schedule[:, :, 2] = -1

    for s_i in range(s + 1):
        schedule[1, s_i, :] = (_FORWARD_REVERSE, 1, 1)
    for s_i in range(1, s + 1):
        for n_i in range(2, n + 1):
            if n_i <= s_i + 1:
                schedule[n_i, s_i, :] = (_WRITE_DATA, 1, n_i)
            elif s_i == 1:
                schedule[n_i, s_i, :] = (_WRITE_ICS, n_i - 1, n_i * (n_i + 1) // 2 - 1)  # noqa: E501
            else:
                for i in range(2, n_i):
                    assert schedule[i, s_i, 2] > 0
                    assert schedule[n_i - i, s_i - 1, 2] > 0
                    m1 = (
                        i
                        + schedule[i, s_i, 2]
                        + schedule[n_i - i, s_i - 1, 2])
                    if schedule[n_i, s_i, 2] < 0 or m1 <= schedule[n_i, s_i, 2]:  # noqa: E501
                        schedule[n_i, s_i, :] = (_WRITE_ICS, i, m1)
                if schedule[n_i, s_i, 2] < 0:
                    raise RuntimeError("Failed to determine total number of "
                                       "steps")
                assert schedule[n_i - 1, s_i - 1, 2] > 0
                m1 = 1 + schedule[n_i - 1, s_i - 1, 2]
                if m1 < schedule[n_i, s_i, 2]:
                    schedule[n_i, s_i, :] = (_WRITE_DATA, 1, m1)
    return schedule


class MixedCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule which mixes storage of forward restart data and
    non-linear dependency data in checkpointing units. Assumes that the data
    required to restart the forward has the same size as the data required to
    advance the adjoint over a step.

    An updated version of the algorithm described in

        - James R. Maddison, 'On the implementation of checkpointing with
          high-level algorithmic differentiation',
          https://arxiv.org/abs/2305.09568v1, 2023

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots: The number of available checkpointing units.
    :arg storage: Checkpointing unit storage location. Either `'RAM'` or
        `'disk'`.
    """

    def __init__(self, max_n, snapshots, *, storage="disk"):
        if snapshots < min(1, max_n - 1):
            raise ValueError("Invalid number of snapshots")
        if storage not in {"RAM", "disk"}:
            raise ValueError("Invalid storage")

        super().__init__(max_n)
        self._exhausted = False
        self._snapshots = min(snapshots, max_n - 1)
        self._storage = storage

    def iter(self):
        snapshot_n = set()
        snapshots = []

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")

        if numba is None:
            warnings.warn("Numba not available -- using memoization",
                          RuntimeWarning)
            schedule = None
        else:
            schedule = mixed_steps_tabulation(self._max_n, self._snapshots)

        while True:
            step_type = StepType.NONE
            while self._n < self._max_n - self._r:
                n0 = self._n
                reuse_snapshot = bool(n0 in snapshot_n)

                if schedule is None:
                    step_type, n1, _ = mixed_step_memoization(
                        self._max_n - self._r - n0,
                        self._snapshots - len(snapshots) + int(reuse_snapshot))
                else:
                    step_type, n1, _ = schedule[
                        self._max_n - self._r - n0,
                        self._snapshots - len(snapshots) + int(reuse_snapshot)]
                n1 += n0
                if reuse_snapshot and \
                        (snapshots[-1][:2] != (step_type, n0)
                         or snapshots[-1][2] < n1):
                    raise RuntimeError("Invalid checkpointing state")

                if step_type == StepType.FORWARD_REVERSE:
                    if n1 > n0 + 1:
                        yield Configure(False, False)
                        self._n = n1 - 1
                        yield Forward(n0, n1 - 1)
                        yield Clear(True, True)
                    elif n1 <= n0:
                        raise RuntimeError("Invalid step")
                    yield Configure(False, True)
                    self._n += 1
                    yield Forward(n1 - 1, n1)
                elif step_type == StepType.FORWARD:
                    if n1 <= n0:
                        raise RuntimeError("Invalid step")
                    yield Configure(False, False)
                    self._n = n1
                    yield Forward(n0, n1)
                    yield Clear(True, True)
                elif step_type == StepType.WRITE_DATA:
                    if n1 != n0 + 1:
                        raise RuntimeError("Invalid step")
                    if reuse_snapshot:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    yield Configure(False, True)
                    self._n = n1
                    yield Forward(n0, n1)
                    snapshot_n.add(n0)
                    snapshots.append((StepType.WRITE_DATA, n0, n1))
                    yield Write(n0, self._storage)
                    yield Clear(True, True)
                elif step_type == StepType.WRITE_ICS:
                    if n1 <= n0 + 1:
                        raise ValueError("Invalid step")
                    yield Configure(not reuse_snapshot, False)
                    self._n = n1
                    yield Forward(n0, n1)
                    if not reuse_snapshot:
                        if len(snapshots) > self._snapshots - 1:
                            raise RuntimeError("Invalid checkpointing state")
                        snapshot_n.add(n0)
                        snapshots.append((StepType.WRITE_ICS, n0, n1))
                        yield Write(n0, self._storage)
                    yield Clear(True, True)
                else:
                    raise RuntimeError("Unexpected step type")
            if self._n != self._max_n - self._r:
                raise RuntimeError("Invalid checkpointing state")
            if step_type not in {StepType.NONE, StepType.FORWARD_REVERSE}:
                raise RuntimeError("Invalid checkpointing state")

            if self._r == 0:
                yield EndForward()

            self._r += 1
            yield Reverse(self._max_n - self._r + 1, self._max_n - self._r)
            yield Clear(True, True)

            if self._r == self._max_n:
                break

            cp_step_type, cp_n, _ = snapshots[-1]
            if cp_step_type not in {StepType.WRITE_ICS, StepType.WRITE_DATA}:
                raise RuntimeError("Invalid checkpointing state")

            if schedule is None:
                next_step_type, _, _ = mixed_step_memoization(
                    self._max_n - self._r - cp_n,
                    self._snapshots - len(snapshots) + 1)
            else:
                next_step_type, _, _ = schedule[
                    self._max_n - self._r - cp_n,
                    self._snapshots - len(snapshots) + 1]
            cp_delete = (cp_step_type != next_step_type)
            if cp_delete:
                snapshot_n.remove(cp_n)
                snapshots.pop()

            if cp_step_type == StepType.WRITE_ICS:
                if cp_n + 1 >= self._max_n - self._r:
                    raise RuntimeError("Invalid checkpointing state")
                self._n = cp_n
            elif cp_step_type == StepType.WRITE_DATA:
                # Non-linear dependency data checkpoint
                if not cp_delete or cp_n + 1 != self._max_n - self._r:
                    # We cannot advance from a loaded non-linear dependency
                    # checkpoint, and so we expect to use it immediately
                    raise RuntimeError("Invalid checkpointing state")
                # Note that we cannot in general restart the forward here
                self._n = cp_n + 1
            yield Read(cp_n, self._storage, cp_delete)
            if cp_step_type == StepType.WRITE_ICS:
                yield Clear(True, True)

        if len(snapshot_n) > 0 or len(snapshots) > 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse(True)

    @property
    def is_exhausted(self):
        return self._exhausted

    @property
    def uses_disk_storage(self):
        return self._max_n > 1 and self._storage == "disk"
