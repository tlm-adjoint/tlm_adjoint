from checkpoint_schedules import StorageType
from checkpoint_schedules import (
    MixedCheckpointSchedule as _MixedCheckpointSchedule)

from .translation import translation

import enum
import functools

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
    READ_DATA = 5
    READ_ICS = 6


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
        if m1 <= m[2]:
            m = (StepType.WRITE_DATA, 1, m1)
        return m


class MixedCheckpointSchedule(translation(_MixedCheckpointSchedule)):
    """A checkpointing schedule which mixes storage of forward restart data and
    non-linear dependency data in checkpointing units. Assumes that the data
    required to restart the forward has the same size as the data required to
    advance the adjoint over a step.

    Described in

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
        super().__init__(
            max_n, snapshots,
            storage={"RAM": StorageType.RAM,
                     "disk": StorageType.DISK}[storage])
