#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from .schedule import CheckpointSchedule, Clear, Configure, Forward, Reverse, \
    Read, Write, EndForward, EndReverse

import enum
import functools


class StepType(enum.Enum):
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
def mixed_step(n, s):
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
                + mixed_step(i, s)[2]
                + mixed_step(n - i, s - 1)[2])
            if m is None or m1 <= m[2]:
                m = (StepType.WRITE_ICS, i, m1)
        if m is None:
            raise RuntimeError("Failed to determine total number of steps")
        m1 = 1 + mixed_step(n - 1, s - 1)[2]
        if m1 <= m[2]:
            m = (StepType.WRITE_DATA, 1, m1)
        return m


def cache_step_0(fn):
    _cache = {}

    @functools.wraps(fn)
    def wrapped_fn(n, s):
        # Avoid some cache misses
        s = min(s, n - 2)
        if (n, s) not in _cache:
            _cache[(n, s)] = fn(n, s)
        return _cache[(n, s)]

    return wrapped_fn


@cache_step_0
def mixed_step_0(n, s):
    if s < 0:
        raise ValueError("Invalid number of snapshots")
    if n < s + 2:
        raise ValueError("Invalid number of steps")

    if s == 0:
        return (StepType.FORWARD_REVERSE, n, n * (n + 1) // 2 - 1)
    else:
        m = None
        for i in range(1, n):
            m1 = (
                i
                + mixed_step(i, s + 1)[2]
                + mixed_step(n - i, s)[2])
            if m is None or m1 <= m[2]:
                m = (StepType.FORWARD, i, m1)
        if m is None:
            raise RuntimeError("Failed to determine total number of steps")
        return m


class MixedCheckpointSchedule(CheckpointSchedule):
    def __init__(self, max_n, snapshots, *, storage="disk"):
        if snapshots < min(1, max_n - 1):
            raise ValueError("Invalid number of snapshots")
        if storage not in ["RAM", "disk"]:
            raise ValueError("Invalid storage")

        super().__init__(max_n)
        self._exhausted = False
        self._snapshots = min(snapshots, max_n - 1)
        self._storage = storage

    def iter(self):
        snapshot_n = set()
        snapshots = []

        steps = 0

        def forward(n0, n1):
            nonlocal steps
            steps += n1 - n0
            return Forward(n0, n1)

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")

        step_type = StepType.NONE
        while True:
            while self._n < self._max_n - self._r:
                n0 = self._n
                if n0 in snapshot_n:
                    # n0 checkpoint exists
                    step_type, n1, _ = mixed_step_0(
                        self._max_n - self._r - n0,
                        self._snapshots - len(snapshots))
                else:
                    # n0 checkpoint does not exist
                    step_type, n1, _ = mixed_step(
                        self._max_n - self._r - n0,
                        self._snapshots - len(snapshots))
                n1 += n0

                if step_type == StepType.FORWARD_REVERSE:
                    if n1 > n0 + 1:
                        yield Configure(False, False)
                        self._n = n1 - 1
                        yield forward(n0, n1 - 1)
                        yield Clear(True, True)
                    elif n1 <= n0:
                        raise RuntimeError("Invalid step")
                    yield Configure(False, True)
                    self._n += 1
                    yield forward(n1 - 1, n1)
                elif step_type == StepType.FORWARD:
                    yield Configure(False, False)
                    self._n = n1
                    yield forward(n0, n1)
                    yield Clear(True, True)
                elif step_type == StepType.WRITE_DATA:
                    if n1 != n0 + 1:
                        raise RuntimeError("Invalid step")
                    yield Configure(False, True)
                    self._n = n1
                    yield forward(n0, n1)
                    if n0 in snapshot_n:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    snapshot_n.add(n0)
                    snapshots.append((StepType.READ_DATA, n0))
                    yield Write(n0, self._storage)
                    yield Clear(True, True)
                elif step_type == StepType.WRITE_ICS:
                    if n1 <= n0 + 1:
                        raise ValueError("Invalid step")
                    yield Configure(True, False)
                    self._n = n1
                    yield forward(n0, n1)
                    if n0 in snapshot_n:
                        raise RuntimeError("Invalid checkpointing state")
                    elif len(snapshots) > self._snapshots - 1:
                        raise RuntimeError("Invalid checkpointing state")
                    snapshot_n.add(n0)
                    snapshots.append((StepType.READ_ICS, n0))
                    yield Write(n0, self._storage)
                    yield Clear(True, True)
                else:
                    raise RuntimeError("Unexpected step type")
            if self._n != self._max_n - self._r:
                raise RuntimeError("Invalid checkpointing state")
            if step_type not in (StepType.FORWARD_REVERSE, StepType.READ_DATA):
                raise RuntimeError("Invalid checkpointing state")

            if self._r == 0:
                yield EndForward()

            self._r += 1
            yield Reverse(self._max_n - self._r + 1, self._max_n - self._r)
            yield Clear(True, True)

            if self._r == self._max_n:
                break

            step_type, cp_n = snapshots[-1]

            # Delete if we have (possibly after deleting this checkpoint)
            # enough storage left to store all non-linear dependency data
            cp_delete = (cp_n >= (self._max_n - self._r - 1
                                  - (self._snapshots - len(snapshots) + 1)))
            if cp_delete:
                snapshot_n.remove(cp_n)
                snapshots.pop()

            self._n = cp_n
            if step_type == StepType.READ_DATA:
                # Non-linear dependency data checkpoint
                if not cp_delete:
                    # We cannot advance from a loaded non-linear dependency
                    # checkpoint, and so we expect to use it immediately
                    raise RuntimeError("Invalid checkpointing state")
                # Note that we cannot in general restart the forward here
                self._n += 1
            elif step_type != StepType.READ_ICS:
                raise RuntimeError("Invalid checkpointing state")
            yield Read(cp_n, self._storage, cp_delete)
            if step_type == StepType.READ_ICS:
                yield Clear(True, True)

        if len(snapshot_n) > 0 or len(snapshots) > 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse(True)

    def is_exhausted(self):
        return self._exhausted

    def uses_disk_storage(self):
        return self._storage == "disk"
