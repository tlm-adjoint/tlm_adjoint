#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .schedule import (
    CheckpointSchedule, Clear, Configure, Forward, Reverse, Read, Write,
    EndForward, EndReverse)

import functools
from operator import itemgetter

try:
    import numba
    from numba import njit
except ImportError:
    numba = None

    def njit(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapped_fn

__all__ = \
    [
        "MultistageCheckpointSchedule",
        "TwoLevelCheckpointSchedule"
    ]


@njit
def n_advance(n, snapshots, *, trajectory="maximum"):
    # GW2000 reference:
    #   Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
    #   implementation of checkpointing for the reverse or adjoint mode of
    #   computational differentiation', ACM Transactions on Mathematical
    #   Software, 26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846

    if n < 1:
        raise ValueError("Require at least one block")
    if snapshots <= 0:
        raise ValueError("Require at least one snapshot")

    # Discard excess snapshots
    snapshots = max(min(snapshots, n - 1), 1)
    # Handle limiting cases
    if snapshots == 1:
        return n - 1  # Minimal storage
    elif snapshots == n - 1:
        return 1  # Maximal storage

    # Find t as in GW2000 Proposition 1 (note 'm' in GW2000 is 'n' here, and
    # 's' in GW2000 is 'snapshots' here). Compute values of beta as in equation
    # (1) of GW2000 as a side effect. We must have a minimal rerun of at least
    # 2 (the minimal rerun of 1 case is maximal storage, handled above) so we
    # start from t = 2.
    t = 2
    b_s_tm2 = 1
    b_s_tm1 = snapshots + 1
    b_s_t = ((snapshots + 1) * (snapshots + 2)) // 2
    while b_s_tm1 >= n or n > b_s_t:
        t += 1
        b_s_tm2 = b_s_tm1
        b_s_tm1 = b_s_t
        b_s_t = (b_s_t * (snapshots + t)) // t

    if trajectory == "maximum":
        # Return the maximal step size compatible with Fig. 4 of GW2000
        b_sm1_tm2 = (b_s_tm2 * snapshots) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm1_tm2:
            return n - b_s_tm1 + b_s_tm2
        b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
        b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm2_tm1 + b_sm1_tm2:
            return b_s_tm2 + b_sm1_tm2
        elif n <= b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
            return n - b_sm1_tm1 - b_sm2_tm1
        else:
            return b_s_tm1
    elif trajectory == "revolve":
        # GW2000, equation at the bottom of p. 34
        b_sm1_tm1 = (b_s_tm1 * snapshots) // (snapshots + t - 1)
        b_sm2_tm1 = (b_sm1_tm1 * (snapshots - 1)) // (snapshots + t - 2)
        if n <= b_s_tm1 + b_sm2_tm1:
            return b_s_tm2
        elif n < b_s_tm1 + b_sm1_tm1 + b_sm2_tm1:
            return n - b_sm1_tm1 - b_sm2_tm1
        else:
            return b_s_tm1
    else:
        raise ValueError("Unexpected trajectory: '{trajectory:s}'")


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
def optimal_extra_steps(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n == 1:
        return 0
    # Equation (2) of
    #   A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
    #   of checkpointing for the reverse or adjoint mode of computational
    #   differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
    #   19--45, 2000
    elif s == 1:
        return n * (n - 1) // 2
    else:
        m = None
        for i in range(1, n):
            m1 = (i
                  + optimal_extra_steps(i, s)
                  + optimal_extra_steps(n - i, s - 1))
            if m is None or m1 < m:
                m = m1
        if m is None:
            raise RuntimeError("Failed to determine number of extra steps")
        return m


def optimal_steps(n, s):
    return n + optimal_extra_steps(n, s)


def allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk, *,
                       write_weight=1.0, read_weight=1.0, delete_weight=0.0,
                       trajectory="maximum"):
    snapshots_in_ram = min(snapshots_in_ram, max_n - 1)
    snapshots_on_disk = min(snapshots_on_disk, max_n - 1)
    snapshots = min(snapshots_in_ram + snapshots_on_disk, max_n - 1)
    weights = [0.0 for _ in range(snapshots)]

    cp_schedule = MultistageCheckpointSchedule(max_n, snapshots, 0,
                                               trajectory=trajectory)

    snapshot_i = -1

    @functools.singledispatch
    def action(cp_action):
        raise TypeError(f"Unexpected checkpointing action: {cp_action}")

    @action.register(Read)
    def action_read(cp_action):
        nonlocal snapshot_i

        if snapshot_i < 0:
            raise RuntimeError("Invalid checkpointing state")
        weights[snapshot_i] += read_weight
        if cp_action.delete:
            weights[snapshot_i] += delete_weight
            snapshot_i -= 1

    @action.register(Write)
    def action_write(cp_action):
        nonlocal snapshot_i

        snapshot_i += 1
        if snapshot_i >= snapshots:
            raise RuntimeError("Invalid checkpointing state")
        weights[snapshot_i] += write_weight

    @action.register(Clear)
    @action.register(Configure)
    @action.register(Forward)
    @action.register(Reverse)
    @action.register(EndForward)
    @action.register(EndReverse)
    def action_pass(cp_action):
        pass

    # Run the schedule, keeping track of the total read/write/delete costs
    # associated with each storage location on the stack of checkpointing units

    while True:
        cp_action = next(cp_schedule)
        action(cp_action)
        if isinstance(cp_action, EndReverse):
            break

    assert snapshot_i == -1

    # Allocate the checkpointing units with highest cost to RAM, and the
    # remaining units to disk. For read and write costs of one and zero delete
    # costs the distribution of storage between RAM and disk is then equivalent
    # to that in
    #   Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal
    #   offline checkpointing', SIAM Journal on Scientific Computing, 31(3),
    #   pp. 1946--1967, 2009, doi: 10.1137/080718036

    allocation = ["disk" for _ in range(snapshots)]
    for i, _ in sorted(enumerate(weights), key=itemgetter(1),
                       reverse=True)[:snapshots_in_ram]:
        allocation[i] = "RAM"

    return tuple(weights), tuple(allocation)


class MultistageCheckpointSchedule(CheckpointSchedule):
    """A binomial checkpointing schedule using the approach described in

        - Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
          implementation of checkpointing for the reverse or adjoint mode of
          computational differentiation', ACM Transactions on Mathematical
          Software, 26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846

    hereafter referred to as GW2000.

    Uses a 'MultiStage' distribution of checkpoints between RAM and disk
    equivalent to that described in

        - Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal
          offline checkpointing', SIAM Journal on Scientific Computing, 31(3),
          pp. 1946--1967, 2009, doi: 10.1137/080718036

    The distribution between RAM and disk is determined using an initial run of
    the schedule.

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots_in_ram: The maximum number of forward restart checkpoints
        to store in memory.
    :arg snapshots_on_disk: The maximum number of forward restart checkpoints
        to store on disk.
    :arg trajectory: When advancing `n` forward steps with `s` checkpointing
        units available there are in general multiple solutions to the problem
        of determining the number of forward steps to advance before storing
        a new forward restart checkpoint -- see Fig. 4 of GW2000. This argument
        selects a solution:

            - `'revolve'`: The standard revolve solution, as specified in the
              equation at the bottom of p. 34 of GW2000.
            - `'maximum'`: The maximum possible number of steps, corresponding
              to the maximum step size compatible with the optimal region in
              Fig. 4 of GW2000.

    The argument names `snaps_in_ram` and `snaps_on_disk` originate from the
    corresponding arguments for the `dolfin_adjoint.solving.adj_checkpointing`
    function in dolfin-adjoint (see e.g. version 2017.1.0).
    """

    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 trajectory="maximum"):
        snapshots_in_ram = min(snapshots_in_ram, max_n - 1)
        snapshots_on_disk = min(snapshots_on_disk, max_n - 1)
        if snapshots_in_ram == 0:
            storage = tuple("disk" for _ in range(snapshots_on_disk))
        elif snapshots_on_disk == 0:
            storage = tuple("RAM" for _ in range(snapshots_in_ram))
        else:
            _, storage = allocate_snapshots(
                max_n, snapshots_in_ram, snapshots_on_disk,
                trajectory=trajectory)

        snapshots_in_ram = storage.count("RAM")
        snapshots_on_disk = storage.count("disk")

        super().__init__(max_n=max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._storage = storage
        self._exhausted = False
        self._trajectory = trajectory

    def iter(self):
        snapshots = []

        def write(n):
            if len(snapshots) >= self._snapshots_in_ram + self._snapshots_on_disk:  # noqa: E501
                raise RuntimeError("Invalid checkpointing state")
            snapshots.append(n)
            return self._storage[len(snapshots) - 1]

        # Forward

        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")
        while self._n < self._max_n - 1:
            yield Configure(True, False)

            n_snapshots = (self._snapshots_in_ram
                           + self._snapshots_on_disk
                           - len(snapshots))
            n0 = self._n
            n1 = n0 + n_advance(self._max_n - n0, n_snapshots,
                                trajectory=self._trajectory)
            assert n1 > n0
            self._n = n1
            yield Forward(n0, n1)

            cp_storage = write(n0)
            yield Write(n0, cp_storage)
            yield Clear(True, True)
        if self._n != self._max_n - 1:
            raise RuntimeError("Invalid checkpointing state")

        # Forward -> reverse

        yield Configure(False, True)

        self._n += 1
        yield Forward(self._n - 1, self._n)

        yield EndForward()

        self._r += 1
        yield Reverse(self._n, self._n - 1)
        yield Clear(True, True)

        # Reverse

        while self._r < self._max_n:
            if len(snapshots) == 0:
                raise RuntimeError("Invalid checkpointing state")
            cp_n = snapshots[-1]
            cp_storage = self._storage[len(snapshots) - 1]
            if cp_n == self._max_n - self._r - 1:
                snapshots.pop()
                self._n = cp_n
                yield Read(cp_n, cp_storage, True)
                yield Clear(True, True)
            else:
                self._n = cp_n
                yield Read(cp_n, cp_storage, False)
                yield Clear(True, True)

                yield Configure(False, False)

                n_snapshots = (self._snapshots_in_ram
                               + self._snapshots_on_disk
                               - len(snapshots) + 1)
                n0 = self._n
                n1 = n0 + n_advance(self._max_n - self._r - n0, n_snapshots,
                                    trajectory=self._trajectory)
                assert n1 > n0
                self._n = n1
                yield Forward(n0, n1)
                yield Clear(True, True)

                while self._n < self._max_n - self._r - 1:
                    yield Configure(True, False)

                    n_snapshots = (self._snapshots_in_ram
                                   + self._snapshots_on_disk
                                   - len(snapshots))
                    n0 = self._n
                    n1 = n0 + n_advance(self._max_n - self._r - n0, n_snapshots,  # noqa: E501
                                        trajectory=self._trajectory)
                    assert n1 > n0
                    self._n = n1
                    yield Forward(n0, n1)

                    cp_storage = write(n0)
                    yield Write(n0, cp_storage)
                    yield Clear(True, True)
                if self._n != self._max_n - self._r - 1:
                    raise RuntimeError("Invalid checkpointing state")

            yield Configure(False, True)

            self._n += 1
            yield Forward(self._n - 1, self._n)

            self._r += 1
            yield Reverse(self._n, self._n - 1)
            yield Clear(True, True)
        if self._r != self._max_n:
            raise RuntimeError("Invalid checkpointing state")
        if len(snapshots) != 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse(True)

    @property
    def is_exhausted(self):
        return self._exhausted

    @property
    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0


class TwoLevelCheckpointSchedule(CheckpointSchedule):
    """A two-level mixed periodic/binomial checkpointing schedule using the
    approach described in

        - Gavin J. Pringle, Daniel C. Jones, Sudipta Goswami, Sri Hari Krishna
          Narayanan, and Daniel Goldberg, 'Providing the ARCHER community with
          adjoint modelling tools for high-performance oceanographic and
          cryospheric computation', version 1.1, EPCC, 2016

    and in the supporting information for

        - D. N. Goldberg, T. A. Smith, S. H. K. Narayanan, P. Heimbach, and M.
          Morlighem, 'Bathymetric influences on Antarctic ice-shelf melt
          rates', Journal of Geophysical Research: Oceans, 125(11),
          e2020JC016370, 2020, doi: 10.1029/2020JC016370

    Online, unlimited adjoint calculations permitted.

    :arg period: Forward restart checkpoints are stored to disk every `period`
        forward steps in the initial forward calculation.
    :arg binomial_snapshots: The maximum number of additional forward restart
        checkpointing units to use when advancing the adjoint between periodic
        disk checkpoints.
    :arg binomial_storage: The storage to use for the additional forward
        restart checkpoints generated when advancing the adjoint between
        periodic disk checkpoints. Either `'RAM'` or `'disk'`.
    :arg binomial_trajectory: See the `trajectory` constructor argument for
        :class:`.MultistageCheckpointSchedule`.
    """

    def __init__(self, period, binomial_snapshots, *,
                 binomial_storage="disk",
                 binomial_trajectory="maximum"):
        if period < 1:
            raise ValueError("period must be positive")
        if binomial_storage not in ["RAM", "disk"]:
            raise ValueError("Invalid storage")

        super().__init__()

        self._period = period
        self._binomial_snapshots = binomial_snapshots
        self._binomial_storage = binomial_storage
        self._trajectory = binomial_trajectory

    def iter(self):
        # Forward

        while self._max_n is None:
            yield Configure(True, False)
            if self._max_n is not None:
                # Unexpected finalize
                raise RuntimeError("Invalid checkpointing state")
            n0 = self._n
            n1 = n0 + self._period
            self._n = n1
            yield Forward(n0, n1)

            # Finalize permitted here

            yield Write(n0, "disk")
            yield Clear(True, True)

        yield EndForward()

        while True:
            # Reverse

            while self._r < self._max_n:
                n = self._max_n - self._r - 1
                n0s = (n // self._period) * self._period
                n1s = min(n0s + self._period, self._max_n)
                if self._r != self._max_n - n1s:
                    raise RuntimeError("Invalid checkpointing state")
                del n, n1s

                snapshots = [n0s]
                while self._r < self._max_n - n0s:
                    if len(snapshots) == 0:
                        raise RuntimeError("Invalid checkpointing state")
                    cp_n = snapshots[-1]
                    if cp_n == self._max_n - self._r - 1:
                        snapshots.pop()
                        self._n = cp_n
                        if cp_n == n0s:
                            yield Read(cp_n, "disk", False)
                        else:
                            yield Read(cp_n, self._binomial_storage, True)
                        yield Clear(True, True)
                    else:
                        self._n = cp_n
                        if cp_n == n0s:
                            yield Read(cp_n, "disk", False)
                        else:
                            yield Read(cp_n, self._binomial_storage, False)
                        yield Clear(True, True)

                        yield Configure(False, False)

                        n_snapshots = (self._binomial_snapshots + 1
                                       - len(snapshots) + 1)
                        n0 = self._n
                        n1 = n0 + n_advance(self._max_n - self._r - n0,
                                            n_snapshots,
                                            trajectory=self._trajectory)
                        assert n1 > n0
                        self._n = n1
                        yield Forward(n0, n1)
                        yield Clear(True, True)

                        while self._n < self._max_n - self._r - 1:
                            yield Configure(True, False)

                            n_snapshots = (self._binomial_snapshots + 1
                                           - len(snapshots))
                            n0 = self._n
                            n1 = n0 + n_advance(self._max_n - self._r - n0,
                                                n_snapshots,
                                                trajectory=self._trajectory)
                            assert n1 > n0
                            self._n = n1
                            yield Forward(n0, n1)

                            if len(snapshots) >= self._binomial_snapshots + 1:
                                raise RuntimeError("Invalid checkpointing "
                                                   "state")
                            snapshots.append(n0)
                            yield Write(n0, self._binomial_storage)
                            yield Clear(True, True)
                        if self._n != self._max_n - self._r - 1:
                            raise RuntimeError("Invalid checkpointing state")

                    yield Configure(False, True)

                    self._n += 1
                    yield Forward(self._n - 1, self._n)

                    self._r += 1
                    yield Reverse(self._n, self._n - 1)
                    yield Clear(True, True)
                if self._r != self._max_n - n0s:
                    raise RuntimeError("Invalid checkpointing state")
                if len(snapshots) != 0:
                    raise RuntimeError("Invalid checkpointing state")
            if self._r != self._max_n:
                raise RuntimeError("Invalid checkpointing state")

            # Reset for new reverse

            self._r = 0
            yield EndReverse(False)

    @property
    def is_exhausted(self):
        return False

    @property
    def uses_disk_storage(self):
        return True
