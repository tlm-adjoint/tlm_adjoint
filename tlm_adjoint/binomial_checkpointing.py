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

# This file implements binomial checkpointing using the approach described in
#   GW2000  A. Griewank and A. Walther, "Algorithm 799: Revolve: An
#           implementation of checkpointing for the reverse or adjoint mode of
#           computational differentiation", ACM Transactions on Mathematical
#           Software, 26(1), pp. 19--45, 2000
# choosing the maximum step size permitted when choosing the next snapshot
# block. This file further implements multi-stage offline checkpointing,
# determined via a brute force search to yield behaviour described in
#   SW2009  P. Stumm and A. Walther, "MultiStage approaches for optimal offline
#           checkpointing", SIAM Journal on Scientific Computing, 31(3),
#           pp. 1946--1967, 2009

from .checkpointing import CheckpointingManager

__all__ = \
    [
        "MultistageManager"
    ]


def n_advance(n, snapshots):
    """
    Determine an optimal offline snapshot interval, taking n steps and with the
    given number of snapshots, using the approach of
       A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
       of checkpointing for the reverse or adjoint mode of computational
       differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
       19--45, 2000
    and choosing the maximal possible step size.
    """

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


def allocate_snapshots(max_n, snapshots_in_ram, snapshots_on_disk, *,
                       write_weight=1.0, read_weight=1.0, delete_weight=0.0):
    """
    Allocate a stack of snapshots based upon the number of read/writes,
    preferentially allocating to RAM. Yields the approach described in
       P. Stumm and A. Walther, "MultiStage approaches for optimal offline
       checkpointing", SIAM Journal on Scientific Computing, 31(3), pp.
       1946--1967, 2009
    but applies a brute force approach to determine the allocation.
    """

    snapshots = snapshots_in_ram + snapshots_on_disk
    weights = [0.0 for i in range(snapshots)]

    cp_manager = MultistageManager(max_n, snapshots, 0)
    cp_iter = iter(cp_manager)

    snapshot_i = -1
    while cp_manager.r() != cp_manager.max_n():
        cp_action, cp_data = next(cp_iter)

        if cp_action == "read":
            _, _, cp_delete = cp_data
            if snapshot_i < 0:
                raise RuntimeError("Invalid checkpointing state")
            weights[snapshot_i] += read_weight
            if cp_delete:
                weights[snapshot_i] += delete_weight
                snapshot_i -= 1
        elif cp_action == "write":
            snapshot_i += 1
            if snapshot_i >= snapshots:
                raise RuntimeError("Invalid checkpointing state")
            weights[snapshot_i] += write_weight
        elif cp_action not in ["clear", "configure", "forward", "reverse"]:
            raise ValueError(f"Unexpected checkpointing action: {cp_action:s}")
    assert snapshot_i == -1

    allocation = ["disk" for i in range(snapshots)]
    for i in [p[0] for p in sorted(enumerate(weights), key=lambda p: p[1],
                                   reverse=True)][:snapshots_in_ram]:
        allocation[i] = "RAM"

    return tuple(weights), tuple(allocation)


class MultistageManager(CheckpointingManager):
    """
    Implements binomial checkpointing using the approach described in
       A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
       of checkpointing for the reverse or adjoint mode of computational
       differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
       19--45, 2000
    Uses a multistage allocation as described in
       P. Stumm and A. Walther, "MultiStage approaches for optimal offline
       checkpointing", SIAM Journal on Scientific Computing, 31(3), pp.
       1946--1967, 2009
    but applies a brute force approach to determine the allocation.
    """

    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk):
        if snapshots_in_ram == 0:
            storage = tuple("disk" for i in range(snapshots_on_disk))
        elif snapshots_on_disk == 0:
            storage = tuple("RAM" for i in range(snapshots_in_ram))
        else:
            _, storage = allocate_snapshots(
                max_n, snapshots_in_ram, snapshots_on_disk)

        super().__init__(max_n=max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._snapshots = []
        self._storage = storage

    def __iter__(self):
        if self._max_n is None:
            raise RuntimeError("Invalid checkpointing state")

        while True:
            if self._r == 0:
                # Forward

                n = self._n

                yield "clear", (True, True)

                if n < self._max_n - 1:
                    yield "configure", (True, False)

                    snapshots = (self._snapshots_in_ram
                                 + self._snapshots_on_disk
                                 - len(self._snapshots))
                    n0 = n
                    n1 = n0 + n_advance(self._max_n - n0, snapshots)
                    assert n1 > n0
                    self._n = n1
                    yield "forward", (n0, n1)

                    cp_storage = self._snapshot(n0)
                    yield "write", (n0, cp_storage)
                elif n == self._max_n - 1:
                    # Forward -> reverse

                    yield "configure", (n == 0, True)

                    self._n = n + 1
                    yield "forward", (n, n + 1)

                    self._r += 1
                    yield "reverse", (n + 1, n)
                else:
                    raise RuntimeError("Invalid checkpointing state")
            elif self._r < self._max_n:
                # Reverse

                if len(self._snapshots) == 0:
                    raise RuntimeError("Invalid checkpointing state")

                n = self._max_n - self._r - 1

                yield "clear", (True, True)

                cp_n = self._snapshots[-1]
                cp_storage = self._storage[len(self._snapshots) - 1]
                cp_delete = cp_n == n
                if cp_delete:
                    self._snapshots.pop()
                self._n = cp_n
                yield "read", (cp_n, cp_storage, cp_delete)

                n0 = cp_n
                while n0 < n:
                    if n0 == cp_n:
                        yield "configure", (False, False)
                    elif n0 > cp_n:
                        yield "clear", (True, True)
                        yield "configure", (True, False)
                    else:
                        raise RuntimeError("Invalid checkpointing state")

                    snapshots = (self._snapshots_in_ram
                                 + self._snapshots_on_disk
                                 - len(self._snapshots))
                    if n0 == cp_n:
                        # Count the snapshot at cp_n
                        snapshots += 1
                    n1 = n0 + n_advance(n + 1 - n0, snapshots)
                    assert n1 > n0
                    self._n = n1
                    yield "forward", (n0, n1)

                    if n0 > cp_n:
                        cp_storage = self._snapshot(n0)
                        yield "write", (n0, cp_storage)

                    n0 = n1
                if n0 != n:
                    raise RuntimeError("Invalid checkpointing state")

                yield "clear", (True, True)
                yield "configure", (n == 0, True)

                self._n = n + 1
                yield "forward", (n, n + 1)

                self._r += 1
                yield "reverse", (n + 1, n)
            elif self._r == self._max_n:
                break
            else:
                raise RuntimeError("Invalid checkpointing state")

    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0

    def single_reverse_run(self):
        return True

    def snapshots_in_ram(self):
        snapshots = 0
        for i in range(len(self._snapshots)):
            if self._storage[i] == "RAM":
                snapshots += 1
        return snapshots

    def snapshots_on_disk(self):
        snapshots = 0
        for i in range(len(self._snapshots)):
            if self._storage[i] == "disk":
                snapshots += 1
        return snapshots

    def _snapshot(self, n):
        assert n >= 0 and n < self._max_n
        if len(self._snapshots) >= \
                self._snapshots_in_ram + self._snapshots_on_disk:
            raise RuntimeError("Invalid checkpointing state")
        self._snapshots.append(n)
        return self._storage[len(self._snapshots) - 1]
