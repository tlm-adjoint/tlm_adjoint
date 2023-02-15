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

from abc import ABC, abstractmethod
import functools

__all__ = \
    [
        "CheckpointAction",
        "Clear",
        "Configure",
        "Forward",
        "Reverse",
        "Read",
        "Write",
        "EndForward",
        "EndReverse",

        "CheckpointSchedule"
    ]


class CheckpointAction:
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args


class Clear(CheckpointAction):
    def __init__(self, clear_ics, clear_data):
        super().__init__(clear_ics, clear_data)

    @property
    def clear_ics(self):
        return self.args[0]

    @property
    def clear_data(self):
        return self.args[1]


class Configure(CheckpointAction):
    def __init__(self, store_ics, store_data):
        super().__init__(store_ics, store_data)

    @property
    def store_ics(self):
        return self.args[0]

    @property
    def store_data(self):
        return self.args[1]


class Forward(CheckpointAction):
    def __init__(self, n0, n1):
        super().__init__(n0, n1)

    def __iter__(self):
        yield from range(self.n0, self.n1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        return self.args[0]

    @property
    def n1(self):
        return self.args[1]


class Reverse(CheckpointAction):
    def __init__(self, n1, n0):
        super().__init__(n1, n0)

    def __iter__(self):
        yield from range(self.n1 - 1, self.n0 - 1, -1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        return self.args[1]

    @property
    def n1(self):
        return self.args[0]


class Read(CheckpointAction):
    def __init__(self, n, storage, delete):
        super().__init__(n, storage, delete)

    @property
    def n(self):
        return self.args[0]

    @property
    def storage(self):
        return self.args[1]

    @property
    def delete(self):
        return self.args[2]


class Write(CheckpointAction):
    def __init__(self, n, storage):
        super().__init__(n, storage)

    @property
    def n(self):
        return self.args[0]

    @property
    def storage(self):
        return self.args[1]


class EndForward(CheckpointAction):
    pass


class EndReverse(CheckpointAction):
    def __init__(self, exhausted):
        super().__init__(exhausted)

    @property
    def exhausted(self):
        return self.args[0]


class CheckpointSchedule(ABC):
    """
    A checkpointing schedule.

    The schedule is defined by iter, which yields actions in a similar manner
    to the approach used in
       A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
       of checkpointing for the reverse or adjoint mode of computational
       differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
       19--45, 2000
    e.g. 'forward', 'read', and 'write' correspond to ADVANCE, RESTORE, and
    TAKESHOT respectively in Griewank and Walther 2000 (although here 'write'
    actions occur *after* forward advancement from snapshots).

    The iter method yields (action, data), with:

    Clear(clear_ics, clear_data)
    Clear checkpoint storage. clear_ics indicates whether stored initial
    condition data should be cleared. clear_data indicates whether stored
    non-linear dependency data should be cleared.

    Configure(store_ics, store_data)
    Configure checkpoint storage. store_ics indicates whether initial condition
    data should be stored. store_data indicates whether non-linear dependency
    data should be stored.

    Forward(n0, n1)
    Run the forward from the start of block n0 to the start of block n1.

    Reverse(n1, n0)
    Run the adjoint from the start of block n1 to the start of block n0.

    Read(n, storage, delete)
    Read checkpoint data associated with block n from the indicated storage.
    delete indicates whether the checkpoint data should be deleted.

    Write(n, storage)
    Write checkpoint data associated with block n to the indicated storage.

    EndForward()
    End the forward calculation.

    EndReverse(exhausted)
    End a reverse calculation. If exhausted is False then a further reverse
    calculation can be performed.
    """

    def __init__(self, max_n=None):
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be positive")

        self._n = 0
        self._r = 0
        self._max_n = max_n

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls_iter = cls.iter

        @functools.wraps(cls_iter)
        def iter(self):
            if not hasattr(self, "_iter"):
                self._iter = cls_iter(self)
            return self._iter

        cls.iter = iter

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter())

    @abstractmethod
    def iter(self):
        raise NotImplementedError

    @abstractmethod
    def is_exhausted(self):
        raise NotImplementedError

    @abstractmethod
    def uses_disk_storage(self):
        raise NotImplementedError

    def n(self):
        return self._n

    def r(self):
        return self._r

    def max_n(self):
        return self._max_n

    def is_running(self):
        return hasattr(self, "_iter")

    def finalize(self, n):
        if n < 1:
            raise ValueError("n must be positive")
        if self._max_n is None:
            if self._n >= n:
                self._n = n
                self._max_n = n
            else:
                raise RuntimeError("Invalid checkpointing state")
        elif self._n != n or self._max_n != n:
            raise RuntimeError("Invalid checkpointing state")
