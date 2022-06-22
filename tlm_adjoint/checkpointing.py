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

from .interface import function_copy, function_get_values, \
    function_global_size, function_id, function_is_checkpointed, \
    function_local_indices, function_new, function_set_values, \
    function_space, function_space_type, space_id, space_new

from abc import ABC, abstractmethod
from collections import deque
import functools
import mpi4py.MPI as MPI
import numpy as np
import os
import pickle
import sys
import weakref

__all__ = \
    [
        "CheckpointStorage",
        "ReplayStorage",

        "Checkpoints",
        "PickleCheckpoints",
        "HDF5Checkpoints",

        "CheckpointingManager",
        "NoneCheckpointingManager",
        "MemoryCheckpointingManager",
        "PeriodicDiskCheckpointingManager"
    ]


class CheckpointStorage:
    def __init__(self, *, store_ics, store_data):
        self._cp_keys = set()
        self._cp = {}
        self._refs_keys = set()
        self._refs = {}
        self._seen_ics = set()

        self._eq_x_keys = {}
        self._data_keys = set()
        self._data = {}

        self._storage = {}

        self.configure(store_ics=store_ics,
                       store_data=store_data)

    def configure(self, *, store_ics, store_data):
        self._store_ics = store_ics
        self._store_data = store_data

    def store_ics(self):
        return self._store_ics

    def store_data(self):
        return self._store_data

    def clear(self, *, clear_ics=True, clear_data=True, clear_refs=False):
        if clear_ics:
            for key in self._cp_keys:
                if key not in self._data_keys:
                    del self._storage[key]
            self._cp_keys.clear()
            self._cp.clear()
            self._seen_ics.clear()
            self._seen_ics.update(self._refs.keys())
        if clear_refs:
            for key in self._refs_keys:
                if key not in self._data_keys:
                    del self._storage[key]
            self._refs_keys.clear()
            self._refs.clear()
            self._seen_ics.clear()
            self._seen_ics.update(self._cp.keys())

        if clear_data:
            for key in self._data_keys:
                if key not in self._cp_keys and key not in self._refs_keys:
                    del self._storage[key]
            self._eq_x_keys.clear()
            self._data_keys.clear()
            self._data.clear()

    def __getitem__(self, key):
        return tuple(self._storage[nl_dep_key]
                     for nl_dep_key in self._data[key])

    def initial_condition(self, x, *, copy=True):
        x_id = function_id(x)
        if x_id in self._cp:
            ic = self._storage[self._cp[x_id]]
        else:
            ic = self._storage[self._refs[x_id]]
        return function_copy(ic) if copy else ic

    def initial_conditions(self, *, cp=True, refs=False, copy=True):
        cp_d = {}
        if cp:
            for x_id, x_key in self._cp.items():
                x = self._storage[x_key]
                cp_d[x_id] = function_copy(x) if copy else x
        if refs:
            for x_id, x_key in self._refs.items():
                x = self._storage[x_key]
                cp_d[x_id] = function_copy(x) if copy else x
        return cp_d

    def add_initial_condition(self, x, value=None, *, _copy=None):
        copy = _copy
        del _copy

        if value is None:
            value = x
        if copy is None:
            copy = function_is_checkpointed(x)

        self._add_initial_condition(x_id=function_id(x), value=value,
                                    copy=copy)

    def _store(self, *, x_id, value, copy):
        key = self._eq_x_keys.get(x_id, (x_id, None))
        if key not in self._storage:
            self._storage[key] = function_copy(value) if copy else value
        return key, self._storage[key]

    def _add_initial_condition(self, *, x_id, value, copy):
        if self._store_ics and x_id not in self._seen_ics:
            key, value = self._store(x_id=x_id, value=value, copy=copy)
            if copy:
                self._cp_keys.add(key)
                self._cp[x_id] = key
            else:
                self._refs_keys.add(key)
                self._refs[x_id] = key
            self._seen_ics.add(x_id)

    def add_equation(self, n, i, eq, *, deps=None, nl_deps=None, _copy=None):
        if _copy is None:
            def copy(x):
                return function_is_checkpointed(x)
        else:
            copy = _copy
        del _copy

        eq_X = eq.X()
        eq_deps = eq.dependencies()
        if deps is None:
            deps = eq_deps

        if self._store_ics:
            for eq_x in eq_X:
                self._seen_ics.add(function_id(eq_x))
            assert len(eq_deps) == len(deps)
            for eq_dep, dep in zip(eq_deps, deps):
                if function_id(eq_dep) not in self._seen_ics:
                    self.add_initial_condition(eq_dep, value=dep,
                                               _copy=copy(eq_dep))

        if self._store_data:
            if (n, i) in self._data:
                raise KeyError("Duplicate key")

            for m, eq_x in enumerate(eq_X):
                eq_x_id = function_id(eq_x)
                self._eq_x_keys[eq_x_id] = (eq_x_id, (n, i, m))

            if nl_deps is None:
                nl_deps = tuple(deps[j]
                                for j in eq.nonlinear_dependencies_map())

            eq_data = []
            eq_nl_deps = eq.nonlinear_dependencies()
            assert len(eq_nl_deps) == len(nl_deps)
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                key, value = self._store(x_id=function_id(eq_dep), value=dep,
                                         copy=copy(eq_dep))
                self._data_keys.add(key)
                eq_data.append(key)
            self._data[(n, i)] = tuple(eq_data)


class ReplayStorage:
    def __init__(self, blocks, N0, N1):
        # Map from dep (id) to (indices of) last equation which depends on dep
        last_eq = {}
        for n in range(N0, N1):
            for i, eq in enumerate(blocks[n]):
                for dep in eq.dependencies():
                    last_eq[function_id(dep)] = (n, i)

        # Ordered container, with each element containing a set of dep ids for
        # which the corresponding equation is the last equation to depend on
        # dep
        eq_last_q = deque()
        eq_last_d = {}
        for n in range(N0, N1):
            for i in range(len(blocks[n])):
                dep_ids = set()
                eq_last_q.append((n, i, dep_ids))
                eq_last_d[(n, i)] = dep_ids
        for dep_id, (n, i) in last_eq.items():
            eq_last_d[(n, i)].add(dep_id)
        del eq_last_d

        self._eq_last = eq_last_q
        self._map = {dep_id: None for dep_id in last_eq.keys()}

    def __len__(self):
        return len(self._map)

    def __contains__(self, x):
        if isinstance(x, int):
            x_id = x
        else:
            x_id = function_id(x)
        return x_id in self._map

    def __getitem__(self, x):
        if isinstance(x, int):
            y = self._map[x]
            if y is None:
                raise RuntimeError("Unable to create new function")
        else:
            x_id = function_id(x)
            y = self._map[x_id]
            if y is None:
                y = self._map[x_id] = function_new(x)
        return y

    def __setitem__(self, x, y):
        if isinstance(x, int):
            x_id = x
        else:
            x_id = function_id(x)
        if self._map[x_id] is not None:  # KeyError if unexpected id
            raise KeyError(f"Key '{x_id:d}' already set")
        self._map[x_id] = y

    def update(self, d, *, copy=True):
        for key, value in d.items():
            if key in self:
                self[key] = function_copy(value) if copy else value

    def pop(self):
        n, i, dep_ids = self._eq_last.popleft()
        for dep_id in dep_ids:
            del self._map[dep_id]
        return (n, i)


class Checkpoints(ABC):
    @abstractmethod
    def __contains__(self, n):
        raise NotImplementedError

    @abstractmethod
    def write(self, n, cp):
        raise NotImplementedError

    @abstractmethod
    def read(self, n, storage):
        raise NotImplementedError

    @abstractmethod
    def delete(self, n):
        raise NotImplementedError


def root_py2f(comm, *, root=0):
    if comm.rank == root:
        py2f = comm.py2f()
    else:
        py2f = None
    return comm.bcast(py2f, root=root)


def root_pid(comm, *, root=0):
    if comm.rank == root:
        pid = os.getpid()
    else:
        pid = None
    return comm.bcast(pid, root=root)


class PickleCheckpoints(Checkpoints):
    def __init__(self, prefix, *, comm=MPI.COMM_WORLD):
        comm = comm.Dup()
        cp_filenames = {}

        def finalize_callback(comm, cp_filenames):
            try:
                for filename in cp_filenames.values():
                    os.remove(filename)
            finally:
                if not MPI.Is_finalized():
                    comm.Free()

        finalize = weakref.finalize(self, finalize_callback,
                                    comm, cp_filenames)
        finalize.atexit = True

        self._prefix = prefix
        self._comm = comm
        self._root_pid = root_pid(comm)
        self._root_py2f = root_py2f(comm)

        self._cp_filenames = cp_filenames
        self._cp_spaces = {}

    def __contains__(self, n):
        assert (n in self._cp_filenames) == (n in self._cp_spaces)
        return n in self._cp_filenames

    def write(self, n, cp):
        if n in self:
            raise RuntimeError("Duplicate checkpoint")

        filename = f"{self._prefix:s}{n:d}_{self._root_pid:d}_" \
                   f"{self._root_py2f:d}_{self._comm.rank:d}.pickle"
        spaces = {}

        cp_data = {}
        for key, F in cp.items():
            F_space = function_space(F)
            F_space_id = space_id(F_space)
            if F_space_id not in spaces:
                spaces[F_space_id] = F_space

            cp_data[key] = (F_space_id,
                            function_space_type(F),
                            function_get_values(F))

        with open(filename, "wb") as h:
            pickle.dump(cp_data, h, protocol=pickle.HIGHEST_PROTOCOL)

        self._cp_filenames[n] = filename
        self._cp_spaces[n] = spaces

    def read(self, n, storage):
        filename = self._cp_filenames[n]
        spaces = self._cp_spaces[n]

        with open(filename, "rb") as h:
            cp_data = pickle.load(h)

        for key in tuple(cp_data.keys()):
            F_space_id, F_space_type, F_values = cp_data.pop(key)
            if key in storage:
                F = space_new(spaces[F_space_id], space_type=F_space_type)
                function_set_values(F, F_values)
                storage[key] = F

    def delete(self, n):
        filename = self._cp_filenames[n]
        os.remove(filename)
        del self._cp_filenames[n]
        del self._cp_spaces[n]


class HDF5Checkpoints(Checkpoints):
    def __init__(self, prefix, *, comm=MPI.COMM_WORLD):
        comm = comm.Dup()
        cp_filenames = {}

        def finalize_callback(comm, rank, cp_filenames):
            try:
                if not MPI.Is_finalized():
                    comm.barrier()
                if rank == 0:
                    for filename in cp_filenames.values():
                        os.remove(filename)
                if not MPI.Is_finalized():
                    comm.barrier()
            finally:
                if not MPI.Is_finalized():
                    comm.Free()

        finalize = weakref.finalize(self, finalize_callback,
                                    comm, comm.rank, cp_filenames)
        finalize.atexit = True

        self._prefix = prefix
        self._comm = comm
        self._root_pid = root_pid(comm)
        self._root_py2f = root_py2f(comm)

        self._cp_filenames = cp_filenames
        self._cp_spaces = {}

        if comm.size > 1:
            self._File_kwargs = {"driver": "mpio", "comm": self._comm}
        else:
            self._File_kwargs = {}

    def __contains__(self, n):
        assert (n in self._cp_filenames) == (n in self._cp_spaces)
        return n in self._cp_filenames

    def write(self, n, cp):
        if n in self:
            raise RuntimeError("Duplicate checkpoint")

        filename = f"{self._prefix:s}{n:d}_{self._root_pid:d}_" \
                   f"{self._root_py2f:d}.hdf5"
        spaces = {}

        import h5py
        with h5py.File(filename, "w", **self._File_kwargs) as h:
            h.create_group("/ics")
            for i, (key, F) in enumerate(cp.items()):
                F_space = function_space(F)
                F_space_id = space_id(F_space)
                if F_space_id not in spaces:
                    spaces[F_space_id] = F_space

                g = h.create_group(f"/ics/{i:d}")

                F_values = function_get_values(F)
                d = g.create_dataset("value", shape=(function_global_size(F),),
                                     dtype=F_values.dtype)
                d[function_local_indices(F)] = F_values

                d = g.create_dataset("space_type", shape=(self._comm.size,),
                                     dtype=np.uint8)
                d[self._comm.rank] = {"primal": 0, "conjugate": 1,
                                      "dual": 2, "conjugate_dual": 3}[function_space_type(F)]  # noqa: E501

                d = g.create_dataset("space_id", shape=(self._comm.size,),
                                     dtype=np.int64)
                d[self._comm.rank] = F_space_id

                d = g.create_dataset("key", shape=(self._comm.size,),
                                     dtype=np.int64)
                d[self._comm.rank] = key

        self._cp_filenames[n] = filename
        self._cp_spaces[n] = spaces

    def read(self, n, storage):
        filename = self._cp_filenames[n]
        spaces = self._cp_spaces[n]

        import h5py
        with h5py.File(filename, "r", **self._File_kwargs) as h:
            for i, (name, g) in enumerate(h["/ics"].items()):
                assert name == f"{i:d}"

                d = g["key"]
                key = int(d[self._comm.rank])
                if key in storage:
                    d = g["space_type"]
                    F_space_type = {0: "primal", 1: "conjugate",
                                    2: "dual", 3: "conjugate_dual"}[d[self._comm.rank]]  # noqa: E501

                    d = g["space_id"]
                    F = space_new(spaces[d[self._comm.rank]],
                                  space_type=F_space_type)

                    d = g["value"]
                    function_set_values(F, d[function_local_indices(F)])

                    storage[key] = F

    def delete(self, n):
        filename = self._cp_filenames[n]
        self._comm.barrier()
        if self._comm.rank == 0:
            os.remove(filename)
        self._comm.barrier()
        del self._cp_filenames[n]
        del self._cp_spaces[n]


class CheckpointingManager(ABC):
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

    action: 'clear'
    data:   (clear_ics, clear_data)
    Clear checkpoint storage. clear_ics indicates whether stored initial
    condition data should be cleared. clear_data indicates whether stored
    non-linear dependency data should be cleared.

    action: 'configure'
    data:   (store_ics, store_data)
    Configure checkpoint storage. store_ics indicates whether initial condition
    data should be stored. store_data indicates whether non-linear dependency
    data should be stored.

    action: 'forward'
    data:   (n0, n1)
    Run the forward from the start of block n0 to the start of block n1.

    action: 'reverse'
    data:   (n1, n0)
    Run the adjoint from the start of block n1 to the start of block n0.

    action: 'read'
    data:   (n, storage, delete)
    Read checkpoint data associated with the start of block n from the
    indicated storage. delete indicates whether the checkpoint data should be
    deleted.

    action: 'write'
    data:   (n, storage)
    Write checkpoint data associated with the start of block n to the indicated
    storage.

    action: 'end_reverse'
    data:   (clear_ics, clear_data, exhausted)
    End a reverse calculation. clear_ics and clear_data are as for the 'clear'
    action. If exhausted is False then a further reverse calculation can be
    performed.
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
        elif self._n != n:
            raise RuntimeError("Invalid checkpointing state")


class NoneCheckpointingManager(CheckpointingManager):
    def iter(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield "clear", (True, True)

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield "configure", (False, False)

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield "forward", (n0, n1)

    def is_exhausted(self):
        return self._max_n is not None

    def uses_disk_storage(self):
        return False


class MemoryCheckpointingManager(CheckpointingManager):
    def iter(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield "clear", (True, True)

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield "configure", (True, True)

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield "forward", (n0, n1)

        while True:
            if self._r == 0:
                # Reverse

                self._r = self._max_n
                yield "reverse", (self._max_n, 0)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield "end_reverse", (False, False, False)
            else:
                raise RuntimeError("Invalid checkpointing state")

    def is_exhausted(self):
        return False

    def uses_disk_storage(self):
        return False


class PeriodicDiskCheckpointingManager(CheckpointingManager):
    def __init__(self, period, *, keep_block_0_ics=False):
        if period < 1:
            raise ValueError("period must be positive")

        super().__init__()
        self._period = period
        self._keep_block_0_ics = keep_block_0_ics

    def iter(self):
        # Forward

        while self._max_n is None:
            yield "clear", (True, True)

            if self._max_n is not None:
                # Unexpected finalize
                raise RuntimeError("Invalid checkpointing state")
            yield "configure", (True, False)
            if self._max_n is not None:
                # Unexpected finalize
                raise RuntimeError("Invalid checkpointing state")
            n0 = self._n
            n1 = n0 + self._period
            self._n = n1
            yield "forward", (n0, n1)

            # Finalize permitted here

            yield "write", (n0, "disk")

        while True:
            # Reverse

            while self._r < self._max_n:
                n = self._max_n - self._r - 1
                n0 = (n // self._period) * self._period
                del n
                n1 = min(n0 + self._period, self._max_n)
                if self._r != self._max_n - n1:
                    raise RuntimeError("Invalid checkpointing state")

                yield "clear", (True, True)

                self._n = n0
                yield "read", (n0, "disk", False)

                if self._keep_block_0_ics and n0 == 0:
                    yield "configure", (True, True)
                    self._n = n0 + 1
                    yield "forward", (n0, n0 + 1)

                    if n1 > n0 + 1:
                        yield "configure", (False, True)
                        self._n = n1
                        yield "forward", (n0 + 1, n1)
                else:
                    yield "configure", (False, True)
                    self._n = n1
                    yield "forward", (n0, n1)

                self._r = self._max_n - n0
                yield "reverse", (n1, n0)
            if self._r != self._max_n:
                raise RuntimeError("Invalid checkpointing state")

            # Reset for new reverse

            self._r = 0
            yield "end_reverse", (not self._keep_block_0_ics, True, False)

    def is_exhausted(self):
        return False

    def uses_disk_storage(self):
        return True
