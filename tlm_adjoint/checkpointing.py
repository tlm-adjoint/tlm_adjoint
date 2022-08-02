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

from .interface import DEFAULT_COMM, function_copy, function_get_values, \
    function_global_size, function_id, function_is_checkpointed, \
    function_local_indices, function_new, function_set_values, \
    function_space, function_space_type, space_id, space_new

from abc import ABC, abstractmethod
from collections import deque
import functools
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import logging
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

        "CheckpointAction",
        "Clear",
        "Configure",
        "Forward",
        "Reverse",
        "Read",
        "Write",
        "EndReverse",

        "CheckpointSchedule",
        "HRevolveCheckpointSchedule",
        "NoneCheckpointSchedule",
        "MemoryCheckpointSchedule",
        "PeriodicDiskCheckpointSchedule"
    ]


class CheckpointStorage:
    def __init__(self, *, store_ics, store_data):
        self._cp_keys = set()
        self._cp = {}
        self._refs_keys = set()
        self._refs = {}
        self._seen_ics = set()

        self._keys = {}
        # Ordering needed in checkpoint_data method
        self._data_keys = {}  # self._data_keys = set()
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
            self._keys.clear()
            self._data_keys.clear()
            self._data.clear()

    def __getitem__(self, key):
        return tuple(self._storage[dep_key] for dep_key in self._data[key])

    def initial_condition(self, x, *, copy=True):
        if isinstance(x, int):
            x_id = x
        else:
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

    def add_initial_condition(self, x, value=None):
        if value is None:
            value = x

        self._add_initial_condition(
            x_id=function_id(x), value=value,
            copy=function_is_checkpointed(x))

    def update_keys(self, n, i, eq):
        for m, x in enumerate(eq.X()):
            x_id = function_id(x)
            self._keys[x_id] = (x_id, (n, i, m))

    def _store(self, *, x_id=None, key=None, value, refs=True, copy):
        if key is None:
            if x_id is None:
                raise TypeError("Require exactly one of x_id or key")
            key = self._keys.setdefault(x_id, (x_id, None))
        else:
            if x_id is not None:
                raise TypeError("Require exactly one of x_id or key")
            x_id = key[0]

        if key not in self._storage:
            if copy:
                self._storage[key] = function_copy(value)
            else:
                self._storage[key] = value
                if refs:
                    self._refs_keys.add(key)
                    self._refs[x_id] = key
                    self._seen_ics.add(x_id)

        return key, self._storage[key]

    def _add_initial_condition(self, *, x_id, value, refs=True, copy):
        if self._store_ics and x_id not in self._seen_ics:
            key, _ = self._store(x_id=x_id, value=value, refs=refs, copy=copy)
            if key not in self._refs_keys:
                self._cp_keys.add(key)
                self._cp[x_id] = key
                self._seen_ics.add(x_id)

    def add_equation(self, n, i, eq, *, deps=None, nl_deps=None):
        eq_deps = eq.dependencies()
        if deps is None:
            deps = eq_deps

        self.update_keys(n, i, eq)

        if self._store_ics:
            for eq_x in eq.X():
                self._seen_ics.add(function_id(eq_x))

            assert len(eq_deps) == len(deps)
            for eq_dep, dep in zip(eq_deps, deps):
                self._add_initial_condition(
                    x_id=function_id(eq_dep), value=dep,
                    copy=function_is_checkpointed(eq_dep))

        self._add_equation_data(
            n, i, eq_deps, deps, eq.nonlinear_dependencies(), nl_deps)

    def add_equation_data(self, n, i, eq, *, nl_deps=None):
        self.update_keys(n, i, eq)

        eq_nl_deps = eq.nonlinear_dependencies()
        if nl_deps is None:
            nl_deps = eq_nl_deps

        if self._store_ics:
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                self._add_initial_condition(
                    x_id=function_id(eq_dep), value=dep,
                    copy=function_is_checkpointed(eq_dep))

        self._add_equation_data(n, i, eq_nl_deps, nl_deps, eq_nl_deps, nl_deps)

    def _add_equation_data(self, n, i, eq_deps, deps, eq_nl_deps, nl_deps=None,
                           *, refs=True, copy=None):
        if copy is None:
            def copy(x):
                return function_is_checkpointed(x)
        else:
            _copy = copy

            def copy(x):
                return _copy

        if self._store_data:
            if (n, i) in self._data:
                raise KeyError("Non-linear dependency data already stored")

            if nl_deps is None:
                assert len(eq_deps) == len(deps)
                deps_map = {function_id(eq_dep): dep
                            for eq_dep, dep in zip(eq_deps, deps)}
                nl_deps = tuple(deps_map[function_id(eq_dep)]
                                for eq_dep in eq_nl_deps)

            eq_data = []
            assert len(eq_nl_deps) == len(nl_deps)
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                key, value = self._store(x_id=function_id(eq_dep), value=dep,
                                         refs=refs, copy=copy(eq_dep))
                self._data_keys[key] = None  # self._data_keys.add(key)
                eq_data.append(key)
            self._data[(n, i)] = tuple(eq_data)

    def checkpoint_data(self, *, ics=True, data=True, copy=True):
        if ics:
            cp_cp = tuple(self._cp.values())
        else:
            cp_cp = ()
        if data:
            cp_data = dict(self._data)
        else:
            cp_data = {}
        cp_storage = {}

        if ics:
            for key in cp_cp:
                value = self._storage[key]
                cp_storage[key] = function_copy(value) if copy else value
        if data:
            for key in self._data_keys:
                if key not in self._refs_keys and key not in cp_storage:
                    value = self._storage[key]
                    cp_storage[key] = function_copy(value) if copy else value

        return (cp_cp, cp_data, cp_storage)

    def update(self, cp, data, storage, *, copy=True):
        keys = set(cp)
        for eq_data in data.values():
            keys.update(eq_data)

        for key, value in storage.items():
            if key in keys:
                if key in self._storage:
                    raise KeyError("Duplicate key")
                self._store(key=key, value=value, refs=False, copy=copy)

        for key in cp:
            if key in self._cp_keys or key in self._refs_keys:
                raise KeyError("Duplicate key")
            if key not in self._storage:
                raise KeyError("Invalid key")
            self._cp_keys.add(key)
            self._cp[key[0]] = key
            self._seen_ics.add(key[0])

        for (n, i), eq_data in data.items():
            if (n, i) in self._data:
                raise KeyError("Non-linear dependency data already stored")
            for key in eq_data:
                if key not in self._storage:
                    raise KeyError("Invalid key")
                self._data_keys[key] = None  # self._data_keys.add(key)
            self._data[(n, i)] = tuple(eq_data)


class ReplayStorage:
    def __init__(self, blocks, N0, N1, *, transpose_deps=None):
        if transpose_deps is None:
            active = {n: np.full(len(blocks[n]), True, dtype=bool)
                      for n in range(N0, N1)}
        else:
            last_eq = {}
            for n in range(N0, N1):
                block = blocks[n]
                for i, eq in enumerate(block):
                    for x in eq.X():
                        x_id = function_id(x)
                        if x_id in last_eq:
                            last_eq[x_id].append((n, i))
                        else:
                            last_eq[x_id] = [(n, i)]

            active = {n: np.full(len(blocks[n]), False, dtype=bool)
                      for n in range(N0, N1)}
            for n in range(N1 - 1, N0 - 1, -1):
                block = blocks[n]
                for i in range(len(block) - 1, - 1, -1):
                    eq = block[i]

                    if transpose_deps.any_is_active(n, i):
                        # Adjoint equation is active, mark forward equations
                        # solving for non-linear dependencies as active
                        for dep in eq.nonlinear_dependencies():
                            dep_id = function_id(dep)
                            if dep_id in last_eq:
                                p, k = last_eq[dep_id][-1]
                                assert n > p or (n == p and i >= k)
                                active[p][k] = True

                    for x in eq.X():
                        x_id = function_id(x)
                        last_eq[x_id].pop()
                        if len(last_eq[x_id]) == 0:
                            del last_eq[x_id]

                    if active[n][i]:
                        # Forward equation is active, mark forward equations
                        # solving for dependencies as active
                        X_ids = {function_id(x) for x in eq.X()}
                        ic_ids = {function_id(dep)
                                  for dep in eq.initial_condition_dependencies()}  # noqa: E501
                        for dep in eq.dependencies():
                            dep_id = function_id(dep)
                            if dep_id in X_ids:
                                if dep_id in ic_ids and dep_id in last_eq:
                                    p, k = last_eq[dep_id][-1]
                                    assert n > p or (n == p and i > k)
                                    active[p][k] = True
                            elif dep_id in last_eq:
                                p, k = last_eq[dep_id][-1]
                                assert n > p or (n == p and i > k)
                                active[p][k] = True

            assert len(last_eq) == 0
            del last_eq

        # Map from dep (id) to (indices of) last equation which needs dep
        last_eq = {}
        for n in range(N0, N1):
            block = blocks[n]
            for i, eq in enumerate(block):
                if active[n][i]:
                    for dep in eq.dependencies():
                        last_eq[function_id(dep)] = (n, i)
                # transpose_deps cannot be None here
                elif transpose_deps.any_is_active(n, i):
                    for dep in eq.nonlinear_dependencies():
                        last_eq[function_id(dep)] = (n, i)

        # Ordered container, with each element containing a set of dep ids for
        # which the corresponding equation is the last equation where dep is
        # needed
        eq_last_q = deque()
        eq_last_d = {}
        for n in range(N0, N1):
            block = blocks[n]
            for i in range(len(block)):
                dep_ids = set()
                eq_last_q.append((n, i, dep_ids))
                eq_last_d[(n, i)] = dep_ids
        for dep_id, (n, i) in last_eq.items():
            eq_last_d[(n, i)].add(dep_id)
        del eq_last_d

        self._active = active
        self._eq_last = eq_last_q
        self._map = {dep_id: None for dep_id in last_eq.keys()}

    def __iter__(self):
        return iter(self._map)

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

    def is_active(self, n, i):
        return self._active[n][i]

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
    def write(self, n, cp, data, storage):
        raise NotImplementedError

    @abstractmethod
    def read(self, n, *, ics=True, data=True, ic_ids=None):
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
    def __init__(self, prefix, *, comm=DEFAULT_COMM):
        comm = comm.Dup()
        cp_filenames = {}

        def finalize_callback(comm, cp_filenames):
            try:
                for filename in cp_filenames.values():
                    os.remove(filename)
            finally:
                if MPI is not None and not MPI.Is_finalized():
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

    def write(self, n, cp, data, storage):
        if n in self:
            raise RuntimeError("Duplicate checkpoint")

        filename = f"{self._prefix:s}{n:d}_{self._root_pid:d}_" \
                   f"{self._root_py2f:d}_{self._comm.rank:d}.pickle"
        spaces = {}

        write_storage = {}
        for key, F in storage.items():
            F_space = function_space(F)
            F_space_id = space_id(F_space)
            spaces.setdefault(F_space_id, F_space)

            write_storage[key] = (F_space_id,
                                  function_space_type(F),
                                  function_get_values(F))

        with open(filename, "wb") as h:
            pickle.dump((cp, data, write_storage),
                        h, protocol=pickle.HIGHEST_PROTOCOL)

        self._cp_filenames[n] = filename
        self._cp_spaces[n] = spaces

    def read(self, n, *, ics=True, data=True, ic_ids=None):
        filename = self._cp_filenames[n]
        spaces = self._cp_spaces[n]

        with open(filename, "rb") as h:
            read_cp, read_data, read_storage = pickle.load(h)

        if ics:
            if ic_ids is not None:
                read_cp = tuple(key for key in read_cp if key[0] in ic_ids)
        else:
            read_cp = ()
        if not data:
            read_data = {}

        keys = set(read_cp)
        for eq_data in read_data.values():
            keys.update(eq_data)
        read_storage = {key: read_storage[key] for key in read_storage
                        if key in keys}

        for key in read_storage:
            F_space_id, F_space_type, F_values = read_storage[key]
            F = space_new(spaces[F_space_id], space_type=F_space_type)
            function_set_values(F, F_values)
            read_storage[key] = F

        return read_cp, read_data, read_storage

    def delete(self, n):
        filename = self._cp_filenames[n]
        os.remove(filename)
        del self._cp_filenames[n]
        del self._cp_spaces[n]


class HDF5Checkpoints(Checkpoints):
    def __init__(self, prefix, *, comm=DEFAULT_COMM):
        comm = comm.Dup()
        cp_filenames = {}

        def finalize_callback(comm, rank, cp_filenames):
            try:
                if MPI is not None and not MPI.Is_finalized():
                    comm.barrier()
                if rank == 0:
                    for filename in cp_filenames.values():
                        os.remove(filename)
                if MPI is not None and not MPI.Is_finalized():
                    comm.barrier()
            finally:
                if MPI is not None and not MPI.Is_finalized():
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

    def write(self, n, cp, data, storage):
        if n in self:
            raise RuntimeError("Duplicate checkpoint")

        filename = f"{self._prefix:s}{n:d}_{self._root_pid:d}_" \
                   f"{self._root_py2f:d}.hdf5"
        spaces = {}

        import h5py
        with h5py.File(filename, "w", **self._File_kwargs) as h:
            g = h.create_group("/cp")

            d = g.create_dataset(
                "keys", shape=(len(cp), 4, self._comm.size),
                dtype=np.int64)
            for j, (x_id, x_indices) in enumerate(cp):
                d[j, 0, self._comm.rank] = x_id
                if x_indices is None:
                    d[j, 1:, self._comm.rank] = -1
                else:
                    d[j, 1:, self._comm.rank] = x_indices

            h.create_group("/data")

            for j, (eq_indices, eq_data) in enumerate(data.items()):
                g = h.create_group(f"/data/{j:d}")

                g.attrs["n"], g.attrs["i"] = eq_indices

                d = g.create_dataset(
                    "keys", shape=(len(eq_data), 4, self._comm.size),
                    dtype=np.int64)
                for k, (x_id, x_indices) in enumerate(eq_data):
                    d[k, 0, self._comm.rank] = x_id
                    if x_indices is None:
                        d[k, 1:, self._comm.rank] = -1
                    else:
                        d[k, 1:, self._comm.rank] = x_indices

            h.create_group("/storage")
            for j, ((x_id, x_indices), F) in enumerate(storage.items()):
                F_space = function_space(F)
                F_space_id = space_id(F_space)
                spaces.setdefault(F_space_id, F_space)

                g = h.create_group(f"/storage/{j:d}")

                d = g.create_dataset(
                    "key", shape=(4, self._comm.size),
                    dtype=np.int64)
                d[0, self._comm.rank] = x_id
                if x_indices is None:
                    d[1:, self._comm.rank] = -1
                else:
                    d[1:, self._comm.rank] = x_indices

                g.attrs["space_type"] = function_space_type(F)

                d = g.create_dataset(
                    "space_id", shape=(self._comm.size,),
                    dtype=np.int64)
                d[self._comm.rank] = F_space_id

                F_values = function_get_values(F)
                d = g.create_dataset(
                    "value", shape=(function_global_size(F),),
                    dtype=F_values.dtype)
                d[function_local_indices(F)] = F_values

        self._cp_filenames[n] = filename
        self._cp_spaces[n] = spaces

    def read(self, n, *, ics=True, data=True, ic_ids=None):
        filename = self._cp_filenames[n]
        spaces = self._cp_spaces[n]

        import h5py
        with h5py.File(filename, "r", **self._File_kwargs) as h:
            read_cp = []
            if ics:
                d = h["/cp/keys"]
                for j in range(d.shape[0]):
                    x_id = int(d[j, 0, self._comm.rank])
                    x_indices = tuple(map(int, d[j, 1:, self._comm.rank]))
                    if x_indices == (-1, -1, -1):
                        x_indices = None
                    key = (x_id, x_indices)

                    if ic_ids is None or x_id in ic_ids:
                        read_cp.append(key)
            read_cp = tuple(read_cp)

            read_data = {}
            if data:
                for j, (name, g) in enumerate(
                        sorted(h["/data"].items(), key=lambda e: int(e[0]))):
                    if name != f"{j:d}":
                        raise RuntimeError("Invalid checkpoint data")

                    eq_indices = tuple(map(int, (g.attrs["n"], g.attrs["i"])))

                    d = g["keys"]
                    eq_data = []
                    for k in range(d.shape[0]):
                        x_id = int(d[k, 0, self._comm.rank])
                        x_indices = tuple(map(int, d[k, 1:, self._comm.rank]))
                        if x_indices == (-1, -1, -1):
                            x_indices = None
                        eq_data.append((x_id, x_indices))

                    read_data[eq_indices] = tuple(eq_data)

            keys = set(read_cp)
            for eq_data in read_data.values():
                keys.update(eq_data)

            read_storage = {}
            for j, (name, g) in enumerate(
                    sorted(h["/storage"].items(), key=lambda e: int(e[0]))):
                if name != f"{j:d}":
                    raise RuntimeError("Invalid checkpoint data")

                d = g["key"]
                x_id = int(d[0, self._comm.rank])
                x_indices = tuple(map(int, d[1:, self._comm.rank]))
                if x_indices == (-1, -1, -1):
                    x_indices = None
                key = (x_id, x_indices)

                if key in keys:
                    F_space_type = g.attrs["space_type"]

                    d = g["space_id"]
                    F = space_new(spaces[d[self._comm.rank]],
                                  space_type=F_space_type)

                    d = g["value"]
                    function_set_values(F, d[function_local_indices(F)])

                    read_storage[key] = F

        return read_cp, read_data, read_storage

    def delete(self, n):
        filename = self._cp_filenames[n]
        self._comm.barrier()
        if self._comm.rank == 0:
            os.remove(filename)
        self._comm.barrier()
        del self._cp_filenames[n]
        del self._cp_spaces[n]


class CheckpointAction:
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"


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


class EndReverse(CheckpointAction):
    def __init__(self, exhausted):
        super().__init__(self, exhausted)

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

    action: "clear"
    data:   (clear_ics, clear_data)
    Clear checkpoint storage. clear_ics indicates whether stored initial
    condition data should be cleared. clear_data indicates whether stored
    non-linear dependency data should be cleared.

    action: "configure"
    data:   (store_ics, store_data)
    Configure checkpoint storage. store_ics indicates whether initial condition
    data should be stored. store_data indicates whether non-linear dependency
    data should be stored.

    action: "forward"
    data:   (n0, n1)
    Run the forward from the start of block n0 to the start of block n1.

    action: "reverse"
    data:   (n1, n0)
    Run the adjoint from the start of block n1 to the start of block n0.

    action: "read"
    data:   (n, storage, delete)
    Read checkpoint data associated with block n from the indicated storage.
    delete indicates whether the checkpoint data should be deleted.

    action: "write"
    data:   (n, storage)
    Write checkpoint data associated with block n to the indicated storage.

    action: "end_reverse"
    data:   (exhausted,)
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


class NoneCheckpointSchedule(CheckpointSchedule):
    def iter(self):
        # Forward

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1)

    def is_exhausted(self):
        return self._max_n is not None

    def uses_disk_storage(self):
        return False


class MemoryCheckpointSchedule(CheckpointSchedule):
    def iter(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield Configure(True, True)

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1)

        while True:
            if self._r == 0:
                # Reverse

                self._r = self._max_n
                yield Reverse(self._max_n, 0)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse(False,)
            else:
                raise RuntimeError("Invalid checkpointing state")

    def is_exhausted(self):
        return False

    def uses_disk_storage(self):
        return False


class PeriodicDiskCheckpointSchedule(CheckpointSchedule):
    def __init__(self, period, *, keep_block_0_ics=False):
        if period < 1:
            raise ValueError("period must be positive")

        super().__init__()
        self._period = period
        self._keep_block_0_ics = keep_block_0_ics

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

        while True:
            # Reverse

            while self._r < self._max_n:
                n = self._max_n - self._r - 1
                n0 = (n // self._period) * self._period
                del n
                n1 = min(n0 + self._period, self._max_n)
                if self._r != self._max_n - n1:
                    raise RuntimeError("Invalid checkpointing state")

                self._n = n0
                yield Read(n0, "disk", False)
                yield Clear(True, True)

                if self._keep_block_0_ics and n0 == 0:
                    yield Configure(True, True)
                    self._n = n0 + 1
                    yield Forward(n0, n0 + 1)

                    if n1 > n0 + 1:
                        yield Configure(False, True)
                        self._n = n1
                        yield Forward(n0 + 1, n1)
                else:
                    yield Configure(False, True)
                    self._n = n1
                    yield Forward(n0, n1)

                self._r = self._max_n - n0
                yield Reverse(n1, n0)
                yield Clear(not self._keep_block_0_ics or n0 != 0, True)
            if self._r != self._max_n:
                raise RuntimeError("Invalid checkpointing state")

            # Reset for new reverse

            self._r = 0
            yield EndReverse(False,)

    def is_exhausted(self):
        return False

    def uses_disk_storage(self):
        return True


class HRevolveCheckpointSchedule(CheckpointSchedule):
    def __init__(self, max_n, snapshots_in_ram, snapshots_on_disk, *,
                 keep_block_0_ics=False,
                 wvect=(0.0, 0.1), rvect=(0.0, 0.1), uf=1.0, ub=2.0, **kwargs):
        super().__init__(max_n)
        self._snapshots_in_ram = snapshots_in_ram
        self._snapshots_on_disk = snapshots_on_disk
        self._exhausted = False
        self._keep_block_0_ics = keep_block_0_ics

        cvect = (snapshots_in_ram, snapshots_on_disk)
        import hrevolve
        schedule = hrevolve.hrevolve(max_n - 1, cvect, wvect, rvect,
                                     uf=uf, ub=ub, **kwargs)
        self._schedule = list(schedule)

        logger = logging.getLogger("tlm_adjoint.checkpointing")
        logger.debug(f"H-Revolve schedule: {str(self._schedule):s}")

    def iter(self):
        def action(i):
            assert i >= 0 and i < len(self._schedule)
            action = self._schedule[i]
            cp_action = action.type
            if cp_action == "Forward":
                n_0 = action.index
                n_1 = n_0 + 1
                storage = None
            elif cp_action == "Forwards":
                cp_action = "Forward"
                n_0, n_1 = action.index
                n_1 += 1
                storage = None
            elif cp_action == "Backward":
                n_0 = action.index
                n_1 = None
                storage = None
            elif cp_action in ["Read", "Write", "Discard"]:
                storage, n_0 = action.index
                n_1 = None
                storage = {0: "RAM", 1: "disk"}[storage]
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")
            return cp_action, (n_0, n_1, storage)

        snapshots = set()
        deferred_cp = None
        for i in range(len(self._schedule)):
            cp_action, (n_0, n_1, storage) = action(i)

            if cp_action == "Forward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")

                yield Clear(True, True)
                yield Configure(n_0 not in snapshots, False)
                self._n = n_1
                yield Forward(n_0, n_1)
            elif cp_action == "Backward":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")
                if n_0 != self._max_n - self._r - 1:
                    raise RuntimeError("Invalid checkpointing state")

                if deferred_cp is not None:
                    snapshots.add(deferred_cp[0])
                    yield Write(*deferred_cp)
                    deferred_cp = None

                yield Clear(True, True)
                yield Configure(self._keep_block_0_ics and n_0 == 0, True)
                self._n = n_0 + 1
                yield Forward(n_0, n_0 + 1)
                self._r += 1
                yield Reverse(n_0 + 1, n_0)
            elif cp_action == "Read":
                if deferred_cp is not None:
                    raise RuntimeError("Invalid checkpointing state")

                if n_0 == self._max_n - self._r - 1:
                    cp_delete = True
                elif i < len(self._schedule) - 2:
                    d_cp_action, (d_n_0, _, d_storage) = action(i + 2)
                    if d_cp_action == "Discard":
                        if d_n_0 != n_0 or d_storage != storage:
                            raise RuntimeError("Invalid schedule")
                        cp_delete = True
                    else:
                        cp_delete = False

                yield Clear(True, True)
                if cp_delete:
                    snapshots.remove(n_0)
                self._n = n_0
                yield Read(n_0, storage, cp_delete)
            elif cp_action == "Write":
                if n_0 != self._n:
                    raise RuntimeError("Invalid checkpointing state")

                if deferred_cp is not None:
                    snapshots.add(deferred_cp[0])
                    yield Write(*deferred_cp)
                    deferred_cp = None

                deferred_cp = (n_0, storage)
            elif cp_action == "Discard":
                if i < 2:
                    raise RuntimeError("Invalid schedule")
                r_cp_action, (r_n_0, _, r_storage) = action(i - 2)
                if r_cp_action != "Read" \
                        or r_n_0 != n_0 \
                        or r_storage != storage:
                    raise RuntimeError("Invalid schedule")
            else:
                raise RuntimeError(f"Unexpected action: {cp_action:s}")

        if len(snapshots) != 0:
            raise RuntimeError("Invalid checkpointing state")

        self._exhausted = True
        yield EndReverse(True,)

    def is_exhausted(self):
        return self._exhausted

    def uses_disk_storage(self):
        return self._snapshots_on_disk > 0
