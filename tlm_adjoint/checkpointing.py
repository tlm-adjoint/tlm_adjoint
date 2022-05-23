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

from .interface import function_copy, function_id, function_is_checkpointed, \
    function_new

from collections import defaultdict, deque

__all__ = \
    [
        "CheckpointStorage",
        "ReplayStorage"
    ]


class CheckpointStorage:
    def __init__(self, *, store_ics=True, store_data=True):
        self._seen_ics = set()
        self._cp = {}
        self._refs = {}

        self._indices = defaultdict(lambda: 0)
        self._dep_keys = {}
        self._data = {}

        self.configure(store_ics=store_ics,
                       store_data=store_data)

    def configure(self, *, store_ics, store_data):
        """
        Configure storage.

        Arguments:

        store_ics   Store initial condition data, used by checkpointing
        store_data  Store equation non-linear dependency data, used in reverse
                    mode
        """

        self._store_ics = store_ics
        self._store_data = store_data

    def store_ics(self):
        return self._store_ics

    def store_data(self):
        return self._store_data

    def clear(self, *, clear_ics=True, clear_data=True, clear_refs=False):
        if clear_refs:
            self._refs.clear()
        if clear_ics:
            self._seen_ics.clear()
            self._cp.clear()

            for x_id in self._refs:
                self._seen_ics.add(x_id)

        if clear_data:
            self._indices.clear()
            self._dep_keys.clear()
            self._data.clear()

            for x_id, x in self._cp.items():
                self._data[self._data_key(x_id)] = x
            for x_id, x in self._refs.items():
                self._data[self._data_key(x_id)] = x

    def __getitem__(self, key):
        return tuple(self._data[dep_key] for dep_key in self._dep_keys[key])

    def initial_condition(self, x, *, copy=True):
        x_id = function_id(x)
        if x_id in self._cp:
            ic = self._cp[x_id]
        else:
            ic = self._refs[x_id]
        if copy:
            ic = function_copy(ic)
        return ic

    def initial_conditions(self, *, cp=True, refs=False, copy=True):
        cp_d = {}
        if cp:
            for x_id, x in self._cp.items():
                cp_d[x_id] = function_copy(x) if copy else x
        if refs:
            for x_id, x in self._refs.items():
                cp_d[x_id] = function_copy(x) if copy else x
        return cp_d

    def _data_key(self, x_id):
        return (x_id, self._indices[x_id])

    def add_initial_condition(self, x, value=None, *, _copy=None):
        copy = _copy
        del _copy

        if value is None:
            value = x
        if copy is None:
            copy = function_is_checkpointed(x)

        self._add_initial_condition(x_id=function_id(x), value=value,
                                    copy=copy)

    def _add_initial_condition(self, *, x_id, value, copy):
        if self._store_ics and x_id not in self._seen_ics:
            assert x_id not in self._cp
            assert x_id not in self._refs

            x_key = self._data_key(x_id)
            if x_key in self._data:
                if copy:
                    self._cp[x_id] = self._data[x_key]
                else:
                    self._refs[x_id] = self._data[x_key]
            else:
                if copy:
                    value = function_copy(value)
                    self._cp[x_id] = value
                else:
                    self._refs[x_id] = value
                self._data[x_key] = value
            self._seen_ics.add(x_id)

    def add_equation(self, key, eq, *, deps=None, nl_deps=None, _copy=None):
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

        for eq_x in eq_X:
            self._indices[function_id(eq_x)] += 1

        if self._store_ics:
            for eq_x in eq_X:
                self._seen_ics.add(function_id(eq_x))
            assert len(eq_deps) == len(deps)
            for eq_dep, dep in zip(eq_deps, deps):
                self.add_initial_condition(eq_dep, value=dep,
                                           _copy=copy(eq_dep))

        if self._store_data:
            if nl_deps is None:
                nl_deps = tuple(deps[i]
                                for i in eq.nonlinear_dependencies_map())

            dep_keys = []
            eq_nl_deps = eq.nonlinear_dependencies()
            assert len(eq_nl_deps) == len(nl_deps)
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                dep_key = self._data_key(function_id(eq_dep))
                if dep_key not in self._data:
                    if copy(eq_dep):
                        self._data[dep_key] = function_copy(dep)
                    else:
                        self._data[dep_key] = dep
                dep_keys.append(dep_key)
            self._dep_keys[key] = tuple(dep_keys)


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
        if x_id in self._map:
            self._map[x_id] = y

    def update(self, d, copy=True):
        for key, value in d.items():
            if key in self:
                self[key] = function_copy(value) if copy else value

    def pop(self):
        n, i, dep_ids = self._eq_last.popleft()
        for dep_id in dep_ids:
            del self._map[dep_id]
        return (n, i)
