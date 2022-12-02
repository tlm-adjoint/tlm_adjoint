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

from .interface import function_caches, function_id, function_state

from .alias import gc_disabled

import functools
import weakref

__all__ = \
    [
        "Cache",
        "CacheRef",
        "Caches",
        "clear_caches",
        "local_caches"
    ]


class CacheRef:
    def __init__(self, value=None):
        self._value = value

    def __call__(self):
        return self._value

    def _clear(self):
        self._value = None


@gc_disabled
def clear_caches(*deps):
    if len(deps) == 0:
        for cache in tuple(Cache._caches.valuerefs()):
            cache = cache()
            if cache is not None:
                cache.clear()
    else:
        for dep in deps:
            function_caches(dep).clear()


def local_caches(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        clear_caches()
        try:
            return fn(*args, **kwargs)
        finally:
            clear_caches()
    return wrapped_fn


class Cache:
    _id_counter = [0]
    _caches = weakref.WeakValueDictionary()

    def __init__(self):
        self._cache = {}
        self._deps_map = {}
        self._dep_caches = {}

        self._id = self._id_counter[0]
        self._id_counter[0] += 1
        self._caches[self._id] = self

        def finalize_callback(self_ref):
            self = self_ref()
            if self is not None:
                for value in self._cache.values():
                    value._clear()

        weakref.finalize(self, finalize_callback,
                         weakref.ref(self))

    def __len__(self):
        return len(self._cache)

    def id(self):
        return self._id

    def clear(self, *deps):
        if len(deps) == 0:
            for value in self._cache.values():
                value._clear()
            self._cache.clear()
            self._deps_map.clear()
            for dep_caches in self._dep_caches.values():
                dep_caches = dep_caches()
                if dep_caches is not None:
                    dep_caches.remove(self)
            self._dep_caches.clear()
        else:
            for dep in deps:
                dep_id = dep if isinstance(dep, int) else function_id(dep)
                del dep
                if dep_id in self._deps_map:
                    # We keep a record of:
                    #   - Cache entries associated with each dependency. The
                    #     cache keys are in self._deps_map[dep_id].keys(), and
                    #     the cache entries in self._cache[key].
                    #   - Dependencies associated with each cache entry. The
                    #     dependency ids are in self._deps_map[dep_id2][key]
                    #     for *each* dependency associated with the cache
                    #     entry.
                    #   - The caches in which dependencies have an associated
                    #     cache entry. A (weak) reference to the caches is in
                    #     self._dep_caches[dep_id2].
                    # To remove a cache item associated with a dependency with
                    # dependency id dep_id we
                    #   1. Clear the cache entries associated with the
                    #      dependency. These are given by self._cache[key] for
                    #      each key in self._deps_map[dep_id].keys().
                    #   2. Remove the dependency ids associated with each cache
                    #      entry. These are given by
                    #      self._deps_map[dep_id2][key] for each dep_id2 in
                    #      self._deps_map[dep_id][key].
                    #  3.  Remove the (weak) reference to this cache for each
                    #      dependency with no further associated cache entries
                    #      in this cache.
                    for key, dep_ids in self._deps_map[dep_id].items():
                        # Step 1.
                        self._cache[key]._clear()
                        del self._cache[key]
                        for dep_id2 in dep_ids:
                            if dep_id2 != dep_id:
                                # Step 2.
                                del self._deps_map[dep_id2][key]
                                if len(self._deps_map[dep_id2]) == 0:
                                    del self._deps_map[dep_id2]
                                    dep_caches = self._dep_caches[dep_id2]()
                                    if dep_caches is not None:
                                        # Step 3.
                                        dep_caches.remove(self)
                                    del self._dep_caches[dep_id2]
                    # Step 2.
                    del self._deps_map[dep_id]
                    dep_caches = self._dep_caches[dep_id]()
                    if dep_caches is not None:
                        # Step 3.
                        dep_caches.remove(self)
                    del self._dep_caches[dep_id]

    def add(self, key, value, deps=None):
        if deps is None:
            deps = []

        if key in self._cache:
            value_ref = self._cache[key]
            value = value_ref()
            if value is None:
                raise RuntimeError("Unexpected cache value state")
            return value_ref, value

        value = value()
        value_ref = CacheRef(value)
        dep_ids = tuple(map(function_id, deps))

        self._cache[key] = value_ref

        assert len(deps) == len(dep_ids)
        for dep, dep_id in zip(deps, dep_ids):
            dep_caches = function_caches(dep)
            dep_caches.add(self)

            if dep_id in self._deps_map:
                self._deps_map[dep_id][key] = dep_ids
                assert dep_id in self._dep_caches
            else:
                self._deps_map[dep_id] = {key: dep_ids}
                self._dep_caches[dep_id] = weakref.ref(dep_caches)

        return value_ref, value

    def get(self, key, default=None):
        return self._cache.get(key, default)


class Caches:
    def __init__(self, x):
        self._caches = weakref.WeakValueDictionary()
        self._id = function_id(x)
        self._state = (self._id, function_state(x))

    def __len__(self):
        return len(self._caches)

    @gc_disabled
    def clear(self):
        for cache in tuple(self._caches.valuerefs()):
            cache = cache()
            if cache is not None:
                cache.clear(self._id)
                assert not cache.id() in self._caches

    def add(self, cache):
        cache_id = cache.id()
        if cache_id not in self._caches:
            self._caches[cache_id] = cache

    def remove(self, cache):
        del self._caches[cache.id()]

    def update(self, x):
        state = (function_id(x), function_state(x))
        if state != self._state:
            self.clear()
            self._state = state
