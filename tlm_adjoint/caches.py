#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import var_caches, var_id, var_state

from .alias import gc_disabled

import functools
import weakref

__all__ = \
    [
        "CacheRef",

        "Cache",
        "Caches",

        "clear_caches",
        "local_caches"
    ]


class CacheRef:
    """A cache entry. Stores a reference to a cached value, which can later be
    cleared. Calling a :class:`.CacheRef` returns the cached object, or `None`
    if no object is referenced.

    :arg value: The object to reference. `None` may be supplied to indicate an
        empty cache entry.
    """

    def __init__(self, value=None):
        self._value = value

    def __call__(self):
        return self._value

    def clear(self):
        """Clear the cache entry. After calling this method, calling the
        :class:`.CacheRef` will return `None`.
        """

        self._value = None


@gc_disabled
def clear_caches(*deps):
    """Clear caches entries.

    :arg deps: A :class:`Sequence` of variables. If non-empty then clear only
        cache entries which depend on the supplied variables. Otherwise clear
        all cache entries.
    """

    if len(deps) == 0:
        for cache_id in sorted(tuple(Cache._caches)):
            cache = Cache._caches.get(cache_id, None)
            if cache is not None:
                cache.clear()
    else:
        for dep in deps:
            var_caches(dep).clear()


def local_caches(fn):
    """Decorator clearing caches before and after calling the decorated
    callable.

    :arg fn: A callable for which caches should be cleared.
    :returns: A callable where caches are cleared before and after calling.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        clear_caches()
        try:
            return fn(*args, **kwargs)
        finally:
            clear_caches()
    return wrapped_fn


class Cache:
    """Stores cache entries.

    Cleared cache entries are removed from the :class:`.Cache`.
    """

    _id_counter = [0]
    _caches = weakref.WeakValueDictionary()

    def __init__(self):
        self._cache = {}
        self._deps_map = {}
        self._dep_caches = {}

        self._id, = self._id_counter
        self._id_counter[0] += 1
        self._caches[self._id] = self

        def finalize_callback(cache):
            for value in cache.values():
                value.clear()

        weakref.finalize(self, finalize_callback,
                         self._cache)

    def __len__(self):
        return len(self._cache)

    @property
    def id(self):
        """A unique :class:`int` ID associated with this :class:`.Cache`.
        """

        return self._id

    def clear(self, *deps):
        """Clear cache entries.

        :arg deps: A :class:`Sequence` of variables. If non-empty then only
            clear cache entries which depend on the supplied variables.
            Otherwise clear all cache entries.
        """

        if len(deps) == 0:
            for value in self._cache.values():
                value.clear()
            self._cache.clear()
            self._deps_map.clear()
            for dep_caches in self._dep_caches.values():
                dep_caches = dep_caches()
                if dep_caches is not None:
                    dep_caches.remove(self)
            self._dep_caches.clear()
        else:
            for dep in deps:
                dep_id = dep if isinstance(dep, int) else var_id(dep)
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
                        self._cache[key].clear()
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
        """Add a cache entry.

        :arg key: The key associated with the cache entry.
        :arg value: A callable, taking no arguments, returning the value
            associated with the cache entry. Only called to if no entry
            associated with `key` exists.
        :arg deps: A :class:`Sequence` of variables, defining dependencies of
            the cache entry.
        :returns: A :class:`tuple` `(value_ref, value)`, where `value` is the
            cache entry value and `value_ref` is a :class:`.CacheRef` storing a
            reference to the value.
        """

        if deps is None:
            deps = ()

        if key in self._cache:
            value_ref = self._cache[key]
            value = value_ref()
            if value is None:
                raise RuntimeError("Unexpected cache value state")
            return value_ref, value

        value = value()
        value_ref = CacheRef(value)
        dep_ids = tuple(map(var_id, deps))

        self._cache[key] = value_ref

        assert len(deps) == len(dep_ids)
        for dep, dep_id in zip(deps, dep_ids):
            dep_caches = var_caches(dep)
            dep_caches.add(self)

            if dep_id in self._deps_map:
                self._deps_map[dep_id][key] = dep_ids
                assert dep_id in self._dep_caches
            else:
                self._deps_map[dep_id] = {key: dep_ids}
                self._dep_caches[dep_id] = weakref.ref(dep_caches)

        return value_ref, value

    def get(self, key, *args):
        """Return the cache entry associated with a given key.

        :arg key: The key.
        :returns: The cache entry or, if supplied, a default value.

        `args` should contain zero or one elements and defines the default
        value. If there is no entry associated with the key then:

            - If `args` has no elements an exception is raised.
            - If `args` has one element then this is returned.
        """

        if len(args) == 0:
            return self._cache[key]
        else:
            default, = args
            return self._cache.get(key, default)


class Caches:
    """Multiple :class:`.Cache` objects, associated with a variable.

    Cache entries may depend on the variable. The variable also defines an
    initial value, and the value is indicated by the variable ID and variable
    state value. The value may be changed either by supplying a new variable
    (changing the ID), or by changing the value of the current variable
    defining the value (which should be indicated by a change to the variable
    state value). Either change invalidates cache entries, in the
    :class:`.Cache` objects, which depend on the original variable.

    The :meth:`.Caches.update` method can be used to check for cache entry
    invalidation, and to clear invalid cache entries.

    :arg x: The variable defining a possible cache entry dependency, and an
        initial value for that dependency.
    """

    def __init__(self, x):
        self._caches = weakref.WeakValueDictionary()
        self._id = var_id(x)
        self._state = (self._id, var_state(x))

    def __len__(self):
        return len(self._caches)

    @gc_disabled
    def clear(self):
        """Clear cache entries which depend on the associated variable.
        """

        for cache_id in sorted(tuple(self._caches)):
            cache = self._caches.get(cache_id, None)
            if cache is not None:
                cache.clear(self._id)
                assert cache.id not in self._caches

    def add(self, cache):
        """Add a new :class:`.Cache` to the :class:`.Caches`.

        :arg cache: The :class:`.Cache` to add to the :class:`.Caches`.
        """

        self._caches.setdefault(cache.id, cache)

    def remove(self, cache):
        """Remove a :class:`.Cache` from the :class:`.Caches`.

        :arg cache: The :class:`.Cache` to remove from the :class:`.Caches`.
        """

        del self._caches[cache.id]

    def update(self, x):
        """Check for cache invalidation associated with a possible change in
        value, and clear invalid cache entries.

        :arg x: A variable which defines a potentially new value.
        """

        state = (var_id(x), var_state(x))
        if state != self._state:
            self.clear()
            self._state = state
