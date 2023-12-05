from .interface import (
    DEFAULT_COMM, VariableStateLockDictionary, comm_dup_cached, space_id,
    space_new, var_assign, var_copy, var_get_values, var_global_size, var_id,
    var_is_scalar, var_is_static, var_local_indices, var_new, var_scalar_value,
    var_set_values, var_space, var_space_type)

from .instructions import Instruction

from abc import ABC, abstractmethod
from collections import deque
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import numpy as np
import os
import pickle
import weakref

__all__ = \
    [
        "CheckpointStorage",
        "ReplayStorage",

        "Checkpoints",
        "PickleCheckpoints",
        "HDF5Checkpoints"
    ]


class CheckpointStorage:
    """A buffer for forward restart data, and a cache for non-linear dependency
    data. Contains three types of data:

        1. References: Dependencies which are stored by reference. Variables
           `x` for which `var_is_static(x)` is `True` are stored by reference.
        2. Forward restart / initial condition data: Dependencies which are
           used to restart and advance the forward calculation.
        3. Non-linear dependency data: Non-linear dependencies of the forward
           which are used to advance the adjoint.

    These may overlap -- for example non-linear dependency data may be stored
    by reference.

    Non-linear dependency data has an associated key `(n, i)`, where `n` is an
    :class:`int` indicating the block index and `i` is an :class:`int`
    indicating the equation index within that block. Non-linear-dependency data
    for an :class:`.Equation` can be accessed via, e.g.

    .. code-block:: python

        nl_deps = cp[(n, i)]

    where `cp` is a :class:`.CheckpointStorage`. Here `nl_deps` is a
    :class:`tuple` of variables storing values associated with
    `eq.nonlinear_dependencies()`, for :class:`.Equation` `i` in block `n`.

    :arg store_ics: Whether to enable storage of forward restart data.
    :arg store_data: Whether to enable storage of non-linear dependency data.
    """

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

        self._storage = VariableStateLockDictionary()

        self.configure(store_ics=store_ics,
                       store_data=store_data)

    def configure(self, *, store_ics, store_data):
        """Enable or disable storage of forward restart and non-linear
        dependency data.

        :arg store_ics: Whether storage of forward restart data should be
            enabled (`store_ics=True`) or disabled (`store_ics=False`).
        :arg store_data: Whether storage of non-linear dependency data should
            be enabled (`store_data=True`) or disabled (`store_data=False`).
        """

        self._store_ics = store_ics
        self._store_data = store_data

    @property
    def store_ics(self):
        """Whether storage of forward restart data is enabled.
        """

        return self._store_ics

    @property
    def store_data(self):
        """Whether storage of non-linear dependency data is enabled.
        """

        return self._store_data

    def clear(self, *, clear_ics=True, clear_data=True, clear_refs=False):
        """Clear stored data.

        :arg clear_ics: Whether forward restart data should be cleared.
        :arg clear_data: Whether non-linear dependency data should be cleared.
        :arg clear_refs: Whether references should be cleared.
        """

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
        """Return the forward restart value associated with a variable `x`.

        :arg x: The variable, or the :class:`int` variable ID, for which the
            forward restart value should be returned.
        :arg copy: If `True` then a copy of the stored value is returned. If
            `False` then an internal variable storing the value is returned.
        :returns: A variable containing the forward restart value for `x`.
        """

        if isinstance(x, int):
            x_id = x
        else:
            x_id = var_id(x)

        if x_id in self._cp:
            ic = self._storage[self._cp[x_id]]
        else:
            ic = self._storage[self._refs[x_id]]
        return var_copy(ic) if copy else ic

    def initial_conditions(self, *, cp=True, refs=False, copy=True):
        """Access stored forward restart data.

        :arg cp: Whether to include forward restart data that is stored by
            value.
        :arg refs: Whether to include data that is stored by reference. May
            include non-linear dependency data that is not forward restart
            data.
        :arg copy: If `True` then a copy of the stored data is returned. If
            `False` then internal variables storing the data are returned.
        :returns: A :class:`.VariableStateLockDictionary`, with items `(x_id:
            x_value)`, where `x_id` is the :class:`int` variable ID and
            `x_value` is a variable storing the data.
        """

        cp_d = VariableStateLockDictionary()
        if cp:
            for x_id, x_key in self._cp.items():
                x = self._storage[x_key]
                cp_d[x_id] = var_copy(x) if copy else x
        if refs:
            for x_id, x_key in self._refs.items():
                x = self._storage[x_key]
                cp_d[x_id] = var_copy(x) if copy else x
        return cp_d

    def add_initial_condition(self, x, value=None):
        """Store forward restart data / an initial condition dependency.

        :arg x: The initial condition dependency variable.
        :arg value: A variable defining the initial condition dependency value.
            `x` is used if not supplied.
        """

        if value is None:
            value = x

        self._add_initial_condition(
            x_id=var_id(x), value=value,
            copy=not var_is_static(x))

    def update_keys(self, n, i, eq):
        """The :class:`.CheckpointStorage` keeps an internal map from forward
        variables to equations in which values for those variables are
        computed. Keys are updated automatically as needed. This method allows
        keys to be updated manually.

        :arg n: The :class:`int` index of the block.
        :arg i: The :class:`int` index of the equation.
        :arg eq: An :class:`.Equation`, equation `i` in block `n`.
        """

        for m, x in enumerate(eq.X()):
            x_id = var_id(x)
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
                self._storage[key] = var_copy(value)
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
        """Store checkpoint data associated with an equation.

        :arg n: The :class:`int` index of the block.
        :arg i: The :class:`int` index of the equation.
        :arg eq: An :class:`.Equation`, equation `i` in block `n`.
        :arg deps: Equation dependency values. `eq.dependencies()` is used if
            not supplied.
        :arg nl_deps: Equation non-linear dependency values. Extracted from
            `deps` if not supplied.
        """

        eq_deps = eq.dependencies()
        if deps is None:
            deps = eq_deps

        self.update_keys(n, i, eq)

        if self._store_ics:
            for eq_x in eq.X():
                self._seen_ics.add(var_id(eq_x))

            assert len(eq_deps) == len(deps)
            for eq_dep, dep in zip(eq_deps, deps):
                self._add_initial_condition(
                    x_id=var_id(eq_dep), value=dep,
                    copy=not var_is_static(eq_dep))

        self._add_equation_data(
            n, i, eq_deps, deps, eq.nonlinear_dependencies(), nl_deps)

    def add_equation_data(self, n, i, eq, *, nl_deps=None):
        """Store checkpoint data associated with an equation. As
        :meth:`.CheckpointStorage.add_equation`, but adds only *non-linear*
        dependency data.

        :arg n: The :class:`int` index of the block.
        :arg i: The :class:`int` index of the equation.
        :arg eq: An :class:`.Equation`, equation `i` in block `n`.
        :arg nl_deps: Equation non-linear dependency values.
            `eq.nonlinear_dependencies()` is used if not supplied.
        """

        self.update_keys(n, i, eq)

        eq_nl_deps = eq.nonlinear_dependencies()
        if nl_deps is None:
            nl_deps = eq_nl_deps

        if self._store_ics:
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                self._add_initial_condition(
                    x_id=var_id(eq_dep), value=dep,
                    copy=not var_is_static(eq_dep))

        self._add_equation_data(n, i, eq_nl_deps, nl_deps, eq_nl_deps, nl_deps)

    def _add_equation_data(self, n, i, eq_deps, deps, eq_nl_deps, nl_deps=None,
                           *, refs=True, copy=None):
        if copy is None:
            def copy(x):
                return not var_is_static(x)
        else:
            _copy = copy

            def copy(x):
                return _copy

        if self._store_data:
            if (n, i) in self._data:
                raise RuntimeError("Non-linear dependency data already stored")

            if nl_deps is None:
                assert len(eq_deps) == len(deps)
                deps_map = {var_id(eq_dep): dep
                            for eq_dep, dep in zip(eq_deps, deps)}
                nl_deps = tuple(deps_map[var_id(eq_dep)]
                                for eq_dep in eq_nl_deps)

            eq_data = []
            assert len(eq_nl_deps) == len(nl_deps)
            for eq_dep, dep in zip(eq_nl_deps, nl_deps):
                key, _ = self._store(x_id=var_id(eq_dep), value=dep,
                                     refs=refs, copy=copy(eq_dep))
                self._data_keys[key] = None  # self._data_keys.add(key)
                eq_data.append(key)
            self._data[(n, i)] = tuple(eq_data)

    def checkpoint_data(self, *, ics=True, data=True, copy=True):
        """Extract checkpoint data.

        :arg ics: Whether to extract forward restart data.
        :arg data: Whether to extract non-linear dependency data.
        :arg copy: If `True` then a copy of the stored data is returned. If
            `False` then internal variables storing the data are returned.
        :returns: A :class:`tuple` `(cp, data, storage)`. Elements of this
            :class:`tuple` are as for the three arguments for the
            :meth:`.CheckpointStorage.update` method, and here `storage` is a
            :class:`.VariableStateLockDictionary`.
        """

        if ics:
            cp_cp = tuple(self._cp.values())
        else:
            cp_cp = ()
        if data:
            cp_data = dict(self._data)
        else:
            cp_data = {}
        cp_storage = VariableStateLockDictionary()

        if ics:
            for key in cp_cp:
                value = self._storage[key]
                cp_storage[key] = var_copy(value) if copy else value
        if data:
            for key in self._data_keys:
                if key not in self._refs_keys and key not in cp_storage:
                    value = self._storage[key]
                    cp_storage[key] = var_copy(value) if copy else value

        return (cp_cp, cp_data, cp_storage)

    def update(self, cp, data, storage, *, copy=True):
        """Update the :class:`.CheckpointStorage` using the provided
        checkpoint data. Used to update the :class:`.CheckpointStorage` from
        loaded data. Note that the :class:`.CheckpointStorage` is *not*
        cleared prior to updating using the provided data.

        :arg cp: A :class:`tuple` of keys. Forward restart data is defined by
            `(storage[key] for key in cp)`.
        :arg data: A :class:`dict`. Items are `((n, i), keys)`, indicating
            that non-linear dependency data for equation `i` in block `n` is
            `(storage[key] for key in keys)`.
        :arg storage: The stored data. A :class:`Mapping` with items `((x_id,
            x_indices), x_value)`. `x_id` is the :class:`int` ID for a variable
            whose value `x_value` is stored. `x_indices` is either `None`, if
            the variable value has not been computed by solving equations with
            forward restart data storage enabled, or a tuple `(n, i, m)`
            indicating that the variable value was computed as component `m` of
            the solution to equation `i` in block `n`.
        :arg copy: Whether the values in `storage` should be copied when being
            stored in the :class:`.CheckpointStorage`.
        """

        keys = set(cp)
        for eq_data in data.values():
            keys.update(eq_data)

        for key, value in storage.items():
            if key in keys:
                if key in self._storage:
                    raise RuntimeError("Duplicate key")
                self._store(key=key, value=value, refs=False, copy=copy)

        for key in cp:
            if key in self._cp_keys or key in self._refs_keys:
                raise RuntimeError("Duplicate key")
            if key not in self._storage:
                raise ValueError("Invalid key")
            self._cp_keys.add(key)
            self._cp[key[0]] = key
            self._seen_ics.add(key[0])

        for (n, i), eq_data in data.items():
            if (n, i) in self._data:
                raise RuntimeError("Non-linear dependency data already stored")
            for key in eq_data:
                if key not in self._storage:
                    raise ValueError("Invalid key")
                self._data_keys[key] = None  # self._data_keys.add(key)
            self._data[(n, i)] = tuple(eq_data)


class ReplayStorage:
    """Storage used when solving forward equations.

    A value for a forward variable can be accessed via

    .. code-block:: python

        x_value = replay_storage[x]

    and set via

    .. code-block:: python

        replay_storage[x] = x_value

    where here `x` is either a variable of an :class:`int` variable ID.
    Containment can also be tested,

    .. code-block:: python

        if x in replay_storage:
            [...]

    :arg blocks: A :class:`Sequence` or :class:`Mapping`, whose elements or
        values are :class:`Sequence` objects containing :class:`.Equation`
        objects. Forward equations.
    :arg N0: An :class:`int`. `(blocks[n] for n in range(N0, N1))` defines the
        forward equations which will be solved.
    :arg N1: An :class:`int`. `(blocks[n] for n in range(N0, N1))` defines the
        forward equations which will be solved.
    :arg transpose_deps: A :class:`.TransposeComputationalGraph`. If supplied
        then an activity analysis is applied.
    """

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
                        x_id = var_id(x)
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

                    if isinstance(eq, Instruction):
                        active[n][i] = True

                    if transpose_deps.any_is_active(n, i):
                        # Adjoint equation is active, mark forward equations
                        # solving for non-linear dependencies as active
                        for dep in eq.nonlinear_dependencies():
                            dep_id = var_id(dep)
                            if dep_id in last_eq:
                                p, k = last_eq[dep_id][-1]
                                assert n > p or (n == p and i >= k)
                                active[p][k] = True

                    for x in eq.X():
                        x_id = var_id(x)
                        last_eq[x_id].pop()
                        if len(last_eq[x_id]) == 0:
                            del last_eq[x_id]

                    if active[n][i]:
                        # Forward equation is active, mark forward equations
                        # solving for dependencies as active
                        X_ids = set(map(var_id, eq.X()))
                        ic_ids = set(map(var_id,
                                         eq.initial_condition_dependencies()))
                        for dep in eq.dependencies():
                            dep_id = var_id(dep)
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
                        last_eq[var_id(dep)] = (n, i)
                # transpose_deps cannot be None here
                elif transpose_deps.any_is_active(n, i):
                    for dep in eq.nonlinear_dependencies():
                        last_eq[var_id(dep)] = (n, i)

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
            x_id = var_id(x)
        return x_id in self._map

    def __getitem__(self, x):
        if isinstance(x, int):
            y = self._map[x]
            if y is None:
                raise RuntimeError("Unable to create new variable")
        else:
            x_id = var_id(x)
            y = self._map[x_id]
            if y is None:
                y = self._map[x_id] = var_new(x)
        return y

    def __setitem__(self, x, y):
        if isinstance(x, int):
            x_id = x
        else:
            x_id = var_id(x)
        if self._map[x_id] is not None:  # KeyError if unexpected id
            raise RuntimeError(f"Key '{x_id:d}' already set")
        self._map[x_id] = y

    def is_active(self, n, i):
        """Return whether the activity analysis indicates that an equation is
        'active'.

        :arg n: The :class:`int` index of the block.
        :arg i: The :class:`int` index of the equation.
        :returns: `True` if the equation is active, and `False` otherwise.
        """

        return self._active[n][i]

    def update(self, d, *, copy=True):
        """Use the supplied :class:`Mapping` to update forward values.

        :arg d: A :class:`Mapping`. Updates values for those keys in `d`
            which are also in the :class:`.ReplayStorage`.
        :arg copy: Whether the values in `d` should be copied when being stored
            in the :class:`.ReplayStorage`.
        """

        for key, value in d.items():
            if key in self:
                self[key] = var_copy(value) if copy else value

    def popleft(self):
        """Remove the first equation. Used to deallocate forward variables
        which are no longer needed as the solution of forward equations
        progresses.

        :returns: A :class:`tuple` `(n, i)`, indicating that equation `i` in
            block `n` has been removed.
        """

        n, i, dep_ids = self._eq_last.popleft()
        for dep_id in dep_ids:
            del self._map[dep_id]
        return (n, i)


class Checkpoints(ABC):
    """Disk checkpointing abstract base class.
    """

    @abstractmethod
    def __contains__(self, n):
        raise NotImplementedError

    @abstractmethod
    def write(self, n, cp, data, storage):
        """Write checkpoint data.

        :arg n: The :class:`int` index of the block with which the checkpoint
            data to be written is associated.
        :arg cp: See :meth:`.CheckpointStorage.update`.
        :arg data: See :meth:`.CheckpointStorage.update`.
        :arg storage: See :meth:`.CheckpointStorage.update`.
        """

        raise NotImplementedError

    @abstractmethod
    def read(self, n, *, ics=True, data=True, ic_ids=None):
        """Read checkpoint data.

        :arg n: The :class:`int` index of the block with which the checkpoint
            data to be read is associated.
        :arg ics: Whether forward restart data should be included.
        :arg data: Whether non-linear dependency data should be included.
        :arg ic_ids: A :class:`Container`. If provided then only variables with
            ID in `ic_ids` are included.
        :returns: A :class:`tuple` `(cp, data, storage)`. Elements of this
            :class:`tuple` are as for the three arguments for the
            :meth:`.CheckpointStorage.update` method.
        """

        raise NotImplementedError

    @abstractmethod
    def delete(self, n):
        """Delete checkpoint data.

        :arg n: The :class:`int` index of the block with which the checkpoint
            data to be deleted is associated.
        """

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
    """Disk checkpointing using the pickle module.

    :arg prefix: Checkpoint files are stored at
        `[prefix]_[root_pid]_[root_py2f]_[rank].pickle`. Here `prefix` is
        defined by this argument, `root_pid` is the process ID on the root
        process (i.e. process 0), `root_py2f` is the Fortran MPI communicator
        on the root process, and `rank` is the process rank.
    :arg comm: A communicator.
    """

    def __init__(self, prefix, *, comm=None):
        if comm is None:
            comm = DEFAULT_COMM

        comm = comm_dup_cached(comm)
        cp_filenames = {}

        def finalize_callback(cp_filenames):
            for filename in cp_filenames.values():
                os.remove(filename)

        weakref.finalize(self, finalize_callback,
                         cp_filenames)

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
            F_space = var_space(F)
            F_space_id = space_id(F_space)
            spaces.setdefault(F_space_id, F_space)

            if var_is_scalar(F):
                F_values = var_scalar_value(F)
            else:
                F_values = var_get_values(F)

            write_storage[key] = (F_space_id,
                                  var_space_type(F),
                                  F_values)

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
            if var_is_scalar(F):
                var_assign(F, F_values)
            else:
                var_set_values(F, F_values)
            read_storage[key] = F

        return read_cp, read_data, read_storage

    def delete(self, n):
        filename = self._cp_filenames[n]
        os.remove(filename)
        del self._cp_filenames[n]
        del self._cp_spaces[n]


class HDF5Checkpoints(Checkpoints):
    """Disk checkpointing using the h5py library.

    :arg prefix: Checkpoint files are stored at
        `[prefix]_[root_pid]_[root_py2f].hdf5`. Here `prefix` is defined by
        this argument, `root_pid` is the process ID on the root process (i.e.
        process 0), and `root_py2f` is the Fortran MPI communicator on the root
        process.
    :arg comm: A communicator.
    """

    def __init__(self, prefix, *, comm=None):
        if comm is None:
            comm = DEFAULT_COMM

        comm = comm_dup_cached(comm, key="HDF5Checkpoints")
        cp_filenames = {}

        def finalize_callback(comm, rank, cp_filenames):
            if MPI is not None and not MPI.Is_finalized():
                comm.barrier()
            if rank == 0:
                for filename in cp_filenames.values():
                    os.remove(filename)
            if MPI is not None and not MPI.Is_finalized():
                comm.barrier()

        weakref.finalize(self, finalize_callback,
                         comm, comm.rank, cp_filenames)

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
                dtype=np.int_)
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
                    dtype=np.int_)
                for k, (x_id, x_indices) in enumerate(eq_data):
                    d[k, 0, self._comm.rank] = x_id
                    if x_indices is None:
                        d[k, 1:, self._comm.rank] = -1
                    else:
                        d[k, 1:, self._comm.rank] = x_indices

            h.create_group("/storage")
            for j, ((x_id, x_indices), F) in enumerate(storage.items()):
                F_space = var_space(F)
                F_space_id = space_id(F_space)
                spaces.setdefault(F_space_id, F_space)

                g = h.create_group(f"/storage/{j:d}")

                d = g.create_dataset(
                    "key", shape=(4, self._comm.size),
                    dtype=np.int_)
                d[0, self._comm.rank] = x_id
                if x_indices is None:
                    d[1:, self._comm.rank] = -1
                else:
                    d[1:, self._comm.rank] = x_indices

                g.attrs["space_type"] = var_space_type(F)

                d = g.create_dataset(
                    "space_id", shape=(self._comm.size,),
                    dtype=np.int_)
                d[self._comm.rank] = F_space_id

                if var_is_scalar(F):
                    F_values = var_scalar_value(F)
                else:
                    F_values = var_get_values(F)
                d = g.create_dataset(
                    "value", shape=(var_global_size(F),),
                    dtype=F_values.dtype)
                d[var_local_indices(F)] = F_values

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
                    if var_is_scalar(F):
                        d, = d
                        var_assign(F, d)
                    else:
                        var_set_values(F, d[var_local_indices(F)])

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
