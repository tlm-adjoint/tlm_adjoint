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

from .interface import function_assign, function_copy, function_get_values, \
    function_global_size, function_id, function_is_checkpointed, \
    function_is_replacement, function_local_indices, function_name, \
    function_new, function_new_tangent_linear, function_set_values, \
    function_space, is_function, space_id, space_new

from .alias import Alias, WeakAlias, gc_disabled
from .binomial_checkpointing import MultistageManager
from .equations import AdjointModelRHS, ControlsMarker, Equation, \
    FunctionalMarker, NullSolver
from .functional import Functional
from .manager import manager as _manager, set_manager

from collections import OrderedDict, defaultdict, deque
from collections.abc import Sequence
import copy
import gc
import logging
import numpy as np
import pickle
import os
import warnings
import weakref

__all__ = \
    [
        "CheckpointStorage",
        "Control",
        "EquationManager",
        "ManagerException"
    ]


class ManagerException(Exception):
    pass


try:
    from mpi4py.MPI import COMM_WORLD as _default_comm
except ImportError:
    # As for mpi4py 3.0.3 API
    class SerialComm:
        _id_counter = [-1]

        def __init__(self):
            self._id = self._id_counter[0]
            self._id_counter[0] -= 1

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 1

        def Dup(self, info=None):
            return SerialComm()

        def Free(self):
            pass

        def allgather(self, sendobj):
            return [copy.deepcopy(sendobj)]

        def barrier(self):
            pass

        def bcast(self, obj, root=0):
            return copy.deepcopy(obj)

        def gather(self, sendobj, root=0):
            assert root == 0
            return [copy.deepcopy(sendobj)]

        def py2f(self):
            return self._id

        def scatter(self, sendobj, root=0):
            assert root == 0
            sendobj, = sendobj
            return copy.deepcopy(sendobj)

    _default_comm = SerialComm()


class Control(Alias):
    def __init__(self, m, manager=None):
        if manager is None:
            manager = _manager()

        if isinstance(m, str):
            m = manager.find_initial_condition(m)

        super().__init__(m)

    def __new__(cls, m, manager=None):
        warnings.warn("Control class is deprecated",
                      DeprecationWarning, stacklevel=2)

        if manager is None:
            manager = _manager()

        if isinstance(m, str):
            m = manager.find_initial_condition(m)

        return super().__new__(cls, m)

    def m(self):
        return self


class CheckpointStorage:
    def __init__(self, store_ics=True, store_data=True):
        self._seen_ics = set()
        self._cp = {}
        self._indices = defaultdict(lambda: 0)
        self._dep_keys = {}
        self._data = {}
        self._refs = {}

        self.configure(store_ics=store_ics,
                       store_data=store_data)

    def configure(self, store_ics=None, store_data=None):
        """
        Configure storage.

        Arguments:

        store_ics   Store initial condition data, used by checkpointing
        store_data  Store equation non-linear dependency data, used in reverse
                    mode
        """

        if store_ics is not None:
            self._store_ics = store_ics
        if store_data is not None:
            self._store_data = store_data

    def store_ics(self):
        return self._store_ics

    def store_data(self):
        return self._store_data

    def clear(self, clear_cp=True, clear_data=True, clear_refs=False):
        if clear_cp:
            self._seen_ics.clear()
            self._cp.clear()
        if clear_data:
            self._indices.clear()
            self._dep_keys.clear()
            self._data.clear()
        if clear_refs:
            self._refs.clear()
        else:
            for x_id, x in self._refs.items():
                # May have been cleared above
                self._seen_ics.add(x_id)

                x_key = self._data_key(x_id)
                if x_key not in self._data:
                    self._data[x_key] = x

    def __getitem__(self, key):
        return tuple(self._data[dep_key] for dep_key in self._dep_keys[key])

    def initial_condition(self, x, copy=True):
        x_id = function_id(x)
        if x_id in self._refs:
            ic = self._refs[x_id]
        else:
            ic = self._cp[x_id]
        if copy:
            ic = function_copy(ic)
        return ic

    def initial_conditions(self, cp=True, refs=False, copy=True):
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

    def add_initial_condition(self, x, value=None, copy=None):
        if value is None:
            value = x
        if copy is None:
            copy = function_is_checkpointed(x)
        self._add_initial_condition(x_id=function_id(x), value=value,
                                    copy=copy)

    def _add_initial_condition(self, x_id, value, copy):
        if self._store_ics and x_id not in self._seen_ics:
            x_key = self._data_key(x_id)
            if x_key in self._data:
                if copy:
                    self._cp[x_id] = self._data[x_key]
                else:
                    assert x_id in self._refs
            else:
                if copy:
                    value = function_copy(value)
                    self._cp[x_id] = value
                else:
                    self._refs[x_id] = value
                self._data[x_key] = value
            self._seen_ics.add(x_id)

    def add_equation(self, key, eq, deps=None, nl_deps=None,
                     copy=lambda x: function_is_checkpointed(x)):
        eq_X = eq.X()
        eq_deps = eq.dependencies()
        if deps is None:
            deps = eq_deps

        for eq_x in eq_X:
            self._indices[function_id(eq_x)] += 1

        if self._store_ics:
            for eq_x in eq_X:
                self._seen_ics.add(function_id(eq_x))
            for eq_dep, dep in zip(eq_deps, deps):
                self.add_initial_condition(eq_dep, value=dep,
                                           copy=copy(eq_dep))

        if self._store_data:
            if nl_deps is None:
                nl_deps = tuple(deps[i]
                                for i in eq.nonlinear_dependencies_map())

            dep_keys = []
            for eq_dep, dep in zip(eq.nonlinear_dependencies(), nl_deps):
                eq_dep_id = function_id(eq_dep)
                dep_key = self._data_key(eq_dep_id)
                if dep_key not in self._data:
                    if copy(eq_dep):
                        self._data[dep_key] = function_copy(dep)
                    else:
                        self._data[dep_key] = dep
                        if eq_dep_id not in self._seen_ics:
                            self._seen_ics.add(eq_dep_id)
                            self._refs[eq_dep_id] = dep
                dep_keys.append(dep_key)
            self._dep_keys[key] = dep_keys


class TangentLinearMap:
    """
    A map from forward to tangent-linear variables.
    """

    def __init__(self, name_suffix=" (tangent-linear)"):
        self._name_suffix = name_suffix
        self._map = {}
        self._finalizes = {}

        @gc_disabled
        def finalize_callback(finalizes):
            for finalize in finalizes.values():
                finalize.detach()
            finalizes.clear()
        finalize = weakref.finalize(self, finalize_callback, self._finalizes)
        finalize.atexit = False

    @gc_disabled
    def __contains__(self, x):
        return function_id(x) in self._map

    @gc_disabled
    def __getitem__(self, x):
        if not is_function(x):
            raise ManagerException("x must be a function")
        assert not isinstance(x, WeakAlias)

        x_id = function_id(x)
        if x_id not in self._map:
            self._map[x_id] = function_new_tangent_linear(
                x, name=f"{function_name(x):s}{self._name_suffix:s}")

            @gc_disabled
            def finalize_callback(self_ref, x_id):
                self = self_ref()
                if self is not None:
                    del self._finalizes[x_id]
                    del self._map[x_id]
            finalize = weakref.finalize(
                x, finalize_callback, weakref.ref(self), x_id)
            finalize.atexit = False
            assert x_id not in self._finalizes
            self._finalizes[x_id] = finalize

        return self._map[x_id]


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
                raise ManagerException("Unable to create new function")
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


class DependencyGraphTranspose:
    def __init__(self, blocks, M, Js, prune_forward=True, prune_adjoint=True):
        # Transpose dependency graph
        last_eq = {}
        transpose_deps = tuple(tuple([None for dep in eq.dependencies()]
                                     for eq in block)
                               for block in blocks)
        for n, block in enumerate(blocks):
            for i, eq in enumerate(block):
                for m, x in enumerate(eq.X()):
                    last_eq[function_id(x)] = (n, i, m)
                for j, dep in enumerate(eq.dependencies()):
                    dep_id = function_id(dep)
                    if dep_id in last_eq:
                        p, k, m = last_eq[dep_id]
                        if p < n or k < i:
                            transpose_deps[n][i][j] = (p, k, m)
        del last_eq

        if prune_forward:
            # Extra reverse traversal to add edges associated with adjoint
            # initial conditions
            last_eq = {}
            transpose_deps_ics = copy.deepcopy(transpose_deps)
            for p in range(len(blocks) - 1, -1, -1):
                block = blocks[p]
                for k in range(len(block) - 1, -1, -1):
                    eq = block[k]
                    X_map = {function_id(x): m for m, x in enumerate(eq.X())}
                    dep_map = {function_id(dep): j
                               for j, dep in enumerate(eq.dependencies())}
                    for dep in eq.adjoint_initial_condition_dependencies():
                        dep_id = function_id(dep)
                        if dep_id in last_eq:
                            n, i, m = last_eq[dep_id]
                            assert n > p or (n == p and i > k)
                            transpose_deps_ics[n][i][m] \
                                = (p, k, dep_map[dep_id])
                    for m, x in enumerate(eq.X()):
                        x_id = function_id(x)
                        last_eq[x_id] = (p, k, X_map[x_id])
            del last_eq

            # Pruning, forward traversal
            active_M = {function_id(dep) for dep in M}
            active_forward = tuple(np.full(len(block), False, dtype=bool)
                                   for block in blocks)
            for n, block in enumerate(blocks):
                for i, eq in enumerate(block):
                    if len(active_M) > 0:
                        X_ids = {function_id(x) for x in eq.X()}
                        for x_id in X_ids:
                            if x_id in active_M:
                                active_M.difference_update(X_ids)
                                active_forward[n][i] = True
                                break
                    if not active_forward[n][i]:
                        for j, dep in enumerate(eq.dependencies()):
                            if transpose_deps_ics[n][i][j] is not None:
                                p, k, m = transpose_deps_ics[n][i][j]
                                if active_forward[p][k]:
                                    active_forward[n][i] = True
                                    break
        else:
            active_forward = tuple(np.full(len(block), True, dtype=bool)
                                   for block in blocks)

        active = {function_id(J): copy.deepcopy(active_forward) for J in Js}

        if prune_adjoint:
            # Pruning, reverse traversal
            for J_id in active:
                active_J = True
                active_adjoint = tuple(np.full(len(block), False, dtype=bool)
                                       for block in blocks)
                for n in range(len(blocks) - 1, -1, -1):
                    block = blocks[n]
                    for i in range(len(block) - 1, -1, -1):
                        eq = block[i]
                        if active_J:
                            for x in eq.X():
                                if function_id(x) == J_id:
                                    active_J = False
                                    active_adjoint[n][i] = True
                                    break
                        if active_adjoint[n][i]:
                            for j, dep in enumerate(eq.dependencies()):
                                if transpose_deps[n][i][j] is not None:
                                    p, k, m = transpose_deps[n][i][j]
                                    active_adjoint[p][k] = True
                        else:
                            active[J_id][n][i] = False

        stored_adj_ics = {function_id(J): tuple(tuple(np.full(len(eq.X()), False, dtype=bool)  # noqa: E501
                          for eq in block) for block in blocks) for J in Js}
        adj_ics = {function_id(J): {} for J in Js}
        for J_id in stored_adj_ics:
            stored = {}
            for n, block in enumerate(blocks):
                for i, eq in enumerate(block):
                    if active[J_id][n][i]:
                        for m, x in enumerate(eq.X()):
                            stored_adj_ics[J_id][n][i][m] = \
                                stored.get(function_id(x), False)

                        adj_ic_ids = {function_id(dep) for dep in eq.adjoint_initial_condition_dependencies()}  # noqa: E501
                        for dep_id in adj_ic_ids:
                            stored[dep_id] = True
                            if dep_id not in adj_ics[J_id]:
                                adj_ics[J_id][dep_id] = True
                        for x in eq.X():
                            x_id = function_id(x)
                            if x_id not in adj_ic_ids:
                                stored[x_id] = False
                                if x_id not in adj_ics[J_id]:
                                    adj_ics[J_id][x_id] = False

        self._transpose_deps = transpose_deps
        self._active = active
        self._stored_adj_ics = stored_adj_ics
        self._adj_ics = adj_ics

    def __contains__(self, key):
        n, i, j = key
        return self._transpose_deps[n][i][j] is not None

    def __getitem__(self, key):
        n, i, j = key
        p, k, m = self._transpose_deps[n][i][j]
        return p, k, m

    def is_active(self, J, n, i):
        if isinstance(J, int):
            J_id = J
        else:
            J_id = function_id(J)
        return self._active[J_id][n][i]

    def has_adj_ic(self, J, x):
        if isinstance(J, int):
            J_id = J
        else:
            J_id = function_id(J)
        if isinstance(x, int):
            x_id = x
        else:
            x_id = function_id(x)
        return self._adj_ics[J_id].get(x_id, False)

    def is_stored_adj_ic(self, J, n, i, m):
        if isinstance(J, int):
            J_id = J
        else:
            J_id = function_id(J)
        return self._stored_adj_ics[J_id][n][i][m]


class EquationManager:
    _id_counter = [0]

    def __init__(self, comm=None, cp_method="memory", cp_parameters={}):
        """
        Manager for tangent-linear and adjoint models.

        Arguments:
        comm  (Optional) Communicator.

        cp_method  (Optional) Checkpointing method. Default "memory".
            Possible methods
                none
                    No storage.
                memory
                    Store everything in RAM.
                periodic_disk
                    Periodically store initial condition data on disk.
                multistage
                    Binomial checkpointing using the approach described in
                        GW2000  A. Griewank and A. Walther, "Algorithm 799:
                                Revolve: An implementation of checkpointing for
                                the reverse or adjoint mode of computational
                                differentiation", ACM Transactions on
                                Mathematical Software, 26(1), pp. 19--45, 2000
                    with a brute force search used to obtain behaviour
                    described in
                        SW2009  P. Stumm and A. Walther, "MultiStage approaches
                                for optimal offline checkpointing", SIAM
                                Journal on Scientific Computing, 31(3),
                                pp. 1946--1967, 2009

        cp_parameters  (Optional) Checkpointing parameters dictionary.
            Parameters for "none" method
                drop_references  Whether to automatically drop references to
                                 internal functions in the provided equations.
                                 Logical, optional, default False.

            Parameters for "memory" method
                drop_references  Whether to automatically drop references to
                                 internal functions in the provided equations.
                                 Logical, optional, default False.

            Parameters for "periodic_disk" method
                path           Directory in which disk checkpoint data should
                               be stored. String, optional, default
                               "checkpoints~".
                format         Disk checkpointing format. One of {"pickle",
                               "hdf5"}, optional, default "hdf5".
                period         Interval between checkpoints. Positive integer,
                               required.

            Parameters for "multistage" method
                path           Directory in which disk checkpoint data should
                               be stored. String, optional, default
                               "checkpoints~".
                format         Disk checkpointing format. One of {"pickle",
                               "hdf5"}, optional, default "hdf5".
                blocks         Total number of blocks. Positive integer,
                               required.
                snaps_in_ram   Number of "snaps" to store in RAM. Non-negative
                               integer, optional, default 0.
                snaps_on_disk  Number of "snaps" to store on disk. Non-negative
                               integer, optional, default 0.
        """
        # "multistage" name, and "snaps_in_ram", and "snaps_on_disk" in
        # "multistage" method, are similar to adj_checkpointing arguments in
        # dolfin-adjoint 2017.1.0

        if comm is None:
            comm = _default_comm
        comm = comm.Dup()

        self._comm = comm
        if self._comm.rank == 0:
            id = self._id_counter[0]
            self._id_counter[0] += 1
            comm_py2f = self._comm.py2f()
        else:
            id = -1
            comm_py2f = -1
        self._id = self._comm.bcast(id, root=0)
        self._comm_py2f = self._comm.bcast(comm_py2f, root=0)

        self._tlm_eqs = {}
        self._to_drop_references = []
        self._finalizes = {}

        self.reset(cp_method=cp_method, cp_parameters=cp_parameters)

        @gc_disabled
        def finalize_callback(comm,
                              to_drop_references, finalizes):
            comm.Free()

            while len(to_drop_references) > 0:
                referrer = to_drop_references.pop()
                referrer._drop_references()
            for finalize in finalizes.values():
                finalize.detach()
            finalizes.clear()
        finalize = weakref.finalize(self, finalize_callback,
                                    self._comm,
                                    self._to_drop_references, self._finalizes)
        finalize.atexit = False

    def comm(self):
        return self._comm

    def info(self, info=print):
        """
        Display information about the equation manager state.

        Arguments:

        info  A callable which displays a provided string.
        """

        info("Equation manager status:")
        info(f"Annotation state: {self._annotation_state:s}")
        info(f"Tangent-linear state: {self._tlm_state:s}")
        info("Equations:")
        blocks = copy.copy(self._blocks)
        if len(self._block) > 0:
            blocks.append(self._block)
        for n, block in enumerate(blocks):
            info(f"  Block {n:d}")
            for i, eq in enumerate(block):
                eq_X = eq.X()
                if len(eq_X) == 1:
                    X_name = function_name(eq_X[0])
                    X_ids = f"id {function_id(eq_X[0]):d}"
                else:
                    X_name = "(%s)" % (",".join(function_name(eq_x)
                                                for eq_x in eq_X))
                    X_ids = "ids (%s)" % (",".join(f"{function_id(eq_x):d}"
                                                   for eq_x in eq_X))
                info("    Equation %i, %s solving for %s (%s)" %
                     (i, type(eq).__name__, X_name, X_ids))
                nl_dep_ids = {function_id(dep)
                              for dep in eq.nonlinear_dependencies()}
                for j, dep in enumerate(eq.dependencies()):
                    info("      Dependency %i, %s (id %i)%s, %s" %
                         (j, function_name(dep), function_id(dep),
                          ", replaced" if function_is_replacement(dep) else "",
                          "non-linear" if function_id(dep) in nl_dep_ids else "linear"))  # noqa: E501
        info("Storage:")
        info(f'  Storing initial conditions: {"yes" if self._cp.store_ics() else "no":s}')  # noqa: E501
        info(f'  Storing equation non-linear dependencies: {"yes" if self._cp.store_data() else "no":s}')  # noqa: E501
        info(f"  Initial conditions stored: {len(self._cp._cp):d}")
        info(f"  Initial conditions referenced: {len(self._cp._refs):d}")
        info("Checkpointing:")
        info(f"  Method: {self._cp_method:s}")
        if self._cp_method in ["none", "memory"]:
            pass
        elif self._cp_method == "periodic_disk":
            info(f"  Function spaces referenced: {len(self._cp_spaces):d}")
        elif self._cp_method == "multistage":
            info(f"  Function spaces referenced: {len(self._cp_spaces):d}")
            info(f"  Snapshots in RAM: {self._cp_manager.snapshots_in_ram():d}")  # noqa: E501
            info(f"  Snapshots on disk: {self._cp_manager.snapshots_on_disk():d}")  # noqa: E501
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{self._cp_method:s}")

    def new(self, cp_method=None, cp_parameters=None):
        """
        Return a new equation manager sharing the communicator of this
        equation manager. Optionally a new checkpointing configuration can be
        provided.
        """

        if cp_method is None:
            cp_method = self._cp_method
        if cp_parameters is None:
            cp_parameters = self._cp_parameters

        return EquationManager(comm=self._comm, cp_method=cp_method,
                               cp_parameters=cp_parameters)

    @gc_disabled
    def reset(self, cp_method=None, cp_parameters=None):
        """
        Reset the equation manager. Optionally a new checkpointing
        configuration can be provided.
        """

        self.drop_references()

        if cp_method is None:
            cp_method = self._cp_method
        if cp_parameters is None:
            cp_parameters = self._cp_parameters

        self._annotation_state = "initial"
        self._tlm_state = "initial"
        self._eqs = {}
        self._blocks = []
        self._block = []

        self._tlm = OrderedDict()
        self._tlm_eqs.clear()

        self.configure_checkpointing(cp_method, cp_parameters=cp_parameters)

    def configure_checkpointing(self, cp_method, cp_parameters={}):
        """
        Provide a new checkpointing configuration.
        """

        if self._annotation_state not in ["initial", "stopped_initial"]:
            raise ManagerException("Cannot configure checkpointing after annotation has started, or after finalization")  # noqa: E501

        cp_parameters = copy.deepcopy(cp_parameters)

        if cp_method in ["none", "memory"]:
            disk_storage = False
        elif cp_method == "periodic_disk":
            disk_storage = True
        elif cp_method == "multistage":
            disk_storage = cp_parameters.get("snaps_on_disk", 0) > 0
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{cp_method:s}")

        if disk_storage:
            cp_parameters["path"] = cp_path = cp_parameters.get("path", "checkpoints~")  # noqa: E501
            cp_parameters["format"] = cp_parameters.get("format", "hdf5")

            if self._comm.rank == 0:
                if not os.path.exists(cp_path):
                    os.makedirs(cp_path)
            self._comm.barrier()

        if cp_method in ["none", "memory"]:
            cp_manager = None
            if "replace" in cp_parameters:
                warnings.warn("'replace' cp_parameters key is deprecated",
                              DeprecationWarning, stacklevel=2)
                if "drop_references" in cp_parameters:
                    if cp_parameters["replace"] != cp_parameters["drop_references"]:  # noqa: E501
                        raise ManagerException("Conflicting cp_parameters "
                                               "values")
                else:
                    cp_parameters["drop_references"] = cp_parameters["replace"]
                del cp_parameters["replace"]
            else:
                cp_parameters["drop_references"] = cp_parameters.get("drop_references", False)  # noqa: E501
        elif cp_method == "periodic_disk":
            cp_manager = set()
        elif cp_method == "multistage":
            cp_blocks = cp_parameters["blocks"]
            cp_parameters["snaps_in_ram"] = cp_snaps_in_ram = cp_parameters.get("snaps_in_ram", 0)  # noqa: E501
            cp_parameters["snaps_on_disk"] = cp_snaps_on_disk = cp_parameters.get("snaps_on_disk", 0)  # noqa: E501

            cp_manager = MultistageManager(cp_blocks,
                                           cp_snaps_in_ram, cp_snaps_on_disk)
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{cp_method:s}")

        self._cp_method = cp_method
        self._cp_parameters = cp_parameters
        self._cp_manager = cp_manager
        self._cp_spaces = {}
        self._cp_memory = {}
        self._cp_disk = {}

        if cp_method == "none":
            self._cp = CheckpointStorage(store_ics=False, store_data=False)
        elif cp_method == "memory":
            self._cp = CheckpointStorage(store_ics=True, store_data=True)
        elif cp_method == "periodic_disk":
            self._cp = CheckpointStorage(store_ics=True, store_data=False)
        elif cp_method == "multistage":
            logger = logging.getLogger("tlm_adjoint.multistage_checkpointing")

            if self._cp_manager.max_n() == 1:
                logger.debug("forward: configuring storage for reverse")
                self._cp = CheckpointStorage(store_ics=True,
                                             store_data=True)
            else:
                logger.debug("forward: configuring storage for snapshot")
                self._cp = CheckpointStorage(store_ics=True,
                                             store_data=False)
                logger.debug(f"forward: deferred snapshot at {self._cp_manager.n():d}")  # noqa: E501
                self._cp_manager.snapshot()
            self._cp_manager.forward()
            logger.debug(f"forward: forward advance to {self._cp_manager.n():d}")  # noqa: E501
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{cp_method:s}")

    def add_tlm(self, M, dM, max_depth=1):
        """
        Add a tangent-linear model computing derivatives with respect to the
        control defined by M in the direction defined by dM.
        """

        if self._tlm_state == "final":
            raise ManagerException("Cannot add a tangent-linear model after finalization")  # noqa: E501

        if is_function(M):
            M = (M,)
        else:
            M = tuple(M)
        if is_function(dM):
            dM = (dM,)
        else:
            dM = tuple(dM)

        if (M, dM) in self._tlm:
            raise ManagerException("Duplicate tangent-linear model")

        if self._tlm_state == "initial":
            self._tlm_state = "deriving"
        elif self._tlm_state == "stopped_initial":
            self._tlm_state = "stopped_deriving"

        if len(M) == 1:
            tlm_map_name_suffix = \
                "_tlm(%s,%s)" % (function_name(M[0]),
                                 function_name(dM[0]))
        else:
            tlm_map_name_suffix = \
                "_tlm((%s),(%s))" % (",".join(function_name(m) for m in M),
                                     ",".join(function_name(dm) for dm in dM))
        self._tlm[(M, dM)] = (TangentLinearMap(tlm_map_name_suffix), max_depth)

    def tlm_enabled(self):
        """
        Return whether addition of tangent-linear models is enabled.
        """

        return self._tlm_state == "deriving"

    def tlm(self, M, dM, x, max_depth=1):
        """
        Return a tangent-linear function associated with the forward function
        x, for the tangent-linear model defined by M and dM.
        """

        if is_function(M):
            M = (M,)
        else:
            M = tuple(M)
        if is_function(dM):
            dM = (dM,)
        else:
            dM = tuple(dM)

        if (M, dM) in self._tlm:
            for depth in range(max_depth):
                x = self._tlm[(M, dM)][0][x]
            return x
        else:
            raise ManagerException("Tangent-linear not found")

    def annotation_enabled(self):
        """
        Return whether the equation manager currently has annotation enabled.
        """

        return self._annotation_state in ["initial", "annotating"]

    def start(self, annotation=True, tlm=True):
        """
        Start annotation or tangent-linear derivation.
        """

        if annotation:
            if self._annotation_state == "stopped_initial":
                self._annotation_state = "initial"
            elif self._annotation_state == "stopped_annotating":
                self._annotation_state = "annotating"

        if tlm:
            if self._tlm_state == "stopped_initial":
                self._tlm_state = "initial"
            elif self._tlm_state == "stopped_deriving":
                self._tlm_state = "deriving"

    def stop(self, annotation=True, tlm=True):
        """
        Pause annotation or tangent-linear derivation. Returns a tuple
        containing:
            (annotation_state, tlm_state)
        where annotation_state is True if the annotation is in state "initial"
        or "annotating" and False otherwise, and tlm_state is True if the
        tangent-linear state is "initial" or "deriving" and False otherwise,
        each evaluated before changing the state.
        """

        state = (self._annotation_state in ["initial", "annotating"],
                 self._tlm_state in ["initial", "deriving"])

        if annotation:
            if self._annotation_state == "initial":
                self._annotation_state = "stopped_initial"
            elif self._annotation_state == "annotating":
                self._annotation_state = "stopped_annotating"

        if tlm:
            if self._tlm_state == "initial":
                self._tlm_state = "stopped_initial"
            elif self._tlm_state == "deriving":
                self._tlm_state = "stopped_deriving"

        return state

    def add_initial_condition(self, x, annotate=None):
        """
        Add an initial condition associated with the function x on the adjoint
        tape.

        annotate (default self.annotation_enabled()):
            Whether to annotate the initial condition on the adjoint tape,
            storing data for checkpointing as required.
        """

        if annotate is None:
            annotate = self.annotation_enabled()
        if annotate:
            if self._annotation_state == "initial":
                self._annotation_state = "annotating"
            elif self._annotation_state == "stopped_initial":
                self._annotation_state = "stopped_annotating"
            elif self._annotation_state == "final":
                raise ManagerException("Cannot add initial conditions after finalization")  # noqa: E501

            self._cp.add_initial_condition(x)

    def initial_condition(self, x):
        """
        Return the value of the initial condition for x recorded for the first
        block. Finalizes the manager.
        """

        self.finalize()

        x_id = function_id(x)
        for eq in self._blocks[0]:
            if x_id in {function_id(dep) for dep in eq.dependencies()}:
                self._restore_checkpoint(0)
                return self._cp.initial_condition(
                    x, copy=function_is_checkpointed(x))
        raise ManagerException("Initial condition not found")

    def add_equation(self, eq, annotate=None, tlm=None, tlm_skip=None):
        """
        Process the provided equation, annotating and / or deriving (and
        solving) tangent-linear equations as required. Assumes that the
        equation has already been solved, and that the initial condition for
        eq.X() has been recorded on the adjoint tape if necessary.

        annotate (default self.annotation_enabled()):
            Whether to annotate the equation on the adjoint tape, storing data
            for checkpointing as required.
        tlm (default self.tlm_enabled()):
            Whether to derive (and solve) associated tangent-linear equations.
        tlm_skip (default None):
            Used for the derivation of higher order tangent-linear equations.
        """

        self.drop_references()

        if annotate is None:
            annotate = self.annotation_enabled()
        if annotate:
            if self._annotation_state == "initial":
                self._annotation_state = "annotating"
            elif self._annotation_state == "stopped_initial":
                self._annotation_state = "stopped_annotating"
            elif self._annotation_state == "final":
                raise ManagerException("Cannot add equations after finalization")  # noqa: E501

            if self._cp_method in ["none", "memory"] \
                    and not self._cp_parameters["drop_references"]:
                eq_id = eq.id()
                if eq_id not in self._eqs:
                    self._eqs[eq_id] = eq
                self._block.append(eq)
            else:
                self._add_equation_finalizes(eq)
                eq_alias = WeakAlias(eq)
                eq_id = eq.id()
                if eq_id not in self._eqs:
                    self._eqs[eq_id] = eq_alias
                self._block.append(eq_alias)
            self._cp.add_equation(
                (len(self._blocks), len(self._block) - 1), eq)

        if tlm is None:
            tlm = self.tlm_enabled()
        if tlm:
            if self._tlm_state == "final":
                raise ManagerException("Cannot add tangent-linear equations after finalization")  # noqa: E501

            depth = 0 if tlm_skip is None else tlm_skip[1]
            for i, (M, dM) in enumerate(reversed(self._tlm)):
                if tlm_skip is not None and i >= tlm_skip[0]:
                    break
                tlm_map, max_depth = self._tlm[(M, dM)]
                tlm_eq = self._tangent_linear(eq, M, dM, tlm_map)
                if tlm_eq is not None:
                    tlm_eq.solve(
                        manager=self, annotate=annotate, tlm=True,
                        _tlm_skip=([i + 1, depth + 1] if max_depth - depth > 1
                                   else [i, 0]))

    @gc_disabled
    def _tangent_linear(self, eq, M, dM, tlm_map):
        eq_id = eq.id()
        X = eq.X()
        if len(set(X).intersection(set(M))) > 0:
            raise ManagerException("Invalid tangent-linear parameter")
        if len(set(X).intersection(set(dM))) > 0:
            raise ManagerException("Invalid tangent-linear direction")

        eq_tlm_eqs = self._tlm_eqs.get(eq_id, None)
        if eq_tlm_eqs is None:
            eq_tlm_eqs = {}
            self._tlm_eqs[eq_id] = eq_tlm_eqs

        tlm_eq = eq_tlm_eqs.get((M, dM), None)
        if tlm_eq is None:
            for dep in eq.dependencies():
                if dep in M or dep in tlm_map:
                    tlm_eq = eq.tangent_linear(M, dM, tlm_map)
                    if tlm_eq is None:
                        tlm_eq = NullSolver([tlm_map[x] for x in X])
                    eq_tlm_eqs[(M, dM)] = tlm_eq
                    break

        return tlm_eq

    @gc_disabled
    def _add_equation_finalizes(self, eq):
        for referrer in eq.referrers():
            assert not isinstance(referrer, WeakAlias)
            referrer_id = referrer.id()
            if referrer_id not in self._finalizes:
                @gc_disabled
                def finalize_callback(self_ref, referrer_alias, referrer_id):
                    self = self_ref()
                    if self is not None:
                        self._to_drop_references.append(referrer_alias)
                        del self._finalizes[referrer_id]
                        if referrer_id in self._tlm_eqs:
                            assert isinstance(referrer_alias, Equation)
                            del self._tlm_eqs[referrer_id]
                finalize = weakref.finalize(
                    referrer, finalize_callback,
                    weakref.ref(self), WeakAlias(referrer), referrer_id)
                finalize.atexit = False
                self._finalizes[referrer_id] = finalize

    @gc_disabled
    def drop_references(self):
        while len(self._to_drop_references) > 0:
            referrer = self._to_drop_references.pop()
            referrer._drop_references()

    def _checkpoint_space_id(self, fn):
        space = function_space(fn)
        id = space_id(space)
        if id not in self._cp_spaces:
            self._cp_spaces[id] = space
        return id

    def _save_memory_checkpoint(self, cp, n):
        if n in self._cp_memory or n in self._cp_disk:
            raise ManagerException("Duplicate checkpoint")

        self._cp_memory[n] = self._cp.initial_conditions(cp=True, refs=False,
                                                         copy=False)

    def _load_memory_checkpoint(self, storage, n, delete=False):
        if delete:
            storage.update(self._cp_memory.pop(n), copy=False)
        else:
            storage.update(self._cp_memory[n], copy=True)

    def _save_disk_checkpoint(self, cp, n):
        if n in self._cp_memory or n in self._cp_disk:
            raise ManagerException("Duplicate checkpoint")

        cp_path = self._cp_parameters["path"]
        cp_format = self._cp_parameters["format"]

        cp = self._cp.initial_conditions(cp=True, refs=False, copy=False)

        if cp_format == "pickle":
            cp_filename = os.path.join(
                cp_path,
                "checkpoint_%i_%i_%i_%i.pickle" % (self._id,
                                                   n,
                                                   self._comm_py2f,
                                                   self._comm.rank))
            self._cp_disk[n] = cp_filename
            h = open(cp_filename, "wb")

            pickle.dump({key: (self._checkpoint_space_id(F),
                               function_get_values(F))
                         for key, F in cp.items()},
                        h, protocol=pickle.HIGHEST_PROTOCOL)

            h.close()
        elif cp_format == "hdf5":
            cp_filename = os.path.join(
                cp_path,
                "checkpoint_%i_%i_%i.hdf5" % (self._id,
                                              n,
                                              self._comm_py2f))
            self._cp_disk[n] = cp_filename
            import h5py
            if self._comm.size > 1:
                h = h5py.File(cp_filename, "w", driver="mpio", comm=self._comm)
            else:
                h = h5py.File(cp_filename, "w")

            h.create_group("/ics")
            for i, (key, F) in enumerate(cp.items()):
                g = h.create_group(f"/ics/{i:d}")

                values = function_get_values(F)
                d = g.create_dataset("value", shape=(function_global_size(F),),
                                     dtype=values.dtype)
                d[function_local_indices(F)] = values
                del values

                d = g.create_dataset("space_id", shape=(self._comm.size,),
                                     dtype=np.int64)
                d[self._comm.rank] = self._checkpoint_space_id(F)

                d = g.create_dataset("key", shape=(self._comm.size,),
                                     dtype=np.int64)
                d[self._comm.rank] = key

            h.close()
        else:
            raise ManagerException(f"Unrecognized checkpointing format: {cp_format:s}")  # noqa: E501

    def _load_disk_checkpoint(self, storage, n, delete=False):
        cp_format = self._cp_parameters["format"]

        if cp_format == "pickle":
            if delete:
                cp_filename = self._cp_disk.pop(n)
            else:
                cp_filename = self._cp_disk[n]
            h = open(cp_filename, "rb")
            cp = pickle.load(h)
            h.close()
            if delete:
                if self._comm.rank == 0:
                    os.remove(cp_filename)
                self._comm.barrier()

            for key in tuple(cp.keys()):
                space_id, values = cp.pop(key)
                if key in storage:
                    F = space_new(self._cp_spaces[space_id])
                    function_set_values(F, values)
                    storage[key] = F
                del space_id, values
        elif cp_format == "hdf5":
            if delete:
                cp_filename = self._cp_disk.pop(n)
            else:
                cp_filename = self._cp_disk[n]
            import h5py
            if self._comm.size > 1:
                h = h5py.File(cp_filename, "r", driver="mpio", comm=self._comm)
            else:
                h = h5py.File(cp_filename, "r")

            for name, g in h["/ics"].items():
                d = g["key"]
                key = int(d[self._comm.rank])
                if key in storage:
                    d = g["space_id"]
                    F = space_new(self._cp_spaces[d[self._comm.rank]])
                    d = g["value"]
                    function_set_values(F, d[function_local_indices(F)])
                    storage[key] = F
                del g, d

            h.close()
            if delete:
                if self._comm.rank == 0:
                    os.remove(cp_filename)
                self._comm.barrier()
        else:
            raise ManagerException(f"Unrecognized checkpointing format: {cp_format:s}")  # noqa: E501

    def _checkpoint(self, final=False):
        if self._cp_method in ["none", "memory"]:
            pass
        elif self._cp_method == "periodic_disk":
            self._periodic_disk_checkpoint(final=final)
        elif self._cp_method == "multistage":
            self._multistage_checkpoint()
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{self._cp_method:s}")

    def _periodic_disk_checkpoint(self, final=False):
        cp_period = self._cp_parameters["period"]

        n = len(self._blocks) - 1
        if final or n % cp_period == cp_period - 1:
            self._save_disk_checkpoint(self._cp,
                                       n=(n // cp_period) * cp_period)
            self._cp.clear()
            self._cp.configure(store_ics=True,
                               store_data=False)

    def _save_multistage_checkpoint(self):
        logger = logging.getLogger("tlm_adjoint.multistage_checkpointing")

        deferred_snapshot = self._cp_manager.deferred_snapshot()
        if deferred_snapshot is not None:
            snapshot_n, snapshot_storage = deferred_snapshot
            if snapshot_storage == "disk":
                if self._cp_manager.r() == 0:
                    logger.debug(f"forward: save snapshot at {snapshot_n:d} on disk")  # noqa: E501
                else:
                    logger.debug(f"reverse: save snapshot at {snapshot_n:d} on disk")  # noqa: E501
                self._save_disk_checkpoint(self._cp, snapshot_n)
            else:
                if self._cp_manager.r() == 0:
                    logger.debug(f"forward: save snapshot at {snapshot_n:d} in RAM")  # noqa: E501
                else:
                    logger.debug(f"reverse: save snapshot at {snapshot_n:d} in RAM")  # noqa: E501
                self._save_memory_checkpoint(self._cp, snapshot_n)

    def _multistage_checkpoint(self):
        logger = logging.getLogger("tlm_adjoint.multistage_checkpointing")

        n = len(self._blocks)
        if n < self._cp_manager.n():
            return
        elif n == self._cp_manager.max_n():
            return
        elif n > self._cp_manager.max_n():
            raise ManagerException("Unexpected number of blocks")

        self._save_multistage_checkpoint()
        self._cp.clear()
        if n == self._cp_manager.max_n() - 1:
            logger.debug("forward: configuring storage for reverse")
            self._cp.configure(store_ics=False,
                               store_data=True)
        else:
            logger.debug("forward: configuring storage for snapshot")
            self._cp.configure(store_ics=True,
                               store_data=False)
            logger.debug(f"forward: deferred snapshot at {self._cp_manager.n():d}")  # noqa: E501
            self._cp_manager.snapshot()
        self._cp_manager.forward()
        logger.debug(f"forward: forward advance to {self._cp_manager.n():d}")

    def _restore_checkpoint(self, n):
        if self._cp_method == "none":
            raise ManagerException("Cannot restore from checkpoint with "
                                   "checkpointing method 'none'")
        elif self._cp_method == "memory":
            pass
        elif self._cp_method == "periodic_disk":
            if n not in self._cp_manager:
                cp_period = self._cp_parameters["period"]

                N0 = (n // cp_period) * cp_period
                N1 = min(((n // cp_period) + 1) * cp_period, len(self._blocks))

                self._cp.clear()
                self._cp_manager.clear()
                storage = ReplayStorage(self._blocks, N0, N1)
                storage.update(self._cp.initial_conditions(cp=False, refs=True,
                                                           copy=False),
                               copy=False)
                self._load_disk_checkpoint(storage, N0, delete=False)

                for n1 in range(N0, N1):
                    self._cp.configure(store_ics=n1 == 0,
                                       store_data=True)

                    for i, eq in enumerate(self._blocks[n1]):
                        eq_deps = eq.dependencies()

                        X = tuple(storage[eq_x] for eq_x in eq.X())
                        deps = tuple(storage[eq_dep] for eq_dep in eq_deps)

                        for eq_dep in eq.initial_condition_dependencies():
                            self._cp.add_initial_condition(
                                eq_dep, value=storage[eq_dep])
                        eq.forward(X, deps=deps)
                        self._cp.add_equation((n1, i), eq, deps=deps)

                        storage_state = storage.pop()
                        assert storage_state == (n1, i)

                    self._cp_manager.add(n1)
                assert len(storage) == 0
        elif self._cp_method == "multistage":
            logger = logging.getLogger("tlm_adjoint.multistage_checkpointing")

            if n == 0 and self._cp_manager.max_n() - self._cp_manager.r() == 0:
                return
            if n != self._cp_manager.max_n() - self._cp_manager.r() - 1:
                raise ManagerException("Invalid checkpointing state")
            if n == self._cp_manager.max_n() - 1:
                logger.debug(f"reverse: adjoint step back to {n:d}")
                assert n + 1 == self._cp_manager.n()
                assert self._cp_manager.r() == 0
                self._cp_manager.reverse()
                assert n == self._cp_manager.max_n() - self._cp_manager.r()
                return

            (snapshot_n,
             snapshot_storage,
             snapshot_delete) = self._cp_manager.load_snapshot()
            self._cp.clear()
            storage = ReplayStorage(self._blocks, snapshot_n, n + 1)
            storage.update(self._cp.initial_conditions(cp=False, refs=True,
                                                       copy=False),
                           copy=False)
            if snapshot_storage == "disk":
                logger.debug(f'reverse: load snapshot at {snapshot_n:d} from disk and {"delete" if snapshot_delete else "keep":s}')  # noqa: E501
                self._load_disk_checkpoint(storage, snapshot_n,
                                           delete=snapshot_delete)
            else:
                logger.debug(f'reverse: load snapshot at {snapshot_n:d} from RAM and {"delete" if snapshot_delete else "keep":s}')  # noqa: E501
                self._load_memory_checkpoint(storage, snapshot_n,
                                             delete=snapshot_delete)

            if snapshot_n < n:
                logger.debug("reverse: no storage")
                self._cp.configure(store_ics=False,
                                   store_data=False)

            snapshot_n_0 = snapshot_n
            while True:
                if snapshot_n == n:
                    logger.debug("reverse: configuring storage for reverse")
                    self._cp.configure(store_ics=n == 0,
                                       store_data=True)
                elif snapshot_n > snapshot_n_0:
                    logger.debug("reverse: configuring storage for snapshot")
                    self._cp .configure(store_ics=True,
                                        store_data=False)
                    logger.debug(f"reverse: deferred snapshot at {self._cp_manager.n():d}")  # noqa: E501
                    self._cp_manager.snapshot()
                self._cp_manager.forward()
                logger.debug(f"reverse: forward advance to {self._cp_manager.n():d}")  # noqa: E501
                for n1 in range(snapshot_n, self._cp_manager.n()):
                    for i, eq in enumerate(self._blocks[n1]):
                        eq_deps = eq.dependencies()

                        X = tuple(storage[eq_x] for eq_x in eq.X())
                        deps = tuple(storage[eq_dep] for eq_dep in eq_deps)

                        for eq_dep in eq.initial_condition_dependencies():
                            self._cp.add_initial_condition(
                                eq_dep, value=storage[eq_dep])
                        eq.forward(X, deps=deps)
                        self._cp.add_equation((n1, i), eq, deps=deps)

                        storage_state = storage.pop()
                        assert storage_state == (n1, i)
                snapshot_n = self._cp_manager.n()
                if snapshot_n > n:
                    break
                self._save_multistage_checkpoint()
                self._cp.clear()
            assert len(storage) == 0

            logger.debug(f"reverse: adjoint step back to {n:d}")
            assert n + 1 == self._cp_manager.n()
            self._cp_manager.reverse()
            assert n == self._cp_manager.max_n() - self._cp_manager.r()
        else:
            raise ManagerException(f"Unrecognized checkpointing method: "
                                   f"{self._cp_method:s}")

    def new_block(self):
        """
        End the current block equation and begin a new block. Ignored if
        "multistage" checkpointing is used and the final block has been
        reached.
        """

        self.drop_references()

        if self._annotation_state in ["stopped_initial",
                                      "stopped_annotating",
                                      "final"]:
            return
        elif self._cp_method == "multistage" \
                and len(self._blocks) == self._cp_parameters["blocks"] - 1:
            # Wait for the finalize
            warnings.warn(
                "Attempting to end the final block without finalising -- "
                "ignored", RuntimeWarning, stacklevel=2)
            return

        self._blocks.append(self._block)
        self._block = []
        self._checkpoint(final=False)

    def finalize(self):
        """
        End the final block equation.
        """

        self.drop_references()

        if self._annotation_state == "final":
            return
        self._annotation_state = "final"
        self._tlm_state = "final"

        self._blocks.append(self._block)
        self._block = []
        if self._cp_method == "multistage" \
                and len(self._blocks) < self._cp_parameters["blocks"]:
            warnings.warn(
                "Insufficient number of blocks -- empty blocks added",
                RuntimeWarning, stacklevel=2)
            while len(self._blocks) < self._cp_parameters["blocks"]:
                self._checkpoint(final=False)
                self._blocks.append([])
        self._checkpoint(final=True)

    def reset_adjoint(self, _warning=True):
        """
        Call the reset_adjoint methods of all annotated Equation objects.
        """

        if _warning:
            warnings.warn("EquationManager.reset_adjoint method is deprecated",
                          DeprecationWarning, stacklevel=2)
        for eq in self._eqs.values():
            eq.reset_adjoint()

    def compute_gradient(self, Js, M, callback=None, prune_forward=True,
                         prune_adjoint=True, adj_ics=None):
        """
        Compute the derivative of one or more functionals with respect to one
        or more control parameters by running adjoint models. Finalizes the
        manager.

        Arguments:

        Js        A Functional or function, or a sequence of these, defining
                  the functionals.
        M         A function, or a sequence of functions, defining the control
                  parameters.
        callback  (Optional) Callable of the form
                      def callback(J_i, n, i, eq, adj_X):
                  where adj_X is None, a function, or a sequence of functions,
                  corresponding to the adjoint solution for the equation eq,
                  which is equation i in block n for the J_i th Functional.
        prune_forward  (Optional) Whether forward traversal graph pruning
                       should be applied.
        prune_adjoint  (Optional) Whether reverse traversal graph pruning
                       should be applied.
        adj_ics   (Optional) Map, or a sequence of maps, from forward functions
                  or function IDs to adjoint initial conditions.
        """

        if not isinstance(M, Sequence):
            if not isinstance(Js, Sequence):
                if adj_ics is not None:
                    adj_ics = [adj_ics]
                ((dJ,),) = self.compute_gradient([Js], [M], callback=callback,
                                                 prune_forward=prune_forward,
                                                 prune_adjoint=prune_adjoint,
                                                 adj_ics=adj_ics)
                return dJ
            else:
                dJs = self.compute_gradient(Js, [M], callback=callback,
                                            prune_forward=prune_forward,
                                            prune_adjoint=prune_adjoint,
                                            adj_ics=adj_ics)
                return tuple(dJ for (dJ,) in dJs)
        elif not isinstance(Js, Sequence):
            if adj_ics is not None:
                adj_ics = [adj_ics]
            dJ, = self.compute_gradient([Js], M, callback=callback,
                                        prune_forward=prune_forward,
                                        prune_adjoint=prune_adjoint,
                                        adj_ics=adj_ics)
            return dJ

        gc.collect()
        self.finalize()
        self.reset_adjoint(_warning=False)

        # Functionals
        Js = tuple(Functional(fn=J) if is_function(J) else J for J in Js)

        # Controls
        M = tuple(M)

        # Derivatives
        dJ = [None for J in Js]

        # Add two additional blocks, one at the start and one at the end of the
        # forward:
        #   Control block   :  Represents the equation "controls = inputs"
        #   Functional block:  Represents the equations "outputs = functionals"
        blocks = ([[ControlsMarker(M)]]
                  + self._blocks
                  + [[FunctionalMarker(J) for J in Js]])
        J_markers = tuple(eq.x() for eq in blocks[-1])

        # Adjoint equation right-hand-sides
        Bs = tuple(AdjointModelRHS(blocks) for J in Js)
        # Adjoint initial condition
        for J_i in range(len(Js)):
            function_assign(Bs[J_i][-1][J_i].b(), 1.0)

        # Transpose dependency graph
        transpose_deps = DependencyGraphTranspose(blocks, M, J_markers,
                                                  prune_forward=prune_forward,
                                                  prune_adjoint=prune_adjoint)

        # Adjoint variables
        adj_Xs = tuple({} for J in Js)
        if adj_ics is not None:
            for J_i, J_marker in enumerate(J_markers):
                for x_id, adj_x in adj_ics[J_i].items():
                    if not isinstance(x_id, int):
                        x_id = function_id(x_id)
                    if transpose_deps.has_adj_ic(J_marker, x_id):
                        adj_Xs[J_i][x_id] = function_copy(adj_x)

        # Reverse (blocks)
        for n in range(len(blocks) - 1, -1, -1):
            cp_n = n - 1  # Forward model block, ignoring the control block
            cp_block = cp_n >= 0 and cp_n < len(self._blocks)
            if cp_block:
                # Load/restore forward model data
                self._restore_checkpoint(cp_n)

            # Reverse (equations in block n)
            for i in range(len(blocks[n]) - 1, -1, -1):
                eq = blocks[n][i]
                # Non-linear dependency data
                nl_deps = self._cp[(cp_n, i)] if cp_block else ()

                for J_i, (J, J_marker) in enumerate(zip(Js, J_markers)):
                    # Adjoint model right-hand-sides
                    B = Bs[J_i]
                    # Adjoint right-hand-side associated with this equation
                    B_state, eq_B = B.pop()
                    assert B_state == (n, i)

                    if transpose_deps.is_active(J_marker, n, i):
                        # Solve adjoint equation, add terms to adjoint
                        # equations
                        if len(eq.adjoint_initial_condition_dependencies()) == 0:  # noqa: E501
                            adj_X = None
                        else:
                            adj_X = []
                            for x in eq.X():
                                adj_x = adj_Xs[J_i].pop(function_id(x), None)
                                if adj_x is None:
                                    adj_x = function_new(x)
                                adj_X.append(adj_x)

                        eq_B = eq_B.B()

                        eq_dep_Bs = {}
                        for j, dep in enumerate(eq.dependencies()):
                            if (n, i, j) in transpose_deps:
                                p, k, m = transpose_deps[(n, i, j)]
                                if transpose_deps.is_active(J_marker, p, k):
                                    eq_dep_Bs[j] = Bs[J_i][p][k][m]

                        adj_X = eq.adjoint(J, adj_X, nl_deps, eq_B, eq_dep_Bs)

                        for m, (x, adj_x) in enumerate(zip(eq.X(), adj_X)):
                            if transpose_deps.is_stored_adj_ic(J_marker,
                                                               n, i, m):
                                adj_Xs[J_i][function_id(x)] \
                                    = function_copy(adj_x)
                    else:
                        # Adjoint solution has no effect on sensitivity
                        adj_X = None

                    if callback is not None and cp_block:
                        if adj_X is None or len(adj_X) > 1:
                            callback(J_i, cp_n, i, eq, adj_X)
                        else:
                            callback(J_i, cp_n, i, eq, adj_X[0])

                    if n == 0 and i == 0:
                        # A requested derivative
                        if adj_X is None:
                            dJ[J_i] = tuple(function_new(m) for m in M)
                        else:
                            dJ[J_i] = tuple(function_copy(adj_x)
                                            for adj_x in adj_X)

            if n > 0:
                # Force finalization of right-hand-sides in the control block
                for B in Bs:
                    B[0].finalize()

        for B in Bs:
            assert B.is_empty()
        for J_i in range(len(Js)):
            assert len(adj_Xs[J_i]) == 0

        if self._cp_method == "multistage":
            self._cp.clear(clear_cp=False, clear_data=True, clear_refs=False)

        return tuple(dJ)

    def find_initial_condition(self, x):
        """
        Find the initial condition function associated with the given function
        name.
        """

        warnings.warn("EquationManager.find_initial_condition method is "
                      "deprecated",
                      DeprecationWarning, stacklevel=2)

        for block in self._blocks + [self._block]:
            for eq in block:
                for dep in eq.dependencies():
                    if function_name(dep) == x:
                        return dep
        raise ManagerException("Initial condition not found")


set_manager(EquationManager())
