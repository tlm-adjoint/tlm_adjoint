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

from .interface import DEFAULT_COMM, check_space_types, function_assign, \
    function_copy, function_id, function_is_replacement, function_name, \
    function_new_tangent_linear, is_function

from .alias import Alias, WeakAlias, gc_disabled
from .binomial_checkpointing import MultistageCheckpointingManager
from .checkpointing import CheckpointStorage, HDF5Checkpoints, \
    MemoryCheckpointingManager, NoneCheckpointingManager, \
    PeriodicDiskCheckpointingManager, PickleCheckpoints, ReplayStorage
from .equations import AdjointModelRHS, ControlsMarker, Equation, \
    FunctionalMarker, NullSolver
from .functional import Functional
from .manager import manager as _manager, restore_manager, set_manager

from collections import OrderedDict
from collections.abc import Sequence
import copy
import gc
import logging
import numpy as np
import os
import warnings
import weakref

__all__ = \
    [
        "Control",
        "EquationManager"
    ]


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


class TangentLinearMap:
    """
    A map from forward to tangent-linear variables.
    """

    def __init__(self, name_suffix=" (tangent-linear)"):
        self._name_suffix = name_suffix

    @gc_disabled
    def __contains__(self, x):
        if hasattr(x, "_tlm_adjoint__tangent_linears"):
            return self in x._tlm_adjoint__tangent_linears
        else:
            return False

    @gc_disabled
    def __getitem__(self, x):
        if not is_function(x):
            raise TypeError("x must be a function")

        if not hasattr(x, "_tlm_adjoint__tangent_linears"):
            x._tlm_adjoint__tangent_linears = weakref.WeakKeyDictionary()
        if self not in x._tlm_adjoint__tangent_linears:
            x._tlm_adjoint__tangent_linears[self] = \
                function_new_tangent_linear(
                    x, name=f"{function_name(x):s}{self._name_suffix:s}")

        return x._tlm_adjoint__tangent_linears[self]


class DependencyGraphTranspose:
    def __init__(self, Js, M, blocks,
                 prune_forward=True, prune_adjoint=True,
                 adj_cache=None):
        if isinstance(blocks, Sequence):
            # Sequence
            blocks_n = tuple(range(len(blocks)))
        else:
            # Mapping
            blocks_n = tuple(sorted(blocks.keys()))

        # Transpose dependency graph
        last_eq = {}
        transpose_deps = {n: tuple([None for dep in eq.dependencies()]
                                   for eq in blocks[n])
                          for n in blocks_n}
        for n in blocks_n:
            block = blocks[n]
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
            for p in reversed(blocks_n):
                block = blocks[p]
                for k in range(len(block) - 1, -1, -1):
                    eq = block[k]
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
                        last_eq[x_id] = (p, k, m)
            del last_eq

            # Pruning, forward traversal
            active_M = {function_id(dep) for dep in M}
            active_forward = {n: np.full(len(blocks[n]), False, dtype=bool)
                              for n in blocks_n}
            for n in blocks_n:
                block = blocks[n]
                for i, eq in enumerate(block):
                    if len(active_M) > 0:
                        X_ids = {function_id(x) for x in eq.X()}
                        if not X_ids.isdisjoint(active_M):
                            active_M.difference_update(X_ids)
                            active_forward[n][i] = True
                    if not active_forward[n][i]:
                        for j, dep in enumerate(eq.dependencies()):
                            if transpose_deps_ics[n][i][j] is not None:
                                p, k, m = transpose_deps_ics[n][i][j]
                                if active_forward[p][k]:
                                    active_forward[n][i] = True
                                    break
        else:
            active_forward = {n: np.full(len(blocks[n]), True, dtype=bool)
                              for n in blocks_n}

        active = {function_id(J): copy.deepcopy(active_forward) for J in Js}

        if prune_adjoint:
            # Pruning, reverse traversal
            for J_id in active:
                active_J = True
                active_adjoint = {n: np.full(len(blocks[n]), False, dtype=bool)
                                  for n in blocks_n}
                for n in reversed(blocks_n):
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

        solved = copy.deepcopy(active)
        if adj_cache is not None:
            for J_i, J in enumerate(Js):
                J_id = function_id(J)
                for n in blocks_n:
                    block = blocks[n]
                    for i, eq in enumerate(block):
                        if adj_cache.has_cached(J_i, n, i):
                            solved[J_id][n][i] = False

        stored_adj_ics = {function_id(J): {n: tuple(np.full(len(eq.X()), False, dtype=bool)  # noqa: E501
                                                    for eq in blocks[n])
                                           for n in blocks_n} for J in Js}
        adj_ics = {function_id(J): {} for J in Js}
        for J_id in stored_adj_ics:
            stored = {}
            for n in blocks_n:
                block = blocks[n]
                for i, eq in enumerate(block):
                    if active[J_id][n][i]:
                        for m, x in enumerate(eq.X()):
                            stored_adj_ics[J_id][n][i][m] = \
                                stored.get(function_id(x), False)

                    adj_ic_ids = {function_id(dep)
                                  for dep in eq.adjoint_initial_condition_dependencies()}  # noqa: E501
                    for x in eq.X():
                        x_id = function_id(x)
                        store_x = solved[J_id][n][i] and x_id in adj_ic_ids
                        stored[x_id] = store_x
                        if x_id not in adj_ics[J_id]:
                            adj_ics[J_id][x_id] = store_x

        self._transpose_deps = transpose_deps
        self._active = active
        self._solved = solved
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

    def any_is_active(self, n, i):
        for J_id in self._active:
            if self._active[J_id][n][i]:
                return True
        return False

    def is_solved(self, J, n, i):
        if isinstance(J, int):
            J_id = J
        else:
            J_id = function_id(J)
        return self._solved[J_id][n][i]

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

    def adj_Bs(self, J, n, i, eq, B):
        if isinstance(J, int):
            J_id = J
        else:
            J_id = function_id(J)

        dep_Bs = {}
        for j, dep in enumerate(eq.dependencies()):
            if (n, i, j) in self:
                p, k, m = self[(n, i, j)]
                if self.is_solved(J_id, p, k):
                    dep_Bs[j] = B[p][k][m]

        return dep_Bs


class AdjointCache:
    def __init__(self):
        self._keys = set()
        self._cache = {}

    def register(self, J_i, n, i):
        self._keys.add((J_i, n, i))

    def has_cached(self, J_i, n, i):
        return (J_i, n, i) in self._cache

    def get_cached(self, J_i, n, i, copy=False):
        adj_X = self._cache[(J_i, n, i)]
        if copy:
            adj_X = tuple(function_copy(adj_x) for adj_x in adj_X)
        return adj_X

    def cache(self, J_i, n, i, adj_X, copy=True, replace=False):
        if (J_i, n, i) in self._keys:
            if replace or (J_i, n, i) not in self._cache:
                if copy:
                    adj_X = tuple(function_copy(adj_x) for adj_x in adj_X)
                else:
                    adj_X = tuple(adj_X)
                self._cache[(J_i, n, i)] = adj_X


class EquationManager:
    _id_counter = [0]

    def __init__(self, comm=None, cp_method="memory", cp_parameters=None):
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
            comm = DEFAULT_COMM
        if cp_parameters is None:
            cp_parameters = {}

        comm = comm.Dup()

        self._comm = comm
        self._to_drop_references = []
        self._finalizes = {}

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

        if self._comm.rank == 0:
            id = self._id_counter[0]
            self._id_counter[0] += 1
        else:
            id = None
        self._id = self._comm.bcast(id, root=0)

        self.reset(cp_method=cp_method, cp_parameters=cp_parameters)

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
        if callable(self._cp_method):
            info("  Method: custom")
        else:
            info(f"  Method: {self._cp_method:s}")

    def new(self, cp_method=None, cp_parameters=None):
        """
        Return a new equation manager sharing the communicator of this
        equation manager. Optionally a new checkpointing configuration can be
        provided.
        """

        if cp_method is None:
            if cp_parameters is not None:
                raise TypeError("cp_parameters can only be supplied if "
                                "cp_method is supplied")
            cp_method = self._cp_method
            cp_parameters = self._cp_parameters
        elif cp_parameters is None:
            raise TypeError("cp_parameters must be supplied if cp_method is "
                            "supplied")

        return EquationManager(comm=self._comm, cp_method=cp_method,
                               cp_parameters=cp_parameters)

    @gc_disabled
    def reset(self, cp_method=None, cp_parameters=None):
        """
        Reset the equation manager. Optionally a new checkpointing
        configuration can be provided.
        """

        if cp_method is None:
            if cp_parameters is not None:
                raise TypeError("cp_parameters can only be supplied if "
                                "cp_method is supplied")
            cp_method = self._cp_method
            cp_parameters = self._cp_parameters
        elif cp_parameters is None:
            raise TypeError("cp_parameters must be supplied if cp_method is "
                            "supplied")

        self.drop_references()

        self._annotation_state = "initial"
        self._tlm_state = "initial"
        self._eqs = {}
        self._blocks = []
        self._block = []

        self._tlm = OrderedDict()
        self._tlm_eqs = {}

        self.configure_checkpointing(cp_method, cp_parameters=cp_parameters)

    def configure_checkpointing(self, cp_method, cp_parameters):
        """
        Provide a new checkpointing configuration.
        """

        if self._annotation_state not in ["initial", "stopped_initial"]:
            raise RuntimeError("Cannot configure checkpointing after "
                               "annotation has started, or after finalization")

        cp_parameters = copy.copy(cp_parameters)

        if not callable(cp_method) and cp_method in ["none", "memory"]:
            if "replace" in cp_parameters:
                warnings.warn("'replace' cp_parameters key is deprecated",
                              DeprecationWarning, stacklevel=2)
                if "drop_references" in cp_parameters:
                    if cp_parameters["replace"] != cp_parameters["drop_references"]:  # noqa: E501
                        raise ValueError("Conflicting cp_parameters values")
                alias_eqs = cp_parameters["replace"]
            else:
                alias_eqs = cp_parameters.get("drop_references", False)
        else:
            alias_eqs = True

        if callable(cp_method):
            cp_manager_kwargs = copy.copy(cp_parameters)
            if "path" in cp_manager_kwargs:
                del cp_manager_kwargs["path"]
            if "format" in cp_manager_kwargs:
                del cp_manager_kwargs["format"]
            cp_manager = cp_method(**cp_manager_kwargs)
        elif cp_method == "none":
            cp_manager = NoneCheckpointingManager()
        elif cp_method == "memory":
            cp_manager = MemoryCheckpointingManager()
        elif cp_method == "periodic_disk":
            cp_manager = PeriodicDiskCheckpointingManager(
                cp_parameters["period"],
                keep_block_0_ics=True)
        elif cp_method == "multistage":
            cp_manager = MultistageCheckpointingManager(
                cp_parameters["blocks"],
                cp_parameters.get("snaps_in_ram", 0),
                cp_parameters.get("snaps_on_disk", 0),
                keep_block_0_ics=True,
                trajectory="maximum")
        else:
            raise ValueError(f"Unrecognized checkpointing method: "
                             f"{cp_method:s}")

        if cp_manager.uses_disk_storage():
            cp_path = cp_parameters.get("path", "checkpoints~")
            cp_format = cp_parameters.get("format", "hdf5")

            self._comm.barrier()
            if self._comm.rank == 0:
                if not os.path.exists(cp_path):
                    os.makedirs(cp_path)
            self._comm.barrier()

            if cp_format == "pickle":
                cp_disk = PickleCheckpoints(
                    os.path.join(cp_path, f"checkpoint_{self._id:d}_"),
                    comm=self._comm)
            elif cp_format == "hdf5":
                cp_disk = HDF5Checkpoints(
                    os.path.join(cp_path, f"checkpoint_{self._id:d}_"),
                    comm=self._comm)
            else:
                raise ValueError(f"Unrecognized checkpointing format: "
                                 f"{cp_format:s}")
        else:
            cp_path = None
            cp_disk = None

        self._cp_method = cp_method
        self._cp_parameters = cp_parameters
        self._alias_eqs = alias_eqs
        self._cp_manager = cp_manager
        self._cp_memory = {}
        self._cp_path = cp_path
        self._cp_disk = cp_disk

        self._cp = CheckpointStorage(store_ics=False,
                                     store_data=False)
        assert len(self._blocks) == 0
        self._checkpoint()

    def add_tlm(self, M, dM, max_depth=1):
        """
        Add a tangent-linear model computing derivatives with respect to the
        control defined by M in the direction defined by dM.
        """

        if self._tlm_state == "final":
            raise RuntimeError("Cannot add a tangent-linear model after "
                               "finalization")

        if is_function(M):
            M = (M,)
        else:
            M = tuple(M)
        if is_function(dM):
            dM = (dM,)
        else:
            dM = tuple(dM)

        if len(M) != len(dM):
            raise ValueError("Invalid tangent-linear model")
        if (M, dM) in self._tlm:
            raise RuntimeError("Duplicate tangent-linear model")
        for m, dm in zip(M, dM):
            check_space_types(m, dm)

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
            raise KeyError("Tangent-linear not found")

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
                raise RuntimeError("Cannot add initial conditions after "
                                   "finalization")

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
                return self._cp.initial_condition(x, copy=True)
        raise KeyError("Initial condition not found")

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
                raise RuntimeError("Cannot add equations after finalization")

            if self._alias_eqs:
                self._add_equation_finalizes(eq)
                eq_alias = WeakAlias(eq)
                eq_id = eq.id()
                if eq_id not in self._eqs:
                    self._eqs[eq_id] = eq_alias
                self._block.append(eq_alias)
            else:
                eq_id = eq.id()
                if eq_id not in self._eqs:
                    self._eqs[eq_id] = eq
                self._block.append(eq)
            self._cp.add_equation(
                len(self._blocks), len(self._block) - 1, eq)

        if tlm is None:
            tlm = self.tlm_enabled()
        if tlm:
            if self._tlm_state == "final":
                raise RuntimeError("Cannot add tangent-linear equations after "
                                   "finalization")

            depth = 0 if tlm_skip is None else tlm_skip[1]
            for i, (M, dM) in enumerate(reversed(self._tlm)):
                if tlm_skip is not None and i >= tlm_skip[0]:
                    break
                tlm_eq = self._tangent_linear(eq, M, dM)
                if tlm_eq is not None:
                    tlm_map, max_depth = self._tlm[(M, dM)]
                    tlm_eq.solve(
                        manager=self, annotate=annotate, tlm=True,
                        _tlm_skip=([i + 1, depth + 1] if max_depth - depth > 1
                                   else [i, 0]))

    def _tangent_linear(self, eq, M, dM):
        if is_function(M):
            M = (M,)
        else:
            M = tuple(M)
        if is_function(dM):
            dM = (dM,)
        else:
            dM = tuple(dM)

        if (M, dM) not in self._tlm:
            raise KeyError("Missing tangent-linear model")
        tlm_map, max_depth = self._tlm[(M, dM)]

        eq_id = eq.id()
        X = eq.X()
        if len(set(X).intersection(set(M))) > 0:
            raise ValueError("Invalid tangent-linear parameter")
        if len(set(X).intersection(set(dM))) > 0:
            raise ValueError("Invalid tangent-linear direction")

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
            if isinstance(referrer, Equation):
                referrer_id = referrer.id()
                if referrer_id in self._tlm_eqs:
                    del self._tlm_eqs[referrer_id]

    def _write_memory_checkpoint(self, n, *, ics=True, data=True):
        if n in self._cp_memory or \
                (self._cp_disk is not None and n in self._cp_disk):
            raise RuntimeError("Duplicate checkpoint")

        self._cp_memory[n] = self._cp.checkpoint_data(
            ics=ics, data=data, copy=False)

    def _read_memory_checkpoint(self, n, *, ic_ids=None, ics=True, data=True,
                                delete=False):
        if ic_ids is None:
            ic_ids = set()

        read_cp, read_data, read_storage = self._cp_memory[n]
        if delete:
            self._cp_memory.pop(n)

        if ics or data:
            if ics:
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

            self._cp.update(read_cp, read_data, read_storage,
                            copy=not delete)

    def _write_disk_checkpoint(self, n, *, ics=True, data=True):
        if n in self._cp_memory or n in self._cp_disk:
            raise RuntimeError("Duplicate checkpoint")

        self._cp_disk.write(
            n, *self._cp.checkpoint_data(ics=ics, data=data, copy=False))

    def _read_disk_checkpoint(self, n, *, ic_ids=None, ics=True, data=True,
                              delete=False):
        if ic_ids is None:
            ic_ids = set()

        if ics or data:
            read_cp, read_data, read_storage = \
                self._cp_disk.read(n, ics=ics, data=data, ic_ids=ic_ids)

            self._cp.update(read_cp, read_data, read_storage,
                            copy=False)

        if delete:
            self._cp_disk.delete(n)

    def _checkpoint(self, final=False):
        assert len(self._block) == 0
        n = len(self._blocks)
        if final:
            self._cp_manager.finalize(n)
        if n < self._cp_manager.n():
            return
        if self._cp_manager.max_n() is not None:
            if n == self._cp_manager.max_n():
                return
            elif n > self._cp_manager.max_n():
                raise RuntimeError("Invalid checkpointing state")

        logger = logging.getLogger("tlm_adjoint.checkpointing")

        while True:
            cp_action, cp_data = next(self._cp_manager)

            if cp_action == "clear":
                clear_ics, clear_data = cp_data
                self._cp.clear(clear_ics=clear_ics,
                               clear_data=clear_data)
            elif cp_action == "configure":
                store_ics, store_data = cp_data
                self._cp.configure(store_ics=store_ics,
                                   store_data=store_data)
            elif cp_action == "forward":
                cp_n0, cp_n1 = cp_data
                logger.debug(f"forward: forward advance to {cp_n1:d}")
                if cp_n0 != n:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_n1 <= n:
                    raise RuntimeError("Invalid checkpointing state")
                break
            elif cp_action == "write":
                cp_w_n, cp_storage = cp_data
                if cp_w_n >= n:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_storage == "disk":
                    logger.debug(f"forward: save snapshot at {cp_w_n:d} "
                                 f"on disk")
                    self._write_disk_checkpoint(cp_w_n)
                elif cp_storage == "RAM":
                    logger.debug(f"forward: save snapshot at {cp_w_n:d} "
                                 f"in RAM")
                    self._write_memory_checkpoint(cp_w_n)
                else:
                    raise ValueError(f"Unrecognized checkpointing storage: "
                                     f"{cp_storage:s}")
            else:
                raise ValueError(f"Unexpected checkpointing action: "
                                 f"{cp_action:s}")

    def _restore_checkpoint(self, n, transpose_deps=None):
        if self._cp_manager.max_n() is None:
            raise RuntimeError("Invalid checkpointing state")
        if n > self._cp_manager.max_n() - self._cp_manager.r() - 1:
            return
        elif n != self._cp_manager.max_n() - self._cp_manager.r() - 1:
            raise RuntimeError("Invalid checkpointing state")

        logger = logging.getLogger("tlm_adjoint.checkpointing")

        storage = None
        initialize_storage_cp = False
        cp_n = None

        while True:
            cp_action, cp_data = next(self._cp_manager)

            if cp_action == "clear":
                clear_ics, clear_data = cp_data
                if initialize_storage_cp:
                    storage.update(self._cp.initial_conditions(cp=True,
                                                               refs=False,
                                                               copy=False),
                                   copy=not clear_ics or not clear_data)
                    initialize_storage_cp = False
                self._cp.clear(clear_ics=clear_ics,
                               clear_data=clear_data)
            elif cp_action == "configure":
                store_ics, store_data = cp_data
                self._cp.configure(store_ics=store_ics,
                                   store_data=store_data)
            elif cp_action == "forward":
                if storage is None or cp_n is None:
                    raise RuntimeError("Invalid checkpointing state")
                if initialize_storage_cp:
                    storage.update(self._cp.initial_conditions(cp=True,
                                                               refs=False,
                                                               copy=False),
                                   copy=True)
                    initialize_storage_cp = False

                cp_n0, cp_n1 = cp_data
                logger.debug(f"reverse: forward advance to {cp_n1:d}")
                if cp_n0 != cp_n:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_n1 > n + 1:
                    raise RuntimeError("Invalid checkpointing state")

                for n1 in range(cp_n0, cp_n1):
                    for i, eq in enumerate(self._blocks[n1]):
                        if storage.is_active(n1, i):
                            X = tuple(storage[eq_x] for eq_x in eq.X())
                            deps = tuple(storage[eq_dep]
                                         for eq_dep in eq.dependencies())

                            for eq_dep in eq.initial_condition_dependencies():
                                self._cp.add_initial_condition(
                                    eq_dep, value=storage[eq_dep])
                            eq.forward(X, deps=deps)
                            self._cp.add_equation(n1, i, eq, deps=deps)
                        elif transpose_deps.any_is_active(n1, i):
                            nl_deps = tuple(storage[eq_dep]
                                            for eq_dep in eq.nonlinear_dependencies())  # noqa: E501

                            self._cp.add_equation_data(
                                n1, i, eq, nl_deps=nl_deps)

                        storage_state = storage.pop()
                        assert storage_state == (n1, i)
                cp_n = cp_n1
            elif cp_action == "reverse":
                cp_n1, cp_n0 = cp_data
                logger.debug(f"reverse: adjoint step back to {cp_n0:d}")
                if cp_n1 != n + 1:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_n0 > n:
                    raise RuntimeError("Invalid checkpointing state")
                if storage is not None:
                    assert len(storage) == 0
                break
            elif cp_action == "read":
                if storage is not None or cp_n is not None:
                    raise RuntimeError("Invalid checkpointing state")

                cp_n, cp_storage, cp_delete = cp_data
                logger.debug(f'reverse: load snapshot at {cp_n:d} from '
                             f'{cp_storage:s} and '
                             f'{"delete" if cp_delete else "keep":s}')

                storage = ReplayStorage(self._blocks, cp_n, n + 1,
                                        transpose_deps=transpose_deps)
                initialize_storage_cp = True
                storage.update(self._cp.initial_conditions(cp=False,
                                                           refs=True,
                                                           copy=False),
                               copy=False)

                if cp_storage == "disk":
                    self._read_disk_checkpoint(cp_n, ic_ids=set(storage),
                                               delete=cp_delete)
                elif cp_storage == "RAM":
                    self._read_memory_checkpoint(cp_n, ic_ids=set(storage),
                                                 delete=cp_delete)
                else:
                    raise ValueError(f"Unrecognized checkpointing storage: "
                                     f"{cp_storage:s}")
            elif cp_action == "write":
                cp_w_n, cp_storage = cp_data
                if cp_w_n >= n:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_storage == "disk":
                    logger.debug(f"reverse: save snapshot at {cp_w_n:d} "
                                 f"on disk")
                    self._write_disk_checkpoint(cp_w_n)
                elif cp_storage == "RAM":
                    logger.debug(f"reverse: save snapshot at {cp_w_n:d} "
                                 f"in RAM")
                    self._write_memory_checkpoint(cp_w_n)
                else:
                    raise ValueError(f"Unrecognized checkpointing storage: "
                                     f"{cp_storage:s}")
            else:
                raise ValueError(f"Unexpected checkpointing action: "
                                 f"{cp_action:s}")

    def new_block(self):
        """
        End the current block equation and begin a new block.
        """

        self.drop_references()

        if self._annotation_state in ["stopped_initial",
                                      "stopped_annotating",
                                      "final"]:
            return
        elif self._cp_manager.max_n() is not None \
                and len(self._blocks) == self._cp_manager.max_n() - 1:
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
        if self._cp_manager.max_n() is not None \
                and len(self._blocks) < self._cp_manager.max_n():
            warnings.warn(
                "Insufficient number of blocks -- empty blocks added",
                RuntimeWarning, stacklevel=2)
            while len(self._blocks) < self._cp_manager.max_n():
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

    @restore_manager
    def compute_gradient(self, Js, M, callback=None, prune_forward=True,
                         prune_adjoint=True, prune_replay=True, adj_ics=None,
                         adj_cache=None):
        """
        Compute the derivative of one or more functionals with respect to one
        or more control parameters by running adjoint models. Finalizes the
        manager. Returns the complex conjugate of the derivative.

        Arguments:

        Js         A Functional or function, or a sequence of these, defining
                   the functionals.
        M          A function, or a sequence of functions, defining the control
                   parameters.
        callback   (Optional) Callable of the form
                       def callback(J_i, n, i, eq, adj_X):
                   where adj_X is None, a function, or a sequence of functions,
                   corresponding to the adjoint solution for the equation eq,
                   which is equation i in block n for the J_i th Functional.
        prune_forward  (Optional) Whether forward traversal graph pruning
                       should be applied.
        prune_adjoint  (Optional) Whether reverse traversal graph pruning
                       should be applied.
        prune_replay   (Optional) Whether graph pruning should be applied in
                       forward replay.
        adj_ics    (Optional) Map, or a sequence of maps, from forward
                   functions or function IDs to adjoint initial conditions.
        adj_cache  (Optional) An AdjointCache.
        """

        if not isinstance(M, Sequence):
            if not isinstance(Js, Sequence):
                ((dJ,),) = self.compute_gradient(
                    (Js,), (M,), callback=callback,
                    prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                    prune_replay=prune_replay,
                    adj_ics=None if adj_ics is None else (adj_ics,),
                    adj_cache=adj_cache)
                return dJ
            else:
                dJs = self.compute_gradient(
                    Js, (M,), callback=callback,
                    prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                    prune_replay=prune_replay,
                    adj_ics=adj_ics,
                    adj_cache=adj_cache)
                return tuple(dJ for (dJ,) in dJs)
        elif not isinstance(Js, Sequence):
            dJ, = self.compute_gradient(
                (Js,), M, callback=callback,
                prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                prune_replay=prune_replay,
                adj_ics=None if adj_ics is None else (adj_ics,),
                adj_cache=adj_cache)
            return dJ

        set_manager(self)
        gc.collect()
        self.finalize()
        self.reset_adjoint(_warning=False)

        # Functionals
        Js = tuple(Functional(_fn=J) if is_function(J) else J for J in Js)

        # Controls
        M = tuple(M)

        # Derivatives
        dJ = [None for J in Js]

        # Add two additional blocks, one at the start and one at the end of the
        # forward:
        #   Control block   :  Represents the equation "controls = inputs"
        #   Functional block:  Represents the equations "outputs = functionals"
        blocks_N = len(self._blocks)
        blocks = {-1: [ControlsMarker(M)]}
        blocks.update({n: block for n, block in enumerate(self._blocks)})
        blocks[blocks_N] = [FunctionalMarker(J) for J in Js]
        J_markers = tuple(eq.x() for eq in blocks[blocks_N])

        # Adjoint equation right-hand-sides
        Bs = tuple(AdjointModelRHS(blocks) for J in Js)
        # Adjoint initial condition
        for J_i in range(len(Js)):
            function_assign(Bs[J_i][blocks_N][J_i].b(), 1.0)

        # Transpose dependency graph
        transpose_deps = DependencyGraphTranspose(
            J_markers, M, blocks,
            prune_forward=prune_forward, prune_adjoint=prune_adjoint,
            adj_cache=adj_cache)

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
        for n in range(blocks_N, -2, -1):
            block = blocks[n]

            cp_block = n >= 0 and n < blocks_N
            if cp_block:
                # Load/restore forward model data
                self._restore_checkpoint(
                    n, transpose_deps=transpose_deps if prune_replay else None)

            # Reverse (equations in block n)
            for i in range(len(block) - 1, -1, -1):
                eq = block[i]
                eq_X = eq.X()

                assert len(Js) == len(J_markers)
                for J_i, (J, J_marker) in enumerate(zip(Js, J_markers)):
                    # Adjoint right-hand-side associated with this equation
                    B_state, eq_B = Bs[J_i].pop()
                    assert B_state == (n, i)

                    # Extract adjoint initial condition
                    adj_X_ic = tuple(adj_Xs[J_i].pop(function_id(x), None)
                                     for x in eq_X)
                    if transpose_deps.is_solved(J_marker, n, i):
                        adj_X_ic_ids = {function_id(dep)
                                        for dep in eq.adjoint_initial_condition_dependencies()}  # noqa: E501
                        assert len(eq_X) == len(adj_X_ic)
                        for x, adj_x_ic in zip(eq_X, adj_X_ic):
                            if function_id(x) not in adj_X_ic_ids:
                                assert adj_x_ic is None
                        del adj_X_ic_ids
                    else:
                        for adj_x_ic in adj_X_ic:
                            assert adj_x_ic is None

                    if transpose_deps.is_solved(J_marker, n, i):
                        # Construct adjoint initial condition
                        if len(eq.adjoint_initial_condition_dependencies()) == 0:  # noqa: E501
                            adj_X = None
                        else:
                            adj_X = []
                            for m, adj_x_ic in enumerate(adj_X_ic):
                                if adj_x_ic is None:
                                    adj_X.append(eq.new_adj_X(m))
                                else:
                                    adj_X.append(adj_x_ic)

                        # Non-linear dependency data
                        nl_deps = self._cp[(n, i)] if cp_block else ()

                        # Solve adjoint equation, add terms to adjoint
                        # equations
                        adj_X = eq.adjoint(
                            J, adj_X, nl_deps,
                            eq_B.B(),
                            transpose_deps.adj_Bs(J_marker, n, i, eq, Bs[J_i]))
                    elif transpose_deps.is_active(J_marker, n, i):
                        # Extract adjoint solution from the cache
                        adj_X = adj_cache.get_cached(J_i, n, i, copy=False)

                        # Non-linear dependency data
                        nl_deps = self._cp[(n, i)] if cp_block else ()

                        # Add terms to adjoint equations
                        eq.adjoint_cached(
                            J, adj_X, nl_deps,
                            transpose_deps.adj_Bs(J_marker, n, i, eq, Bs[J_i]))
                    else:
                        # Adjoint solution has no effect on sensitivity
                        adj_X = None

                    if adj_X is not None:
                        # Store adjoint initial conditions
                        assert len(eq_X) == len(adj_X)
                        for m, (x, adj_x) in enumerate(zip(eq_X, adj_X)):
                            if transpose_deps.is_stored_adj_ic(J_marker, n, i, m):  # noqa: E501
                                adj_Xs[J_i][function_id(x)] = function_copy(adj_x)  # noqa: E501

                        if adj_cache is not None:
                            # Store adjoint solution in the cache, if needed
                            adj_cache.cache(J_i, n, i, adj_X,
                                            copy=True, replace=False)

                    if callback is not None:
                        # Diagnostic callback
                        if adj_X is None:
                            callback(J_i, n, i, eq,
                                     None)
                        elif len(adj_X) == 1:
                            callback(J_i, n, i, eq,
                                     function_copy(adj_X[0]))
                        else:
                            callback(J_i, n, i, eq,
                                     tuple(function_copy(adj_x)
                                           for adj_x in adj_X))

                    if n == -1:
                        assert i == 0
                        # A requested derivative
                        if adj_X is None:
                            dJ[J_i] = eq.new_adj_X()
                        else:
                            dJ[J_i] = tuple(function_copy(adj_x)
                                            for adj_x in adj_X)
                    else:
                        # Finalize right-hand-sides in the control block
                        Bs[J_i][-1].finalize()

        for B in Bs:
            assert B.is_empty()
        for J_i in range(len(adj_Xs)):
            assert len(adj_Xs[J_i]) == 0

        if self._cp_manager.max_n() is None \
                or self._cp_manager.r() != self._cp_manager.max_n():
            raise RuntimeError("Invalid checkpointing state")

        while True:
            cp_action, cp_data = next(self._cp_manager)

            if cp_action == "clear":
                clear_ics, clear_data = cp_data
                self._cp.clear(clear_ics=clear_ics,
                               clear_data=clear_data)
            elif cp_action == "end_reverse":
                break
            else:
                raise ValueError(f"Unexpected checkpointing action: "
                                 f"{cp_action:s}")

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
        raise KeyError("Initial condition not found")


set_manager(EquationManager())
