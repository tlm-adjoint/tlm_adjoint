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

from .alias import WeakAlias, gc_disabled
from .binomial_checkpointing import MultistageCheckpointingManager
from .checkpointing import CheckpointStorage, HDF5Checkpoints, \
    MemoryCheckpointingManager, NoneCheckpointingManager, \
    PeriodicDiskCheckpointingManager, PickleCheckpoints, ReplayStorage
from .equations import AdjointModelRHS, ControlsMarker, Equation, \
    FunctionalMarker, NullSolver
from .functional import Functional
from .manager import restore_manager, set_manager

from collections import defaultdict, deque
from collections.abc import Sequence
import copy
import enum
import itertools
import logging
import numpy as np
from operator import itemgetter
import os
import warnings
import weakref

__all__ = \
    [
        "EquationManager"
    ]


def tlm_key(M, dM):
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
    for m, dm in zip(M, dM):
        check_space_types(m, dm)

    return ((M, dM),
            (tuple(function_id(m) for m in M),
             tuple(function_id(dm) for dm in dM)))


def tlm_keys(*args):
    M_dM_keys = tuple(map(lambda arg: tlm_key(*arg), args))
    for ks in itertools.chain.from_iterable(
            distinct_combinations_indices((key for _, key in M_dM_keys), j)
            for j in range(1, len(M_dM_keys) + 1)):
        yield tuple(M_dM_keys[k] for k in ks)


class TangentLinear:
    def __init__(self, *, annotate=True):
        self._children = {}
        self._annotate = annotate

    def __contains__(self, key):
        _, key = tlm_key(*key)
        return key in self._children

    def __getitem__(self, key):
        _, key = tlm_key(*key)
        return self._children[key][1]

    def __iter__(self):
        yield from self.keys()

    def __len__(self):
        return len(self._children)

    def keys(self):
        for (M, dM), _ in self._children.values():
            yield (M, dM)

    def values(self):
        for _, child in self._children.values():
            yield child

    def items(self):
        yield from zip(self.keys(), self.values())

    def add(self, M, dM, *, annotate=True):
        (M, dM), key = tlm_key(M, dM)
        if key not in self._children:
            self._children[key] = ((M, dM), TangentLinear(annotate=annotate))

    def remove(self, M, dM):
        _, key = tlm_key(M, dM)
        del self._children[key]

    def clear(self):
        self._children.clear()

    def is_annotated(self):
        return self._annotate

    def set_is_annotated(self, annotate):
        self._annotate = annotate


class TangentLinearMap:
    """
    A map from forward to tangent-linear variables.
    """

    def __init__(self, M, dM):
        (M, dM), _ = tlm_key(M, dM)

        if len(M) == 1:
            self._name_suffix = \
                "_tlm(%s,%s)" % (function_name(M[0]),
                                 function_name(dM[0]))
        else:
            self._name_suffix = \
                "_tlm((%s),(%s))" % (",".join(map(function_name, M)),
                                     ",".join(map(function_name, dM)))

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
            tau_x = function_new_tangent_linear(
                x, name=f"{function_name(x):s}{self._name_suffix:s}")
            if tau_x is not None:
                tau_x._tlm_adjoint__tlm_root_id = getattr(
                    x, "_tlm_adjoint__tlm_root_id", function_id(x))
            x._tlm_adjoint__tangent_linears[self] = tau_x

        return x._tlm_adjoint__tangent_linears[self]


class DependencyGraphTranspose:
    def __init__(self, Js, M, blocks, *,
                 prune_forward=True, prune_adjoint=True):
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
                        X_ids = set(map(function_id, eq.X()))
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

        active = {J_i: copy.deepcopy(active_forward) for J_i in range(len(Js))}

        if prune_adjoint:
            # Pruning, reverse traversal
            for J_i, J in enumerate(Js):
                J_id = function_id(J)
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
                            active[J_i][n][i] = False

        solved = copy.deepcopy(active)

        stored_adj_ics = {J_i: {n: tuple([None for x in eq.X()]
                                         for eq in blocks[n])
                                for n in blocks_n} for J_i in range(len(Js))}
        adj_ics = {J_i: {} for J_i in range(len(Js))}
        for J_i in range(len(Js)):
            for n in blocks_n:
                block = blocks[n]
                for i, eq in enumerate(block):
                    adj_ic_ids = set(map(function_id,
                                         eq.adjoint_initial_condition_dependencies()))  # noqa: E501
                    for m, x in enumerate(eq.X()):
                        x_id = function_id(x)

                        stored_adj_ics[J_i][n][i][m] = adj_ics[J_i].get(x_id, None)  # noqa: E501

                        if x_id in adj_ic_ids:
                            adj_ics[J_i][x_id] = (n, i)
                        elif x_id in adj_ics[J_i]:
                            del adj_ics[J_i][x_id]

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

    def is_active(self, J_i, n, i):
        return self._active[J_i][n][i]

    def any_is_active(self, n, i):
        for J_i in self._active:
            if self._active[J_i][n][i]:
                return True
        return False

    def is_solved(self, J_i, n, i):
        return self._solved[J_i][n][i]

    def set_not_solved(self, J_i, n, i):
        self._solved[J_i][n][i] = False

    def has_adj_ic(self, J_i, x):
        if isinstance(x, int):
            x_id = x
        else:
            x_id = function_id(x)

        if x_id in self._adj_ics[J_i]:
            n, i = self._adj_ics[J_i][x_id]
            return self.is_solved(J_i, n, i)
        else:
            return False

    def is_stored_adj_ic(self, J_i, n, i, m):
        stored_adj_ics = self._stored_adj_ics[J_i][n][i][m]
        if stored_adj_ics is None:
            return False
        else:
            p, k = stored_adj_ics
            return self.is_solved(J_i, p, k)

    def adj_Bs(self, J_i, n, i, eq, B):
        dep_Bs = {}
        for j, dep in enumerate(eq.dependencies()):
            if (n, i, j) in self:
                p, k, m = self[(n, i, j)]
                if self.is_solved(J_i, p, k):
                    dep_Bs[j] = B[p][k][m]

        return dep_Bs


def distinct_combinations_indices(iterable, r):
    class Comparison:
        def __init__(self, key, value):
            self._key = key
            self._value = value

        def __eq__(self, other):
            if isinstance(other, Comparison):
                return self._key == other._key
            else:
                return NotImplemented

        def __hash__(self):
            return hash(self._key)

        def value(self):
            return self._value

    t = tuple(Comparison(value, i) for i, value in enumerate(iterable))

    try:
        import more_itertools
    except ImportError:
        # Basic implementation likely suffices for most cases in practice
        seen = set()
        for combination in itertools.combinations(t, r):
            if combination not in seen:
                seen.add(combination)
                yield tuple(e.value() for e in combination)
        return

    for combination in more_itertools.distinct_combinations(t, r):
        yield tuple(e.value() for e in combination)


def J_tangent_linears(Js, blocks, *, max_adjoint_degree=None):
    if isinstance(blocks, Sequence):
        # Sequence
        blocks_n = tuple(range(len(blocks)))
    else:
        # Mapping
        blocks_n = tuple(sorted(blocks.keys()))

    J_is = {function_id(J): J_i for J_i, J in enumerate(Js)}
    J_roots = list(Js)
    J_root_ids = {J_id: J_id for J_id in map(function_id, Js)}
    remaining_Js = dict(enumerate(Js))
    tlm_adj = defaultdict(lambda: [])

    for n in reversed(blocks_n):
        block = blocks[n]
        for i in range(len(block) - 1, -1, -1):
            eq = block[i]

            if isinstance(eq, ControlsMarker):
                continue
            elif isinstance(eq, FunctionalMarker):
                J, J_root = eq.dependencies()
                J_id = function_id(J)
                if J_id in J_root_ids:
                    assert J_root_ids[J_id] == J_id
                    J_roots[J_is[J_id]] = J_root
                    J_root_ids[J_id] = function_id(J_root)
                    assert J_root_ids[J_id] != J_id
                del J, J_root, J_id
                continue

            eq_X_ids = set(map(function_id, eq.X()))
            eq_tlm_key = getattr(eq, "_tlm_adjoint__tlm_key", ())

            found_Js = []
            for J_i, J in remaining_Js.items():
                if J_root_ids[function_id(J)] in eq_X_ids:
                    found_Js.append(J_i)
                    J_max_adjoint_degree = len(eq_tlm_key) + 1
                    if max_adjoint_degree is not None:
                        assert max_adjoint_degree >= 0
                        J_max_adjoint_degree = min(J_max_adjoint_degree,
                                                   max_adjoint_degree)
                    for ks in itertools.chain.from_iterable(
                            distinct_combinations_indices(eq_tlm_key, j)
                            for j in range(len(eq_tlm_key) + 1 - J_max_adjoint_degree,  # noqa: E501
                                           len(eq_tlm_key) + 1)):
                        tlm_key = tuple(eq_tlm_key[k] for k in ks)
                        ks = set(ks)
                        adj_tlm_key = tuple(eq_tlm_key[k]
                                            for k in range(len(eq_tlm_key))
                                            if k not in ks)
                        tlm_adj[tlm_key].append((J_i, adj_tlm_key))
            for J_i in found_Js:
                del remaining_Js[J_i]

            if len(remaining_Js) == 0:
                break
        if len(remaining_Js) == 0:
            break

    return (tuple(J_roots),
            {tlm_key: tuple(sorted(adj_key, key=itemgetter(0)))
             for tlm_key, adj_key in tlm_adj.items()})


class AdjointCache:
    def __init__(self):
        self._cache = {}
        self._keys = {}
        self._cache_key = None

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key):
        J_i, n, i = key
        return (J_i, n, i) in self._cache

    def clear(self):
        self._cache.clear()
        self._keys.clear()
        self._cache_key = None

    def get(self, J_i, n, i, *, copy=True):
        adj_X = self._cache[(J_i, n, i)]
        if copy:
            adj_X = tuple(function_copy(adj_x) for adj_x in adj_X)
        return adj_X

    def pop(self, J_i, n, i, *, copy=True):
        adj_X = self._cache.pop((J_i, n, i))
        if copy:
            adj_X = tuple(function_copy(adj_x) for adj_x in adj_X)
        return adj_X

    def remove(self, J_i, n, i):
        del self._cache[(J_i, n, i)]

    def cache(self, J_i, n, i, adj_X, *, copy=True, store=False):
        if (J_i, n, i) in self._keys \
                and (store or len(self._keys[(J_i, n, i)]) > 0):
            if (J_i, n, i) in self._cache:
                adj_X = self._cache[(J_i, n, i)]
            elif copy:
                adj_X = tuple(function_copy(adj_x) for adj_x in adj_X)
            else:
                adj_X = tuple(adj_X)

            if store:
                self._cache[(J_i, n, i)] = adj_X
            for J_j, p, k in self._keys[(J_i, n, i)]:
                self._cache[(J_j, p, k)] = adj_X

    def initialize(self, Js, blocks, transpose_deps, *,
                   cache_degree=None):
        J_roots, tlm_adj = J_tangent_linears(Js, blocks,
                                             max_adjoint_degree=cache_degree)
        J_root_ids = tuple(getattr(J, "_tlm_adjoint__tlm_root_id", function_id(J))  # noqa: E501
                           for J in J_roots)

        cache_key = tuple((J_root_ids[J_i], adj_tlm_key)
                          for J_i, adj_tlm_key
                          in sorted(itertools.chain.from_iterable(tlm_adj.values())))  # noqa: E501

        if self._cache_key is None or self._cache_key != cache_key:
            self.clear()

        self._keys.clear()
        self._cache_key = None

        if cache_degree is None or cache_degree > 0:
            self._cache_key = cache_key

            if isinstance(blocks, Sequence):
                # Sequence
                blocks_n = tuple(range(len(blocks)))
            else:
                # Mapping
                blocks_n = tuple(sorted(blocks.keys()))

            eqs = defaultdict(lambda: [])
            for n in reversed(blocks_n):
                block = blocks[n]
                for i in range(len(block) - 1, -1, -1):
                    eq = block[i]

                    if isinstance(eq, (ControlsMarker, FunctionalMarker)):
                        continue

                    eq_id = eq.id()
                    eq_tlm_root_id = getattr(eq, "_tlm_adjoint__tlm_root_id", eq_id)  # noqa: E501
                    eq_tlm_key = getattr(eq, "_tlm_adjoint__tlm_key", ())

                    for J_i, adj_tlm_key in tlm_adj.get(eq_tlm_key, ()):
                        if transpose_deps.is_solved(J_i, n, i) \
                                or (J_i, n, i) in self._cache:
                            if cache_degree is not None:
                                assert len(adj_tlm_key) < cache_degree
                            eqs[eq_tlm_root_id].append(
                                ((J_i, n, i),
                                 (J_root_ids[J_i], adj_tlm_key)))

                    eq_root = {}
                    for (J_j, p, k), adj_key in eqs.pop(eq_id, []):
                        assert transpose_deps.is_solved(J_j, p, k) \
                            or (J_j, p, k) in self._cache
                        if adj_key in eq_root:
                            self._keys[eq_root[adj_key]].append((J_j, p, k))
                            if (J_j, p, k) in self._cache \
                                    and eq_root[adj_key] not in self._cache:
                                self._cache[eq_root[adj_key]] = self._cache[(J_j, p, k)]  # noqa: E501
                        else:
                            eq_root[adj_key] = (J_j, p, k)
                            self._keys[eq_root[adj_key]] = []
            assert len(eqs) == 0

        for (J_i, n, i) in self._cache:
            transpose_deps.set_not_solved(J_i, n, i)
        for eq_root in self._keys:
            for (J_i, n, i) in self._keys[eq_root]:
                transpose_deps.set_not_solved(J_i, n, i)


class AnnotationState(enum.Enum):
    STOPPED = "stopped"
    ANNOTATING = "annotating"
    FINAL = "final"


class TangentLinearState(enum.Enum):
    STOPPED = "stopped"
    DERIVING = "deriving"
    FINAL = "final"


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
                nl_dep_ids = set(map(function_id,
                                     eq.nonlinear_dependencies()))
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

        self._annotation_state = AnnotationState.ANNOTATING
        self._tlm_state = TangentLinearState.DERIVING
        self._eqs = {}
        self._blocks = []
        self._block = []

        self._tlm = TangentLinear()
        self._tlm_map = {}
        self._tlm_eqs = {}

        self._adj_cache = AdjointCache()

        self.configure_checkpointing(cp_method, cp_parameters=cp_parameters)

    def configure_checkpointing(self, cp_method, cp_parameters):
        """
        Provide a new checkpointing configuration.
        """

        if len(self._block) != 0 or len(self._blocks) != 0:
            raise RuntimeError("Cannot configure checkpointing after "
                               "equations have been recorded")

        cp_parameters = copy.copy(cp_parameters)

        if not callable(cp_method) and cp_method in ["none", "memory"]:
            if "replace" in cp_parameters:
                warnings.warn("replace cp_parameters key is deprecated",
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
                cp_parameters["period"])
        elif cp_method == "multistage":
            cp_manager = MultistageCheckpointingManager(
                cp_parameters["blocks"],
                cp_parameters.get("snaps_in_ram", 0),
                cp_parameters.get("snaps_on_disk", 0),
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

    def configure_tlm(self, *args, annotate=None, tlm=True):
        """
        Configure the tangent-linear tree.

        Arguments:

        args      ((M_0, dM_0), [...]). Identifies a node of the tangent-linear
                  tree.
        annotate  (Optional, default tlm) If true then enable annotation for
                  the tangent-linear model associated with the node, and enable
                  annotation for all tangent-linear models on which it depends.
                  If false then disable annotation for the tangent-linear
                  model associated with the node, all tangent-linear models
                  which depend on it, and any tangent-linear models associated
                  with new nodes.
        tlm       (Optional) If true then add the tangent-linear model
                  associated with the node, and add all tangent-linear models
                  on which it depends. If false then remove the tangent-linear
                  model associated with the node, and remove all tangent-linear
                  models which depend on it.
        """

        if self._tlm_state == TangentLinearState.FINAL:
            raise RuntimeError("Cannot configure tangent-linear models after "
                               "finalization")

        if annotate is None:
            annotate = tlm
        if annotate and not tlm:
            raise ValueError("Invalid annotate/tlm combination")

        if tlm:
            # Could be optimized to avoid encountering parent nodes multiple
            # times
            for M_dM_keys in tlm_keys(*args):
                node = self._tlm
                for (M, dM), key in M_dM_keys:
                    if (M, dM) in node:
                        if annotate:
                            node[(M, dM)].set_is_annotated(True)
                    else:
                        node.add(M, dM, annotate=annotate)
                        if key not in self._tlm_map:
                            self._tlm_map[key] = TangentLinearMap(M, dM)
                    node = node[(M, dM)]

        if not annotate or not tlm:
            def depends(keys_a, keys_b):
                j = 0
                for i, key_a in enumerate(keys_a):
                    if j >= len(keys_b):
                        return True
                    elif key_a == keys_b[j]:
                        j += 1
                return j >= len(keys_b)

            keys = tuple(key
                         for _, key in map(lambda arg: tlm_key(*arg), args))
            remaining_nodes = [(self._tlm,
                                (tlm_key(*child_M_dM)[1],),
                                child_M_dM,
                                child)
                               for child_M_dM, child in self._tlm.items()]
            while len(remaining_nodes) > 0:
                parent, node_keys, node_M_dM, node = remaining_nodes.pop()
                if depends(node_keys, keys):
                    if not tlm:
                        parent.remove(*node_M_dM)
                    elif not annotate:
                        node.set_is_annotated(False)
                if node_M_dM in parent:
                    remaining_nodes.extend(
                        (node,
                         tuple(list(node_keys) + [tlm_key(*child_M_dM)[1]]),
                         child_M_dM,
                         child)
                        for child_M_dM, child in node.items())

    def add_tlm(self, M, dM, max_depth=1, *, _warning=True):
        if _warning:
            warnings.warn("EquationManager.add_tlm method is deprecated -- "
                          "use EquationManager.configure_tlm instead",
                          DeprecationWarning, stacklevel=2)

        if self._tlm_state == TangentLinearState.FINAL:
            raise RuntimeError("Cannot configure tangent-linear models after "
                               "finalization")

        (M, dM), key = tlm_key(M, dM)

        for depth in range(max_depth):
            remaining_nodes = [self._tlm]
            while len(remaining_nodes) > 0:
                node = remaining_nodes.pop()
                remaining_nodes.extend(node.values())
                node.add(M, dM, annotate=True)

        if key not in self._tlm_map:
            self._tlm_map[key] = TangentLinearMap(M, dM)

    def tlm_enabled(self):
        """
        Return whether derivation of tangent-linear equations is enabled.
        """

        return self._tlm_state == TangentLinearState.DERIVING

    def function_tlm(self, x, *args):
        """
        Return a tangent-linear function associated with the function x.
        """

        tau = x
        for _, key in map(lambda arg: tlm_key(*arg), args):
            tau = self._tlm_map[key][tau]
        return tau

    def tlm(self, M, dM, x, max_depth=1, *, _warning=True):
        if _warning:
            warnings.warn("EquationManager.tlm method is deprecated -- "
                          "use EquationManager.function_tlm instead",
                          DeprecationWarning, stacklevel=2)

        return self.function_tlm(x, *[(M, dM) for depth in range(max_depth)])

    def annotation_enabled(self):
        """
        Return whether the equation manager currently has annotation enabled.
        """

        return self._annotation_state == AnnotationState.ANNOTATING

    def start(self, annotation=True, tlm=True):
        """
        Start annotation or tangent-linear derivation.
        """

        if annotation:
            self._annotation_state \
                = {AnnotationState.STOPPED: AnnotationState.ANNOTATING,
                   AnnotationState.ANNOTATING: AnnotationState.ANNOTATING}[self._annotation_state]  # noqa: E501

        if tlm:
            self._tlm_state \
                = {TangentLinearState.STOPPED: TangentLinearState.DERIVING,
                   TangentLinearState.DERIVING: TangentLinearState.DERIVING}[self._tlm_state]  # noqa: E501

    def stop(self, annotation=True, tlm=True):
        """
        Pause annotation or tangent-linear derivation. Returns a tuple
        containing:
            (annotation_state, tlm_state)
        where annotation_state indicates whether annotation is enabled, and
        tlm_state indicates whether tangent-linear equation derivation is
        enabled, each evaluated before changing the state.
        """

        state = (self.annotation_enabled(), self.tlm_enabled())

        if annotation:
            self._annotation_state \
                = {AnnotationState.STOPPED: AnnotationState.STOPPED,
                   AnnotationState.ANNOTATING: AnnotationState.STOPPED,
                   AnnotationState.FINAL: AnnotationState.FINAL}[self._annotation_state]  # noqa: E501

        if tlm:
            self._tlm_state \
                = {TangentLinearState.STOPPED: TangentLinearState.STOPPED,
                   TangentLinearState.DERIVING: TangentLinearState.STOPPED,
                   TangentLinearState.FINAL: TangentLinearState.FINAL}[self._tlm_state]  # noqa: E501

        return state

    def add_initial_condition(self, x, annotate=None):
        """
        Record an initial condition associated with the function x.

        annotate (default self.annotation_enabled()):
            Whether to record the initial condition, storing data for
            checkpointing as required.
        """

        if annotate is None:
            annotate = self.annotation_enabled()
        if annotate:
            if self._annotation_state == AnnotationState.FINAL:
                raise RuntimeError("Cannot add initial conditions after "
                                   "finalization")

            self._cp.add_initial_condition(x)

    def add_equation(self, eq, annotate=None, tlm=None):
        """
        Process the provided equation, deriving (and solving) tangent-linear
        equations as required. Assumes that the equation has already been
        solved, and that the initial condition for eq.X() has been recorded if
        necessary.

        annotate (default self.annotation_enabled()):
            Whether to record the equation, storing data for checkpointing as
            required.
        tlm (default self.tlm_enabled()):
            Whether to derive (and solve) associated tangent-linear equations.
        """

        self.drop_references()

        if annotate is None:
            annotate = self.annotation_enabled()
        if annotate:
            if self._annotation_state == AnnotationState.FINAL:
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
            if self._tlm_state == TangentLinearState.FINAL:
                raise RuntimeError("Cannot add tangent-linear equations after "
                                   "finalization")

            remaining_eqs = deque((eq, child_M_dM, child)
                                  for child_M_dM, child in self._tlm.items())
            while len(remaining_eqs) > 0:
                parent_eq, (node_M, node_dM), node = remaining_eqs.popleft()

                node_eq = self._tangent_linear(parent_eq, node_M, node_dM)
                if node_eq is not None:
                    node_eq.solve(
                        manager=self,
                        annotate=node.is_annotated(),
                        tlm=False)
                    remaining_eqs.extend(
                        (node_eq, child_M_dM, child)
                        for child_M_dM, child in node.items())

    def _tangent_linear(self, eq, M, dM):
        (M, dM), key = tlm_key(M, dM)

        X = eq.X()
        X_ids = set(map(function_id, X))
        if not X_ids.isdisjoint(set(key[0])):
            raise ValueError("Invalid tangent-linear parameter")
        if not X_ids.isdisjoint(set(key[1])):
            raise ValueError("Invalid tangent-linear direction")

        eq_id = eq.id()
        eq_tlm_eqs = self._tlm_eqs.get(eq_id, None)
        if eq_tlm_eqs is None:
            eq_tlm_eqs = {}
            self._tlm_eqs[eq_id] = eq_tlm_eqs

        tlm_map = self._tlm_map[key]
        tlm_eq = eq_tlm_eqs.get(key, None)
        if tlm_eq is None:
            for dep in eq.dependencies():
                if dep in M or dep in tlm_map:
                    tlm_eq = eq.tangent_linear(M, dM, tlm_map)
                    if tlm_eq is None:
                        tlm_eq = NullSolver([tlm_map[x] for x in X])
                    tlm_eq._tlm_adjoint__tlm_root_id = getattr(
                        eq, "_tlm_adjoint__tlm_root_id", eq.id())
                    tlm_eq._tlm_adjoint__tlm_key = tuple(
                        list(getattr(eq, "_tlm_adjoint__tlm_key", ()))
                        + [key])

                    eq_tlm_eqs[key] = tlm_eq
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
        read_cp, read_data, read_storage = self._cp_memory[n]
        if delete:
            del self._cp_memory[n]

        if ics or data:
            if ics:
                read_cp = tuple(key for key in read_cp
                                if ic_ids is None or key[0] in ic_ids)
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
                        else:
                            self._cp.update_keys(
                                n1, i, eq)

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

        if self._annotation_state in [AnnotationState.STOPPED,
                                      AnnotationState.FINAL]:
            return

        if self._cp_manager.max_n() is not None \
                and len(self._blocks) == self._cp_manager.max_n() - 1:
            # Wait for the finalize
            warnings.warn(
                "Attempting to end the final block without finalizing -- "
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

        if self._annotation_state == AnnotationState.FINAL:
            return

        self._annotation_state = AnnotationState.FINAL
        self._tlm_state = TangentLinearState.FINAL

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

    def reset_adjoint(self, *, _warning=True):
        if _warning:
            warnings.warn("EquationManager.reset_adjoint method is deprecated",
                          DeprecationWarning, stacklevel=2)

        for eq in self._eqs.values():
            eq.reset_adjoint()

    @restore_manager
    def compute_gradient(self, Js, M, callback=None, prune_forward=True,
                         prune_adjoint=True, prune_replay=True,
                         cache_adjoint_degree=None, store_adjoint=False,
                         adj_ics=None):
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
        cache_adjoint_degree
                       (Optional) Cache and reuse adjoint solutions of this
                       degree and lower. If not supplied then caching is
                       applied for all degrees.
        store_adjoint  (Optional) Whether adjoint solutions should be retained
                       for use by a later call to compute_gradient.
        adj_ics    (Optional) Map, or a sequence of maps, from forward
                   functions or function IDs to adjoint initial conditions.
        """

        if not isinstance(M, Sequence):
            if not isinstance(Js, Sequence):
                ((dJ,),) = self.compute_gradient(
                    (Js,), (M,), callback=callback,
                    prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                    prune_replay=prune_replay,
                    cache_adjoint_degree=cache_adjoint_degree,
                    store_adjoint=store_adjoint,
                    adj_ics=None if adj_ics is None else (adj_ics,))
                return dJ
            else:
                dJs = self.compute_gradient(
                    Js, (M,), callback=callback,
                    prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                    prune_replay=prune_replay,
                    cache_adjoint_degree=cache_adjoint_degree,
                    store_adjoint=store_adjoint,
                    adj_ics=adj_ics)
                return tuple(dJ for (dJ,) in dJs)
        elif not isinstance(Js, Sequence):
            dJ, = self.compute_gradient(
                (Js,), M, callback=callback,
                prune_forward=prune_forward, prune_adjoint=prune_adjoint,
                prune_replay=prune_replay,
                cache_adjoint_degree=cache_adjoint_degree,
                store_adjoint=store_adjoint,
                adj_ics=None if adj_ics is None else (adj_ics,))
            return dJ

        set_manager(self)
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
            prune_forward=prune_forward, prune_adjoint=prune_adjoint)

        # Initialize the adjoint cache
        self._adj_cache.initialize(J_markers, blocks, transpose_deps,
                                   cache_degree=cache_adjoint_degree)

        # Adjoint variables
        adj_Xs = tuple({} for J in Js)
        if adj_ics is not None:
            for J_i in range(len(Js)):
                for x_id, adj_x in adj_ics[J_i].items():
                    if not isinstance(x_id, int):
                        x_id = function_id(x_id)
                    if transpose_deps.has_adj_ic(J_i, x_id):
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

                for J_i, J in enumerate(Js):
                    # Adjoint right-hand-side associated with this equation
                    B_state, eq_B = Bs[J_i].pop()
                    assert B_state == (n, i)

                    # Extract adjoint initial condition
                    adj_X_ic = tuple(adj_Xs[J_i].pop(function_id(x), None)
                                     for x in eq_X)
                    if transpose_deps.is_solved(J_i, n, i):
                        adj_X_ic_ids = set(map(function_id,
                                               eq.adjoint_initial_condition_dependencies()))  # noqa: E501
                        assert len(eq_X) == len(adj_X_ic)
                        for x, adj_x_ic in zip(eq_X, adj_X_ic):
                            if function_id(x) not in adj_X_ic_ids:
                                assert adj_x_ic is None
                        del adj_X_ic_ids
                    else:
                        for adj_x_ic in adj_X_ic:
                            assert adj_x_ic is None

                    if transpose_deps.is_solved(J_i, n, i):
                        assert (J_i, n, i) not in self._adj_cache

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
                            transpose_deps.adj_Bs(J_i, n, i, eq, Bs[J_i]))
                    elif transpose_deps.is_active(J_i, n, i):
                        # Extract adjoint solution from the cache
                        if store_adjoint:
                            adj_X = self._adj_cache.get(J_i, n, i,
                                                        copy=False)
                        else:
                            adj_X = self._adj_cache.pop(J_i, n, i,
                                                        copy=False)

                        # Non-linear dependency data
                        nl_deps = self._cp[(n, i)] if cp_block else ()

                        # Add terms to adjoint equations
                        eq.adjoint_cached(
                            J, adj_X, nl_deps,
                            transpose_deps.adj_Bs(J_i, n, i, eq, Bs[J_i]))
                    else:
                        if not store_adjoint \
                                and (J_i, n, i) in self._adj_cache:
                            self._adj_cache.remove(J_i, n, i)

                        # Adjoint solution has no effect on sensitivity
                        adj_X = None

                    if adj_X is not None:
                        # Store adjoint initial conditions
                        assert len(eq_X) == len(adj_X)
                        for m, (x, adj_x) in enumerate(zip(eq_X, adj_X)):
                            if transpose_deps.is_stored_adj_ic(J_i, n, i, m):
                                adj_Xs[J_i][function_id(x)] = function_copy(adj_x)  # noqa: E501

                        # Store adjoint solution in the cache
                        self._adj_cache.cache(J_i, n, i, adj_X,
                                              copy=True, store=store_adjoint)

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
        if not store_adjoint:
            assert len(self._adj_cache) == 0

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


set_manager(EquationManager())
