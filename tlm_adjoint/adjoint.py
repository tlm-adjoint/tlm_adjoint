from .interface import (
    VariableStateLockDictionary, space_new, subtract_adjoint_derivative_action,
    var_copy, var_id, var_space, var_space_type)

from .instructions import Instruction
from .markers import ControlsMarker, FunctionalMarker
from .tangent_linear import J_tangent_linears

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import itertools
import numpy as np

__all__ = \
    [
        "AdjointRHS",
        "AdjointEquationRHS",
        "AdjointBlockRHS",
        "AdjointModelRHS"
    ]


class AdjointRHS:
    """Adjoint equation right-hand-side component.

    The right-hand-side of an adjoint equation, for an adjoint variable
    associated with an operation computing a forward variable `x`.

    Parameters
    ----------

    x : variable
        The forward variable.
    """

    def __init__(self, x):
        self._space = var_space(x)
        self._space_type = var_space_type(x, rel_space_type="conjugate_dual")
        self._b = None

    def b(self, *, copy=False):
        """Access the adjoint right-hand-side.

        Parameters
        ----------

        copy : bool
            Whether to return a copy of the right-hand-side.

        Returns
        -------

        variable
            The right-hand-side.
        """

        self._initialize()
        if copy:
            return var_copy(self._b)
        else:
            return self._b

    def _initialize(self):
        if self._b is None:
            self._b = space_new(self._space, space_type=self._space_type)

    def sub(self, b):
        """Subtract a term from the adjoint right-hand-side.

        Parameters
        ----------

        b : object
            The term to subtract.

        See also
        --------

        :func:`.subtract_adjoint_derivative_action`
        """

        if b is not None:
            self._initialize()
            subtract_adjoint_derivative_action(self._b, b)


class AdjointEquationRHS(Sequence):
    """Adjoint equation right-hand-side.

    The right-hand-side of an adjoint equation, for adjoint variables
    associated with an operation computing multiple forward variables.

    :class:`Sequence` of :class:`.AdjointRHS` objects.

    Parameters
    ----------

    eq : :class:`.Equation`
        The adjoint right-hand-sides are associated with the operation defined
        by this :class:`.Equation`.
    """

    def __init__(self, eq):
        self._B = tuple(map(AdjointRHS, eq.X()))

    def __getitem__(self, key):
        return self._B[key]

    def __len__(self):
        return len(self._B)

    def b(self, *, copy=False):
        """For the case where there is a single forward variable, return
        the associated adjoint right-hand-side.

        Parameters
        ----------

        copy : bool
            Whether to return a copy of the right-hand-side.

        Returns
        -------

        variable
            The right-hand-side.
        """

        b, = self._B
        return b.b(copy=copy)

    def B(self, *, copy=False):
        """Return adjoint right-hand-side components.

        Parameters
        ----------

        copy : bool
            Whether to return copies of the right-hand-side components.

        Returns
        -------

        tuple[variable, ...]
            The right-hand-side components.
        """

        return tuple(B.b(copy=copy) for B in self._B)


class AdjointBlockRHS(Sequence):
    """Multiple adjoint equation right-hand-sides.

    :class:`Sequence` of :class:`.AdjointEquationRHS` objects.

    Parameters
    ----------

    block : Sequence[:class:`.Equation`, ...]
        The adjoint right-hand-sides are associated with the operations defined
        by these :class:`.Equation` objects.
    """

    def __init__(self, block):
        self._block = tuple(block)
        self._B = {}
        self._k = len(self._block) - 1

    def __getitem__(self, key):
        if key not in self._B and key <= self._k:
            self._B[key] = AdjointEquationRHS(self._block[key])
        return self._B[key]

    def __len__(self):
        return len(self._block)

    def pop(self):
        """Remove and return the last :class:`.AdjointEquationRHS`. Decreases
        the length of this :class:`.AdjointBlockRHS` by one.

        Returns
        -------

        int
            The index of the removed :class:`.AdjointEquationRHS`.
        :class:`.AdjointEquationRHS`
            The removed :class:`.AdjointEquationRHS`.
        """

        if self._k < 0:
            return -1, [].pop()

        k, B = self._k, self[self._k]
        del self._B[k]
        self._k -= 1
        return k, B

    def is_empty(self):
        """Return whether this :class:`.AdjointBlockRHS` has length zero.

        Returns
        -------

        bool
            Whether this :class:`.AdjointBlockRHS` has length zero.
        """

        return self._k < 0


class AdjointModelRHS(Mapping):
    """Multiple blocks of adjoint right-hand-sides.

    :class:`Mapping` from :class:`int` to :class:`.AdjointBlockRHS` objects.

    Parameters
    ----------

    blocks : Sequence[Sequence[:class:`.Equation`, ...], ...] or \
           Mapping[int, Sequence[:class:`.Equation`, ...], ...]
       The adjoint right-hand-sides are associated with the operations defined
       by these blocks of :class:`.Equation` objects. Any trailing empty
       :class:`.AdjointBlockRHS` objects are removed automatically.
    """

    def __init__(self, blocks):
        if isinstance(blocks, Sequence):
            # Sequence
            self._blocks_n = list(range(len(blocks)))
        else:
            # Mapping
            self._blocks_n = sorted(blocks.keys())
        self._B = {n: AdjointBlockRHS(blocks[n]) for n in self._blocks_n}
        self._pop_empty()

    def __getitem__(self, key):
        return self._B[key]

    def __iter__(self):
        yield from self._B

    def __len__(self):
        return len(self._B)

    def pop(self):
        """Remove and return the last :class:`.AdjointEquationRHS` in the
        last :class:`.AdjointBlockRHS` in this :class:`.AdjointModelRHS`. Then
        remove any trailing empty :class:`.AdjointBlockRHS` objects.

        Returns
        -------

        (n, i) : tuple[int, int]
            The removed right-hand-side is associated with :class:`.Equation`
            `i` in block `n`.
        :class:`.AdjointEquationRHS`
            The removed :class:`.AdjointEquationRHS`.
        """

        n = self._blocks_n[-1]
        i, B = self._B[n].pop()
        self._pop_empty()
        return (n, i), B

    def _pop_empty(self):
        while len(self._B) > 0 and self._B[self._blocks_n[-1]].is_empty():
            del self._B[self._blocks_n.pop()]

    def is_empty(self):
        """Return whether this :class:`.AdjointModelRHS` has length zero.

        Returns
        -------

        bool
            Whether this :class:`.AdjointModelRHS` has length zero.
        """

        return len(self._B) == 0


class TransposeComputationalGraph:
    def __init__(self, Js, M, blocks, *,
                 prune_forward=True, prune_adjoint=True):
        if isinstance(blocks, Sequence):
            # Sequence
            blocks_n = tuple(range(len(blocks)))
        else:
            # Mapping
            blocks_n = tuple(sorted(blocks.keys()))

        # Transpose computational graph
        last_eq = {}
        transpose_deps = {n: {} for n in blocks_n}
        for n in blocks_n:
            block = blocks[n]
            for i, eq in enumerate(block):
                for m, x in enumerate(eq.X()):
                    last_eq[var_id(x)] = (n, i, m)
                for j, dep in enumerate(eq.dependencies()):
                    dep_id = var_id(dep)
                    if dep_id in last_eq:
                        p, k, m = last_eq[dep_id]
                        if p < n or k < i:
                            transpose_deps[(n, i, j)] = (p, k, m)
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
                    dep_map = {var_id(dep): j
                               for j, dep in enumerate(eq.dependencies())}
                    adj_ic_ids = set(map(var_id,
                                         eq.adjoint_initial_condition_dependencies()))  # noqa: E501
                    for m, x in enumerate(eq.X()):
                        adj_x_type = var_space_type(x, rel_space_type=eq.adj_X_type(m))  # noqa: E501
                        x_id = var_id(x)
                        if x_id in adj_ic_ids and x_id in last_eq:
                            adj_x_n_i_j_type, (n, i, j) = last_eq[x_id]
                            if adj_x_type == adj_x_n_i_j_type:
                                assert n > p or (n == p and i > k)
                                transpose_deps_ics[(n, i, j)] = (p, k, m)
                        last_eq[x_id] = (adj_x_type, (p, k, dep_map[x_id]))
            del last_eq

            # Pruning, forward traversal
            active_M = set(map(var_id, M))
            active_forward = {n: np.full(len(blocks[n]), False, dtype=bool)
                              for n in blocks_n}
            for n in blocks_n:
                block = blocks[n]
                for i, eq in enumerate(block):
                    if isinstance(eq, Instruction):
                        active_forward[n][i] = True
                    if len(active_M) > 0:
                        X_ids = set(map(var_id, eq.X()))
                        if not X_ids.isdisjoint(active_M):
                            active_M.difference_update(X_ids)
                            active_forward[n][i] = True
                    if not active_forward[n][i]:
                        for j, dep in enumerate(eq.dependencies()):
                            if (n, i, j) in transpose_deps_ics:
                                p, k, m = transpose_deps_ics[(n, i, j)]
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
                J_id = var_id(J)
                active_J = True
                active_adjoint = {n: np.full(len(blocks[n]), False, dtype=bool)
                                  for n in blocks_n}
                for n in reversed(blocks_n):
                    block = blocks[n]
                    for i in range(len(block) - 1, -1, -1):
                        eq = block[i]
                        if active_J:
                            for x in eq.X():
                                if var_id(x) == J_id:
                                    active_J = False
                                    active_adjoint[n][i] = True
                                    break
                        if active_adjoint[n][i]:
                            for j, dep in enumerate(eq.dependencies()):
                                if (n, i, j) in transpose_deps:
                                    p, k, m = transpose_deps[(n, i, j)]
                                    active_adjoint[p][k] = True
                        elif not isinstance(eq, Instruction):
                            active[J_i][n][i] = False

        solved = copy.deepcopy(active)

        stored_adj_ics = {J_i: {} for J_i in range(len(Js))}
        adj_ics = {J_i: {} for J_i in range(len(Js))}
        for J_i in range(len(Js)):
            last_eq = {}
            for n in blocks_n:
                block = blocks[n]
                for i, eq in enumerate(block):
                    adj_ic_ids = set(map(var_id,
                                         eq.adjoint_initial_condition_dependencies()))  # noqa: E501
                    for m, x in enumerate(eq.X()):
                        adj_x_type = var_space_type(x, rel_space_type=eq.adj_X_type(m))  # noqa: E501
                        x_id = var_id(x)
                        if x_id in last_eq:
                            adj_x_p_k_m_type, (p, k) = last_eq[x_id]
                            if adj_x_type == adj_x_p_k_m_type:
                                stored_adj_ics[J_i][(n, i, m)] = (p, k)
                        if x_id in adj_ic_ids:
                            adj_ics[J_i][x_id] = (n, i)
                            last_eq[x_id] = (adj_x_type, (n, i))
                        else:
                            adj_ics[J_i].pop(x_id, None)
                            last_eq.pop(x_id, None)
            del last_eq

        self._transpose_deps = transpose_deps
        self._active = active
        self._solved = solved
        self._stored_adj_ics = stored_adj_ics
        self._adj_ics = adj_ics

    def __contains__(self, key):
        n, i, j = key
        return (n, i, j) in self._transpose_deps

    def __getitem__(self, key):
        n, i, j = key
        p, k, m = self._transpose_deps[(n, i, j)]
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
            x_id = var_id(x)

        if x_id in self._adj_ics[J_i]:
            n, i = self._adj_ics[J_i][x_id]
            return self.is_solved(J_i, n, i)
        else:
            return False

    def is_stored_adj_ic(self, J_i, n, i, m):
        stored_adj_ics = self._stored_adj_ics[J_i].get((n, i, m), None)
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


class AdjointCache:
    def __init__(self):
        self._cache = VariableStateLockDictionary()
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
            adj_X = tuple(map(var_copy, adj_X))
        return adj_X

    def pop(self, J_i, n, i, *, copy=True):
        adj_X = self._cache.pop((J_i, n, i))
        if copy:
            adj_X = tuple(map(var_copy, adj_X))
        return adj_X

    def remove(self, J_i, n, i):
        del self._cache[(J_i, n, i)]

    def cache(self, J_i, n, i, adj_X, *, copy=True, store=False):
        if (J_i, n, i) in self._keys \
                and (store or len(self._keys[(J_i, n, i)]) > 0):
            if (J_i, n, i) in self._cache:
                adj_X = self._cache[(J_i, n, i)]
            elif copy:
                adj_X = tuple(map(var_copy, adj_X))
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
        J_root_ids = tuple(getattr(J, "_tlm_adjoint__tlm_root_id", var_id(J))
                           for J in J_roots)

        # Clear the cache if we are computing different (conjugate) derivatives
        #   J_root_ids[J_i]  The id of a functional being differentiated
        #   adj_tlm_key      Defines a (conjugate) derivative of the functional
        #                    computed by the adjoint
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

            eqs = defaultdict(list)
            for n in reversed(blocks_n):
                block = blocks[n]
                for i in range(len(block) - 1, -1, -1):
                    eq = block[i]

                    if isinstance(eq, (ControlsMarker, FunctionalMarker)):
                        continue

                    eq_id = eq.id
                    eq_tlm_root_id = getattr(eq, "_tlm_adjoint__tlm_root_id", eq_id)  # noqa: E501
                    eq_tlm_key = getattr(eq, "_tlm_adjoint__tlm_key", ())

                    # The root (forward) equation eqs[eq_tlm_root_id] always
                    # appears first on the tape. We go through equations in
                    # reverse order and build a list of the (conjugate)
                    # derivatives stored by adjoint variables.
                    for J_i, adj_tlm_key in tlm_adj.get(eq_tlm_key, ()):
                        if transpose_deps.is_solved(J_i, n, i) \
                                or (J_i, n, i) in self._cache:
                            assert cache_degree is None or len(adj_tlm_key) < cache_degree  # noqa: E501
                            eqs[eq_tlm_root_id].append(
                                ((J_i, n, i),
                                 (J_root_ids[J_i], adj_tlm_key)))

                    # When we reach the root (forward) equation we can now
                    # build a map between adjoint variables which store the
                    # same (conjugate) derivatives
                    eq_root = {}
                    for (J_j, p, k), adj_key in eqs.pop(eq_id, []):
                        assert transpose_deps.is_solved(J_j, p, k) or (J_j, p, k) in self._cache  # noqa: E501
                        if adj_key in eq_root:
                            self._keys[eq_root[adj_key]].append((J_j, p, k))
                            if (J_j, p, k) in self._cache \
                                    and eq_root[adj_key] not in self._cache:
                                # Corner case: A value is already cached in a
                                # non-root adjoint variable
                                self._cache[eq_root[adj_key]] = self._cache[(J_j, p, k)]  # noqa: E501
                        else:
                            # Now one of the adjoint variables is marked as the
                            # 'root', and all others reuse its value
                            eq_root[adj_key] = (J_j, p, k)
                            self._keys[eq_root[adj_key]] = []
            assert len(eqs) == 0

        for (J_i, n, i) in self._cache:
            transpose_deps.set_not_solved(J_i, n, i)
        for eq_root in self._keys:
            for (J_i, n, i) in self._keys[eq_root]:
                transpose_deps.set_not_solved(J_i, n, i)
