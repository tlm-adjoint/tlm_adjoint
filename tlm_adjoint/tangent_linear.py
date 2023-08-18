#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import check_space_types, function_id, function_name, \
    function_new_tangent_linear, is_function

from .alias import gc_disabled
from .markers import ControlsMarker, FunctionalMarker

from collections import defaultdict
from collections.abc import Sequence
import itertools
from operator import itemgetter
import weakref

__all__ = \
    [
        "TangentLinearMap",
        "get_tangent_linear"
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

    if len(set(M)) != len(M):
        raise ValueError("Invalid tangent-linear model")
    if len(M) != len(dM):
        raise ValueError("Invalid tangent-linear model")
    for m, dm in zip(M, dM):
        check_space_types(m, dm)

    return ((M, dM),
            (tuple(function_id(m) for m in M),
             tuple(function_id(dm) for dm in dM)))


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
    """Defines a map from forward variables to associated tangent-linear
    variables.

    The map is used via e.g.

    .. code-block:: python

        tau_x = tlm_map[x]

    where `x` is a function associated with the forward variable. If the
    function `x` is 'checkpointed', meaning that it is stored by value by a
    :class:`tlm_adjoint.checkpointing.CheckpointStorage`, then `tau_x` is a
    function associated with the tangent-linear variable -- a new function is
    instantiated if needed. If the function `x` is not 'checkpointed', meaning
    that it is stored by reference by a
    :class:`tlm_adjoint.checkpointing.CheckpointStorage`, then `tau_x` is
    `None`.

    Containment can also be tested

    .. code-block:: python

        if x in tlm_map:
            [...]

    and returns `True` if a tangent-linear function associated with `x` has
    been instantiated.

    Note that a :class:`TangentLinearMap` should not be used to map from the
    control `M` to the derivative direction `dM`. Typically a
    :class:`TangentLinearMap` should not be used directly, and instead
    :func:`get_tangent_linear` should be used, which *does* map from the
    control to the direction.

    :arg M: A function or :class:`Sequence` of functions defining the control.
    :arg dM: A function or :class:`Sequence` of functions defining the
        derivative direction. The tangent-linear model computes directional
        derivatives with respect to the control defined by `M` and with
        direction defined by `dM`.
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


def get_tangent_linear(x, M, dM, tlm_map):
    """Return a tangent-linear variable associated with a variable `x`.

    This function should be used in place of accessing via the
    :class:`TangentLinearMap`, if the variable `x` may be a control variable.

    :arg x: A function defining the variable for which a tangent-linear
        variable should be returned.
    :arg M: A :class:`Sequence` of functions defining the control.
    :arg dM: A :class:`Sequence` of functions defining the derivative
        direction. The tangent-linear model computes directional derivatives
        with respect to the control defined by `M` and with direction defined
        by `dM`.
    :arg tlm_map: A :class:`TangentLinearMap` storing values for tangent-linear
        variables.
    :returns: If `x` is a control variable then returns the associated
        direction. If `x` is not a control variable then returns a function
        corresponding to a tangent-linear variable if `x` is 'checkpointed'
        (i.e. stored by value by a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage`), and `None`
        otherwise.
    """

    if x in M:
        return dM[M.index(x)]
    else:
        return tlm_map[x]


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
