from .interface import (
    check_space_types, is_var, packed, var_id, var_is_replacement, var_name,
    var_new_tangent_linear)

from .alias import gc_disabled
from .markers import ControlsMarker, FunctionalMarker

from collections import defaultdict
from collections.abc import Sequence
import itertools
from operator import itemgetter
import weakref

__all__ = \
    [
        "TangentLinearMap"
    ]


def tlm_key(M, dM):
    M = packed(M)
    dM = packed(dM)
    if any(map(var_is_replacement, M)):
        raise ValueError("Invalid tangent-linear")
    if any(map(var_is_replacement, dM)):
        raise ValueError("Invalid tangent-linear")

    if len(set(M)) != len(M):
        raise ValueError("Invalid tangent-linear")
    if len(M) != len(dM):
        raise ValueError("Invalid tangent-linear")

    for m, dm in zip(M, dM):
        check_space_types(m, dm)

    return ((M, dM), (tuple(map(var_id, M)), tuple(map(var_id, dM))))


def distinct_combinations_indices(iterable, r):
    class Comparison:
        def __init__(self, key, value):
            self._key = key
            self._value = value

        def __eq__(self, other):
            return self._key == other._key

        def __hash__(self):
            return hash(self._key)

        @property
        def value(self):
            return self._value

    t = tuple(Comparison(value, i) for i, value in enumerate(iterable))

    try:
        import more_itertools
    except ModuleNotFoundError:
        # Basic implementation likely suffices for most cases in practice
        seen = set()
        for combination in itertools.combinations(t, r):
            if combination not in seen:
                seen.add(combination)
                yield tuple(e.value for e in combination)
        return

    for combination in more_itertools.distinct_combinations(t, r):
        yield tuple(e.value for e in combination)


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

    where `x` is a forward variable.

        - If `x` defines a component of the control, then `tau_x` is a variable
          defining the associated component of the direction.
        - If `x` does not define a component of the control and is not
          'static', then `tau_x` is a tangent-linear variable. A new variable
          is instantiated if needed.
        - Otherwise `tau_x` is `None`, indicating that the tangent-linear
          variable is zero.

    Containment can be tested

    .. code-block:: python

        if x in tlm_map:
            [...]

    and returns `True` if `x` defines a component of the control, or a
    tangent-linear variable associated with `x` has been instantiated.

    :arg M: A variable or :class:`Sequence` of variables defining the control.
    :arg dM: A variable or :class:`Sequence` of variables defining the
        derivative direction. The tangent-linear computes directional
        derivatives with respect to the control defined by `M` and with
        direction defined by `dM`.
    """

    _id_counter = itertools.count()

    def __init__(self, M, dM):
        (M, dM), _ = tlm_key(M, dM)

        self._id = next(self._id_counter)

        self._X = weakref.WeakValueDictionary()
        self._M = M
        self._dM = dM

        @gc_disabled
        def finalize_callback(X, tlm_map_id):
            for x_id in sorted(tuple(X)):
                x = X.get(x_id, None)
                if x is not None:
                    getattr(x, "_tlm_adjoint__tangent_linears", {}).pop(tlm_map_id, None)  # noqa: E501

        weakref.finalize(self, finalize_callback,
                         self._X, self._id)

        if len(M) == 1:
            self._name_suffix = \
                "_tlm(%s,%s)" % (var_name(M[0]),  # noqa: UP031
                                 var_name(dM[0]))
        else:
            self._name_suffix = \
                "_tlm((%s),(%s))" % (",".join(map(var_name, M)),  # noqa: UP031
                                     ",".join(map(var_name, dM)))

        assert len(M) == len(dM)
        for m, dm in zip(M, dM):
            if not hasattr(m, "_tlm_adjoint__tangent_linears"):
                self._X[var_id(m)] = m
                m._tlm_adjoint__tangent_linears = {}
            # Do not set _tlm_adjoint__tlm_root_id, as dm cannot appear as the
            # solution to an Equation
            m._tlm_adjoint__tangent_linears[self.id] = dm

    def __contains__(self, x):
        if not is_var(x):
            raise TypeError("x must be a variable")
        if var_is_replacement(x):
            raise ValueError("x cannot be a replacement")

        return self.id in getattr(x, "_tlm_adjoint__tangent_linears", {})

    def __getitem__(self, x):
        if not is_var(x):
            raise TypeError("x must be a variable")
        if var_is_replacement(x):
            raise ValueError("x cannot be a replacement")

        if not hasattr(x, "_tlm_adjoint__tangent_linears"):
            self._X[var_id(x)] = x
            x._tlm_adjoint__tangent_linears = {}
        if self.id not in x._tlm_adjoint__tangent_linears:
            tau_x = var_new_tangent_linear(
                x, name=f"{var_name(x):s}{self._name_suffix:s}")
            if tau_x is not None:
                tau_x._tlm_adjoint__tlm_root_id = getattr(
                    x, "_tlm_adjoint__tlm_root_id", var_id(x))
            x._tlm_adjoint__tangent_linears[self.id] = tau_x

        return x._tlm_adjoint__tangent_linears[self.id]

    @property
    def id(self):
        """A unique :class:`int` ID associated with this
        :class:`.TangentLinearMap`.
        """

        return self._id

    @property
    def M(self):
        """A :class:`Sequence` of variables defining the control.
        """

        return self._M

    @property
    def dM(self):
        """A :class:`Sequence` of variables defining the derivative direction.
        """

        return self._dM


def J_tangent_linears(Js, blocks, *, max_adjoint_degree=None):
    if isinstance(blocks, Sequence):
        # Sequence
        blocks_n = tuple(range(len(blocks)))
    else:
        # Mapping
        blocks_n = tuple(sorted(blocks.keys()))

    J_is = {var_id(J): J_i for J_i, J in enumerate(Js)}
    J_roots = list(Js)
    J_root_ids = {J_id: J_id for J_id in map(var_id, Js)}
    remaining_Js = dict(enumerate(Js))
    tlm_adj = defaultdict(list)

    for n in reversed(blocks_n):
        block = blocks[n]
        for i in range(len(block) - 1, -1, -1):
            eq = block[i]

            if isinstance(eq, ControlsMarker):
                continue
            elif isinstance(eq, FunctionalMarker):
                # When we run the adjoint we compute the derivative of a
                # 'marker' variable, which appears as a solution in the
                # functional block. Here we need to find the original 'root'
                # variable which actually stores the value of the functional.
                J, J_root = eq.dependencies()
                J_id = var_id(J)
                if J_id in J_root_ids:
                    assert J_root_ids[J_id] == J_id
                    J_roots[J_is[J_id]] = J_root
                    J_root_ids[J_id] = var_id(J_root)
                    assert J_root_ids[J_id] != J_id
                del J, J_root, J_id
                continue

            eq_X_ids = set(map(var_id, eq.X()))
            eq_tlm_key = getattr(eq, "_tlm_adjoint__tlm_key", ())

            # For an operation computing a forward or tangent-linear variable,
            # computing derivatives defined by 'tlm_key' (which may be empty),
            # here we identify the (conjugate) derivatives stored by all
            # associated adjoint variables. These are (conjugate) derivatives
            # of the functional with index J_i and with directions defined by
            # adj_tlm_key.
            found_Js = []
            for J_i, J in remaining_Js.items():
                if J_root_ids[var_id(J)] in eq_X_ids:
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
