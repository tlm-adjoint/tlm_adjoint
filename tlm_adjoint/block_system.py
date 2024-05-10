from .interface import (
    DEFAULT_COMM, space_default_space_type, space_id, space_new, var_comm,
    var_get_values, var_global_size, var_local_size, var_set_values)


from collections import deque
from collections.abc import Sequence
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None


def iter_sub(iterable, *, expand=None):
    if expand is None:
        def expand(e):
            return e

    q = deque(map(expand, iterable))
    while len(q) > 0:
        e = q.popleft()
        if isinstance(e, Sequence) and not isinstance(e, str):
            q.extendleft(map(expand, reversed(e)))
        else:
            yield e


def tuple_sub(iterable, sequence):
    iterator = iter_sub(iterable)

    def tuple_sub(iterator, value):
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(tuple_sub(iterator, e) for e in value)
        return next(iterator)

    t = tuple_sub(iterator, sequence)

    try:
        next(iterator)
        raise ValueError("Non-equal lengths")
    except StopIteration:
        pass

    return t


def zip_sub(*iterables):
    iterators = tuple(map(iter_sub, iterables))
    yield from zip(*iterators)

    for iterator in iterators:
        try:
            next(iterator)
            raise ValueError("Non-equal lengths")
        except StopIteration:
            pass


class MixedSpace(Sequence):
    """Used to map between different versions of a mixed space.

    This class defines two representations for the space:

        1. As a 'split space': A tree defining the mixed space. Stored using
           space and :class:`tuple` objects, each corresponding to a node in
           the tree. Spaces correspond to leaf nodes, and :class:`tuple`
           objects to other nodes in the tree.
        2. As a 'flattened space': A :class:`Sequence` containing leaf nodes of
           the split space with an ordering determined using a depth first
           search.

    Provides methods to allow data to be copied to and from a compatible
    :class:`petsc4py.PETSc.Vec`. This allows, for example, the construction
    (with Firedrake):

    .. code-block:: python

        u_0 = Function(space_0, name='u_0')
        u_1 = Function(space_1, name='u_1')
        u_2 = Function(space_2, name='u_2')

        mixed_space = MixedSpace(((space_0, space_1), space_2))

    and then data can be copied to a compatible :class:`petsc4py.PETSc.Vec` via

    .. code-block:: python

        mixed_space.to_petsc(u_petsc, ((u_0, u_1), u_2))

    and from a compatible :class:`petsc4py.PETSc.Vec` via

    .. code-block:: python

        mixed_space.from_petsc(u_petsc, ((u_0, u_1), u_2))

    :arg spaces: The split space.
    """

    def __init__(self, spaces, *, space_types=None):
        if isinstance(spaces, MixedSpace):
            spaces = spaces.split_space
        elif isinstance(spaces, Sequence):
            spaces = tuple(spaces)
        else:
            spaces = (spaces,)
        spaces = tuple_sub(spaces, spaces)
        flattened_spaces = tuple(iter_sub(spaces))

        if space_types is None:
            space_types = tuple(map(space_default_space_type,
                                    flattened_spaces))
        elif space_types in ["primal", "conjugate", "dual", "conjugate_dual"]:
            space_types = tuple(space_types for _ in flattened_spaces)
        space_types = tuple(iter_sub(space_types))
        if len(space_types) != len(flattened_spaces):
            raise ValueError("Invalid space types")
        for space_type in space_types:
            if space_type not in {"primal", "conjugate",
                                  "dual", "conjugate_dual"}:
                raise ValueError("Invalid space types")

        comm = None
        indices = []
        n = 0
        N = 0
        for space, space_type in zip(flattened_spaces, space_types):
            u_i = space_new(space, space_type=space_type)
            if comm is None:
                comm = var_comm(u_i)
            indices.append((n, n + var_local_size(u_i)))
            n += var_local_size(u_i)
            N += var_global_size(u_i)
        if comm is None:
            comm = DEFAULT_COMM if MPI is None else MPI.COMM_SELF

        self._spaces = spaces
        self._flattened_spaces = flattened_spaces
        self._space_types = space_types
        self._comm = comm
        self._indices = indices
        self._n = n
        self._N = N

    def __len__(self):
        return len(self.split_space)

    def __getitem__(self, key):
        if len(self) == 1:
            return (self,)[key]
        else:
            return MixedSpace(self.split_space[key])

    def __eq__(self, other):
        self_ids = self.tuple_sub(map(space_id, self.flattened_space))
        other_ids = other.tuple_sub(map(space_id, other.flattened_space))
        return (self_ids == other_ids
                and self.space_types == other.space_types)

    @property
    def comm(self):
        """The communicator associated with the mixed space.
        """

        return self._comm

    @property
    def split_space(self):
        """The split space representation.
        """

        return self._spaces

    @property
    def flattened_space(self):
        """The flattened space representation.
        """

        return self._flattened_spaces

    @property
    def space_types(self):
        """Space types for the flattened space representation.
        """

        return self._space_types

    def tuple_sub(self, u):
        """
        :arg u: An :class:`Iterable`.
        :returns: A :class:`tuple` storing elements in `u` using the tree
            structure of the split space.
        """

        return tuple_sub(u, self.split_space)

    def new_split(self):
        """
        :returns: A new element in the split space.
        """

        u = []
        for space, space_type in zip(self.flattened_space, self.space_types):
            u.append(space_new(space, space_type=space_type))
        return self.tuple_sub(u)

    @property
    def local_size(self):
        """The number of local degrees of freedom.
        """

        return self._n

    @property
    def global_size(self):
        """The global number of degrees of freedom.
        """

        return self._N

    def from_petsc(self, u_petsc, u):
        """Copy data from a compatible :class:`petsc4py.PETSc.Vec`.

        :arg u_petsc: The :class:`petsc4py.PETSc.Vec`.
        :arg u: An element of the split space.
        """

        if PETSc is None:
            raise RuntimeError("PETSc not available")

        u_a = u_petsc.getArray(True)

        if not np.can_cast(u_a, PETSc.ScalarType):
            raise ValueError("Invalid dtype")
        if len(u_a.shape) != 1:
            raise ValueError("Invalid shape")

        i0 = 0
        for j, u_i in zip_sub(range(len(self._indices)), u):
            i1 = i0 + var_local_size(u_i)
            if i1 > u_a.shape[0]:
                raise ValueError("Invalid shape")
            if (i0, i1) != self._indices[j]:
                raise ValueError("Invalid shape")
            var_set_values(u_i, u_a[i0:i1])
            i0 = i1
        if i0 != u_a.shape[0]:
            raise ValueError("Invalid shape")

    def to_petsc(self, u_petsc, u):
        """Copy data to a compatible :class:`petsc4py.PETSc.Vec`. Does not
        update the ghost.

        :arg u_petsc: The :class:`petsc4py.PETSc.Vec`.
        :arg u: An element of the split space.
        """

        if PETSc is None:
            raise RuntimeError("PETSc not available")

        u_a = np.zeros(self.local_size, dtype=PETSc.ScalarType)

        i0 = 0
        for j, u_i in zip_sub(range(len(self._indices)), iter_sub(u)):
            u_i_a = var_get_values(u_i)

            if not np.can_cast(u_i_a, PETSc.ScalarType):
                raise ValueError("Invalid dtype")
            if len(u_i_a.shape) != 1:
                raise ValueError("Invalid shape")

            i1 = i0 + u_i_a.shape[0]
            if i1 > u_a.shape[0]:
                raise ValueError("Invalid shape")
            if (i0, i1) != self._indices[j]:
                raise ValueError("Invalid shape")
            u_a[i0:i1] = u_i_a
            i0 = i1
        if i0 != u_a.shape[0]:
            raise ValueError("Invalid shape")

        u_petsc.setArray(u_a)
