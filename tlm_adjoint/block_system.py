from .interface import (
    DEFAULT_COMM, space_default_space_type, space_comm, space_id, space_new)
from .petsc import PETScVecInterface

from collections import deque
from collections.abc import Sequence
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
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


class MixedSpace(PETScVecInterface, Sequence):
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

        if len(flattened_spaces) > 0:
            comm = space_comm(flattened_spaces[0])
        elif MPI is None:
            comm = DEFAULT_COMM
        else:
            comm = MPI.COMM_SELF

        super().__init__(
            tuple(space_new(space, space_type=space_type)
                  for space, space_type in zip(flattened_spaces, space_types)))
        self._spaces = spaces
        self._flattened_spaces = flattened_spaces
        self._space_types = space_types
        self._comm = comm

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

    def from_petsc(self, y, X):
        super().from_petsc(y, tuple(iter_sub(X)))

    def to_petsc(self, x, Y):
        super().to_petsc(x, tuple(iter_sub(Y)))
