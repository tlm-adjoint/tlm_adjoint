r"""Solvers for linear systems defined in mixed spaces.

Given a linear problem with a potentially singular matrix :math:`A`

.. math::

    A u = b,

a :class:`.LinearSolver` instead solves the linear problem

.. math::

    \left[ (I - M U (U^* M U)^{-1} U^*) A (I - V (V^* C V)^{-1} V^* C)
        + M U S V^* C \right] u = (I - M U (U^* M U)^{-1} U^*) b.

Here

    - :math:`U` is a full rank matrix whose columns span the left nullspace for
      a modified system matrix :math:`\tilde{A}`.
    - :math:`V` is a full rank matrix with the same number of columns as
      :math:`U`, whose columns span the nullspace for :math:`\tilde{A}`.
    - :math:`V^* C V` and :math:`S` are invertible matrices.
    - :math:`M` is a Hermitian positive definite matrix.

Here the left nullspace for a matrix is defined to be the nullspace for its
Hermitian transpose, and the modified system matrix :math:`\tilde{A}` is
defined

.. math::

    \tilde{A} = (I - M U (U^* M U)^{-1} U^*) A (I - V (V^* C V)^{-1} V^* C).

This has two primary use cases:

    1. Where a matrix :math:`A` and right-hand-side :math:`b` are constructed
       via finite element assembly on superspaces of the test space and trial
       space. The typical example is in the application of homogeneous
       essential Dirichlet boundary conditions.

    2. Where the matrix :math:`A` is singular and :math:`b` is orthogonal to
       the left nullspace of :math:`A`. Typically one would then choose
       :math:`U` and :math:`V` so that their columns respectively span the left
       nullspace and nullspace of :math:`A`, and the :class:`.LinearSolver`
       then seeks a solution to the original problem subject to the linear
       constraints :math:`V^* C u = 0`.

Spaces are defined via backend spaces or :class:`.TypedSpace` objects, and
:class:`Sequence` objects containing backend spaces, :class:`.TypedSpace`
objects, or similar :class:`Sequence` objects. Similarly variables are defined
via backend variables, or :class:`Sequence` objects containing backend
variables, or similar :class:`Sequence` objects. This defines a basic tree
structure which is useful e.g. when defining block matrices in terms of
sub-block matrices.

Elements of the tree are accessed in a consistent order using a depth first
search. Hence e.g.

.. code-block:: python

    ((u_0, u_1), u_2)

and

.. code-block:: python

    (u_0, u_1, u_2)

where `u_0`, `u_1`, and `u_2` are backend variables, are both valid
representations of a mixed space solution.
"""

from .interface import (
    Packed, comm_dup_cached, packed, paused_space_type_checking, space_comm,
    space_default_space_type, space_eq, space_new, var_assign, var_axpy,
    var_locked, var_zero)
from .manager import manager_disabled
from .petsc import (
    PETScOptions, PETScVec, PETScVecInterface, attach_destroy_finalizer)

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from collections import deque
import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None
try:
    import slepc4py.SLEPc as SLEPc
except ModuleNotFoundError:
    SLEPc = None


__all__ = \
    [
        "TypedSpace",
        "MixedSpace",

        "Nullspace",
        "NoneNullspace",
        "BlockNullspace",

        "Matrix",
        "MatrixFreeMatrix",
        "BlockMatrix",

        "LinearSolver",
        "Eigensolver",
        "MatrixFunctionSolver"
    ]


def iter_sub(iterable, *, expand=None):
    if expand is None:
        def expand(e):
            return e

    q = deque(map(expand, iterable))
    while len(q) > 0:
        e = Packed(q.popleft())
        if e.is_packed:
            e, = e
            yield e
        else:
            q.extendleft(map(expand, reversed(e)))


def tuple_sub(iterable, sequence):
    iterator = iter_sub(iterable)

    def tuple_sub(iterator, value):
        value = Packed(value)
        if value.is_packed:
            return next(iterator)
        else:
            return tuple(tuple_sub(iterator, e) for e in value)

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


class TypedSpace:
    """A space with an associated space type.

    :arg space: The space.
    :arg space_types: The space type.
    """

    def __init__(self, space, *, space_type=None):
        if isinstance(space, TypedSpace):
            if space_type is None:
                space_type = space.space_type
            space = space.space

        if space_type is None:
            space_type = space_default_space_type(space)
        if space_type not in {"primal", "conjugate",
                              "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")

        self._space = space
        self._space_type = space_type

    def __eq__(self, other):
        return (isinstance(other, TypedSpace)
                and self.space_type == other.space_type
                and space_eq(self.space, other.space))

    def __ne__(self, other):
        return not (self == other)

    @property
    def comm(self):
        """The communicator associated with the space.
        """

        return space_comm(self.space)

    @property
    def space(self):
        """The backend space.
        """

        return self._space

    @property
    def space_type(self):
        """The space type.
        """

        return self._space_type

    def new(self):
        """Return a new variable in the space.
        """

        return space_new(self.space, space_type=self.space_type)


class MixedSpace(PETScVecInterface, Sequence):
    """Used to map between different versions of a mixed space.

    This class defines two representations for the space:

        1. As a 'split space': A tree defining the mixed space. Stored using
           :class:`.TypedSpace` and :class:`tuple` objects, each corresponding
           to a node in the tree. :class:`.TypedSpace` objects correspond to
           leaf nodes, and :class:`tuple` objects to other nodes in the tree.
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

    :arg spaces: Defines the split space.
    """

    def __init__(self, spaces):
        if isinstance(spaces, MixedSpace):
            spaces = spaces.split_space
        else:
            spaces = packed(spaces)
        flattened_spaces = tuple(space if isinstance(space, TypedSpace)
                                 else TypedSpace(space)
                                 for space in iter_sub(spaces))
        spaces = tuple_sub(flattened_spaces, spaces)

        super().__init__(tuple(space.space for space in flattened_spaces))
        self._spaces = spaces
        self._flattened_spaces = flattened_spaces

    def __len__(self):
        return len(self.split_space)

    def __getitem__(self, key):
        space = self.split_space[key]
        if isinstance(space, TypedSpace):
            return space
        else:
            return MixedSpace(space)

    def __eq__(self, other):
        if not isinstance(other, MixedSpace):
            return False
        self_shape = self.tuple_sub(None for _ in self.flattened_space)
        other_shape = other.tuple_sub(None for _ in other.flattened_space)
        if self_shape != other_shape:
            return False
        assert len(self.flattened_space) == len(other.flattened_space)
        for self_space, other_space in zip(self.flattened_space,
                                           other.flattened_space):
            if self_space != other_space:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

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

    def tuple_sub(self, u):
        """
        :arg u: An :class:`Iterable`.
        :returns: A :class:`tuple` storing elements in `u` using the tree
            structure of the split space.
        """

        return tuple_sub(u, self.split_space)

    def new(self):
        """
        :returns: A new element in the split space.
        """

        u = tuple(space.new() for space in self.flattened_space)
        return self.tuple_sub(u)

    def from_petsc(self, y, X):
        super().from_petsc(y, tuple(iter_sub(X)))

    def to_petsc(self, x, Y):
        super().to_petsc(x, tuple(iter_sub(Y)))


class Nullspace(ABC):
    """Represents a nullspace and left nullspace for a square matrix.
    """

    @abstractmethod
    def apply_nullspace_transformation_lhs_right(self, x):
        r"""Apply the nullspace transformation associated with a matrix action
        on :math:`x`,

        .. math::

            x \rightarrow (I - V (V^* C V)^{-1} V^* C) x.

        :arg x: Defines :math:`x`.
        """

        raise NotImplementedError

    @abstractmethod
    def apply_nullspace_transformation_lhs_left(self, y):
        r"""Apply the left nullspace transformation associated with a matrix
        action,

        .. math::

            y \rightarrow (I - M U (U^* M U)^{-1} U^*) y.

        :arg y: Defines :math:`y`.
        """

        raise NotImplementedError

    @abstractmethod
    def constraint_correct_lhs(self, x, y):
        r"""Add the linear constraint term to :math:`y`,

        .. math::

            y \rightarrow y + M U S V^* C x.

        :arg x: Defines :math:`x`.
        :arg y: Defines :math:`y`.
        """

        raise NotImplementedError

    @abstractmethod
    def pc_constraint_correct_soln(self, u, b):
        r"""Add the preconditioner linear constraint term to :math:`u`,

        .. math::

            u \rightarrow u + V \tilde{S}^{-1} U^* b,

        with

        .. math::

            \tilde{S}^{-1} =
                \left( V^* C V \right)^{-1}
                S^{-1}
                \left( U^* M U \right)^{-1}.

        :arg u: Defines :math:`u`.
        :arg b: Defines :math:`b`.
        """

        raise NotImplementedError

    def correct_soln(self, x):
        """Correct the linear system solution so that it is orthogonal to
        space spanned by the columns of :math:`V`.

        :arg x: The linear system solution, to be corrected.
        """

        self.apply_nullspace_transformation_lhs_right(x)

    def pre_mult_correct_lhs(self, x):
        """Apply the pre-left-multiplication nullspace transformation.

        :arg x: Defines the vector on which the matrix action is computed.
        """

        self.apply_nullspace_transformation_lhs_right(x)

    def post_mult_correct_lhs(self, x, y):
        """Apply the post-left-multiplication nullspace transformation, and add
        the linear constraint term.

        :arg x: Defines the vector on which the matrix action is computed, and
            used to add the linear constraint term. If `None` is supplied then
            the linear constraint term is not added.
        :arg y: Defines the result of the matrix action on `x`.
        """

        self.apply_nullspace_transformation_lhs_left(y)
        if x is not None:
            self.constraint_correct_lhs(x, y)

    def correct_rhs(self, b):
        """Correct the linear system right-hand-side so that it is orthogonal
        to the space spanned by the columns of :math:`U`.

        :arg b: The linear system right-hand-side, to be corrected.
        """

        self.apply_nullspace_transformation_lhs_left(b)

    def pc_pre_mult_correct(self, b):
        """Apply the pre-preconditioner-application nullspace transformation.

        :arg b: Defines the vector on which the preconditioner action is
            computed.
        """

        self.apply_nullspace_transformation_lhs_left(b)

    def pc_post_mult_correct(self, u, b):
        """Apply the post-preconditioner-application left nullspace
        transformation, and add the linear constraint term.

        :arg u: Defines the result of the preconditioner action on `b`.
        :arg b: Defines the vector on which the preconditioner action is
            computed, and used to add the linear constraint term. If `None` is
            supplied then the linear constraint term is not added.
        """

        self.apply_nullspace_transformation_lhs_right(u)
        if b is not None:
            self.pc_constraint_correct_soln(u, b)


class NoneNullspace(Nullspace):
    """An empty nullspace and left nullspace.
    """

    def apply_nullspace_transformation_lhs_right(self, x):
        pass

    def apply_nullspace_transformation_lhs_left(self, y):
        pass

    def constraint_correct_lhs(self, x, y):
        pass

    def pc_constraint_correct_soln(self, u, b):
        pass


class BlockNullspace(Nullspace, Sequence):
    """Nullspaces for a square :class:`.BlockMatrix`.

    :arg nullspaces: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace. `None` indicates a
        :class:`.NoneNullspace`.
    """

    def __init__(self, nullspaces):
        nullspaces = list(packed(nullspaces))
        for i, nullspace in enumerate(nullspaces):
            if nullspace is None:
                nullspaces[i] = NoneNullspace()
        nullspaces = tuple(nullspaces)

        super().__init__()
        self._nullspaces = nullspaces

    def __new__(cls, nullspaces, *args, **kwargs):
        nullspaces = packed(nullspaces)
        for nullspace in nullspaces:
            if nullspace is not None \
                    and not isinstance(nullspace, NoneNullspace):
                break
        else:
            return NoneNullspace()
        return super().__new__(cls)

    def __getitem__(self, key):
        return self._nullspaces[key]

    def __len__(self):
        return len(self._nullspaces)

    def apply_nullspace_transformation_lhs_right(self, x):
        if len(x) != len(self):
            raise ValueError("Invalid x")
        for nullspace, x_i in zip(self, x):
            nullspace.apply_nullspace_transformation_lhs_right(x_i)

    def apply_nullspace_transformation_lhs_left(self, y):
        if len(y) != len(self):
            raise ValueError("Invalid y")
        for nullspace, y_i in zip(self, y):
            nullspace.apply_nullspace_transformation_lhs_left(y_i)

    def constraint_correct_lhs(self, x, y):
        if len(x) != len(self):
            raise ValueError("Invalid x")
        if len(y) != len(self):
            raise ValueError("Invalid y")
        with var_locked(*iter_sub(x)):
            for nullspace, x_i, y_i in zip(self, x, y):
                nullspace.constraint_correct_lhs(x_i, y_i)

    def pc_constraint_correct_soln(self, u, b):
        if len(u) != len(self):
            raise ValueError("Invalid u")
        if len(b) != len(self):
            raise ValueError("Invalid b")
        with var_locked(*iter_sub(b)):
            for nullspace, u_i, b_i in zip(self, u, b):
                nullspace.pc_constraint_correct_soln(u_i, b_i)


class Matrix:
    r"""Represents a matrix defining a mapping
    :math:`A` mapping :math:`V \rightarrow W`.

    :arg arg_space: Defines the space :math:`V`.
    :arg action_space: Defines the space :math:`W`.
    """

    def __init__(self, arg_space, action_space):
        if not isinstance(arg_space, (TypedSpace, MixedSpace)):
            if isinstance(arg_space, Sequence):
                arg_space = MixedSpace(arg_space)
            else:
                arg_space = TypedSpace(arg_space)
        if not isinstance(action_space, (TypedSpace, MixedSpace)):
            if isinstance(action_space, Sequence):
                action_space = MixedSpace(action_space)
            else:
                action_space = TypedSpace(action_space)

        self._arg_space = arg_space
        self._action_space = action_space

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.mult is Matrix.mult and cls.mult_add is Matrix.mult_add:
            raise RuntimeError("Must override at least one of mult or "
                               "mult_add")

    @property
    def arg_space(self):
        """The space defining :math:`V`.
        """

        return self._arg_space

    @property
    def action_space(self):
        """The space defining :math:`W`.
        """

        return self._action_space

    def mult(self, x, y):
        """Compute :math:`y = A x`.

        :arg x: Defines :math:`x`. Should not be modified.
        :arg y: Defines :math:`y`.
        """

        x = Packed(x)
        y = Packed(y)

        for y_i in iter_sub(y):
            var_zero(y_i)
        with var_locked(*iter_sub(x)):
            self.mult_add(x.unpack(x), y.unpack(y))

    def mult_add(self, x, y):
        """Add :math:`A x` to :math:`y`.

        :arg x: Defines :math:`x`. Should not be modified.
        :arg y: Defines :math:`y`.
        """

        x = Packed(x)
        y = Packed(y)
        y_term = packed(self.action_space.new())

        with var_locked(*iter_sub(x)):
            self.mult(x.unpack(x), y.unpack(y_term))
        for y_i, y_term_i in zip_sub(y, y_term):
            var_axpy(y_i, 1.0, y_term_i)


class MatrixFreeMatrix(Matrix):
    def __init__(self, arg_space, action_space, mult):
        super().__init__(arg_space, action_space)
        self._mult = mult

    def mult(self, x, y):
        with var_locked(*iter_sub(packed(x))):
            self._mult(x, y)


class BlockMatrix(Matrix, MutableMapping):
    r"""A matrix defining a mapping :math:`A` mapping :math:`V \rightarrow W`,
    where :math:`V` and :math:`W` are defined by mixed spaces.

    :arg arg_space: Defines the space :math:`V`.
    :arg action_space: Defines the space :math:`W`.
    :arg block: A :class:`Mapping` defining the blocks of the matrix. Items are
        `((i, j), block)` where the block in the `i` th and `j` th column is
        defined by `block`. Each `block` is a
        :class:`tlm_adjoint.block_system.Matrix`, or `None` to indicate a zero
        block.
    """

    def __init__(self, arg_space, action_space, blocks=None):
        if not isinstance(arg_space, MixedSpace):
            arg_space = MixedSpace(arg_space)
        if not isinstance(action_space, MixedSpace):
            action_space = MixedSpace(action_space)
        if not isinstance(blocks, Mapping):
            blocks = {(0, 0): blocks}

        super().__init__(arg_space, action_space)
        self._blocks = {}

        if blocks is not None:
            self.update(blocks)

    def __iter__(self):
        yield from sorted(self._blocks)

    def __getitem__(self, key):
        i, j = key
        return self._blocks[(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        if value is None:
            self.arg_space[j], self.action_space[i]
            self.pop((i, j), None)
        else:
            if value.arg_space != self.arg_space[j]:
                raise ValueError("Invalid space")
            if value.action_space != self.action_space[i]:
                raise ValueError("Invalid space")
            self._blocks[(i, j)] = value

    def __delitem__(self, key):
        i, j = key
        del self._blocks[(i, j)]

    def __len__(self):
        return len(self._blocks)

    def mult_add(self, x, y):
        with var_locked(*iter_sub(x)):
            for (i, j), block in self.items():
                block.mult_add(x[j], y[i])


class PETScSquareMatInterface:
    def __init__(self, arg_space, action_space, *,
                 nullspace=None, nullspace_constraint=True):
        if not isinstance(arg_space, MixedSpace):
            arg_space = MixedSpace(arg_space)
        if not isinstance(action_space, MixedSpace):
            action_space = MixedSpace(action_space)
        if nullspace is None:
            nullspace = NoneNullspace()
        elif not isinstance(nullspace, (NoneNullspace, BlockNullspace)):
            nullspace = BlockNullspace(nullspace)
        nullspace_constraint = (nullspace_constraint
                                and not isinstance(nullspace, NoneNullspace))

        self._arg_space = arg_space
        self._action_space = action_space
        self._nullspace = nullspace
        self._nullspace_constraint = nullspace_constraint

        self._x = arg_space.new()
        self._y = action_space.new()
        self._x_c = arg_space.new() if nullspace_constraint else self._x

    @property
    def arg_space(self):
        return self._arg_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def nullspace(self):
        return self._nullspace

    def _pre_mult(self, x_petsc):
        self.arg_space.from_petsc(x_petsc, self._x)

        if self._nullspace_constraint:
            for x_i, x_c_i in zip_sub(self._x, self._x_c):
                var_assign(x_c_i, x_i)

    def _post_mult(self, y_petsc):
        self.action_space.to_petsc(y_petsc, self._y)


class SystemMatrix(PETScSquareMatInterface):
    def __init__(self, matrix, *,
                 nullspace=None, nullspace_constraint=True):
        super().__init__(matrix.arg_space, matrix.action_space,
                         nullspace=nullspace,
                         nullspace_constraint=nullspace_constraint)
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    def mult(self, A, x, y):
        self._pre_mult(x)

        if not isinstance(self.nullspace, NoneNullspace):
            self.nullspace.pre_mult_correct_lhs(self._x_c)
        self.matrix.mult(self._x_c, self._y)
        if not isinstance(self.nullspace, NoneNullspace):
            self.nullspace.post_mult_correct_lhs(
                self._x if self._nullspace_constraint else None, self._y)

        self._post_mult(y)


class Preconditioner(PETScSquareMatInterface):
    def __init__(self, matrix, *,
                 nullspace=None, nullspace_constraint=True):
        super().__init__(matrix.arg_space, matrix.action_space,
                         nullspace=nullspace,
                         nullspace_constraint=nullspace_constraint)
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    def apply(self, pc, x, y):
        self._pre_mult(x)

        if not isinstance(self.nullspace, NoneNullspace):
            self.nullspace.pc_pre_mult_correct(self._x_c)
        self.matrix.mult(self._x_c, self._y)
        if not isinstance(self.nullspace, NoneNullspace):
            self.nullspace.pc_post_mult_correct(
                self._y, self._x if self._nullspace_constraint else None)

        self._post_mult(y)


def petsc_ksp(A, *, comm=None, solver_parameters=None, pc_fn=None):
    if comm is None:
        comm = A.arg_space.comm
    if solver_parameters is None:
        solver_parameters = {}

    comm = comm_dup_cached(comm, key="petsc")

    A_mat = PETSc.Mat().createPython(
        ((A.action_space.local_size, A.action_space.global_size),
         (A.arg_space.local_size, A.arg_space.global_size)), A,
        comm=comm)
    A_mat.setUp()

    if pc_fn is not None:
        A_pc = Preconditioner(MatrixFreeMatrix(A.action_space, A.arg_space,
                                               lambda b, u: pc_fn(u, b)),
                              nullspace=A.nullspace)
        pc = PETSc.PC().createPython(A_pc, comm=comm)
        pc.setOperators(A_mat)
        pc.setUp()
    else:
        pc = None

    ksp = PETSc.KSP().create(comm=comm)
    options = PETScOptions(f"_tlm_adjoint__{ksp.name:s}_")
    options.update(solver_parameters)
    ksp.setOptionsPrefix(options.options_prefix)
    if pc is not None:
        ksp.setPC(pc)
    ksp.setOperators(A_mat)

    ksp.setFromOptions()
    ksp.setUp()

    attach_destroy_finalizer(ksp, pc, A_mat)

    return ksp


class LinearSolver:
    """Solver for a linear system

    .. math::

        A u = b,

    using PETSc.

    :arg A: A :class:`tlm_adjoint.block_system.Matrix` defining :math:`A`.
    :arg nullspace: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace and left nullspace
        of :math:`A`. `None` indicates a :class:`.NoneNullspace`.
    :arg solver_parameters: A :class:`Mapping` defining Krylov solver
        parameters.
    :arg pc_fn: Defines the application of a preconditioner. A callable

        .. code-block:: python

            def pc_fn(u, b):

        The preconditioner is applied to `b`, and the result stored in `u`.
        Defaults to an identity.
    :arg comm: Communicator.
    """

    def __init__(self, A, *, nullspace=None, solver_parameters=None,
                 pc_fn=None, comm=None):
        if nullspace is None:
            nullspace = NoneNullspace()
        elif isinstance(nullspace, Sequence):
            nullspace = BlockNullspace(nullspace)
        if not isinstance(A, BlockMatrix):
            A = BlockMatrix((A.arg_space,), (A.action_space,), A)
            if not isinstance(nullspace, NoneNullspace):
                nullspace = BlockNullspace((nullspace,))
        if solver_parameters is None:
            solver_parameters = {}
        if pc_fn is None:
            pc_pc_fn = None
        else:
            def pc_pc_fn(u, b):
                pc_fn, = self._pc_pc_fn
                with var_locked(*iter_sub(b)):
                    pc_fn(u, b)

        A = SystemMatrix(A, nullspace=nullspace)
        ksp = petsc_ksp(
            A, solver_parameters=solver_parameters, pc_fn=pc_pc_fn, comm=comm)

        self._A = A
        self._ksp = ksp
        self._pc_fn = pc_fn
        self._pc_pc_fn = [pc_fn]

        attach_destroy_finalizer(self, ksp)

    @property
    def ksp(self):
        """The class:`petsc4py.PETSc.KSP` used to solve the linear problem.
        """

        return self._ksp

    @manager_disabled()
    def solve(self, u, b, *,
              correct_initial_guess=True, correct_solution=True):
        """Solve the linear system.

        :arg u: Defines the solution :math:`u`.
        :arg b: Defines the right-hand-side :math:`b`.
        :arg correct_initial_guess: Whether to apply a nullspace correction to
            the initial guess.
        :arg correct_solution: Whether to apply a nullspace correction to
            the solution.
        """

        u_packed = Packed(u)
        b_packed = Packed(b)
        u = tuple(u_packed)
        b = tuple(b_packed)

        pc_fn = self._pc_fn
        if u_packed.is_packed:
            pc_fn_u = pc_fn

            def pc_fn(u, b):
                u, = tuple(iter_sub(u))
                pc_fn_u(u, b)

        if b_packed.is_packed:
            pc_fn_b = pc_fn

            def pc_fn(u, b):
                b, = tuple(iter_sub(b))
                pc_fn_b(u, b)

        u = self._A.arg_space.tuple_sub(u)
        b = self._A.action_space.tuple_sub(b)

        b_c = self._A.action_space.new()
        for b_c_i, b_i in zip_sub(b_c, b):
            if b_i is not None:
                var_assign(b_c_i, b_i)

        if correct_initial_guess:
            self._A.nullspace.correct_soln(u)
        self._A.nullspace.correct_rhs(b_c)

        u_petsc = PETScVec(self._A.arg_space)
        u_petsc.to_petsc(u)
        b_petsc = PETScVec(self._A.action_space)
        b_petsc.to_petsc(b_c)
        del b_c

        try:
            self._pc_pc_fn[0] = pc_fn
            self.ksp.solve(b_petsc.vec, u_petsc.vec)
        finally:
            self._pc_pc_fn[0] = self._pc_fn
        del b_petsc

        u_petsc.from_petsc(u)
        del u_petsc

        if correct_solution:
            self._A.nullspace.correct_soln(u)

        if self.ksp.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")


def slepc_eps(A, B, *, B_inv=None, solver_parameters=None, comm=None):
    if solver_parameters is None:
        solver_parameters = {}
    if comm is None:
        comm = A.arg_space.comm

    comm = comm_dup_cached(comm, key="slepc")

    A_mat = PETSc.Mat().createPython(
        ((A.action_space.local_size, A.action_space.global_size),
         (A.arg_space.local_size, A.arg_space.global_size)), A,
        comm=comm)
    A_mat.setUp()
    if B is not None:
        B_mat = PETSc.Mat().createPython(
            ((B.action_space.local_size, B.action_space.global_size),
             (B.arg_space.local_size, B.arg_space.global_size)), B,
            comm=comm)
        B_mat.setUp()
    else:
        B_mat = None

    eps = SLEPc.EPS().create(comm=comm)
    options = PETScOptions(f"_tlm_adjoint__{eps.name:s}_")
    options.update(solver_parameters)
    eps.setOptionsPrefix(options.options_prefix)
    if B is None:
        eps.setOperators(A_mat)
    else:
        eps.setOperators(A_mat, B_mat)

    eps.setFromOptions()

    if B_inv is not None:
        B_pc = PETSc.PC().createPython(B_inv, comm=comm)
        B_pc.setOperators(B_mat)
        B_pc.setUp()

        B_ksp = eps.getST().getKSP()
        B_ksp.setType(PETSc.KSP.Type.PREONLY)
        B_ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
        B_ksp.setPC(B_pc)
        B_ksp.setUp()
    else:
        B_pc = None
        B_ksp = None

    eps.setUp()

    attach_destroy_finalizer(eps, A_mat, B_ksp, B_pc, B_mat)

    return eps


class Eigensolver:
    r"""Solver for an eigenproblem

    .. math::

        A v = \lambda B v

    using SLEPc.

    :arg A: A :class:`tlm_adjoint.block_system.Matrix` defining :math:`A`.
    :arg B: A :class:`tlm_adjoint.block_system.Matrix` defining :math:`B`. If
        supplied then a generalized eigenproblem is solved. Otherwise a
        standard eigenproblem (with :math:`B` equal to an identity) is solved.
    :arg B_inv: A :class:`tlm_adjoint.block_system.Matrix` defining the inverse
        of :math:`B`.
    :arg nullspace: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace and left nullspace
        of :math:`A` and :math:`B`. `None` indicates a :class:`.NoneNullspace`.
    :arg solver_parameters: A :class:`Mapping` defining solver parameters.
    :arg comm: Communicator.
    """

    def __init__(self, A, B=None, *, B_inv=None, nullspace=None,
                 solver_parameters=None, comm=None):
        if nullspace is None:
            nullspace = NoneNullspace()
        elif isinstance(nullspace, Sequence):
            nullspace = BlockNullspace(nullspace)
        packed = Packed(A).mapped(lambda a: None)
        if not isinstance(A, BlockMatrix):
            A = BlockMatrix((A.arg_space,), (A.action_space,), A)
            if not isinstance(nullspace, NoneNullspace):
                nullspace = BlockNullspace((nullspace,))
            if B is not None:
                B = BlockMatrix((B.arg_space,), (B.action_space,), B)
            if B_inv is not None:
                B_inv = BlockMatrix((B_inv.arg_space,), (B_inv.action_space,), B_inv)  # noqa: E501
        if B is not None and (B.arg_space != A.arg_space
                              or B.action_space != A.action_space):
            raise ValueError("Invalid space")
        if B_inv is not None and (B_inv.arg_space != A.action_space
                                  or B_inv.action_space != A.arg_space):
            raise ValueError("Invalid space")

        A = SystemMatrix(A, nullspace=nullspace, nullspace_constraint=False)
        if B is not None:
            B = SystemMatrix(B, nullspace=nullspace)
        if B_inv is not None:
            B_inv = Preconditioner(B_inv, nullspace=nullspace)

        self._packed = packed
        self._A = A
        self._B = B
        self._B_inv = B_inv
        self._eps = slepc_eps(A, B, B_inv=B_inv,
                              solver_parameters=solver_parameters, comm=comm)

        attach_destroy_finalizer(self, self.eps)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return self.eps.getConverged()

    def _unpack(self, obj):
        return self._packed.unpack(obj)

    def _eigenpair(self, key):
        x_r = self._A.arg_space.new_petsc()
        x_i = self._A.arg_space.new_petsc()
        lam = self.eps.getEigenpair(key, x_r, x_i)

        if self.is_hermitian_and_positive():
            if lam.imag != 0.0:
                raise ValueError("Unexpected complex eigenvalue")
            lam = lam.real
        if self.is_hermitian_and_positive() \
                or issubclass(PETSc.ScalarType, np.complexfloating):
            if x_i.norm(norm_type=PETSc.NormType.NORM_INFINITY) != 0.0:
                raise ValueError("Unexpected complex eigenvector component")
            x_i = None

        return lam, (x_r, x_i)

    def __getitem__(self, key):
        lam, (x_r, x_i) = self._eigenpair(key)
        v_r = self._A.arg_space.new()
        self._A.arg_space.from_petsc(x_r, v_r)
        if x_i is None:
            v_i = None
        else:
            v_i = self._A.arg_space.new()
            self._A.arg_space.from_petsc(x_i, v_i)

        return lam, (self._unpack(v_r),
                     None if v_i is None else self._unpack(v_i))

    @property
    def eps(self):
        """The class:`slepc4py.SLEPc.EPS` used to solve the eigenproblem.
        """

        return self._eps

    def is_hermitian_and_positive(self):
        """
        :returns: Whether the eigenproblem is Hermitian with positive
            semi-definite :math:`B`.
        """

        return (self.eps.isHermitian()
                and (not self.eps.isGeneralized() or self.eps.isPositive()))

    @manager_disabled()
    def solve(self):
        """Solve the eigenproblem.
        """

        self.eps.solve()
        if self.eps.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")
        N, _, _ = self.eps.getDimensions()
        if len(self) < N:
            raise RuntimeError("Convergence failure")

    def eigenvalues(self):
        """Return converged eigenvalues.

        :returns: A :class:`numpy.ndarray` containing eigenvalues.
        """

        Lam = np.zeros(len(self), dtype=(PETSc.RealType
                                         if self.is_hermitian_and_positive()
                                         else PETSc.ComplexType))
        for i in range(len(self)):
            lam = self.eps.getEigenvalue(i)
            if self.is_hermitian_and_positive():
                if lam.imag != 0.0:
                    raise ValueError("Unexpected complex eigenvalue")
                lam = lam.real
            Lam[i] = lam

        return Lam

    def eigenpairs(self):
        """Return converged eigenpairs.

        :returns: A :class:`tuple` `(Lam, V)`, where

            - `Lam` is a :class:`numpy.ndarray` containing eigenvalues.
            - `V` is a :class:`Sequence` of length two :class:`tuple` objects,
              `(v_r, v_i)`, defining corresponding eigenvectors. Each
              eigenvectors is defined by `v_r + 1.0j v_i`. `v_i` is `None`,
              to indicate a value of zero, for Hermitian eigenvalue problems
              or with a complex PETSc build.
        """

        Lam = np.zeros(len(self), dtype=(PETSc.RealType
                                         if self.is_hermitian_and_positive()
                                         else PETSc.ComplexType))
        V = []
        for i, (lam, (v_r, v_i)) in enumerate(self):
            Lam[i] = lam
            V.append((v_r, v_i))
        return Lam, tuple(V)

    def B_orthonormality_test(self):
        r"""Test :math:`B` orthonormality of the eigenvectors for a Hermitian
        eigenvalue problem.

        :returns: :math:`\left| V^* B V - I \right|_\infty` where
            :math:`V` is the matrix whose columns are the eigenvectors.
        """

        if not self.is_hermitian_and_positive():
            raise ValueError("Hermitian and positive eigenproblem required")

        if self._B is None:
            B = None
        else:
            _, B = self.eps.getOperators()
            z = self._A.arg_space.new_petsc()

        error_norm = 0.0
        for i in range(self.eps.getConverged()):
            _, (x, _) = self._eigenpair(i)
            for j in range(self.eps.getConverged()):
                _, (y, _) = self._eigenpair(j)
                if B is None:
                    z = y
                else:
                    B.mult(y, z)
                error_norm = max(error_norm, abs(z.dot(x) - int(i == j)))
        return error_norm


def slepc_mfn(A, *, solver_parameters=None, comm=None):
    if solver_parameters is None:
        solver_parameters = {}
    if comm is None:
        comm = A.arg_space.comm

    comm = comm_dup_cached(comm, key="slepc")

    A_mat = PETSc.Mat().createPython(
        ((A.action_space.local_size, A.action_space.global_size),
         (A.arg_space.local_size, A.arg_space.global_size)), A,
        comm=comm)
    A_mat.setUp()

    mfn = SLEPc.MFN().create(comm=comm)
    options = PETScOptions(f"_tlm_adjoint__{mfn.name:s}_")
    options.update(solver_parameters)
    mfn.setOptionsPrefix(options.options_prefix)
    mfn.setOperator(A_mat)

    mfn.setFromOptions()
    mfn.setUp()

    attach_destroy_finalizer(mfn, A_mat)

    return mfn


class MatrixFunctionSolver:
    r"""Matrix function action evaluation

    .. math::

        v = f ( A ) u

    using SLEPc.

    :arg A: A :class:`tlm_adjoint.block_system.Matrix` defining :math:`A`.
    :arg nullspace: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace and left nullspace
        of :math:`A` and :math:`B`. `None` indicates a :class:`.NoneNullspace`.
    :arg solver_parameters: A :class:`Mapping` defining solver parameters.
    :arg comm: Communicator.
    """

    def __init__(self, A, *, nullspace=None,
                 solver_parameters=None, comm=None):
        if nullspace is None:
            nullspace = NoneNullspace()
        elif isinstance(nullspace, Sequence):
            nullspace = BlockNullspace(nullspace)
        if not isinstance(A, BlockMatrix):
            A = BlockMatrix((A.arg_space,), (A.action_space,), A)
            if not isinstance(nullspace, NoneNullspace):
                nullspace = BlockNullspace((nullspace,))
        if solver_parameters is None:
            solver_parameters = {}

        A = SystemMatrix(A, nullspace=nullspace, nullspace_constraint=False)

        self._A = A
        self._mfn = slepc_mfn(A, solver_parameters=solver_parameters,
                              comm=comm)

        attach_destroy_finalizer(self, self.mfn)

    @property
    def mfn(self):
        """The class:`slepc4py.SLEPc.MFN` used to compute the matrix function
        action.
        """

        return self._mfn

    def solve(self, u, v):
        """Compute the matrix function action.

        :arg u: Defines the argument :math:`u`.
        :arg v: Defines the result :math:`v`.
        """

        u = packed(u)
        v = packed(v)

        u_petsc = PETScVec(self._A.arg_space)
        u_petsc.to_petsc(u)
        v_petsc = PETScVec(self._A.arg_space)
        v_petsc.to_petsc(v)

        self.mfn.solve(u_petsc.vec, v_petsc.vec)
        if self.mfn.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")

        with paused_space_type_checking():
            v_petsc.from_petsc(v)
