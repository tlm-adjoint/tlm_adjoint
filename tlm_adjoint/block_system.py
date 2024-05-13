r"""Solvers for linear systems defined in mixed spaces.

The :class:`.System` class defines the block structure of the linear system,
and solves the system using an outer Krylov solver.

Given a linear problem with a potentially singular matrix :math:`A`

.. math::

    A u = b,

a :class:`.System` instead solves the linear problem

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
       nullspace and nullspace of :math:`A`, and the :class:`.System` then
       seeks a solution to the original problem subject to the linear
       constraints :math:`V^* C u = 0`.

Spaces are defined via backend spaces, and :class:`Sequence` objects containing
backend spaces or similar :class:`Sequence` objects. Similarly variables are
defined via backend variables, or :class:`Sequence` objects containing backend
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
    DEFAULT_COMM, space_default_space_type, space_comm, space_id, space_new,
    var_assign, var_zero)
from .petsc import PETScVecInterface

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from collections import deque
from collections.abc import Sequence
from functools import wraps
import logging
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None


__all__ = \
    [
        "MixedSpace",

        "Nullspace",
        "NoneNullspace",
        "BlockNullspace",

        "Matrix",
        "BlockMatrix",

        "System"
    ]


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
        if len(self) == 1 and not isinstance(self.split_space[0], Sequence):
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


class Nullspace(ABC):
    """Represents a matrix nullspace and left nullspace.
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


class BlockNullspace(Nullspace):
    """Nullspaces for a mixed space.

    :arg nullspaces: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace. `None` indicates a
        :class:`.NoneNullspace`.
    """

    def __init__(self, nullspaces):
        if not isinstance(nullspaces, Sequence):
            nullspaces = (nullspaces,)

        nullspaces = list(nullspaces)
        for i, nullspace in enumerate(nullspaces):
            if nullspace is None:
                nullspaces[i] = NoneNullspace()
        nullspaces = tuple(nullspaces)

        super().__init__()
        self._nullspaces = nullspaces

    def __new__(cls, nullspaces, *args, **kwargs):
        if not isinstance(nullspaces, Sequence):
            nullspaces = (nullspaces,)
        for nullspace in nullspaces:
            if nullspace is not None \
                    and not isinstance(nullspace, NoneNullspace):
                break
        else:
            return NoneNullspace()
        return super().__new__(cls)

    def __getitem__(self, key):
        return self._nullspaces[key]

    def __iter__(self):
        yield from self._nullspaces

    def __len__(self):
        return len(self._nullspaces)

    def apply_nullspace_transformation_lhs_right(self, x):
        assert len(self._nullspaces) == len(x)
        for nullspace, x_i in zip(self._nullspaces, x):
            nullspace.apply_nullspace_transformation_lhs_right(x_i)

    def apply_nullspace_transformation_lhs_left(self, y):
        assert len(self._nullspaces) == len(y)
        for nullspace, y_i in zip(self._nullspaces, y):
            nullspace.apply_nullspace_transformation_lhs_left(y_i)

    def constraint_correct_lhs(self, x, y):
        assert len(self._nullspaces) == len(x)
        assert len(self._nullspaces) == len(y)
        for nullspace, x_i, y_i in zip(self._nullspaces, x, y):
            nullspace.constraint_correct_lhs(x_i, y_i)

    def pc_constraint_correct_soln(self, u, b):
        assert len(self._nullspaces) == len(u)
        assert len(self._nullspaces) == len(b)
        for nullspace, u_i, b_i in zip(self._nullspaces, u, b):
            nullspace.pc_constraint_correct_soln(u_i, b_i)


class Matrix(ABC):
    r"""Represents a matrix :math:`A` mapping :math:`V \rightarrow W`.

    :arg arg_space: Defines the space `V`.
    :arg action_space: Defines the space `W`.
    """

    def __init__(self, arg_space, action_space):
        if not isinstance(arg_space, MixedSpace):
            arg_space = MixedSpace(arg_space)
        if not isinstance(action_space, MixedSpace):
            action_space = MixedSpace(action_space)

        self._arg_space = arg_space
        self._action_space = action_space

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

    @abstractmethod
    def mult_add(self, x, y):
        """Add :math:`A x` to :math:`y`.

        :arg x: Defines :math:`x`. Should not be modified.
        :arg y: Defines :math:`y`.
        """

        raise NotImplementedError


class BlockMatrix(Matrix, MutableMapping):
    r"""A matrix :math:`A` mapping :math:`V \rightarrow W`, where :math:`V` and
    :math:`W` are defined by mixed spaces.

    :arg arg_spaces: Defines the space `V`.
    :arg action_spaces: Defines the space `W`.
    :arg block: A :class:`Mapping` defining the blocks of the matrix. Items are
        `((i, j), block)` where the block in the `i` th and `j` th column is
        defined by `block`. Each `block` is a
        :class:`tlm_adjoint.block_system.Matrix`, or `None` to indicate a zero
        block.
    """

    def __init__(self, arg_spaces, action_spaces, blocks=None):
        if not isinstance(blocks, Mapping):
            blocks = {(0, 0): blocks}

        super().__init__(arg_spaces, action_spaces)
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
        for (i, j), block in self.items():
            block.mult_add(x[j], y[i])


class PETScInterface:
    def __init__(self, arg_space, action_space, nullspace):
        if not isinstance(arg_space, MixedSpace):
            arg_space = MixedSpace(arg_space)
        if not isinstance(action_space, MixedSpace):
            action_space = MixedSpace(action_space)

        self._arg_space = arg_space
        self._action_space = action_space
        self._nullspace = nullspace

        self._x = arg_space.new_split()
        self._y = action_space.new_split()

        if isinstance(self._nullspace, NoneNullspace):
            self._x_c = self._x
        else:
            self._x_c = arg_space.new_split()

    @property
    def arg_space(self):
        return self._arg_space

    @property
    def action_space(self):
        return self._action_space

    def _pre_mult(self, x_petsc):
        self.arg_space.from_petsc(x_petsc, self._x)

        if not isinstance(self._nullspace, NoneNullspace):
            for x_i, x_c_i in zip_sub(self._x, self._x_c):
                var_assign(x_c_i, x_i)

        for y_i in iter_sub(self._y):
            var_zero(y_i)

    def _post_mult(self, y_petsc):
        self.action_space.to_petsc(y_petsc, self._y)


class SystemMatrix(PETScInterface):
    def __init__(self, matrix, nullspace):
        super().__init__(matrix.arg_space, matrix.action_space, nullspace)
        self._matrix = matrix

    def mult(self, A, x, y):
        self._pre_mult(x)

        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.pre_mult_correct_lhs(self._x_c)
        self._matrix.mult_add(self._x_c, self._y)
        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.post_mult_correct_lhs(self._x, self._y)

        self._post_mult(y)


class Preconditioner(PETScInterface):
    def __init__(self, arg_space, action_space, pc_fn, nullspace):
        super().__init__(arg_space, action_space, nullspace)
        self._pc_fn = pc_fn

    def apply(self, pc, x, y):
        self._pre_mult(x)

        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.pc_pre_mult_correct(self._x_c)
        self._pc_fn(self._y, self._x_c)
        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.pc_post_mult_correct(
                self._y, self._x)

        self._post_mult(y)


class System:
    """A linear system

    .. math::

        A u = b.

    :arg arg_spaces: Defines the space for `u`.
    :arg action_spaces: Defines the space for `b`.
    :arg blocks: One of

        - A :class:`tlm_adjoint.block_system.Matrix` or :class:`ufl.Form`
          defining :math:`A`.
        - A :class:`Mapping` with items `((i, j), block)` where the matrix
          associated with the block in the `i` th and `j` th column is defined
          by `block`. Each `block` is a
          :class:`tlm_adjoint.block_system.Matrix` or :class:`ufl.Form`, or
          `None` to indicate a zero block.

    :arg nullspaces: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace and left nullspace
        of :math:`A`. `None` indicates a :class:`.NoneNullspace`.
    :arg comm: Communicator.
    """

    def __init__(self, arg_spaces, action_spaces, blocks, *,
                 nullspaces=None, comm=None):
        if isinstance(blocks, BlockMatrix):
            matrix = blocks
        else:
            matrix = BlockMatrix(arg_spaces, action_spaces, blocks)
        arg_spaces = matrix.arg_space
        action_spaces = matrix.action_space

        nullspace = BlockNullspace(nullspaces)
        if isinstance(nullspace, BlockNullspace):
            if len(nullspace) != len(arg_spaces):
                raise ValueError("Invalid space")
            if len(nullspace) != len(action_spaces):
                raise ValueError("Invalid space")

        if comm is None:
            comm = arg_spaces.comm

        self._comm = comm
        self._arg_space = arg_spaces
        self._action_space = action_spaces
        self._matrix = matrix
        self._nullspace = nullspace

    def solve(self, u, b, *,
              solver_parameters=None, pc_fn=None,
              pre_callback=None, post_callback=None,
              correct_initial_guess=True, correct_solution=True):
        """Solve the linear system.

        :arg u: Defines the solution :math:`u`.
        :arg b: Defines the right-hand-side :math:`b`.
        :arg solver_parameters: A :class:`Mapping` defining outer Krylov solver
            parameters. Parameters (a number of which are based on FEniCS
            solver parameters) are:

            - `'linear_solver'`: The Krylov solver type, default `'fgmres'`.
            - `'pc_side'`: Overrides the PETSc default preconditioning side.
            - `'relative_tolerance'`: Relative tolerance. Required.
            - `'absolute_tolerance'`: Absolute tolerance. Required.
            - `'divergence_limit'`: Overrides the default divergence limit.
            - `'maximum_iterations'`: Maximum number of iterations. Default
              1000.
            - `'norm_type'`: Overrides the default convergence norm definition.
            - `'nonzero_initial_guess'`: Whether to use a non-zero initial
              guess, defined by the input `u`. Default `True`.
            - `'gmres_restart'`: Overrides the default GMRES restart parameter.

        :arg pc_fn: Defines the application of a preconditioner. A callable

            .. code-block:: python

                def pc_fn(u, b):

            The preconditioner is applied to `b`, and the result stored in `u`.
            Defaults to an identity.
        :arg pre_callback: A callable accepting a single
            :class:`petsc4py.PETSc.KSP` argument. Used for detailed manual
            configuration. Called after all other configuration options are
            set, but before the :meth:`petsc4py.PETSc.KSP.setUp` method is
            called.
        :arg post_callback: A callable accepting a single
            :class:`petsc4py.PETSc.KSP` argument. Called after the
            :meth:`petsc4py.PETSc.KSP.solve` method has been called.
        :arg correct_initial_guess: Whether to apply a nullspace correction to
            the initial guess.
        :arg correct_solution: Whether to apply a nullspace correction to
            the solution.
        :returns: The number of Krylov iterations.
        """

        if solver_parameters is None:
            solver_parameters = {}

        if isinstance(u, Sequence):
            u = tuple(u)
        else:
            u = (u,)

            if pc_fn is not None:
                pc_fn_u = pc_fn

                @wraps(pc_fn_u)
                def pc_fn(u, b):
                    u, = tuple(iter_sub(u))
                    return pc_fn_u(u, b)
        u = self._arg_space.tuple_sub(u)

        if isinstance(b, Sequence):
            b = tuple(b)
        else:
            b = (b,)

            if pc_fn is not None:
                pc_fn_b = pc_fn

                @wraps(pc_fn_b)
                def pc_fn(u, b):
                    b, = tuple(iter_sub(b))
                    return pc_fn_b(u, b)
        b = self._action_space.tuple_sub(b)

        if tuple(u_i.function_space() for u_i in iter_sub(u)) \
                != self._arg_space.flattened_space:
            raise ValueError("Invalid space")
        for b_i, space in zip_sub(b, self._action_space.split_space):
            if b_i is not None and b_i.function_space() != space:
                raise ValueError("Invalid space")

        b_c = self._action_space.new_split()
        for b_c_i, b_i in zip_sub(b_c, b):
            if b_i is not None:
                var_assign(b_c_i, b_i)

        A = SystemMatrix(self._matrix, self._nullspace)

        mat_A = PETSc.Mat().createPython(
            ((self._action_space.local_size, self._action_space.global_size),
             (self._arg_space.local_size, self._arg_space.global_size)), A,
            comm=self._comm)
        mat_A.setUp()

        if pc_fn is not None:
            A_pc = Preconditioner(self._action_space, self._arg_space,
                                  pc_fn, self._nullspace)
            pc = PETSc.PC().createPython(
                A_pc, comm=self._comm)
            pc.setOperators(mat_A)
            pc.setUp()

        ksp_solver = PETSc.KSP().create(comm=self._comm)
        ksp_solver.setType(solver_parameters.get("linear_solver", "fgmres"))
        if pc_fn is not None:
            ksp_solver.setPC(pc)
        if "pc_side" in solver_parameters:
            ksp_solver.setPCSide(solver_parameters["pc_side"])
        ksp_solver.setOperators(mat_A)
        ksp_solver.setTolerances(
            rtol=solver_parameters["relative_tolerance"],
            atol=solver_parameters["absolute_tolerance"],
            divtol=solver_parameters.get("divergence_limit", None),
            max_it=solver_parameters.get("maximum_iterations", 1000))
        ksp_solver.setInitialGuessNonzero(
            solver_parameters.get("nonzero_initial_guess", True))
        ksp_solver.setNormType(
            solver_parameters.get(
                "norm_type", PETSc.KSP.NormType.DEFAULT))
        if "gmres_restart" in solver_parameters:
            ksp_solver.setGMRESRestart(solver_parameters["gmres_restart"])

        logger = logging.getLogger("tlm_adjoint.System")

        def monitor(ksp_solver, it, r_norm):
            logger.debug(f"KSP: "
                         f"iteration {it:d}, "
                         f"residual norm {r_norm:.16e}")

        ksp_solver.setMonitor(monitor)

        if correct_initial_guess:
            self._nullspace.correct_soln(u)
        self._nullspace.correct_rhs(b_c)

        u_petsc = mat_A.createVecRight()
        self._arg_space.to_petsc(u_petsc, u)
        b_petsc = mat_A.createVecLeft()
        self._action_space.to_petsc(b_petsc, b_c)
        del b_c

        if pre_callback is not None:
            pre_callback(ksp_solver)
        ksp_solver.setUp()
        ksp_solver.solve(b_petsc, u_petsc)
        if post_callback is not None:
            post_callback(ksp_solver)
        del b_petsc

        self._arg_space.from_petsc(u_petsc, u)
        del u_petsc

        if correct_solution:
            # Not needed if the linear problem were to be solved exactly
            self._nullspace.correct_soln(u)

        if ksp_solver.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")
        ksp_its = ksp_solver.getIterationNumber()

        ksp_solver.destroy()
        mat_A.destroy()
        if pc_fn is not None:
            pc.destroy()

        return ksp_its
