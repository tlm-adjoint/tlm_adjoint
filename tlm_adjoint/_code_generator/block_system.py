#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""This module is used by both the FEniCS and Firedrake backends, and
implements solvers for linear systems defined in mixed spaces.

The :class:`System` class defines the block structure of the linear system, and
solves the system using an outer Krylov solver. A custom preconditioner can be
defined via the `pc_fn` callback to :meth:`System.solve`, and this
preconditioner can itself e.g. make use of further Krylov solvers. This
provides a Python interface for custom block preconditioners.

Given a linear problem with a potentially singular matrix :math:`A`

.. math::

    A u = b,

a :class:`System` instead solves the linear problem

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

  1. Where a matrix :math:`A` and right-hand-side :math:`b` are constructed via
     finite element assembly on superspaces of the test space and trial space.
     The typical example is in the application of homogeneous essential
     Dirichlet boundary conditions.

  2. Where the matrix :math:`A` is singular and :math:`b` is orthogonal to the
     left nullspace of :math:`A`. Typically one would then choose :math:`U` and
     :math:`V` so that their columns respectively span the left nullspace and
     nullspace of :math:`A`, and the :class:`System` then seeks a solution to
     the original problem subject to the linear constraints :math:`V^* C u =
     0`.

Function spaces are defined via backend function spaces, and :class:`Sequence`
objects containing backend function spaces or similar :class:`Sequence`
objects. Similarly functions are defined via backend :class:`Function` objects,
or :class:`Sequence` objects containing backend :class:`Function` objects or
similar :class:`Sequence` objects. This defines a basic tree structure which is
useful e.g. when defining block matrices in terms of sub-block matrices.

Elements of the tree are accessed in a consistent order using a depth first
search. Hence e.g.

.. code-block:: python

    ((u_0, u_1), u_2)

and

.. code-block:: python

    (u_0, u_1, u_2)

where `u_0`, `u_1`, and `u_2` are backend :class:`Function` objects, are both
valid representations of a mixed space solution.

Code in this module is written to use only backend functionality, and does not
use tlm_adjoint interfaces. Consequently if used directly, and in combination
with other tlm_adjoint code, space type warnings may be encountered.
"""

# This is the only import from other tlm_adjoint modules that is permitted in
# this module
from .backend import backend

if backend == "Firedrake":
    from firedrake import (Constant, DirichletBC, Function, FunctionSpace,
                           TestFunction, assemble)
elif backend == "FEniCS":
    from fenics import (Constant, DirichletBC, Function, FunctionSpace,
                        TestFunction, assemble)
    from fenics import FunctionAssigner, as_backend_type
    from dolfin.cpp.function import Constant as cpp_Constant
else:
    raise ImportError(f"Unexpected backend: {backend}")

import petsc4py.PETSc as PETSc
import ufl

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from contextlib import contextmanager
from enum import Enum
from functools import cached_property, wraps
import logging

__all__ = \
    [
        "MixedSpace",
        "BackendMixedSpace",

        "Nullspace",
        "NoneNullspace",
        "ConstantNullspace",
        "UnityNullspace",
        "DirichletBCNullspace",
        "BlockNullspace",

        "Matrix",
        "PETScMatrix",
        "BlockMatrix",
        "form_matrix",

        "ConvergenceError",
        "System"
    ]

_error_flag = False


def flag_errors(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        global _error_flag
        try:
            return fn(*args, **kwargs)
        except Exception:
            _error_flag = True
            raise
    return wrapped_fn


# Following naming of PyOP2 Dat.vec_context access types
class Access(Enum):
    RW = "RW"
    READ = "READ"
    WRITE = "WRITE"


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


def zip_sub(*iterables):
    iterators = map(iter_sub, iterables)
    yield from zip(*iterators)

    for iterator in iterators:
        try:
            next(iterator)
            raise ValueError("Non-equal lengths")
        except StopIteration:
            pass


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


class MixedSpace(ABC):
    """Used to map between mixed and split versions of spaces.

    This class defines three representations for the space:

        1. As a 'mixed space': A single function space defined using a UFL
           :class:`MixedElement`.
        2. As a 'split space': A tree defining the mixed space. Stored using
           backend function space and :class:`tuple` objects, each
           corresponding to a node in the tree. Function spaces correspond to
           leaf nodes, and :class:`tuple` objects to other nodes in the tree.
        3. As a 'flattened space': A :class:`Sequence` containing leaf nodes of
           the split space with an ordering determined using a depth first
           search.

    This allows, for example, the construction:

    .. code-block:: python

        u_0 = Function(space_0, name='u_0')
        u_1 = Function(space_1, name='u_1')
        u_2 = Function(space_2, name='u_2')

        mixed_space = BackendMixedSpace(((space_0, space_1), space_2))
        u_fn = mixed_space.new_mixed()

    and then data can be copied to the function in the mixed space via

    .. code-block:: python

        mixed_space.split_to_mixed(u_fn, ((u_0, u_1), u_2))

    and from the function in the mixed space via

    .. code-block:: python

        mixed_space.mixed_to_split(((u_0, u_1), u_2), u_fn)

    :arg spaces: The split space.
    """

    def __init__(self, spaces):
        if isinstance(spaces, Sequence):
            spaces = tuple(spaces)
        else:
            spaces = (spaces,)
        spaces = tuple_sub(spaces, spaces)
        flattened_spaces = tuple(iter_sub(spaces))

        mesh = flattened_spaces[0].mesh()
        for space in flattened_spaces[1:]:
            if space.mesh() != mesh:
                raise ValueError("Invalid mesh")

        if len(flattened_spaces) == 1:
            mixed_space, = flattened_spaces
        else:
            mixed_element = ufl.classes.MixedElement(
                *(space.ufl_element() for space in flattened_spaces))
            mixed_space = FunctionSpace(mesh, mixed_element)

        with vec(Function(mixed_space), Access.READ) as v:
            n = v.getLocalSize()
            N = v.getSize()

        self._mesh = mesh
        self._spaces = spaces
        self._flattened_spaces = flattened_spaces
        self._mixed_space = mixed_space
        self._sizes = (n, N)

    def mesh(self):
        """
        :returns: The mesh associated with the space.
        """

        return self._mesh

    def split_space(self):
        """
        :returns: The split space.
        """

        return self._spaces

    def flattened_space(self):
        """
        :returns: The flattened space.
        """

        return self._flattened_spaces

    def mixed_space(self):
        """
        :returns: The mixed space.
        """

        return self._mixed_space

    def new_split(self, *args, **kwargs):
        """
        :returns: A new function in the split space.

        Arguments are passed to the backend :class:`Function` constructor.
        """

        return tuple_sub((Function(space, *args, **kwargs)
                          for space in self._flattened_spaces),
                         self._spaces)

    def new_mixed(self, *args, **kwargs):
        """
        :returns: A new function in the mixed space.

        Arguments are passed to the backend :class:`Function` constructor.
        """

        return Function(self._mixed_space, *args, **kwargs)

    def sizes(self):
        """
        :returns: A :class:`tuple`, `(n, N)`, where `n` is the number of
            process local degrees of freedom and `N` is the number of global
            degrees of freedom, each for the mixed space.
        """

        return self._sizes

    @abstractmethod
    def mixed_to_split(self, u, u_fn):
        """Copy data out of the mixed space representation.

        :arg u: A function in a compatible split space.
        :arg u_fn: The function in the mixed space.
        """

        raise NotImplementedError

    @abstractmethod
    def split_to_mixed(self, u_fn, u):
        """Copy data into the mixed space representation.

        :arg u_fn: The function in the mixed space.
        :arg u: A function in a compatible split space.
        """

        raise NotImplementedError


if backend == "Firedrake":
    def mesh_comm(mesh):
        return mesh.comm

    class BackendMixedSpace(MixedSpace):
        @staticmethod
        def _iter_sub_fn(iterable):
            def expand(e):
                if isinstance(e, Function):
                    space = e.function_space()
                    if hasattr(space, "num_sub_spaces"):
                        return tuple(e.sub(i)
                                     for i in range(space.num_sub_spaces()))
                return e

            return iter_sub(iterable, expand=expand)

        def mixed_to_split(self, u, u_fn):
            if len(self._flattened_spaces) == 1:
                u, = tuple(iter_sub(u))
                u.assign(u_fn)
            else:
                for i, u_i in enumerate(self._iter_sub_fn(u)):
                    u_i.assign(u_fn.sub(i))

        def split_to_mixed(self, u_fn, u):
            if len(self._flattened_spaces) == 1:
                u, = tuple(iter_sub(u))
                u_fn.assign(u)
            else:
                for i, u_i in enumerate(self._iter_sub_fn(u)):
                    u_fn.sub(i).assign(u_i)

    def mat(a):
        return a.petscmat

    @contextmanager
    def vec(u, mode=Access.RW):
        attribute_name = {Access.RW: "vec",
                          Access.READ: "vec_ro",
                          Access.WRITE: "vec_wo"}[mode]
        with getattr(u.dat, attribute_name) as u_v:
            yield u_v

    def bc_space(bc):
        return bc.function_space()

    def bc_is_homogeneous(bc):
        return isinstance(bc.function_arg, ufl.classes.Zero)

    def bc_domain_args(bc):
        return (bc.sub_domain,)

    def apply_bcs(u, bcs):
        if not isinstance(bcs, Sequence):
            bcs = (bcs,)
        for bc in bcs:
            bc.apply(u)
elif backend == "FEniCS":
    def mesh_comm(mesh):
        return mesh.mpi_comm()

    class BackendMixedSpace(MixedSpace):
        @cached_property
        def _mixed_to_split_assigner(self):
            return FunctionAssigner(list(self._flattened_spaces),
                                    self._mixed_space)

        @cached_property
        def _split_to_mixed_assigner(self):
            return FunctionAssigner(self._mixed_space,
                                    list(self._flattened_spaces))

        def mixed_to_split(self, u, u_fn):
            if len(self._flattened_spaces) == 1:
                u, = tuple(iter_sub(u))
                u.assign(u_fn)
            else:
                self._mixed_to_split_assigner.assign(list(iter_sub(u)),
                                                     u_fn)

        def split_to_mixed(self, u_fn, u):
            if len(self._flattened_spaces) == 1:
                u, = tuple(iter_sub(u))
                u_fn.assign(u)
            else:
                self._split_to_mixed_assigner.assign(u_fn,
                                                     list(iter_sub(u)))

    def mat(a):
        matrix = as_backend_type(a).mat()
        if not isinstance(matrix, PETSc.Mat):
            raise RuntimeError("PETSc backend required")
        return matrix

    @contextmanager
    def vec(u, mode=Access.RW):
        if isinstance(u, Function):
            u = u.vector()
        u_v = as_backend_type(u).vec()
        if not isinstance(u_v, PETSc.Vec):
            raise RuntimeError("PETSc backend required")

        yield u_v

        if mode in {Access.RW, Access.WRITE}:
            u.update_ghost_values()

    def bc_space(bc):
        return FunctionSpace(bc.function_space())

    def bc_is_homogeneous(bc):
        # A weaker check with FEniCS, as the constant might be modified
        return (isinstance(bc.value(), cpp_Constant)
                and (bc.value().values() == 0.0).all())

    def bc_domain_args(bc):
        return (bc.sub_domain, bc.method())

    def apply_bcs(u, bcs):
        if not isinstance(bcs, Sequence):
            bcs = (bcs,)
        for bc in bcs:
            bc.apply(u.vector())
else:
    raise ImportError(f"Unexpected backend: {backend}")


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


class ConstantNullspace(Nullspace):
    r"""A nullspace and left nullspace spanned by the vector of ones.

    Here :math:`V = U`, :math:`U` is a single column matrix whose elements are
    ones, :math:`C = M`, and :math:`M` is an identity matrix.

    :arg alpha: Defines the linear constraint matrix :math:`S = \left( \alpha /
        N \right)` where :math:`N` is the length of the vector of ones.
    """

    def __init__(self, *, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    @staticmethod
    def _correct(x, y, *, alpha=1.0):
        with vec(x, Access.READ) as x_v:
            x_sum = x_v.sum()
            N = x_v.getSize()

        with vec(y) as y_v:
            y_v.shift(alpha * x_sum / float(N))

    def apply_nullspace_transformation_lhs_right(self, x):
        self._correct(x, x, alpha=-1.0)

    def apply_nullspace_transformation_lhs_left(self, y):
        self._correct(y, y, alpha=-1.0)

    def constraint_correct_lhs(self, x, y):
        self._correct(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._correct(b, u, alpha=1.0 / self._alpha)


class UnityNullspace(Nullspace):
    r"""A nullspace and left nullspace defined by the unity-valued function.

    Here :math:`V = U`, :math:`U` is a single column matrix containing the
    degree-of-freedom vector for the unity-valued function, :math:`C = M`,
    and :math:`M` is the mass matrix.

    :arg space: A scalar-valued function space containing the unity-valued
        function.
    :arg alpha: Defines the linear constraint matrix :math:`S = \alpha \left(
        U^* M U \right)^{-1}`.
    """

    def __init__(self, space, *, alpha=1.0):
        U = Function(space, name="U")
        U.assign(Constant(1.0))
        MU = assemble(ufl.inner(U, TestFunction(space)) * ufl.dx)
        UMU = assemble(ufl.inner(U, U) * ufl.dx)

        self._alpha = alpha
        self._U = U
        self._MU = MU
        self._UMU = UMU

    @staticmethod
    def _correct(x, y, u, v, *, alpha=1.0):
        with vec(x, Access.READ) as x_v, vec(u, Access.READ) as u_v:
            u_x = x_v.dot(u_v)

        with vec(y) as y_v, vec(v, Access.READ) as v_v:
            y_v.axpy(alpha * u_x, v_v)

    def apply_nullspace_transformation_lhs_right(self, x):
        self._correct(
            x, x, self._MU, self._U, alpha=-1.0 / self._UMU)

    def apply_nullspace_transformation_lhs_left(self, y):
        self._correct(
            y, y, self._U, self._MU, alpha=-1.0 / self._UMU)

    def constraint_correct_lhs(self, x, y):
        self._correct(
            x, y, self._MU, self._MU, alpha=self._alpha / self._UMU)

    def pc_constraint_correct_soln(self, u, b):
        self._correct(
            b, u, self._U, self._U, alpha=1.0 / (self._alpha * self._UMU))


class DirichletBCNullspace(Nullspace):
    r"""A nullspace and left nullspace associated with homogeneous Dirichlet
    boundary conditions.

    Here :math:`V = U`, :math:`U` is a zero-one matrix with exactly one
    non-zero per column corresponding to one boundary condition
    degree-of-freedom, :math:`C = M`, and :math:`M` is an identity matrix.

    :arg bcs: The Dirichlet boundary conditions.
    :arg alpha: Defines the linear constraint matrix :math:`S = \alpha M`.
    """

    def __init__(self, bcs, *, alpha=1.0):
        if isinstance(bcs, Sequence):
            bcs = tuple(bcs)
        else:
            bcs = (bcs,)

        space = bc_space(bcs[0])
        for bc in bcs:
            if bc_space(bc) != space:
                raise ValueError("Invalid space")
            if not bc_is_homogeneous(bc):
                raise ValueError("Homogeneous boundary conditions required")

        super().__init__()
        self._bcs = bcs
        self._alpha = alpha
        self._c = Function(space)

    def apply_nullspace_transformation_lhs_right(self, x):
        apply_bcs(x, self._bcs)

    def apply_nullspace_transformation_lhs_left(self, y):
        apply_bcs(y, self._bcs)

    def _constraint_correct_lhs(self, x, y, *, alpha=1.0):
        with vec(self._c, Access.WRITE) as c_v:
            c_v.zeroEntries()

        apply_bcs(self._c,
                  tuple(DirichletBC(x.function_space(), x, *bc_domain_args(bc))
                        for bc in self._bcs))

        with vec(self._c, Access.READ) as c_v, vec(y) as y_v:
            y_v.axpy(alpha, c_v)

    def constraint_correct_lhs(self, x, y):
        self._constraint_correct_lhs(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._constraint_correct_lhs(b, u, alpha=1.0 / self._alpha)


class BlockNullspace(Nullspace):
    """Nullspaces for a mixed space.

    :arg nullspaces: A :class:`Nullspace` or a :class:`Sequence` of
        :class:`Nullspace` objects defining the nullspace. `None` indicates a
        :class:`NoneNullspace`.
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

    def __new__(cls, nullspaces):
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

    Note that :math:`V` and :math:`W` need not correspond directly to discrete
    function spaces as defined by `arg_space` and `action_space`, but may
    instead e.g. be defined via one or more antidual spaces.

    :arg arg_space: Defines the space `V`.
    :arg action_space: Defines the space `W`.
    """

    def __init__(self, arg_space, action_space):
        if isinstance(arg_space, Sequence):
            arg_space = tuple(arg_space)
            arg_space = tuple_sub(arg_space, arg_space)
        if isinstance(action_space, Sequence):
            action_space = tuple(action_space)
            action_space = tuple_sub(action_space, action_space)

        self._arg_space = arg_space
        self._action_space = action_space

    def arg_space(self):
        """
        :returns: The space defining :math:`V`.
        """

        return self._arg_space

    def action_space(self):
        """
        :returns: The space defining :math:`W`.
        """

        return self._action_space

    @abstractmethod
    def mult_add(self, x, y):
        """Add :math:`A x` to :math:`y`.

        :arg x: Defines :math:`x`. Should not be modified.
        :arg y: Defines :math:`y`.
        """

        raise NotImplementedError


class PETScMatrix(Matrix):
    r"""A :class:`Matrix` associated with a PETSc matrix :math:`A` mapping
    :math:`V \rightarrow W`.

    :arg arg_space: Defines the space `V`.
    :arg action_space: Defines the space `W`.
    :arg a: The PETSc matrix.
    """

    def __init__(self, arg_space, action_space, a):
        super().__init__(arg_space, action_space)
        self._matrix = a

    def mult_add(self, x, y):
        matrix = mat(self._matrix)
        with vec(x, Access.READ) as x_v, vec(y) as y_v:
            matrix.multAdd(x_v, y_v, y_v)


def form_matrix(a, *args, **kwargs):
    """Return a :class:`PETScMatrix` associated with a given sesquilinear form.

    :arg a: A UFL :class:`Form` defining the sesquilinear form.

    Remaining arguments are passed to the backend :func:`assemble`.
    """

    test, trial = a.arguments()
    assert test.number() < trial.number()

    return PETScMatrix(
        trial.function_space(), test.function_space(),
        assemble(a, *args, **kwargs))


class BlockMatrix(Matrix):
    r"""A matrix :math:`A` mapping :math:`V \rightarrow W`, where :math:`V` and
    :math:`W` are defined by mixed spaces.

    :arg arg_spaces: Defines the space `V`.
    :arg action_spaces: Defines the space `W`.
    :arg block: A :class:`Mapping` defining the blocks of the matrix. Items are
        `((i, j), block)` defining a UFL :class:`Form` or :class:`Matrix` for
        the block in row `i` and column `j`. A value for `block` of `None`
        indicates a zero block.
    """

    def __init__(self, arg_spaces, action_spaces, blocks=None):
        if not isinstance(blocks, BlockMatrix) \
                and isinstance(blocks, (Matrix, ufl.classes.Form)):
            blocks = {(0, 0): blocks}

        super().__init__(arg_spaces, action_spaces)
        self._blocks = {}

        if blocks is not None:
            self.update(blocks)

    def __contains__(self, key):
        i, j = key
        return (i, j) in self._blocks

    def __iter__(self):
        yield from self.keys()

    def __getitem__(self, key):
        i, j = key
        return self._blocks[(i, j)]

    def __setitem__(self, key, value):
        i, j = key
        if value is None:
            self.pop((i, j))
        else:
            if isinstance(value, ufl.classes.Form):
                value = form_matrix(value)
            if value.arg_space() != self._arg_space[j]:
                raise ValueError("Invalid space")
            if value.action_space() != self._action_space[i]:
                raise ValueError("Invalid space")
            self._blocks[(i, j)] = value

    def __delitem__(self, key):
        i, j = key
        del self._blocks[(i, j)]

    def __len__(self):
        return len(self._blocks)

    def keys(self):
        yield from sorted(self._blocks)

    def values(self):
        for (i, j) in self:
            yield self[(i, j)]

    def items(self):
        for (i, j) in self:
            yield ((i, j), self[(i, j)])

    def update(self, other):
        for (i, j), block in other.items():
            self[(i, j)] = block

    def pop(self, key, default=None):
        i, j = key
        return self._blocks.pop((i, j), default)

    def mult_add(self, x, y):
        for (i, j), block in self.items():
            block.mult_add(x[j], y[i])


class PETScInterface:
    def __init__(self, arg_space, action_space, nullspace):
        self._arg_space = arg_space
        self._action_space = action_space
        self._nullspace = nullspace

        self._x = arg_space.new_split()
        self._y = action_space.new_split()

        if len(arg_space.flattened_space()) == 1:
            self._x_fn, = tuple(iter_sub(self._x))
        else:
            self._x_fn = arg_space.new_mixed()
        if len(action_space.flattened_space()) == 1:
            self._y_fn, = tuple(iter_sub(self._y))
        else:
            self._y_fn = action_space.new_mixed()

        if isinstance(self._nullspace, NoneNullspace):
            self._x_c = self._x
        else:
            self._x_c = arg_space.new_split()

    def _pre_mult(self, x_petsc):
        with vec(self._x_fn, Access.WRITE) as x_v:
            # assert x_petsc.getSizes() == x_v.getSizes()
            x_petsc.copy(result=x_v)
        if len(self._arg_space.flattened_space()) != 1:
            self._arg_space.mixed_to_split(self._x, self._x_fn)

        if not isinstance(self._nullspace, NoneNullspace):
            for x_i, x_c_i in zip_sub(self._x, self._x_c):
                x_c_i.assign(x_i)

        for y_i in iter_sub(self._y):
            with vec(y_i, Access.WRITE) as y_i_v:
                y_i_v.zeroEntries()

    def _post_mult(self, y_petsc):
        if len(self._action_space.flattened_space()) != 1:
            self._action_space.split_to_mixed(self._y_fn, self._y)

        with vec(self._y_fn, Access.READ) as y_v:
            assert y_petsc.getSizes() == y_v.getSizes()
            y_v.copy(result=y_petsc)


class SystemMatrix(PETScInterface):
    def __init__(self, arg_space, action_space, matrix, nullspace):
        if matrix.arg_space() != arg_space.split_space():
            raise ValueError("Invalid space")
        if matrix.action_space() != action_space.split_space():
            raise ValueError("Invalid space")

        super().__init__(arg_space, action_space, nullspace)
        self._matrix = matrix

    @flag_errors
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

    @flag_errors
    def apply(self, pc, x, y):
        self._pre_mult(x)

        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.pc_pre_mult_correct(self._x_c)
        self._pc_fn(self._y, self._x_c)
        if not isinstance(self._nullspace, NoneNullspace):
            self._nullspace.pc_post_mult_correct(
                self._y, self._x)

        self._post_mult(y)


class ConvergenceError(RuntimeError):
    """An outer Krylov solver convergence error. The PETSc KSP can be accessed
    via the `ksp` attribute.
    """

    def __init__(self, *args, ksp, **kwargs):
        super().__init__(*args, **kwargs)
        self.ksp = ksp


class System:
    """A linear system

    .. math::

        A u = b.

    :arg arg_spaces: Defines the space for `u`.
    :arg action_spaces: Defines the space for `b`.
    :arg blocks: One of

        - A :class:`Matrix` or UFL :class:`Form` defining :math:`A`.
        - A :class:`Mapping` with items `((i, j), block)` where the matrix
          associated with the block in the `i` th and `j` th column is defined
          by `block`. Each `block` is a :class:`Matrix` or UFL :class:`Form`,
          or `None` to indicate a zero block.

    :arg nullspaces: A :class:`Nullspace` or a :class:`Sequence` of
        :class:`Nullspace` objects defining the nullspace and left nullspace of
        :math:`A`. `None` indicates a :class:`NoneNullspace`.
    :arg comm: MPI communicator.
    """

    def __init__(self, arg_spaces, action_spaces, blocks, *,
                 nullspaces=None, comm=None):
        if isinstance(arg_spaces, MixedSpace):
            arg_space = arg_spaces
        else:
            arg_space = BackendMixedSpace(arg_spaces)
        arg_spaces = arg_space.split_space()
        if isinstance(action_spaces, MixedSpace):
            action_space = action_spaces
        else:
            action_space = BackendMixedSpace(action_spaces)
        action_spaces = action_space.split_space()

        matrix = BlockMatrix(arg_spaces, action_spaces, blocks)

        nullspace = BlockNullspace(nullspaces)
        if isinstance(nullspace, BlockNullspace):
            if len(nullspace) != len(arg_spaces):
                raise ValueError("Invalid space")
            if len(nullspace) != len(action_spaces):
                raise ValueError("Invalid space")

        if comm is None:
            self._comm = mesh_comm(arg_space.mesh())
        else:
            self._comm = comm
        self._arg_space = arg_space
        self._action_space = action_space
        self._matrix = matrix
        self._nullspace = nullspace

    def solve(self, u, b, *,
              solver_parameters=None, pc_fn=None, configure=None,
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
        :arg configure: A callable accepting the PETSc :class:`KSP`.

            .. code-block:: python

                def configure(ksp):

            Called after all other configuration of the :class:`KSP`, but
            before :meth:`KSP.setUp`.
        :arg correct_initial_guess: Whether to apply a nullspace correction to
            the initial guess.
        :arg correct_solution: Whether to apply a nullspace correction to
            the solution.
        :returns: The PETSc :class:`KSP`.
        """

        global _error_flag

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
        u = tuple_sub(u, self._arg_space.split_space())

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
        b = tuple_sub(b, self._action_space.split_space())

        if tuple(u_i.function_space() for u_i in iter_sub(u)) \
                != self._arg_space.flattened_space():
            raise ValueError("Invalid space")
        for b_i, space in zip_sub(b, self._action_space.split_space()):
            if b_i is not None and b_i.function_space() != space:
                raise ValueError("Invalid space")

        b_c = self._action_space.new_split()
        for b_c_i, b_i in zip_sub(b_c, b):
            if b_i is not None:
                b_c_i.assign(b_i)

        A = SystemMatrix(self._arg_space, self._action_space,
                         self._matrix, self._nullspace)

        mat_A = PETSc.Mat().createPython(
            (self._action_space.sizes(), self._arg_space.sizes()), A,
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

        if configure is not None:
            configure(ksp_solver)
        ksp_solver.setUp()

        if correct_initial_guess:
            self._nullspace.correct_soln(u)
        self._nullspace.correct_rhs(b_c)

        if len(self._arg_space.flattened_space()) == 1:
            u_fn, = tuple(iter_sub(u))
        else:
            u_fn = self._arg_space.new_mixed()
            self._arg_space.split_to_mixed(u_fn, u)
        if len(self._action_space.flattened_space()) == 1:
            b_fn, = tuple(iter_sub(b_c))
        else:
            b_fn = self._action_space.new_mixed()
            self._action_space.split_to_mixed(b_fn, b_c)
        del b_c

        _error_flag = False
        with vec(u_fn) as u_v, vec(b_fn) as b_v:
            ksp_solver.solve(b_v, u_v)
        del b_fn

        if len(self._arg_space.flattened_space()) != 1:
            self._arg_space.mixed_to_split(u, u_fn)
        del u_fn

        if correct_solution:
            # Not needed if the linear problem were to be solved exactly
            self._nullspace.correct_soln(u)

        if ksp_solver.getConvergedReason() <= 0:
            raise ConvergenceError("Solver failed to converge",
                                   ksp=ksp_solver)
        if _error_flag:
            raise ConvergenceError("Error encountered in PETSc solve",
                                   ksp=ksp_solver)

        return ksp_solver
