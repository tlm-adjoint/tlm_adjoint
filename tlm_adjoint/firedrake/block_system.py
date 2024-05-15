"""Firedrake specific extensions to :mod:`tlm_adjoint.block_system`.
"""

from ..block_system import (
    BlockMatrix as _BlockMatrix, BlockNullspace, LinearSolver as _LinearSolver,
    Matrix, MixedSpace, NoneNullspace, Nullspace, TypedSpace)
from ..interface import space_eq

from firedrake import Constant, DirichletBC, Function, TestFunction, assemble

import ufl

from collections.abc import Sequence
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None

__all__ = \
    [
        "TypedSpace",
        "MixedSpace",

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

        "LinearSolver"
    ]


def apply_bcs(u, bcs):
    if not isinstance(bcs, Sequence):
        bcs = (bcs,)
    if len(bcs) > 0 and not isinstance(u.function_space(), type(bcs[0].function_space())):  # noqa: E501
        u_bc = u.riesz_representation("l2")
    else:
        u_bc = u
    for bc in bcs:
        bc.apply(u_bc)


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
        with x.dat.vec_ro as x_v:
            x_sum = x_v.sum()
            N = x_v.getSize()

        with y.dat.vec as y_v:
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
        U.interpolate(Constant(1.0))
        MU = assemble(ufl.inner(U, TestFunction(space)) * ufl.dx)
        UMU = assemble(ufl.inner(U, U) * ufl.dx)

        self._alpha = alpha
        self._U = U
        self._MU = MU
        self._UMU = UMU

    @staticmethod
    def _correct(x, y, u, v, *, alpha=1.0):
        with x.dat.vec_ro as x_v, u.dat.vec_ro as u_v:
            u_x = x_v.dot(u_v)

        with y.dat.vec as y_v, v.dat.vec_ro as v_v:
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

    :arg bcs: A :class:`firedrake.bcs.DirichletBC`, or a :class:`Sequence` of
        :class:`firedrake.bcs.DirichletBC` objects.
    :arg alpha: Defines the linear constraint matrix :math:`S = \alpha M`.
    """

    def __init__(self, bcs, *, alpha=1.0):
        if isinstance(bcs, Sequence):
            bcs = tuple(bcs)
        else:
            bcs = (bcs,)

        space = bcs[0].function_space()
        for bc in bcs:
            if not space_eq(bc.function_space(), space):
                raise ValueError("Invalid space")
            if not isinstance(bc._function_arg, ufl.classes.Zero):
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
        with self._c.dat.vec_wo as c_v:
            c_v.zeroEntries()

        apply_bcs(self._c,
                  tuple(DirichletBC(x.function_space(), x, bc.sub_domain)
                        for bc in self._bcs))

        with self._c.dat.vec_ro as c_v, y.dat.vec as y_v:
            y_v.axpy(alpha, c_v)

    def constraint_correct_lhs(self, x, y):
        self._constraint_correct_lhs(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._constraint_correct_lhs(b, u, alpha=1.0 / self._alpha)


class PETScMatrix(Matrix):
    r"""A :class:`tlm_adjoint.block_system.Matrix` associated with a
    :class:`firedrake.matrix.Matrix` :math:`A` mapping :math:`V \rightarrow W`.

    :arg arg_space: Defines the space `V`.
    :arg action_space: Defines the space `W`.
    :arg a: The :class:`firedrake.matrix.Matrix`.
    """

    def __init__(self, arg_space, action_space, a):
        super().__init__(arg_space, action_space)
        self._matrix = a

    def mult_add(self, x, y):
        matrix = self._matrix.petscmat
        with x.dat.vec_ro as x_v, y.dat.vec as y_v:
            matrix.multAdd(x_v, y_v, y_v)


def form_matrix(a, *args, **kwargs):
    """Construct a :class:`.PETScMatrix` associated with a given sesquilinear
    form.

    :arg a: A :class:`ufl.Form` defining the sesquilinear form.
    :returns: The :class:`.PETScMatrix`.

    Remaining arguments are passed to the :func:`firedrake.assemble.assemble`
    function.
    """

    test, trial = a.arguments()
    assert test.number() < trial.number()

    return PETScMatrix(
        trial.function_space(), test.function_space().dual(),
        assemble(a, *args, **kwargs))


class BlockMatrix(_BlockMatrix):
    def __setitem__(self, key, value):
        if isinstance(value, ufl.classes.Form):
            value = form_matrix(value)
        super().__setitem__(key, value)


class LinearSolver(_LinearSolver):
    def __init__(self, A, *args, **kwargs):
        if isinstance(A, ufl.classes.Form):
            A = form_matrix(A)
        super().__init__(A, *args, **kwargs)
