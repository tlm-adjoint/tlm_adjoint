"""Firedrake specific extensions to :mod:`tlm_adjoint.block_system`.
"""

from .backend import (
    LinearSolver as backend_LinearSolver, TestFunction, TrialFunction,
    backend_Cofunction, backend_DirichletBC, backend_assemble, dx, inner)
from ..interface import (
    packed, space_dtype, space_eq, var_axpy, var_copy, var_dtype,
    var_get_values, var_inner, var_local_size, var_new, var_set_values)

from ..block_system import (
    BlockMatrix as _BlockMatrix, BlockNullspace, Eigensolver,
    LinearSolver as _LinearSolver, Matrix, MatrixFunctionSolver,
    MatrixFreeMatrix, MixedSpace, NoneNullspace, Nullspace, TypedSpace)
from ..petsc import flattened_options

from .backend_interface import assemble, matrix_multiply
from .variables import Cofunction, Constant, Function

from functools import cached_property
import numpy as np
import ufl

__all__ = \
    [
        "TypedSpace",
        "MixedSpace",

        "Nullspace",
        "NoneNullspace",
        "BlockNullspace",
        "ConstantNullspace",
        "UnityNullspace",
        "DirichletBCNullspace",

        "Matrix",
        "MatrixFreeMatrix",
        "BlockMatrix",
        "PETScMatrix",
        "form_matrix",

        "LinearSolver",
        "Eigensolver",
        "MatrixFunctionSolver",

        "WhiteNoiseSampler"
    ]


def apply_bcs(u, bcs):
    bcs = packed(bcs)
    if isinstance(u, backend_Cofunction):
        u = u.riesz_representation("l2")
    for bc in bcs:
        if not space_eq(bc.function_space(), u.function_space()):
            raise ValueError("Invalid space")
    for bc in bcs:
        bc.apply(u)


class ConstantNullspace(Nullspace):
    r"""Nullspace and left nullspace spanned by the vector of ones.

    Here :math:`V = U`, :math:`U` is a single column matrix whose elements are
    ones, :math:`C = M`, and :math:`M` is an identity matrix.

    Parameters
    ----------

    alpha : scalar
        Defines the linear constraint matrix :math:`S = \left( \alpha / N
        \right)` where :math:`N` is the length of the vector of ones.
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
    r"""Nullspace and left nullspace spanned by the unity-valued function.

    Here :math:`V = U`, :math:`U` is a single column matrix containing the
    degree-of-freedom vector for the unity-valued function, :math:`C = M`,
    and :math:`M` is the mass matrix.

    Parameters
    ----------

    space : :class:`firedrake.functionspaceimpl.WithGeometry`
        A scalar-valued function space containing the unity-valued function.
    alpha : scalar
        Defines the linear constraint matrix :math:`S = \alpha \left( U^* M U
        \right)^{-1}`.
    """

    def __init__(self, space, *, alpha=1.0):
        U = Function(space, name="U")
        U.interpolate(Constant(1.0))
        MU = assemble(ufl.inner(U, TestFunction(space)) * ufl.dx)
        UMU = assemble(ufl.inner(U, U) * ufl.dx)

        self._space = space
        self._alpha = alpha
        self._U = U
        self._MU = MU
        self._UMU = UMU

    @staticmethod
    def _correct(x, y, u, v, *, alpha=1.0):
        u_x = var_inner(x, u)
        var_axpy(y, alpha * u_x, v)

    def apply_nullspace_transformation_lhs_right(self, x):
        if not space_eq(x.function_space(), self._space):
            raise ValueError("Invalid space")
        self._correct(
            x, x, self._MU, self._U, alpha=-1.0 / self._UMU)

    def apply_nullspace_transformation_lhs_left(self, y):
        if not space_eq(y.function_space(), self._space.dual()):
            raise ValueError("Invalid space")
        self._correct(
            y, y, self._U, self._MU, alpha=-1.0 / self._UMU)

    def constraint_correct_lhs(self, x, y):
        if not space_eq(x.function_space(), self._space):
            raise ValueError("Invalid space")
        if not space_eq(y.function_space(), self._space.dual()):
            raise ValueError("Invalid space")
        self._correct(
            x, y, self._MU, self._MU, alpha=self._alpha / self._UMU)

    def pc_constraint_correct_soln(self, u, b):
        if not space_eq(u.function_space(), self._space):
            raise ValueError("Invalid space")
        if not space_eq(b.function_space(), self._space.dual()):
            raise ValueError("Invalid space")
        self._correct(
            b, u, self._U, self._U, alpha=1.0 / (self._alpha * self._UMU))


class DirichletBCNullspace(Nullspace):
    """Nullspace and left nullspace associated with homogeneous Dirichlet
    boundary conditions.

    Here :math:`V = U`, :math:`U` is a zero-one matrix with exactly one
    non-zero per column corresponding to one boundary condition
    degree-of-freedom, :math:`C = M`, and :math:`M` is an identity matrix.

    Parameters
    ----------

    bcs : :class:`firedrake.bcs.DirichletBC` or \
            Sequence[:class:`firedrake.bcs.DirichletBC`, ...]
        Homogeneous Dirichlet boundary conditions.
    alpha : scalar
        Defines the linear constraint matrix :math:`S = \\alpha M`.
    """

    def __init__(self, bcs, *, alpha=1.0):
        bcs = packed(bcs)
        space = bcs[0].function_space()
        for bc in bcs:
            if not space_eq(bc.function_space(), space):
                raise ValueError("Invalid space")
            if not isinstance(bc._function_arg, ufl.classes.Zero):
                raise ValueError("Homogeneous boundary conditions required")

        super().__init__()
        self._space = space
        self._bcs = bcs
        self._alpha = alpha

    def apply_nullspace_transformation_lhs_right(self, x):
        apply_bcs(x, self._bcs)

    def apply_nullspace_transformation_lhs_left(self, y):
        apply_bcs(y, self._bcs)

    def _constraint_correct_lhs(self, x, y, *, alpha=1.0):
        if isinstance(x, backend_Cofunction):
            x = x.riesz_representation("l2")
        c = var_new(y)
        apply_bcs(
            c,
            tuple(backend_DirichletBC(x.function_space(), x, bc.sub_domain)
                  for bc in self._bcs))
        var_axpy(y, alpha, c)

    def constraint_correct_lhs(self, x, y):
        self._constraint_correct_lhs(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._constraint_correct_lhs(b, u, alpha=1.0 / self._alpha)


class PETScMatrix(Matrix):
    r"""A :class:`tlm_adjoint.block_system.Matrix` associated with a
    :class:`firedrake.matrix.Matrix` :math:`A` defining a mapping
    :math:`V \rightarrow W`.

    Parameters
    ----------

    A : :class:`firedrake.matrix.Matrix`
    """

    def __init__(self, A):
        test, trial = A.form.arguments()
        assert test.number() < trial.number()
        super().__init__(trial.function_space(), test.function_space().dual())
        self._A = A

    def mult_add(self, x, y):
        matrix_multiply(self._A, x, tensor=y, addto=True)


def form_matrix(a, *args, **kwargs):
    """Construct a :class:`.PETScMatrix` associated with a given sesquilinear
    form.

    Parameters
    ----------

    a : :class:`ufl.Form`
        Defines the sesquilinear form.
    args, kwargs
        Passed to the :func:`firedrake.assemble.assemble` function.

    Returns
    -------

    :class:`.PETScMatrix`.
        :class:`.PETScMatrix` defined by the assembled sesquilinear form.
    """

    return PETScMatrix(backend_assemble(a, *args, **kwargs))


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


class WhiteNoiseSampler:
    r"""White noise sampling.

    Utility class for drawing independent spatial white noise samples.
    Generates a sample using

    .. math::

        X = M^{-1} \Xi^{-T} \sqrt{ \Xi^T M \Xi } Z,

    where

        - :math:`M` is the mass matrix.
        - :math:`\Xi` is a preconditioner.
        - :math:`Z` is a vector whose elements are independent standard
          Gaussian samples.

    The matrix square root is computed using SLEPc.

    Parameters
    ----------

    space : :class:`firedrake.functionspaceimpl.WithGeometry`
        The function space.
    rng : :class:`numpy.random.Generator`
        Pseudorandom number generator.
    precondition : :class:`bool`
        If `True` then :math:`\Xi` is set equal to the inverse of the
        (principal) square root of the diagonal of :math:`M`. Otherwise it is
        set equal to the identity.
    M : :class:`firedrake.matrix.Matrix`
        Mass matrix. Constructed by finite element assembly if not supplied.
    mfn_solver_parameters : :class:`Mapping`
        :class:`slepc4py.SLEPc.MFN` solver parameters, used for the matrix
        square root action.
    ksp_solver_parameters : :class:`Mapping`
        Solver parameters, used for :math:`M^{-1}`.

    Attributes
    ----------

    space : :class:`firedrake.functionspaceimpl.WithGeometry`
        The function space.
    rng : :class:`numpy.random.Generator`
        Pseudorandom number generator.
    """

    def __init__(self, space, rng, *, precondition=True, M=None,
                 mfn_solver_parameters=None, ksp_solver_parameters=None):
        if mfn_solver_parameters is None:
            mfn_solver_parameters = {}
        else:
            mfn_solver_parameters = dict(mfn_solver_parameters)
        if mfn_solver_parameters.get("mfn_type", "krylov") != "krylov":
            raise ValueError("Invalid mfn_type")
        if mfn_solver_parameters.get("fn_type", "sqrt") != "sqrt":
            raise ValueError("Invalid fn_type")
        mfn_solver_parameters.update({"mfn_type": "krylov",
                                      "fn_type": "sqrt"})
        if ksp_solver_parameters is None:
            ksp_solver_parameters = {}

        if not issubclass(space_dtype(space), np.floating):
            raise ValueError("Real space required")
        if M is None:
            test = TestFunction(space)
            trial = TrialFunction(space)
            M = assemble(inner(trial, test) * dx)

        if precondition:
            M_diag = M.petscmat.getDiagonal()
            pc = np.sqrt(M_diag.getArray(True))
        else:
            pc = None

        self._space = space
        self._M = M
        self._rng = rng
        self._pc = pc
        self._mfn_solver_parameters = dict(flattened_options(mfn_solver_parameters))  # noqa: E501
        self._ksp_solver_parameters = dict(flattened_options(ksp_solver_parameters))  # noqa: E501

    @cached_property
    def _mfn(self):
        def mult(x, y):
            if self._pc is not None:
                x = var_copy(x)
                var_set_values(x, var_get_values(x) / self._pc)
            matrix_multiply(self._M, x, tensor=y)
            if self._pc is not None:
                var_set_values(y, var_get_values(y) / self._pc)

        return MatrixFunctionSolver(
            MatrixFreeMatrix(self.space, self.space, mult),
            solver_parameters=self._mfn_solver_parameters)

    @cached_property
    def _ksp(self):
        return backend_LinearSolver(
            self._M, solver_parameters=self._ksp_solver_parameters)

    @property
    def space(self):
        return self._space

    @property
    def rng(self):
        return self._rng

    def dual_sample(self):
        r"""Generate a new sample in the dual space.

        The result is given by

        .. math::

            X = \Xi^{-T} \sqrt{ \Xi^T M \Xi } Z.

        Returns
        -------
        :class:`firedrake.cofunction.Cofunction`
            The sample.
        """

        Z = Function(self.space)
        var_set_values(
            Z, self.rng.standard_normal(var_local_size(Z), dtype=var_dtype(Z)))
        X = Cofunction(self.space.dual())
        self._mfn.solve(Z, X)
        if self._pc is not None:
            var_set_values(X, var_get_values(X) * self._pc)
        return X

    def sample(self):
        r"""Generate a new sample.

        The result is given by

        .. math::

            X = M^{-1} \Xi^{-T} \sqrt{ \Xi^T M \Xi } Z.

        Returns
        -------
        :class:`firedrake.function.Function`
            The sample.
        """

        Y = self.dual_sample()
        X = Function(self.space)
        self._ksp.solve(X, Y)
        return X
