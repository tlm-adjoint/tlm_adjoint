#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..interface import (
    check_space_types, check_space_types_conjugate, comm_dup_cached,
    function_assign, function_axpy, function_copy, function_dtype,
    function_get_values, function_inner, function_new_conjugate,
    function_set_values, function_space, function_space_type, is_function,
    space_comm, space_dtype)

from ..eigendecomposition import eigendecompose

from .block_system import (
    BackendMixedSpace, BlockNullspace, Matrix, NoneNullspace, Preconditioner,
    System, iter_sub, tuple_sub)
from .functions import Function

from collections.abc import Sequence
import numpy as np
import petsc4py.PETSc as PETSc
import warnings

__all__ = \
    [
        "HessianSystem",
        "hessian_eigendecompose",
        "B_inv_orthonormality_test",
        "hessian_eigendecomposition_pc",
    ]


class MixedSpace(BackendMixedSpace):
    def __init__(self, spaces, space_types=None):
        if isinstance(spaces, Sequence):
            spaces = tuple(spaces)
        else:
            spaces = (spaces,)
        spaces = tuple_sub(spaces, spaces)

        if space_types is None:
            space_types = "primal"
        if space_types in {"primal", "conjugate", "dual", "conjugate_dual"}:
            space_types = tuple(space_types for _ in iter_sub(spaces))
        else:
            space_types = tuple(iter_sub(space_types))

        super().__init__(spaces)
        if len(space_types) != len(self.flattened_space()):
            raise ValueError("Invalid space types")
        self._space_types = space_types

    def new_split(self, *args, **kwargs):
        flattened_space = self.flattened_space()
        assert len(flattened_space) == len(self._space_types)
        return tuple_sub((Function(space, *args, space_type=space_type, **kwargs)  # noqa: E501
                          for space, space_type in zip(flattened_space, self._space_types)),  # noqa: E501
                         self.split_space())

    def new_mixed(self, *args, **kwargs):
        space_type = self._space_types[0]
        return Function(self.mixed_space(), space_type=space_type,
                        *args, **kwargs)


# Complex note: It is convenient to define a Hessian matrix action in terms of
# the *conjugate* of the action, i.e. (H \zeta)^{*,T}, e.g. this is the form
# returned by reverse-over-forward AD. However complex conjugation is then
# needed in a number of places (e.g. one cannot define an eigenproblem directly
# in terms of the conjugate of an action, as this is antilinear, rather than
# linear). The following functions are used to apply complex conjugation where
# needed. In the real case these are used to avoid space type warnings.


def function_copy_conjugate(x):
    y = function_new_conjugate(x)
    function_set_values(y, function_get_values(x).conjugate())
    return y


def function_assign_conjugate(x, y):
    check_space_types_conjugate(x, y)
    function_set_values(x, function_get_values(y).conjugate())


def function_axpy_conjugate(y, alpha, x, /):
    check_space_types_conjugate(y, x)
    function_set_values(
        y, function_get_values(y) + alpha * function_get_values(x).conjugate())


class HessianMatrix(Matrix):
    def __init__(self, H, M):
        if is_function(M):
            M = (M,)
        else:
            M = tuple(M)
        space = tuple(map(function_space, M))

        super().__init__(space, space)
        self._H = H
        self._M = M

    def mult_add(self, x, y):
        if is_function(x):
            x = (x,)
        if is_function(y):
            y = (y,)

        if len(x) != len(self._M):
            raise ValueError("Invalid Hessian argument")
        for x_i, m in zip(x, self._M):
            check_space_types(x_i, m)

        _, _, ddJ = self._H.action(self._M, x)

        if len(y) != len(ddJ):
            raise ValueError("Invalid Hessian action")
        for y_i, ddJ_i in zip(y, ddJ):
            function_axpy_conjugate(y_i, 1.0, ddJ_i)


class HessianSystem(System):
    """Defines a linear system involving a Hessian matrix,

    .. math::

        H u = b.

    :arg H: A :class:`tlm_adjoint.hessian.Hessian` defining :math:`H`.
    :arg M: A function or a :class:`Sequence` of functions defining the
        control.
    :arg nullspace: A
        :class:`tlm_adjoint._code_generator.block_system.Nullspace` or a
        :class:`Sequence` of
        :class:`tlm_adjoint._code_generator.block_system.Nullspace` objects
        defining the nullspace and left nullspace of the Hessian matrix. `None`
        indicates a
        :class:`tlm_adjoint._code_generator.block_system.NoneNullspace`.
    :arg comm: MPI communicator.
    """

    def __init__(self, H, M, *,
                 nullspace=None, comm=None):
        if is_function(M):
            M = (M,)

        arg_spaces = MixedSpace(
            (tuple(map(function_space, M)),),
            space_types=tuple(map(function_space_type, M)))
        action_spaces = MixedSpace(
            (tuple(map(function_space, M)),),
            space_types=tuple(function_space_type(m, rel_space_type="dual")
                              for m in M))

        matrix = HessianMatrix(H, M)

        if comm is None:
            comm = space_comm(arg_spaces.mixed_space())
        comm = comm_dup_cached(comm, key="HessianSystem")

        super().__init__(
            arg_spaces, action_spaces, matrix,
            nullspaces=BlockNullspace(nullspace), comm=comm)

    def solve(self, u, b, **kwargs):
        """Solve a linear system involving a Hessian matrix,

        .. math::

            H u = b.

        :arg u: A function or a :class:`Sequence` of functions defining the
            solution :math:`u`.
        :arg b: A function or a :class:`Sequence` of functions defining the
            conjugate of the right-hand-side :math:`b`.

        Remaining arguments are handed to the base class :meth:`solve` method.
        """

        if is_function(b):
            b = function_copy_conjugate(b)
        else:
            b = tuple_sub(map(function_copy_conjugate, iter_sub(b)), b)
        return super().solve(u, b, **kwargs)


def hessian_eigendecompose(
        H, m, B_inv_action, B_action, *,
        nullspace=None, problem_type=None, configure=None,
        correct_eigenvectors=True, **kwargs):
    r"""Interface with SLEPc via slepc4py, for the matrix free solution of
    generalized eigenproblems

    .. math::

        H v = \lambda B^{-1} v,

    where :math:`H` is a Hessian matrix.

    Despite the notation :math:`B^{-1}` may be singular, defining an inverse
    operator only on an appropriate subspace.

    :arg H: A :class:`tlm_adjoint.hessian.Hessian`.
    :arg m: A backend :class:`Function` defining the control.
    :arg B_inv_action: A callable accepting a backend :class:`Function`
        defining `v` and computing the conjugate of the action of
        :math:`B^{-1}` on :math:`v`, returning the result as a backend
        :class:`Function`.
    :arg B_action: A callable accepting a backend :class:`Function` defining
        :math:`v` and computing the action of :math:`B` on the conjugate of
        :math:`v`, returning the result as a backend :class:`Function`.
    :arg nullspace: A
        :class:`tlm_adjoint._code_generator.block_system.Nullspace` defining
        the nullspace and left nullspace of :math:`H` and :math:`B^{-1}`.
    :arg problem_type: The eigenproblem type -- see
        :class:`slepc4py.SLEPc.EPS.ProblemType`. Defaults to
        `slepc4py.SLEPc.EPS.ProblemType.GHEP` in the real case and
        `slepc4py.SLEPc.EPS.ProblemType.GNHEP` in the complex case.
    :arg configure: A callable accepting a single :class:`slepc4py.SLEPc.EPS`
        argument. Used for detailed manual configuration. Called after all
        other configuration options are set, but before the :meth:`EPS.setUp`
        method is called.
    :arg correct_eigenvectors: Whether to apply a nullspace correction to the
        eigenvectors.

    Remaining keyword arguments are passed to
    :func:`tlm_adjoint.eigendecomposition.eigendecompose`.
    """

    space = function_space(m)

    arg_space_type = function_space_type(m)
    arg_space = MixedSpace(space, space_types=arg_space_type)
    assert arg_space.split_space() == (space,)
    assert arg_space.flattened_space() == (space,)
    assert arg_space.mixed_space() == space

    action_space_type = function_space_type(m, rel_space_type="dual")
    action_space = MixedSpace(space, space_types=action_space_type)
    assert action_space.split_space() == (space,)
    assert action_space.flattened_space() == (space,)
    assert action_space.mixed_space() == space

    if nullspace is None:
        nullspace = NoneNullspace()

    def H_action(x):
        x = function_copy(x)
        nullspace.pre_mult_correct_lhs(x)
        _, _, y = H.action(m, x)
        y = function_copy_conjugate(y)
        nullspace.post_mult_correct_lhs(None, y)
        return y

    B_inv_action_arg = B_inv_action

    def B_inv_action(x):
        x = function_copy(x)
        nullspace.pre_mult_correct_lhs(x)
        y = B_inv_action_arg(function_copy(x))
        y = function_copy_conjugate(y)
        nullspace.post_mult_correct_lhs(x, y)
        return y

    B_action_arg = B_action

    def B_action(x, y):
        x, = x
        y, = tuple(map(function_copy_conjugate, y))
        # Nullspace corrections applied by the Preconditioner class
        function_assign(x, B_action_arg(y))

    configure_arg = configure

    def configure(eps):
        _, B_inv = eps.getOperators()
        ksp_solver = eps.getST().getKSP()

        B_pc = Preconditioner(
            action_space, arg_space,
            B_action, BlockNullspace(nullspace))
        pc = PETSc.PC().createPython(
            B_pc, comm=ksp_solver.comm)
        pc.setOperators(B_inv)
        pc.setUp()

        ksp_solver.setType(PETSc.KSP.Type.PREONLY)
        ksp_solver.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
        ksp_solver.setPC(pc)
        ksp_solver.setUp()

        if hasattr(eps, "setPurify"):
            eps.setPurify(False)
        else:
            warnings.warn("slepc4py.SLEPc.EPS.setPurify not available",
                          RuntimeWarning)

        if configure_arg is not None:
            configure_arg(eps)

    if problem_type is None:
        import slepc4py.SLEPc as SLEPc
        if issubclass(space_dtype(space), (float, np.floating)):
            problem_type = SLEPc.EPS.ProblemType.GHEP
        else:
            problem_type = SLEPc.EPS.ProblemType.GNHEP

    Lam, V = eigendecompose(
        space, H_action, B_action=B_inv_action, arg_space_type=arg_space_type,
        action_space_type=action_space_type, problem_type=problem_type,
        configure=configure, **kwargs)

    if correct_eigenvectors:
        if len(V) > 0 and isinstance(V[0], Sequence):
            assert len(V[0]) == len(V[1])
            for V_r, V_i in zip(*V):
                nullspace.correct_soln(V_r)
                nullspace.correct_soln(V_i)
        else:
            for V_r in V:
                nullspace.correct_soln(V_r)

    return Lam, V


def B_inv_orthonormality_test(V, B_inv_action):
    """Check for :math:`B^{-1}`-orthonormality.

    Requires real spaces.

    :arg B_inv_action: A callable accepting a backend :class:`Function`
        defining `v` and computing the action of :math:`B^{-1}` on :math:`v`,
        returning the result as a backend :class:`Function`.
    :arg V: A :class:`Sequence` of functions to test for
        :math:`B^{-1}`-orthonormality.
    :returns: A :class:`tuple` `(max_diagonal_error_norm,
        max_off_diagonal_error_norm)` with

            - `max_diagonal_error_norm`: The maximum :math:`B^{-1}`
              normalization error magnitude.
            - `max_diagonal_error_norm`: The maximum :math:`B^{-1}`
              orthogonality error magnitude.
    """

    if len(V) > 0 and isinstance(V[0], Sequence):
        raise ValueError("Cannot supply separate real/complex eigenvector "
                         "components")

    B_inv_V = []
    for v in V:
        if not issubclass(function_dtype(v), (float, np.floating)):
            raise ValueError("Real dtype required")
        B_inv_V.append(B_inv_action(function_copy(v)))
        if not issubclass(function_dtype(B_inv_V[-1]), (float, np.floating)):
            raise ValueError("Real dtype required")

    max_diagonal_error_norm = 0.0
    max_off_diagonal_error_norm = 0.0
    assert len(V) == len(B_inv_V)
    for i, v in enumerate(V):
        for j, B_inv_v in enumerate(B_inv_V):
            if i == j:
                max_diagonal_error_norm = max(
                    max_diagonal_error_norm,
                    abs(function_inner(v, B_inv_v) - 1.0))
            else:
                max_off_diagonal_error_norm = max(
                    max_off_diagonal_error_norm,
                    abs(function_inner(v, B_inv_v)))

    return max_diagonal_error_norm, max_off_diagonal_error_norm


def hessian_eigendecomposition_pc(B_action, Lam, V):
    r"""Construct a Hessian matrix preconditioner using a partial spectrum
    generalized eigendecomposition. Assumes that the Hessian matrix consists of
    two terms

    .. math::

        H = R^{-1} + B^{-1},

    where :math:`R` and :math:`B` are symmetric.

    Assumes real spaces. Despite the notation :math:`R^{-1}` and :math:`B^{-1}`
    (and later :math:`H^{-1}`) may be singular, defining inverse operators only
    on an appropriate subspace. :math:`B` is assumed to define a symmetric
    positive definite operator on that subspace.

    The approximation is defined via

    .. math::

        H^{-1} \approx B + V \Lambda \left( I + \lambda \right)^{-1} V^T

    where

    .. math::

        R^{-1} V = B^{-1} V \Lambda,

    and where :math:`\Lambda` is a diagonal matrix and :math:`V` has
    :math:`B^{-1}`-orthonormal columns, :math:`V^T B^{-1} V = I`.

    This low rank update approximation for the Hessian matrix inverse is
    described in

        - Tobin Isaac, Noemi Petra, Georg Stadler, and Omar Ghattas, 'Scalable
          and efficient algorithms for the propagation of uncertainty from data
          through inference to prediction for large-scale problems, with
          application to flow of the Antarctic ice sheet', Journal of
          Computational Physics, 296, pp. 348--368, 2015, doi:
          10.1016/j.jcp.2015.04.047

    See in particular their equation (20).

    :arg B_action: A callable accepting a backend :class:`Function` defining
        :math:`v` and computing the action of :math:`B` on :math:`v`, returning
        the result as a backend :class:`Function`.
    :arg Lam: A :class:`Sequence` defining the diagonal of :math:`\Lambda`.
    :arg V: A :class:`Sequence` of backend :class:`Function` objects defining
        the columns of :math:`V`.
    :returns: A callable suitable for use as the `pc_fn` argument to
        :meth:`HessianSystem.solve`.
    """

    if len(V) > 0 and isinstance(V[0], Sequence):
        raise ValueError("Cannot supply separate real/complex eigenvector "
                         "components")

    Lam = tuple(Lam)
    V = tuple(V)
    if len(Lam) != len(V):
        raise ValueError("Invalid eigenpairs")

    def pc_fn(u, b):
        b = function_copy_conjugate(b)
        function_assign(u, B_action(function_copy(b)))

        assert len(Lam) == len(V)
        for lam, v in zip(Lam, V):
            alpha = -(lam / (1.0 + lam)) * function_inner(b, v)
            function_axpy(u, alpha, v)

    return pc_fn
