from ..interface import (
    is_var, var_assign, var_assign_conjugate, var_axpy, var_copy,
    var_copy_conjugate, var_dtype, var_inner)

from ..block_system import TypedSpace, eigensolve

from collections.abc import Sequence
import numpy as np

__all__ = \
    [
        "hessian_eigendecompose",
        "B_inv_orthonormality_test",
        "hessian_eigendecomposition_pc",
    ]


def hessian_eigendecompose(
        H, m, B_inv_action, B_action, *args, **kwargs):
    r"""Interface with SLEPc via slepc4py, for the matrix free solution of
    generalized eigenproblems

    .. math::

        H v = \lambda B^{-1} v,

    where :math:`H` is a Hessian matrix.

    Despite the notation :math:`B^{-1}` may be singular, defining an inverse
    operator only on an appropriate subspace.

    :arg H: A :class:`.Hessian`.
    :arg m: A :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` defining the control.
    :arg B_inv_action: A callable accepting a
        :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` defining :math:`v` and
        computing the conjugate of the action of :math:`B^{-1}` on :math:`v`,
        returning the result as a :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`.
    :arg B_action: A callable accepting a :class:`firedrake.function.Function`
        or :class:`firedrake.cofunction.Cofunction` defining :math:`v` and
        computing the action of :math:`B` on the conjugate of :math:`v`,
        returning the result as a :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`.

    Remaining keyword arguments are passed to :func:`.eigendecompose`.
    """

    space = m.function_space()

    def H_action(x, y):
        _, _, ddJ = H.action(m, x)
        var_assign_conjugate(y, ddJ)

    B_inv_action_arg = B_inv_action

    def B_inv_action(x, y):
        var_assign_conjugate(y, B_inv_action_arg(x))

    B_action_arg = B_action

    def B_action(x, y):
        var_assign(y, B_action_arg(var_copy_conjugate(x)))

    return eigensolve(space, TypedSpace(space.dual(), space_type="dual"),
                      H_action, B_inv_action, B_inv_action=B_action,
                      *args, **kwargs)


def B_inv_orthonormality_test(V, B_inv_action):
    """Check for :math:`B^{-1}`-orthonormality.

    Requires real spaces.

    :arg B_inv_action: A callable accepting a
        :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` defining :math:`v` and
        computing the action of :math:`B^{-1}` on :math:`v`, returning the
        result as a :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`.
    :arg V: A :class:`Sequence` of :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` objects to test for
        :math:`B^{-1}`-orthonormality.
    :returns: A :class:`tuple` `(max_diagonal_error_norm,
        max_off_diagonal_error_norm)` with

            - `max_diagonal_error_norm`: The maximum :math:`B^{-1}`
              normalization error magnitude.
            - `max_diagonal_error_norm`: The maximum :math:`B^{-1}`
              orthogonality error magnitude.
    """

    if len(V) == 2 \
            and not is_var(V[0]) and isinstance(V[0], Sequence) \
            and not is_var(V[1]) and isinstance(V[1], Sequence):
        raise ValueError("Cannot supply separate real/complex eigenvector "
                         "components")

    B_inv_V = []
    for v in V:
        if not issubclass(var_dtype(v), np.floating):
            raise ValueError("Real dtype required")
        B_inv_V.append(B_inv_action(var_copy(v)))
        if not issubclass(var_dtype(B_inv_V[-1]), np.floating):
            raise ValueError("Real dtype required")

    max_diagonal_error_norm = 0.0
    max_off_diagonal_error_norm = 0.0
    assert len(V) == len(B_inv_V)
    for i, v in enumerate(V):
        for j, B_inv_v in enumerate(B_inv_V):
            if i == j:
                max_diagonal_error_norm = max(
                    max_diagonal_error_norm,
                    abs(var_inner(v, B_inv_v) - 1.0))
            else:
                max_off_diagonal_error_norm = max(
                    max_off_diagonal_error_norm,
                    abs(var_inner(v, B_inv_v)))

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

        H^{-1} \approx B + V \Lambda \left( I + \Lambda \right)^{-1} V^T

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

    :arg B_action: A callable accepting a :class:`firedrake.function.Function`
        or :class:`firedrake.cofunction.Cofunction` defining :math:`v` and
        computing the action of :math:`B` on :math:`v`, returning the result as
        a :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`.
    :arg Lam: A :class:`Sequence` defining the diagonal of :math:`\Lambda`.
    :arg V: A :class:`Sequence` of :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` objects defining the columns
        of :math:`V`.
    :returns: A callable suitable for use as the `pc_fn` argument to
        :meth:`.HessianLinearSolver.solve`.
    """

    if len(V) == 2 \
            and not is_var(V[0]) and isinstance(V[0], Sequence) \
            and not is_var(V[1]) and isinstance(V[1], Sequence):
        raise ValueError("Cannot supply separate real/complex eigenvector "
                         "components")

    Lam = tuple(Lam)
    V = tuple(V)
    if len(Lam) != len(V):
        raise ValueError("Invalid eigenpairs")

    def pc_fn(u, b):
        b = var_copy_conjugate(b)
        var_assign(u, B_action(var_copy(b)))

        assert len(Lam) == len(V)
        for lam, v in zip(Lam, V):
            alpha = -(lam / (1.0 + lam)) * var_inner(b, v)
            var_axpy(u, alpha, v)

    return pc_fn
