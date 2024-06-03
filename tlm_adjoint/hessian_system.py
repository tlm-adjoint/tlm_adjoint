from .interface import (
    Packed, packed, var_copy_conjugate, var_increment_state_lock, var_copy,
    var_is_cached, var_is_static, var_locked, var_space, vars_assign,
    vars_assign_conjugate, vars_axpy, vars_axpy_conjugate, vars_inner,
    var_new_conjugate_dual)

from .block_system import (
    Eigensolver, LinearSolver, Matrix, MatrixFreeMatrix, TypedSpace)
from .manager import manager_disabled

__all__ = \
    [
        "HessianLinearSolver",
        "HessianEigensolver"
    ]


# Complex note: It is convenient to define a Hessian action in terms of the
# *conjugate* of the action, i.e. (H \zeta)^{*,T}, e.g. this is the form
# returned by reverse-over-forward AD. However complex conjugation is then
# needed in a number of places (e.g. one cannot define an eigenproblem directly
# in terms of the conjugate of an action, as this is antilinear, rather than
# linear).


class HessianMatrix(Matrix):
    def __init__(self, H, M):
        M_packed = Packed(M)
        M = tuple(M_packed)
        arg_space = tuple(map(var_space, M))
        action_space = tuple(TypedSpace(var_space(m), space_type="dual")
                             for m in M)

        super().__init__(M_packed.unpack(arg_space),
                         M_packed.unpack(action_space))
        self._H = H
        self._M = M

        var_increment_state_lock(self, *M)

    def mult_add(self, x, y):
        x = packed(x)
        y = packed(y)

        assert len(x) == len(self._M)
        dM = tuple(var_copy(x_i, static=var_is_static(m),
                            cache=var_is_cached(m))
                   for x_i, m in zip(x, self._M))

        _, _, ddJ = self._H.action(self._M, dM)
        vars_axpy_conjugate(y, 1.0, ddJ)


class HessianLinearSolver(LinearSolver):
    """Defines a linear system involving a Hessian matrix,

    .. math::

        H u = b.

    :arg H: A :class:`.Hessian` defining :math:`H`.
    :arg M: A variable or a :class:`Sequence` of variables defining the
        control and its value.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.block_system.LinearSolver` constructor.
    """

    def __init__(self, H, M, *args, **kwargs):
        super().__init__(HessianMatrix(H, M), *args, **kwargs)

    @manager_disabled()
    def solve(self, u, b, *args, **kwargs):
        """Solve a linear system involving a Hessian matrix,

        .. math::

            H u = b.

        :arg u: Defines the solution :math:`u`.
        :arg b: Defines the conjugate of the right-hand-side :math:`b`.

        Remaining arguments are handed to the
        :meth:`tlm_adjoint.block_system.LinearSolver.solve` method.
        """

        b_conj = Packed(b).mapped(var_copy_conjugate)
        super().solve(u, b_conj.unpack(b_conj), **kwargs)


class HessianEigensolver(Eigensolver):
    r"""Defines an eigenproblem involving a Hessian matrix

    .. math::

        H v = \lambda B v.

    :arg H: A :class:`.Hessian` defining :math:`H`.
    :arg M: A variable or a :class:`Sequence` of variables defining the
        control and its value.
    :arg B_action: A callable defining the conjugate of the action of :math:`B`
        on some direction. Accepts one or more variables as arguments, defining
        the direction, and returns a variable or a :class:`Sequence` of
        variables defining the conjugate of the action on this direction.
        Arguments should not be modified.
    :arg B_inv_action: A callable defining the the action of :math:`B^{-1}` on
        the conjugate of some direction. Accepts one or more variables as
        arguments, defining the direction, and returns a variable or a
        :class:`Sequence` of variables defining the action on this direction.
        Arguments should not be modified.

    Remaining arguments are passed to the :class:`.Eigensolver` constructor.
    """

    def __init__(self, H, M, B_action, B_inv_action, *args, **kwargs):
        B_action_arg = B_action

        def B_action(x, y):
            x = packed(x)
            y = packed(y)

            with var_locked(*x):
                vars_assign_conjugate(y, packed(B_action_arg(*x)))

        B_inv_action_arg = B_inv_action

        def B_inv_action(x, y):
            x = packed(x)
            y = packed(y)

            with var_locked(*x):
                vars_assign(y, packed(B_inv_action_arg(*x)))

        A = HessianMatrix(H, M)
        B = MatrixFreeMatrix(A.arg_space, A.action_space, B_action)
        B_inv = MatrixFreeMatrix(A.action_space, A.arg_space, B_inv_action)
        super().__init__(A, B, B_inv=B_inv, *args, **kwargs)

    def spectral_approximation_solve(self, b):
        r"""Compute the approximate action of an inverse Hessian action -- see
        :meth:`HessianEigensolver.spectral_pc_fn`. Note that this computes the
        approximation of the action of :math:`(A + B)^{-1}` where :math:`A` and
        :math:`B` are the matrices defining the eigenproblem solved by this
        :class:`HessianEigensolver`.

        :arg b: A variable or :class:`Sequence` of variables defining the
            direction for which the action is evaluated.
        :returns: A variable or :class:`Sequence` of variables storing the
            action.
        """

        if not self.is_hermitian_and_positive():
            raise ValueError("Hermitian and positive eigenproblem required")

        b_packed = Packed(b)
        b = tuple(b_packed)

        u = tuple(map(var_new_conjugate_dual, b))
        self._B_inv.matrix.mult(b, u)

        for lam_i, (v, _) in self:
            v = packed(v)
            alpha = -(lam_i / (1.0 + lam_i)) * vars_inner(b, v)
            vars_axpy(u, alpha, v)

        return b_packed.unpack(u)

    def spectral_pc_fn(self):
        r"""Construct a Hessian matrix preconditioner using a partial spectrum
        generalized eigendecomposition. Assumes that the Hessian matrix
        consists of two terms

        .. math::

            H = A + B,

        where :math:`A` and :math:`B` are the matrices defining the
        eigenproblem solved by this :class:`HessianEigensolver`.

        The approximation is defined via

        .. math::

            H^{-1} \approx B
                - V \Lambda \left( I + \Lambda \right)^{-1} V^*

        where

        .. math::

            A V = B V \Lambda,

        and where :math:`\Lambda` is a diagonal matrix and :math:`V` has
        :math:`B`-orthonormal columns, :math:`V^* B V = I`.

        This low rank update approximation for the Hessian matrix inverse is
        described in

            - Tobin Isaac, Noemi Petra, Georg Stadler, and Omar Ghattas,
              'Scalable and efficient algorithms for the propagation of
              uncertainty from data through inference to prediction for
              large-scale problems, with application to flow of the Antarctic
              ice sheet', Journal of Computational Physics, 296, pp. 348--368,
              2015, doi: 10.1016/j.jcp.2015.04.047

        See in particular their equation (20).

        :returns: A callable suitable for use as the `pc_fn` argument to
            :meth:`.HessianLinearSolver.solve`.
        """

        if not self.is_hermitian_and_positive():
            raise ValueError("Hermitian and positive eigenproblem required")

        lam, V = self.eigenpairs()

        def pc_fn(u, b):
            u = Packed(u)
            b = Packed(b).mapped(var_copy_conjugate)

            self._B_inv.matrix.mult(b, u)

            assert len(lam) == len(V)
            for lam_i, (v, _) in zip(lam, V):
                v = packed(v)
                alpha = -(lam_i / (1.0 + lam_i)) * vars_inner(b, v)
                vars_axpy(u, alpha, v)

        return pc_fn
