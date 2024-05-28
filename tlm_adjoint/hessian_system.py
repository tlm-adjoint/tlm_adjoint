from .interface import (
    Packed, packed, var_copy_conjugate, var_increment_state_lock, var_locked,
    var_space, vars_assign, vars_assign_conjugate, vars_axpy_conjugate,
    vars_copy_conjugate)

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

        for m in M:
            var_increment_state_lock(m, self)

    def mult_add(self, x, y):
        x = packed(x)
        y = packed(y)

        _, _, ddJ = self._H.action(self._M, x)
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
        variables defining the action on this direction. Arguments should not
        be modified.
    :arg B_inv_action: A callable defining the conjugate of the action of
        :math:`B^{-1}` on some direction. Accepts one or more variables as
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
            x = vars_copy_conjugate(packed(x))
            y = packed(y)

            with var_locked(*x):
                vars_assign(y, packed(B_inv_action_arg(*x)))

        A = HessianMatrix(H, M)
        B = MatrixFreeMatrix(A.arg_space, A.action_space, B_action)
        B_inv = MatrixFreeMatrix(A.action_space, A.arg_space, B_inv_action)
        super().__init__(A, B, B_inv=B_inv, *args, **kwargs)
