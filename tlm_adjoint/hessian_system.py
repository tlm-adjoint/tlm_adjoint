from .interface import (
    Packed, packed, var_axpy_conjugate, var_copy_conjugate,
    var_increment_state_lock, var_space)

from .block_system import LinearSolver, Matrix, TypedSpace
from .manager import manager_disabled

from collections import Sequence

__all__ = \
    [
        "HessianMatrix",
        "HessianLinearSolver"
    ]


# Complex note: It is convenient to define a Hessian action in terms of the
# *conjugate* of the action, i.e. (H \zeta)^{*,T}, e.g. this is the form
# returned by reverse-over-forward AD. However complex conjugation is then
# needed in a number of places (e.g. one cannot define an eigenproblem directly
# in terms of the conjugate of an action, as this is antilinear, rather than
# linear).


class HessianMatrix(Matrix):
    """A :class:`tlm_adjoint.block_system.Matrix` wrapping a :class:`.Hessian`.

    :arg H: The :class:`.Hessian`.
    :arg M: A variable or a :class:`Sequence` of variables defining the
        control and its value.
    """

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
        X = x if isinstance(x, Sequence) else (x,)
        Y = y if isinstance(y, Sequence) else (y,)

        _, _, ddJ = self._H.action(self._M, X)
        assert len(Y) == len(ddJ)
        for y_i, ddJ_i in zip(Y, ddJ):
            var_axpy_conjugate(y_i, 1.0, ddJ_i)


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
