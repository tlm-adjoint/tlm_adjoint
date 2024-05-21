from .interface import (
    var_axpy_conjugate, var_copy_conjugate, var_increment_state_lock,
    var_space)

from .block_system import (
    BlockNullspace, LinearSolver, Matrix, NoneNullspace, TypedSpace)
from .manager import manager_disabled

from collections.abc import Sequence

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
        if isinstance(M, Sequence):
            M = tuple(M)
        else:
            M = (M,)
        arg_space = tuple(TypedSpace(var_space(m)) for m in M)
        action_space = tuple(TypedSpace(var_space(m), space_type="dual") for m in M)  # noqa: E501

        super().__init__(arg_space, action_space)
        self._H = H
        self._M = M

        for m in M:
            var_increment_state_lock(m, self)

    def mult_add(self, x, y):
        _, _, ddJ = self._H.action(self._M, x)
        assert len(y) == len(ddJ)
        for y_i, ddJ_i in zip(y, ddJ):
            var_axpy_conjugate(y_i, 1.0, ddJ_i)


class HessianLinearSolver(LinearSolver):
    """Defines a linear system involving a Hessian matrix,

    .. math::

        H u = b.

    :arg H: A :class:`.Hessian` defining :math:`H`.
    :arg M: A variable or a :class:`Sequence` of variables defining the
        control and its value.
    :arg nullspaces: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace. `None` indicates a
        :class:`.NoneNullspace`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.block_system.LinearSolver` constructor.
    """

    def __init__(self, H, M, *args, nullspace=None, **kwargs):
        if nullspace is None:
            nullspace = NoneNullspace()
        elif not isinstance(nullspace, (NoneNullspace, BlockNullspace)):
            nullspace = BlockNullspace(nullspace)
        super().__init__(HessianMatrix(H, M), *args, nullspace=nullspace,
                         **kwargs)

    @manager_disabled()
    def solve(self, u, b, **kwargs):
        """Solve a linear system involving a Hessian matrix,

        .. math::

            H u = b.

        :arg u: Defines the solution :math:`u`.
        :arg b: Defines the conjugate of the right-hand-side :math:`b`.

        Remaining arguments are handed to the
        :meth:`tlm_adjoint.block_system.LinearSolver.solve` method.
        """

        if isinstance(b, Sequence):
            b = tuple(map(var_copy_conjugate, b))
        else:
            b = var_copy_conjugate(b)
        return super().solve(u, b, **kwargs)
