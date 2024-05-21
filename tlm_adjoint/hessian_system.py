from .interface import (
    check_space_types, check_space_types_conjugate, var_axpy_conjugate,
    var_space)

from .block_system import Matrix, TypedSpace

from collections.abc import Sequence

__all__ = \
    [
        "HessianMatrix"
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
        control.
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

    def mult_add(self, x, y):
        if isinstance(x, Sequence):
            x = tuple(x)
        else:
            x = (x,)
        if isinstance(y, Sequence):
            y = tuple(y)
        else:
            y = (y,)

        if len(x) != len(self._M):
            raise ValueError("Invalid length")
        for x_i, m in zip(x, self._M):
            check_space_types(x_i, m)

        _, _, ddJ = self._H.action(self._M, x)

        if len(y) != len(ddJ):
            raise ValueError("Invalid length")
        for y_i, ddJ_i in zip(y, ddJ):
            check_space_types_conjugate(y_i, ddJ_i)
        for y_i, ddJ_i in zip(y, ddJ):
            var_axpy_conjugate(y_i, 1.0, ddJ_i)
