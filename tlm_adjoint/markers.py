from .interface import is_var, var_new

from .equation import Equation

__all__ = \
    [
        "ControlsMarker",
        "FunctionalMarker"
    ]


class ControlsMarker(Equation):
    r"""Represents

    .. math::

        m = m_\text{input},

    where :math:`m` is the control and :math:`m_\text{input}` the input value
    for the control. The forward residual is defined

    .. math::

        \mathcal{F} \left( m \right) = m - m_\text{input}.

    :arg M: A variable or a :class:`Sequence` of variables defining the
        control :math:`m`. May be static.
    """

    def __init__(self, M):
        if is_var(M):
            M = (M,)

        super(Equation, self).__init__()
        self._X = tuple(M)
        self._deps = tuple(M)
        self._nl_deps = ()
        self._ic_deps = ()
        self._adj_ic_deps = ()
        self._adj_X_type = tuple("conjugate_dual" for m in M)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B


class FunctionalMarker(Equation):
    r"""Represents

    .. math::

        J_\text{output} = J,

    where :math:`J` is the functional and :math:`J_\text{output}` is the output
    value for the functional. The forward residual is defined

    .. math::

        \mathcal{F} \left( J_\text{output}, J \right) = J_\text{output} - J.

    :arg J: A variable defining the functional :math:`J`.
    """

    def __init__(self, J):
        # Extra variable allocation could be avoided
        J_ = var_new(J)
        super().__init__([J_], [J_, J], nl_deps=[], ic=False, adj_ic=False)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise ValueError("Unexpected dep_index")

        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b
