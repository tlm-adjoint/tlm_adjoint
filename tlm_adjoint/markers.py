from .interface import (
    Packed, packed, check_space_types_conjugate_dual, var_assign, var_inner,
    var_is_scalar, var_new, var_scalar_value)

from .equation import Equation, ZeroAssignment

__all__ = \
    [
        "ControlsMarker",
        "FunctionalMarker",
        "AdjointActionMarker"
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
        M_packed = Packed(M)
        M = tuple(M_packed)

        super(Equation, self).__init__()
        self._packed = M_packed.mapped(lambda m: None)
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
        if not var_is_scalar(J):
            raise ValueError("Functional must be a scalar variable")

        # Extra variable allocation could be avoided
        J_ = var_new(J)
        super().__init__(J_, [J_, J], nl_deps=[], ic=False, adj_ic=False)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise ValueError("Unexpected dep_index")

        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b


class AdjointActionMarker(Equation):
    r"""Represents

    .. math::

        J_\text{output} = \lambda_x^* x,

    with forward residual

    .. math::

        \mathcal{F} \left( J_\text{output}, x \right)
            = J_\text{output} - \lambda_x^* x.

    Note that :math:`\lambda_x` is *not* treated as a dependency.

    Can be used to initialize an adjoint calculation, and compute adjoint
    Jacobian actions, via the construction

    .. code-block:: python

        start_manager()
        X = forward(M)
        with paused_manager():
            adj_X = ...
        J = Float(name="J")
        AdjointRHSMarker(J, X, adj_X).solve()
        stop_manager()

        # Compute the action of the adjoint of the Jacobian on the direction
        # defined by adj_X
        dJ = compute_gradient(J, M)

    :arg J: A variable defining the functional :math:`J`.
    :arg X: A variable or :class:`Sequence` of variables defining :math:`x`.
    :arg adj_X: A variable or :class:`Sequence` of variables defining
        :math:`\lambda_x`.
    """

    def __init__(self, J, X, adj_X):
        if not var_is_scalar(J):
            raise ValueError("Functional must be a scalar variable")
        X = packed(X)
        adj_X = packed(adj_X)
        if len(X) != len(adj_X):
            raise ValueError("Invalid length")
        for x, adj_x in zip(X, adj_X):
            check_space_types_conjugate_dual(x, adj_x)

        super().__init__(J, [J] + list(X), nl_deps=X, ic=False, adj_ic=False)
        self._adj_X = tuple(adj_X)

    def forward_solve(self, x, deps=None):
        J = x
        X = (self.dependencies() if deps is None else deps)[1:]

        v = 0.0
        assert len(X) == len(self._adj_X)
        for x, adj_x in zip(X, self._adj_X):
            v += var_inner(x, adj_x)
        var_assign(J, v)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            raise ValueError("Unexpected dep_index")
        return (-var_scalar_value(adj_x), self._adj_X[dep_index - 1])

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        J = self.x()
        X = self.dependencies()[1:]

        tau_X = []
        adj_X = []
        assert len(X) == len(self._adj_X)
        for x, adj_x in zip(X, self._adj_X):
            tau_x = tlm_map[x]
            if tau_x is not None:
                tau_X.append(tau_x)
                adj_X.append(adj_x)

        if len(tau_X) == 0:
            return ZeroAssignment(tlm_map[J])
        else:
            return AdjointActionMarker(tlm_map[J], tau_X, adj_X)
