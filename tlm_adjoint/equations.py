from .interface import (
    check_space_types, check_space_types_conjugate_dual,
    check_space_types_dual, var_assign, var_axpy, var_axpy_conjugate, var_dot,
    var_dtype, var_get_values, var_id, var_is_scalar, var_inner,
    var_local_size, var_new_conjugate_dual, var_replacement, var_scalar_value,
    var_set_values, var_zero)

from .equation import Equation, ZeroAssignment
from .linear_equation import LinearEquation, RHS

import numpy as np

__all__ = \
    [
        "EmptyEquation",

        "Assignment",
        "Axpy",
        "Conversion",
        "LinearCombination",

        "DotProductRHS",
        "DotProduct",
        "InnerProductRHS",
        "InnerProduct"
    ]


class EmptyEquation(Equation):
    """An adjoint tape record with no associated solution variables.
    """

    def __init__(self):
        super().__init__([], [], nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, X, deps=None):
        pass


class Assignment(Equation):
    r"""Represents an assignment

    .. math::

        x = y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y \right) = x - y.

    :arg x: A variable defining the forward solution :math:`x`.
    :arg y: A variable defining :math:`y`.
    """

    def __init__(self, x, y):
        check_space_types(x, y)
        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        var_assign(x, y)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise ValueError("Unexpected dep_index")

        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        x, y = self.dependencies()
        tau_y = tlm_map[y]
        if tau_y is None:
            return ZeroAssignment(tlm_map[x])
        else:
            return Assignment(tlm_map[x], tau_y)


class Conversion(Equation):
    r"""Represents degree of freedom assignment

    .. math::

        \tilde{x} = \tilde{y}

    where :math:`\tilde{x}` and :math:`\tilde{y}` are vectors of degrees of
    freedom for :math:`x` and :math:`y` respectively. Can be used to convert
    between different variable types.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y \right) = \tilde{x} - \tilde{y}.

    :arg x: A variable defining the forward solution :math:`x`.
    :arg y: A variable defining :math:`y`.
    """

    def __init__(self, x, y):
        if var_local_size(x) != var_local_size(y):
            raise ValueError("Invalid shape")
        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        var_set_values(x, var_get_values(y))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise ValueError("Unexpected dep_index")

        _, y = self.dependencies()
        F = var_new_conjugate_dual(y)
        var_set_values(F, var_get_values(adj_x))
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        x, y = self.dependencies()
        tau_y = tlm_map[y]
        if tau_y is None:
            return ZeroAssignment(tlm_map[x])
        else:
            return Conversion(tlm_map[x], tau_y)


class LinearCombination(Equation):
    r"""Represents an assignment

    .. math::

        x = \sum_i \alpha_i y_i.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y_1, y_2, \ldots \right)
            = x - \sum_i \alpha_i y_i.

    :arg x: A variable defining the forward solution :math:`x`.
    :arg args: A :class:`tuple` of two element :class:`Sequence` objects. The
        :math:`i` th element consists of `(alpha_i, y_i)`, where `alpha_i` is a
        scalar corresponding to :math:`\alpha_i` and `y_i` a variable
        corresponding to :math:`y_i`.
    """

    def __init__(self, x, *args):
        alpha = []
        Y = []
        for a, y in args:
            if a.imag == 0.0:
                a = a.real
            check_space_types(x, y)

            alpha.append(a)
            Y.append(y)

        super().__init__(x, [x] + Y, nl_deps=[], ic=False, adj_ic=False)
        self._alpha = tuple(alpha)

    def forward_solve(self, x, deps=None):
        deps = self.dependencies() if deps is None else tuple(deps)
        var_zero(x)
        assert len(self._alpha) == len(deps[1:])
        for alpha, y in zip(self._alpha, deps[1:]):
            var_axpy(x, alpha, y)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index <= 0 or dep_index > len(self._alpha):
            raise ValueError("Unexpected dep_index")

        return (-self._alpha[dep_index - 1].conjugate(), adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        deps = self.dependencies()
        x, ys = deps[0], deps[1:]
        args = []
        assert len(self._alpha) == len(ys)
        for alpha, y in zip(self._alpha, ys):
            tau_y = tlm_map[y]
            if tau_y is not None:
                args.append((alpha, tau_y))
        return LinearCombination(tlm_map[x], *args)


class Axpy(LinearCombination):
    r"""Represents an assignment

    .. math::

        y_\text{new} = y_\text{old} + \alpha x.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( y_\text{new}, y_\text{old}, x \right)
            = y_\text{new} - y_\text{old} - \alpha x.

    :arg y_new: A variable defining the forward solution :math:`y_\text{new}`.
    :arg y_old: A variable defining :math:`y_\text{old}`.
    :arg alpha: A scalar defining :math:`\alpha`.
    :arg x: A variable defining :math:`x`.
    """

    def __init__(self, y_new, y_old, alpha, x):
        super().__init__(y_new, (1.0, y_old), (alpha, x))


class DotProduct(LinearEquation):
    r"""Represents an assignment

    .. math::

        x = \alpha z^T y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y, z \right) = x - \alpha z^T y.

    :arg x: A variable whose degrees of freedom define the forward solution
        :math:`x`.
    :arg y: A variable whose degrees of freedom define :math:`y`.
    :arg z: A variable whose degrees of freedom define :math:`z`. May be the
        same variable as `y`.
    :arg alpha: A scalar defining :math:`\alpha`.
    """

    def __init__(self, x, y, z, *, alpha=1.0):
        super().__init__(x, DotProductRHS(y, z, alpha=alpha))


class InnerProduct(LinearEquation):
    r"""Represents an assignment

    .. math::

        x = \alpha z^* M y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y, z \right) = x - \alpha z^* M y.

    :arg x: A variable whose degrees of freedom define the forward solution
        :math:`x`.
    :arg y: A variable whose degrees of freedom define :math:`y`.
    :arg z: A variable whose degrees of freedom define :math:`z`. May be the
        same variable as `y`.
    :arg alpha: A scalar defining :math:`\alpha`.
    :arg M: A :class:`tlm_adjoint.linear_equation.Matrix` defining :math:`M`.
        Must have no dependencies. Defaults to an identity matrix.
    """

    def __init__(self, x, y, z, *, alpha=1.0, M=None):
        super().__init__(x, InnerProductRHS(y, z, alpha=alpha, M=M))


class DotProductRHS(RHS):
    r"""Represents a right-hand-side term

    .. math::

        \alpha y^T x.

    :arg x: A variable whose degrees of freedom define :math:`x`.
    :arg y: A variable whose degrees of freedom define :math:`y`. May be the
        same variable as `x`.
    :arg alpha: A scalar defining :math:`\alpha`.
    """

    def __init__(self, x, y, *, alpha=1.0):
        check_space_types_dual(x, y)

        x_equals_y = var_id(x) == var_id(y)
        if x_equals_y:
            deps = [x]
        else:
            deps = [x, y]

        super().__init__(deps, nl_deps=deps)
        self._x = x
        self._y = y
        self._x_equals_y = x_equals_y
        self._alpha = alpha

    def drop_references(self):
        super().drop_references()
        self._x = var_replacement(self._x)
        self._y = var_replacement(self._y)

    def add_forward(self, b, deps):
        if not var_is_scalar(b):
            raise ValueError("Scalar variable required")

        if self._x_equals_y:
            (x,), (y,) = deps, deps
        else:
            x, y = deps

        if var_local_size(y) != var_local_size(x):
            raise ValueError("Invalid space")
        check_space_types_dual(x, y)

        var_assign(b, var_scalar_value(b) + self._alpha * var_dot(x, y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if not var_is_scalar(adj_x):
            raise ValueError("Scalar variable required")

        if self._x_equals_y:
            if dep_index == 0:
                x, = nl_deps
                alpha = -2.0 * self._alpha.conjugate() * var_scalar_value(adj_x)  # noqa: E501
                var_axpy_conjugate(b, alpha, x)
            else:
                raise ValueError("Unexpected dep_index")
        elif dep_index == 0:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * var_scalar_value(adj_x)
            var_axpy_conjugate(b, alpha, y)
        elif dep_index == 1:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * var_scalar_value(adj_x)
            var_axpy_conjugate(b, alpha, x)
        else:
            raise ValueError("Unexpected dep_index")

    def tangent_linear_rhs(self, tlm_map):
        tlm_B = []

        if self._x_equals_y:
            x, = self.dependencies()

            tlm_x = tlm_map[x]
            if tlm_x is not None:
                tlm_B.append(DotProductRHS(tlm_x, x, alpha=2.0 * self._alpha))
        else:
            x, y = self.dependencies()

            tlm_x = tlm_map[x]
            if tlm_x is not None:
                tlm_B.append(DotProductRHS(tlm_x, y, alpha=self._alpha))

            tlm_y = tlm_map[y]
            if tlm_y is not None:
                tlm_B.append(DotProductRHS(x, tlm_y, alpha=self._alpha))

        return tlm_B


class InnerProductRHS(RHS):
    r"""Represents a right-hand-side term

    .. math::

        \alpha y^* M x.

    :arg x: A variable whose degrees of freedom define :math:`x`.
    :arg y: A variable whose degrees of freedom define :math:`y`. May be the
        same variable as `x`.
    :arg alpha: A scalar defining :math:`\alpha`.
    :arg M: A :class:`tlm_adjoint.linear_equation.Matrix` defining :math:`M`.
        Must have no dependencies. Defaults to an identity matrix.
    """

    def __init__(self, x, y, *, alpha=1.0, M=None):
        if M is None:
            check_space_types_conjugate_dual(x, y)
        else:
            check_space_types(x, y)
        if M is not None and len(M.nonlinear_dependencies()) > 0:
            raise NotImplementedError("Non-linear matrix dependencies not "
                                      "supported")

        norm_sq = var_id(x) == var_id(y)
        if norm_sq:
            deps = [x]
        else:
            deps = [x, y]

        super().__init__(deps, nl_deps=deps)
        self._x = x
        self._y = y
        self._norm_sq = norm_sq
        self._alpha = alpha
        self._M = M

        if M is not None:
            self.add_referrer(M)

    def drop_references(self):
        super().drop_references()
        self._x = var_replacement(self._x)
        self._y = var_replacement(self._y)
        if self._M is not None:
            self._M = self._M._weak_alias

    def add_forward(self, b, deps):
        if not var_is_scalar(b):
            raise ValueError("Scalar variable required")

        if self._norm_sq:
            x, y = deps[0], deps[0]
            M_deps = deps[1:]
        else:
            x, y = deps[:2]
            M_deps = deps[2:]

        if self._M is None:
            Y = y
        else:
            Y = var_new_conjugate_dual(x)
            self._M.adjoint_action(M_deps, y, Y, method="assign")
        check_space_types_conjugate_dual(x, Y)

        var_assign(b, var_scalar_value(b) + self._alpha * var_inner(x, Y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if not var_is_scalar(adj_x):
            raise ValueError("Scalar variable required")

        if self._norm_sq:
            if dep_index == 0:
                x = nl_deps[0]
                if not issubclass(var_dtype(x), np.floating):
                    raise RuntimeError("Not complex differentiable")
                M_deps = nl_deps[1:]

                if self._M is None:
                    X = x
                else:
                    X = var_new_conjugate_dual(x)
                    self._M.adjoint_action(M_deps, x, X, method="assign")

                var_axpy(
                    b, -self._alpha.conjugate() * var_scalar_value(adj_x), X)

                if self._M is None:
                    X = x
                else:
                    X = var_new_conjugate_dual(x)
                    self._M.forward_action(M_deps, x, X, method="assign")

                var_axpy(
                    b, -self._alpha.conjugate() * var_scalar_value(adj_x), X)
            else:
                raise ValueError("Unexpected dep_index")
        elif dep_index == 0:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                Y = y
            else:
                Y = var_new_conjugate_dual(x)
                self._M.adjoint_action(M_deps, y, Y, method="assign")

            var_axpy(b, -self._alpha.conjugate() * var_scalar_value(adj_x), Y)
        elif dep_index == 1:
            x, y = nl_deps[:2]
            if not issubclass(var_dtype(y), np.floating):
                raise RuntimeError("Not complex differentiable")
            M_deps = nl_deps[2:]

            if self._M is None:
                X = x
            else:
                X = var_new_conjugate_dual(y)
                self._M.forward_action(M_deps, x, X, method="assign")

            var_axpy(b, -self._alpha.conjugate() * var_scalar_value(adj_x), X)
        else:
            raise ValueError("Unexpected dep_index")

    def tangent_linear_rhs(self, tlm_map):
        tlm_B = []

        if self._norm_sq:
            x = self.dependencies()[0]
            tlm_x = tlm_map[x]
            if tlm_x is not None:
                if not issubclass(var_dtype(x), np.floating):
                    raise RuntimeError("Not complex differentiable")
                tlm_B.append(InnerProductRHS(tlm_x, x, alpha=self._alpha,
                                             M=self._M))
                tlm_B.append(InnerProductRHS(x, tlm_x, alpha=self._alpha,
                                             M=self._M))
        else:
            x, y = self.dependencies()[:2]

            tlm_x = tlm_map[x]
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(tlm_x, y, alpha=self._alpha,
                                             M=self._M))

            tlm_y = tlm_map[y]
            if tlm_y is not None:
                if not issubclass(var_dtype(y), np.floating):
                    raise RuntimeError("Not complex differentiable")
                tlm_B.append(InnerProductRHS(x, tlm_y, alpha=self._alpha,
                                             M=self._M))

        return tlm_B
