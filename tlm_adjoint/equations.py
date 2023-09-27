#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    check_space_types, check_space_types_conjugate_dual,
    check_space_types_dual, is_var, var_assign, var_axpy, var_comm, var_dtype,
    var_get_values, var_id, var_inner, var_local_size, var_new_conjugate_dual,
    var_replacement, var_set_values, var_sum, var_zero)

from .alias import WeakAlias
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
        "InnerProduct",
        "MatrixActionRHS"
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
        if dep_index == 0:
            return adj_x
        elif dep_index == 1:
            return (-1.0, adj_x)
        else:
            raise IndexError("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
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
    between different backends.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y \right) = \tilde{x} - \tilde{y}.

    :arg x: A variable defining the forward solution :math:`x`.
    :arg y: A variable defining :math:`y`.
    """

    def __init__(self, x, y):
        if var_local_size(x) != var_local_size(y):
            raise ValueError("Invalid shape")
        check_space_types(x, y)
        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        var_set_values(x, var_get_values(y))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        elif dep_index == 1:
            _, y = self.dependencies()
            F = var_new_conjugate_dual(y)
            var_set_values(F, var_get_values(adj_x))
            return (-1.0, F)
        else:
            raise IndexError("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
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
        if dep_index == 0:
            return adj_x
        elif dep_index <= len(self._alpha):
            return (-self._alpha[dep_index - 1].conjugate(), adj_x)
        else:
            raise IndexError("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
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


class MatrixActionRHS(RHS):
    """Represents a right-hand-side term

    .. math::

        A x.

    :arg A: A :class:`tlm_adjoint.linear_equation.Matrix` defining :math:`A`.
    :arg x: A variable or a :class:`Sequence` of variables defining :math:`x`.
    """

    def __init__(self, A, X):
        if is_var(X):
            X = (X,)
        if len(set(map(var_id, X))) != len(X):
            raise ValueError("Invalid dependency")

        A_nl_deps = A.nonlinear_dependencies()
        if len(A_nl_deps) == 0:
            x_indices = {i: i for i in range(len(X))}
            super().__init__(X, nl_deps=[])
        else:
            nl_deps = list(A_nl_deps)
            nl_dep_ids = {var_id(dep): i for i, dep in enumerate(nl_deps)}
            x_indices = {}
            for i, x in enumerate(X):
                x_id = var_id(x)
                if x_id not in nl_dep_ids:
                    nl_deps.append(x)
                    nl_dep_ids[x_id] = len(nl_deps) - 1
                x_indices[nl_dep_ids[x_id]] = i
            super().__init__(nl_deps, nl_deps=nl_deps)

        self._A = A
        self._x_indices = x_indices

        self.add_referrer(A)

    def drop_references(self):
        super().drop_references()
        self._A = WeakAlias(self._A)

    def add_forward(self, B, deps):
        if is_var(B):
            B = (B,)
        X = tuple(deps[j] for j in self._x_indices)
        self._A.forward_action(deps[:len(self._A.nonlinear_dependencies())],
                               X[0] if len(X) == 1 else X,
                               B[0] if len(B) == 1 else B,
                               method="add")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        if is_var(adj_X):
            adj_X = (adj_X,)
        if dep_index < 0 or dep_index >= len(self.dependencies()):
            raise IndexError("dep_index out of bounds")
        N_A_nl_deps = len(self._A.nonlinear_dependencies())
        if dep_index < N_A_nl_deps:
            X = [nl_deps[j] for j in self._x_indices]
            self._A.adjoint_derivative_action(
                nl_deps[:N_A_nl_deps], dep_index,
                X[0] if len(X) == 1 else X,
                adj_X[0] if len(adj_X) == 1 else adj_X,
                b, method="sub")
        if dep_index in self._x_indices:
            self._A.adjoint_action(nl_deps[:N_A_nl_deps],
                                   adj_X[0] if len(adj_X) == 1 else adj_X,
                                   b, b_index=self._x_indices[dep_index],
                                   method="sub")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        deps = self.dependencies()
        N_A_nl_deps = len(self._A.nonlinear_dependencies())

        X = tuple(deps[j] for j in self._x_indices)
        tlm_X = tuple(tlm_map[x] for x in X)
        tlm_B = [MatrixActionRHS(self._A, tlm_X)]

        if N_A_nl_deps > 0:
            tlm_b = self._A.tangent_linear_rhs(M, dM, tlm_map, X)
            if tlm_b is None:
                pass
            elif isinstance(tlm_b, RHS):
                tlm_B.append(tlm_b)
            else:
                tlm_B.extend(tlm_b)

        return tlm_B


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
        if self._x_equals_y:
            (x,), (y,) = deps, deps
        else:
            x, y = deps

        if var_local_size(y) != var_local_size(x):
            raise ValueError("Invalid space")
        check_space_types_dual(x, y)

        d = (var_get_values(y) * var_get_values(x)).sum()
        comm = var_comm(b)
        if comm.size > 1:
            import mpi4py.MPI as MPI
            d = comm.allreduce(d, op=MPI.SUM)

        var_set_values(b, var_get_values(b) + self._alpha * d)

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._x_equals_y:
            if dep_index == 0:
                x, = nl_deps
                alpha = -2.0 * self._alpha.conjugate() * var_sum(adj_x)
                var_set_values(
                    b,
                    var_get_values(b)
                    + alpha * var_get_values(x).conjugate())
            else:
                raise IndexError("dep_index out of bounds")
        elif dep_index == 0:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * var_sum(adj_x)
            var_set_values(
                b,
                var_get_values(b)
                + alpha * var_get_values(y).conjugate())
        elif dep_index == 1:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * var_sum(adj_x)
            var_set_values(
                b,
                var_get_values(b)
                + alpha * var_get_values(x).conjugate())
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
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
            self._M = WeakAlias(self._M)

    def add_forward(self, b, deps):
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

        var_set_values(b, var_get_values(b) + self._alpha * var_inner(x, Y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._norm_sq:
            if dep_index == 0:
                x = nl_deps[0]
                if not issubclass(var_dtype(x), (float, np.floating)):
                    raise RuntimeError("Not complex differentiable")
                M_deps = nl_deps[1:]

                if self._M is None:
                    X = x
                else:
                    X = var_new_conjugate_dual(x)
                    self._M.adjoint_action(M_deps, x, X, method="assign")

                var_axpy(
                    b, -self._alpha.conjugate() * var_sum(adj_x), X)

                if self._M is None:
                    X = x
                else:
                    X = var_new_conjugate_dual(x)
                    self._M.forward_action(M_deps, x, X, method="assign")

                var_axpy(
                    b, -self._alpha.conjugate() * var_sum(adj_x), X)
            else:
                raise IndexError("dep_index out of bounds")
        elif dep_index == 0:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                Y = y
            else:
                Y = var_new_conjugate_dual(x)
                self._M.adjoint_action(M_deps, y, Y, method="assign")

            var_axpy(b, -self._alpha.conjugate() * var_sum(adj_x), Y)
        elif dep_index == 1:
            x, y = nl_deps[:2]
            if not issubclass(var_dtype(y), (float, np.floating)):
                raise RuntimeError("Not complex differentiable")
            M_deps = nl_deps[2:]

            if self._M is None:
                X = x
            else:
                X = var_new_conjugate_dual(y)
                self._M.forward_action(M_deps, x, X, method="assign")

            var_axpy(b, -self._alpha.conjugate() * var_sum(adj_x), X)
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        tlm_B = []

        if self._norm_sq:
            x = self.dependencies()[0]
            tlm_x = tlm_map[x]
            if tlm_x is not None:
                if not issubclass(var_dtype(x), (float, np.floating)):
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
                if not issubclass(var_dtype(y), (float, np.floating)):
                    raise RuntimeError("Not complex differentiable")
                tlm_B.append(InnerProductRHS(x, tlm_y, alpha=self._alpha,
                                             M=self._M))

        return tlm_B
