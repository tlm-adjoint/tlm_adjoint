#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from .interface import check_space_types, check_space_types_conjugate_dual, \
    check_space_types_dual, function_assign, function_axpy, function_comm, \
    function_dtype, function_get_values, function_id, function_inner, \
    function_local_size, function_new_conjugate_dual, function_replacement, \
    function_set_values, function_sum, function_zero, is_function

from .alias import WeakAlias
from .equation import Equation, ZeroAssignment
from .linear_equation import LinearEquation, RHS
from .tangent_linear import get_tangent_linear

import numpy as np
import warnings

__all__ = \
    [
        "Assignment",
        "Axpy",
        "LinearCombination",

        "DotProductRHS",
        "DotProduct",
        "InnerProductRHS",
        "InnerProduct",
        "MatrixActionRHS",

        "AssignmentSolver",
        "AxpySolver",
        "DotProductSolver",
        "InnerProductSolver",
        "LinearCombinationSolver",
        "MatrixActionSolver",
        "NormSqRHS",
        "NormSqSolver",
        "ScaleSolver",
        "SumRHS",
        "SumSolver"
    ]


class Assignment(Equation):
    r"""Represents an assignment

    .. math::

        x = y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y \right) = x - y.

    :arg x: A function defining the forward solution :math:`x`.
    :arg y: A function defining :math:`y`.
    """

    def __init__(self, x, y):
        check_space_types(x, y)
        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        function_assign(x, y)

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
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return ZeroAssignment(tlm_map[x])
        else:
            return Assignment(tlm_map[x], tau_y)


class AssignmentSolver(Assignment):
    ""

    def __init__(self, y, x):
        warnings.warn("AssignmentSolver is deprecated -- "
                      "use Assignment instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y)


class LinearCombination(Equation):
    r"""Represents an assignment

    .. math::

        x = \sum_i \alpha_i y_i.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y_1, y_2, \ldots \right)
            = x - \sum_i \alpha_i y_i.

    :arg x: A function defining the forward solution :math:`x`.
    :arg args: A :class:`tuple` of two element :class:`Sequence` objects. The
        :math:`i` th element consists of `(alpha_i, y_i)`, where `alpha_i` is a
        scalar corresponding to :math:`\alpha_i` and `y_i` a function
        corresponding :math:`y_i`.
    """

    def __init__(self, x, *args):
        alpha = []
        Y = []
        for a, y in args:
            a = function_dtype(x)(a)
            if a.imag == 0.0:
                a = a.real
            check_space_types(x, y)

            alpha.append(a)
            Y.append(y)

        super().__init__(x, [x] + Y, nl_deps=[], ic=False, adj_ic=False)
        self._alpha = tuple(alpha)

    def forward_solve(self, x, deps=None):
        deps = self.dependencies() if deps is None else tuple(deps)
        function_zero(x)
        assert len(self._alpha) == len(deps[1:])
        for alpha, y in zip(self._alpha, deps[1:]):
            function_axpy(x, alpha, y)

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
            tau_y = get_tangent_linear(y, M, dM, tlm_map)
            if tau_y is not None:
                args.append((alpha, tau_y))
        return LinearCombination(tlm_map[x], *args)


class LinearCombinationSolver(LinearCombination):
    ""

    def __init__(self, x, *args):
        warnings.warn("LinearCombinationSolver is deprecated -- "
                      "use LinearCombination instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, *args)


class ScaleSolver(LinearCombination):
    ""

    def __init__(self, alpha, y, x):
        warnings.warn("ScaleSolver is deprecated -- "
                      "use LinearCombination instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, (alpha, y))


class Axpy(LinearCombination):
    r"""Represents an assignment

    .. math::

        y_\text{new} = y_\text{old} + \alpha x.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( y_\text{new}, y_\text{old}, x \right)
            = y_\text{new} - y_\text{old} - \alpha x.

    :arg y_new: A function defining the forward solution :math:`y_\text{new}`.
    :arg y_old: A function defining :math:`y_\text{old}`.
    :arg alpha: A scalar defining :math:`\alpha`.
    :arg x: A function defining :math:`x`.
    """

    def __init__(self, y_new, y_old, alpha, x):
        super().__init__(y_new, (1.0, y_old), (alpha, x))


class AxpySolver(Axpy):
    ""

    def __init__(self, y_old, alpha, x, y_new, /):
        warnings.warn("AxpySolver is deprecated -- "
                      "use Axpy instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(y_new, y_old, alpha, x)


class MatrixActionSolver(LinearEquation):
    ""

    def __init__(self, Y, A, X):
        warnings.warn("MatrixActionSolver is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(X, MatrixActionRHS(A, Y))


class DotProduct(LinearEquation):
    r"""Represents an assignment

    .. math::

        x = \alpha z^T y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y, z \right) = x - \alpha z^T y.

    :arg x: A function whose degrees of freedom define the forward solution
        :math:`x`.
    :arg y: A function whose degrees of freedom define :math:`y`.
    :arg z: A function whose degrees of freedom define :math:`z`. May be the
        same function as `y`.
    :arg alpha: A scalar defining :math:`\alpha`.
    """

    def __init__(self, x, y, z, *, alpha=1.0):
        super().__init__(x, DotProductRHS(y, z, alpha=alpha))


class DotProductSolver(DotProduct):
    ""

    def __init__(self, y, z, x, alpha=1.0):
        warnings.warn("DotProductSolver is deprecated -- "
                      "use DotProduct instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, z, alpha=alpha)


class InnerProduct(LinearEquation):
    r"""Represents an assignment

    .. math::

        x = \alpha z^* M y.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y, z \right) = x - \alpha z^* M y.

    :arg x: A function whose degrees of freedom define the forward solution
        :math:`x`.
    :arg y: A function whose degrees of freedom define :math:`y`.
    :arg z: A function whose degrees of freedom define :math:`z`. May be the
        same function a `y`.
    :arg alpha: A scalar defining :math:`\alpha`.
    :arg M: A :class:`tlm_adjoint.linear_equation.Matrix` defining :math:`M`.
        Must have no dependencies. Defaults to an identity matrix.
    """

    def __init__(self, x, y, z, *, alpha=1.0, M=None):
        super().__init__(x, InnerProductRHS(y, z, alpha=alpha, M=M))


class InnerProductSolver(InnerProduct):
    ""

    def __init__(self, y, z, x, alpha=1.0, M=None):
        warnings.warn("InnerProductSolver is deprecated -- "
                      "use InnerProduct instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, z, alpha=alpha, M=M)


class NormSqSolver(InnerProduct):
    ""

    def __init__(self, y, x, alpha=1.0, M=None):
        warnings.warn("NormSqSolver is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, y, alpha=alpha, M=M)


class SumSolver(LinearEquation):
    ""

    def __init__(self, y, x):
        warnings.warn("SumSolver is deprecated",
                      DeprecationWarning, stacklevel=2)

        super().__init__(x, SumRHS(y))


class MatrixActionRHS(RHS):
    """Represents a right-hand-side term

    .. math::

        A x.

    :arg A: A :class:`tlm_adjoint.linear_equation.Matrix` defining :math:`A`.
    :arg x: A function or a :class:`Sequence` of functions defining :math:`x`.
    """

    def __init__(self, A, X):
        if is_function(X):
            X = (X,)
        if len({function_id(x) for x in X}) != len(X):
            raise ValueError("Invalid dependency")

        A_nl_deps = A.nonlinear_dependencies()
        if len(A_nl_deps) == 0:
            x_indices = {i: i for i in range(len(X))}
            super().__init__(X, nl_deps=[])
        else:
            nl_deps = list(A_nl_deps)
            nl_dep_ids = {function_id(dep): i for i, dep in enumerate(nl_deps)}
            x_indices = {}
            for i, x in enumerate(X):
                x_id = function_id(x)
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
        if is_function(B):
            B = (B,)
        X = [deps[j] for j in self._x_indices]
        self._A.forward_action(deps[:len(self._A.nonlinear_dependencies())],
                               X[0] if len(X) == 1 else X,
                               B[0] if len(B) == 1 else B,
                               method="add")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        if is_function(adj_X):
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

        X = [deps[j] for j in self._x_indices]
        tlm_X = tuple(get_tangent_linear(x, M, dM, tlm_map) for x in X)
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

    :arg x: A function whose degrees of freedom define :math:`x`.
    :arg y: A function whose degrees of freedom define :math:`y`. May be the
        same function as `x`.
    :arg alpha: A scalar defining :math:`\alpha`.
    """

    def __init__(self, x, y, *, alpha=1.0):
        check_space_types_dual(x, y)

        x_equals_y = function_id(x) == function_id(y)
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
        self._x = function_replacement(self._x)
        self._y = function_replacement(self._y)

    def add_forward(self, b, deps):
        if self._x_equals_y:
            (x,), (y,) = deps, deps
        else:
            x, y = deps

        if function_local_size(y) != function_local_size(x):
            raise ValueError("Invalid space")
        check_space_types_dual(x, y)

        d = (function_get_values(y) * function_get_values(x)).sum()
        comm = function_comm(b)
        if comm.size > 1:
            import mpi4py.MPI as MPI
            d_local = np.array([d], dtype=function_dtype(b))
            d_global = np.full((1,), np.NAN, dtype=function_dtype(b))
            comm.Allreduce(d_local, d_global, op=MPI.SUM)
            d, = d_global
            del d_local, d_global

        function_set_values(b, function_get_values(b) + self._alpha * d)

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._x_equals_y:
            if dep_index == 0:
                x, = nl_deps
                alpha = -2.0 * self._alpha.conjugate() * function_sum(adj_x)
                function_set_values(
                    b,
                    function_get_values(b)
                    + alpha * function_get_values(x).conjugate())
            else:
                raise IndexError("dep_index out of bounds")
        elif dep_index == 0:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * function_sum(adj_x)
            function_set_values(
                b,
                function_get_values(b)
                + alpha * function_get_values(y).conjugate())
        elif dep_index == 1:
            x, y = nl_deps
            alpha = -self._alpha.conjugate() * function_sum(adj_x)
            function_set_values(
                b,
                function_get_values(b)
                + alpha * function_get_values(x).conjugate())
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        tlm_B = []

        if self._x_equals_y:
            x, = self.dependencies()

            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(DotProductRHS(tlm_x, x, alpha=2.0 * self._alpha))
        else:
            x, y = self.dependencies()

            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(DotProductRHS(tlm_x, y, alpha=self._alpha))

            tlm_y = get_tangent_linear(y, M, dM, tlm_map)
            if tlm_y is not None:
                tlm_B.append(DotProductRHS(x, tlm_y, alpha=self._alpha))

        return tlm_B


class InnerProductRHS(RHS):
    r"""Represents a right-hand-side term

    .. math::

        \alpha y^* M x.

    :arg x: A function whose degrees of freedom define :math:`x`.
    :arg y: A function whose degrees of freedom define :math:`y`. May be the
        same function as `x`.
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

        norm_sq = function_id(x) == function_id(y)
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
        self._x = function_replacement(self._x)
        self._y = function_replacement(self._y)
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
            Y = function_new_conjugate_dual(x)
            self._M.adjoint_action(M_deps, y, Y, method="assign")
        check_space_types_conjugate_dual(x, Y)

        function_set_values(b,
                            function_get_values(b) + self._alpha
                            * function_inner(x, Y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._norm_sq:
            if dep_index == 0:
                x = nl_deps[0]
                if not issubclass(function_dtype(x), (float, np.floating)):
                    raise RuntimeError("Not complex differentiable")
                M_deps = nl_deps[1:]

                if self._M is None:
                    X = x
                else:
                    X = function_new_conjugate_dual(x)
                    self._M.adjoint_action(M_deps, x, X, method="assign")

                function_axpy(
                    b, -self._alpha.conjugate() * function_sum(adj_x), X)

                if self._M is None:
                    X = x
                else:
                    X = function_new_conjugate_dual(x)
                    self._M.forward_action(M_deps, x, X, method="assign")

                function_axpy(
                    b, -self._alpha.conjugate() * function_sum(adj_x), X)
            else:
                raise IndexError("dep_index out of bounds")
        elif dep_index == 0:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                Y = y
            else:
                Y = function_new_conjugate_dual(x)
                self._M.adjoint_action(M_deps, y, Y, method="assign")

            function_axpy(b, -self._alpha.conjugate() * function_sum(adj_x), Y)
        elif dep_index == 1:
            x, y = nl_deps[:2]
            if not issubclass(function_dtype(y), (float, np.floating)):
                raise RuntimeError("Not complex differentiable")
            M_deps = nl_deps[2:]

            if self._M is None:
                X = x
            else:
                X = function_new_conjugate_dual(y)
                self._M.forward_action(M_deps, x, X, method="assign")

            function_axpy(b, -self._alpha.conjugate() * function_sum(adj_x), X)
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        tlm_B = []

        if self._norm_sq:
            x = self.dependencies()[0]
            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                if not issubclass(function_dtype(x), (float, np.floating)):
                    raise RuntimeError("Not complex differentiable")
                tlm_B.append(InnerProductRHS(tlm_x, x, alpha=self._alpha,
                                             M=self._M))
                tlm_B.append(InnerProductRHS(x, tlm_x, alpha=self._alpha,
                                             M=self._M))
        else:
            x, y = self.dependencies()[:2]

            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(tlm_x, y, alpha=self._alpha,
                                             M=self._M))

            tlm_y = get_tangent_linear(y, M, dM, tlm_map)
            if tlm_y is not None:
                if not issubclass(function_dtype(y), (float, np.floating)):
                    raise RuntimeError("Not complex differentiable")
                tlm_B.append(InnerProductRHS(x, tlm_y, alpha=self._alpha,
                                             M=self._M))

        return tlm_B


class NormSqRHS(InnerProductRHS):
    ""

    def __init__(self, x, alpha=1.0, M=None):
        warnings.warn("NormSqRHS is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, x, alpha=alpha, M=M)


class SumRHS(RHS):
    ""

    def __init__(self, x):
        warnings.warn("SumRHS is deprecated",
                      DeprecationWarning, stacklevel=2)

        super().__init__([x], nl_deps=[])

    def add_forward(self, b, deps):
        y, = deps
        function_set_values(b, function_get_values(b) + function_sum(y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if dep_index == 0:
            function_set_values(b,
                                function_get_values(b) - function_sum(adj_x))
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        y, = self.dependencies()
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return None
        else:
            return SumRHS(tau_y)
