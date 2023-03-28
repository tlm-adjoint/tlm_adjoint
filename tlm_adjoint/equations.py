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
    check_space_types_dual, conjugate_dual_space_type, function_assign, \
    function_axpy, function_comm, function_dtype, function_get_values, \
    function_id, function_inner, function_local_size, \
    function_new_conjugate_dual, function_replacement, function_set_values, \
    function_space, function_space_type, function_sum, function_zero, \
    is_function, space_new

from .alias import WeakAlias
from .equation import Equation, Referrer
from .tangent_linear import get_tangent_linear

from collections.abc import Sequence
import inspect
import numpy as np
import warnings

__all__ = \
    [
        "Assignment",
        "Axpy",
        "LinearCombination",
        "ZeroAssignment",

        "LinearEquation",
        "Matrix",
        "RHS",

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
        "NullSolver",
        "ScaleSolver",
        "SumRHS",
        "SumSolver"
    ]


class ZeroAssignment(Equation):
    r"""Represents an assignment

    .. math::

        x = 0.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x \right) = x.

    :arg X: A function or a :class:`Sequence` of functions defining the forward
        solution :math:`x`.
    """

    def __init__(self, X):
        if is_function(X):
            X = (X,)
        super().__init__(X, X, nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)
        for x in X:
            function_zero(x)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)
        if dep_index < len(adj_X):
            return adj_X[dep_index]
        else:
            raise IndexError("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        return ZeroAssignment([tlm_map[x] for x in self.X()])


class NullSolver(ZeroAssignment):
    ""

    def __init__(self, X):
        warnings.warn("NullSolver is deprecated -- "
                      "use ZeroAssignment instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(X)


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
    :arg args: A :class:`Sequence` of two element :class:`Sequence` objects.
        The :math:`i` th element consists of `(alpha_i, y_i)`, where `alpha_i`
        is a scalar corresponding to :math:`\alpha_i` and `y_i` a function
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


class LinearEquation(Equation):
    def __init__(self, X, B, *, A=None, adj_type=None):
        if isinstance(X, RHS) \
                or (isinstance(X, Sequence) and len(X) > 0
                    and isinstance(X[0], RHS)):
            warnings.warn("LinearEquation(B, X, *, A=None, adj_type=None) "
                          "signature is deprecated -- use "
                          "LinearEquation(X, B, *, A=None, adj_type=None) "
                          "instead",
                          DeprecationWarning, stacklevel=2)
            X, B = B, X

        if is_function(X):
            X = (X,)
        if isinstance(B, RHS):
            B = (B,)
        if adj_type is None:
            if A is None:
                adj_type = "conjugate_dual"
            else:
                adj_type = "primal"

        deps = []
        dep_ids = {}
        nl_deps = []
        nl_dep_ids = {}

        x_ids = set()
        for x in X:
            x_id = function_id(x)
            if x_id in x_ids:
                raise ValueError("Duplicate solve")
            x_ids.add(x_id)
            deps.append(x)
            dep_ids[x_id] = len(deps) - 1

        b_dep_indices = tuple([] for b in B)
        b_nl_dep_indices = tuple([] for b in B)

        for i, b in enumerate(B):
            for dep in b.dependencies():
                dep_id = function_id(dep)
                if dep_id in x_ids:
                    raise ValueError("Invalid dependency in linear Equation")
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                b_dep_indices[i].append(dep_ids[dep_id])
            for dep in b.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                b_nl_dep_indices[i].append(nl_dep_ids[dep_id])

        b_dep_ids = tuple({function_id(b_dep): i
                           for i, b_dep in enumerate(b.dependencies())}
                          for b in B)

        if A is not None:
            A_dep_indices = []
            A_nl_dep_indices = []
            for dep in A.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                A_dep_indices.append(dep_ids[dep_id])
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                A_nl_dep_indices.append(nl_dep_ids[dep_id])

            A_nl_dep_ids = {function_id(A_nl_dep): i
                            for i, A_nl_dep
                            in enumerate(A.nonlinear_dependencies())}

            if len(A.nonlinear_dependencies()) > 0:
                A_x_indices = []
                for x in X:
                    x_id = function_id(x)
                    if x_id not in nl_dep_ids:
                        nl_deps.append(x)
                        nl_dep_ids[x_id] = len(nl_deps) - 1
                    A_x_indices.append(nl_dep_ids[x_id])

        del x_ids, dep_ids, nl_dep_ids

        super().__init__(
            X, deps, nl_deps=nl_deps,
            ic=A is not None and A.has_initial_condition(),
            adj_ic=A is not None and A.adjoint_has_initial_condition(),
            adj_type=adj_type)
        self._B = tuple(B)
        self._b_dep_indices = b_dep_indices
        self._b_nl_dep_indices = b_nl_dep_indices
        self._b_dep_ids = b_dep_ids
        self._A = A
        if A is not None:
            self._A_dep_indices = A_dep_indices
            self._A_nl_dep_indices = A_nl_dep_indices
            self._A_nl_dep_ids = A_nl_dep_ids
            if len(A.nonlinear_dependencies()) > 0:
                self._A_x_indices = A_x_indices

        self.add_referrer(*B)
        if A is not None:
            self.add_referrer(A)

    def drop_references(self):
        super().drop_references()
        self._B = tuple(WeakAlias(b) for b in self._B)
        if self._A is not None:
            self._A = WeakAlias(self._A)

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)
        if deps is None:
            deps = self.dependencies()

        if self._A is None:
            for x in X:
                function_zero(x)
            B = X
        else:
            def b_space_type(m):
                space_type = function_space_type(
                    self.X(m), rel_space_type=self.adj_X_type(m))
                return conjugate_dual_space_type(space_type)

            B = tuple(space_new(function_space(x), space_type=b_space_type(m))
                      for m, x in enumerate(X))

        for i, b in enumerate(self._B):
            b.add_forward(B[0] if len(B) == 1 else B,
                          [deps[j] for j in self._b_dep_indices[i]])

        if self._A is not None:
            self._A.forward_solve(X[0] if len(X) == 1 else X,
                                  [deps[j] for j in self._A_dep_indices],
                                  B[0] if len(B) == 1 else B)

    _reset_adjoint_warning = False

    def reset_adjoint(self):
        if self._A is not None:
            self._A.reset_adjoint()

    _initialize_adjoint_warning = False

    def initialize_adjoint(self, J, nl_deps):
        if self._A is not None:
            self._A.initialize_adjoint(J, nl_deps)

    _finalize_adjoint_warning = False

    def finalize_adjoint(self, J):
        if self._A is not None:
            self._A.finalize_adjoint(J)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        if self._A is None:
            return B
        else:
            return self._A.adjoint_solve(
                adj_X, [nl_deps[j] for j in self._A_nl_dep_indices], B)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)

        eq_deps = self.dependencies()
        if dep_index < 0 or dep_index >= len(eq_deps):
            raise IndexError("dep_index out of bounds")
        elif dep_index < len(self.X()):
            if self._A is None:
                return adj_X[dep_index]
            else:
                dep = eq_deps[dep_index]
                F = function_new_conjugate_dual(dep)
                self._A.adjoint_action([nl_deps[j]
                                        for j in self._A_nl_dep_indices],
                                       adj_X[0] if len(adj_X) == 1 else adj_X,
                                       F, b_index=dep_index, method="assign")
                return F
        else:
            dep = eq_deps[dep_index]
            dep_id = function_id(dep)
            F = function_new_conjugate_dual(dep)
            assert len(self._B) == len(self._b_dep_ids)
            for i, (b, b_dep_ids) in enumerate(zip(self._B, self._b_dep_ids)):
                if dep_id in b_dep_ids:
                    b_dep_index = b_dep_ids[dep_id]
                else:
                    continue
                b_nl_deps = [nl_deps[j] for j in self._b_nl_dep_indices[i]]
                b.subtract_adjoint_derivative_action(
                    b_nl_deps, b_dep_index,
                    adj_X[0] if len(adj_X) == 1 else adj_X,
                    F)
            if self._A is not None and dep_id in self._A_nl_dep_ids:
                A_nl_dep_index = self._A_nl_dep_ids[dep_id]
                A_nl_deps = [nl_deps[j] for j in self._A_nl_dep_indices]
                X = [nl_deps[j] for j in self._A_x_indices]
                self._A.adjoint_derivative_action(
                    A_nl_deps, A_nl_dep_index,
                    X[0] if len(X) == 1 else X,
                    adj_X[0] if len(adj_X) == 1 else adj_X,
                    F, method="add")
            return F

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()

        if self._A is None:
            tlm_B = []
        else:
            tlm_B = self._A.tangent_linear_rhs(M, dM, tlm_map,
                                               X[0] if len(X) == 1 else X)
            if tlm_B is None:
                tlm_B = []
            elif isinstance(tlm_B, RHS):
                tlm_B = [tlm_B]
        for b in self._B:
            tlm_b = b.tangent_linear_rhs(M, dM, tlm_map)
            if tlm_b is None:
                pass
            elif isinstance(tlm_b, RHS):
                tlm_B.append(tlm_b)
            else:
                tlm_B.extend(tlm_b)

        if len(tlm_B) == 0:
            return ZeroAssignment([tlm_map[x] for x in self.X()])
        else:
            return LinearEquation([tlm_map[x] for x in self.X()], tlm_B,
                                  A=self._A, adj_type=self.adj_X_type())


class Matrix(Referrer):
    def __init__(self, nl_deps=None, *, has_ic_dep=None, ic=None, adj_ic=True):
        if nl_deps is None:
            nl_deps = []
        if len({function_id(dep) for dep in nl_deps}) != len(nl_deps):
            raise ValueError("Duplicate non-linear dependency")

        if has_ic_dep is not None:
            warnings.warn("has_ic_dep argument is deprecated -- use ic "
                          "instead",
                          DeprecationWarning, stacklevel=2)
            if ic is not None:
                raise TypeError("Cannot pass both has_ic_dep and ic arguments")
            ic = has_ic_dep
        elif ic is None:
            ic = True

        super().__init__()
        self._nl_deps = tuple(nl_deps)
        self._ic = ic
        self._adj_ic = adj_ic

    _reset_adjoint_warning = True
    _initialize_adjoint_warning = True
    _finalize_adjoint_warning = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "reset_adjoint"):
            if cls._reset_adjoint_warning:
                warnings.warn("Matrix.reset_adjoint method is deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._reset_adjoint_warning = False
            cls.reset_adjoint = lambda self: None

        if hasattr(cls, "initialize_adjoint"):
            if cls._initialize_adjoint_warning:
                warnings.warn("Matrix.initialize_adjoint method is "
                              "deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._initialize_adjoint_warning = False
            cls.initialize_adjoint = lambda self, J, nl_deps: None

        if hasattr(cls, "finalize_adjoint"):
            if cls._finalize_adjoint_warning:
                warnings.warn("Matrix.finalize_adjoint method is deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._finalize_adjoint_warning = False
            cls.finalize_adjoint = lambda self, J: None

        adj_solve_sig = inspect.signature(cls.adjoint_solve)
        if tuple(adj_solve_sig.parameters.keys()) in [("self", "nl_deps", "b"),
                                                      ("self", "nl_deps", "B")]:  # noqa: E501
            warnings.warn("Matrix.adjoint_solve(self, nl_deps, b/B) method "
                          "signature is deprecated",
                          DeprecationWarning, stacklevel=2)

            def adjoint_solve(self, adj_X, nl_deps, B):
                return adjoint_solve_orig(self, nl_deps, B)
            adjoint_solve_orig = cls.adjoint_solve
            cls.adjoint_solve = adjoint_solve

    def drop_references(self):
        self._nl_deps = tuple(function_replacement(dep)
                              for dep in self._nl_deps)

    def nonlinear_dependencies(self):
        return self._nl_deps

    def has_initial_condition_dependency(self):
        warnings.warn("Matrix.has_initial_condition_dependency method is "
                      "deprecated -- use Matrix.has_initial_condition instead",
                      DeprecationWarning, stacklevel=2)
        return self._ic

    def has_initial_condition(self):
        return self._ic

    def adjoint_has_initial_condition(self):
        return self._adj_ic

    def forward_action(self, nl_deps, X, B, method="assign"):
        """
        Evaluate the (forward) action of the matrix.

        The form:
            forward_action(self, nl_deps, x, b, method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A sequence of functions defining the values of non-linear
                     dependencies.
        x/X          The argument of the matrix action.
        b/B          The result of the matrix action.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise NotImplementedError("Method not overridden")

    def adjoint_action(self, nl_deps, adj_X, b, b_index=0, method="assign"):
        """
        Evaluate the adjoint action of the matrix.

        The form:
            adjoint_action(self, nl_deps, adj_x, b, b_index=0, method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A sequence of functions defining the values of non-linear
                     dependencies.
        adj_x/adj_X  The argument of the matrix action.
        b            The result of the matrix action.
        b_index      (Optional) The element of the matrix action B to compute.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise NotImplementedError("Method not overridden")

    def forward_solve(self, X, nl_deps, B):
        raise NotImplementedError("Method not overridden")

    def adjoint_derivative_action(self, nl_deps, nl_dep_index, X, adj_X, b,
                                  method="assign"):
        """
        Evaluate the action of the adjoint of a derivative of the matrix
        action.

        The form:
            adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b,
                                      method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A sequence of functions defining the values of non-linear
                     dependencies.
        nl_dep_index The index of the dependency in
                     self.nonlinear_dependencies() with respect to which a
                     derivative should be taken.
        x/X          The argument of the forward matrix action.
        adj_x/adj_X  A function or sequence of functions on which the adjoint
                     of the derivative acts.
        b            The result.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise NotImplementedError("Method not overridden")

    def adjoint_solve(self, adj_X, nl_deps, B):
        raise NotImplementedError("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map, X):
        raise NotImplementedError("Method not overridden")


class RHS(Referrer):
    def __init__(self, deps, nl_deps=None):
        dep_ids = set(map(function_id, deps))
        if len(dep_ids) != len(deps):
            raise ValueError("Duplicate dependency")

        if nl_deps is None:
            nl_deps = tuple(deps)
        nl_dep_ids = set(map(function_id, nl_deps))
        if len(nl_dep_ids) != len(nl_deps):
            raise ValueError("Duplicate non-linear dependency")
        if len(dep_ids.intersection(nl_dep_ids)) != len(nl_deps):
            raise ValueError("Non-linear dependency is not a dependency")

        super().__init__()
        self._deps = tuple(deps)
        self._nl_deps = tuple(nl_deps)

    def drop_references(self):
        self._deps = tuple(function_replacement(dep) for dep in self._deps)
        self._nl_deps = tuple(function_replacement(dep)
                              for dep in self._nl_deps)

    def dependencies(self):
        return self._deps

    def nonlinear_dependencies(self):
        return self._nl_deps

    def add_forward(self, B, deps):
        raise NotImplementedError("Method not overridden")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        raise NotImplementedError("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        raise NotImplementedError("Method not overridden")


class MatrixActionSolver(LinearEquation):
    def __init__(self, Y, A, X):
        warnings.warn("MatrixActionSolver is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(X, MatrixActionRHS(A, Y))


class DotProduct(LinearEquation):
    def __init__(self, x, y, z, *, alpha=1.0):
        super().__init__(x, DotProductRHS(y, z, alpha=alpha))


class DotProductSolver(DotProduct):
    def __init__(self, y, z, x, alpha=1.0):
        warnings.warn("DotProductSolver is deprecated -- "
                      "use DotProduct instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, z, alpha=alpha)


class InnerProduct(LinearEquation):
    def __init__(self, x, y, z, *, alpha=1.0, M=None):
        super().__init__(x, InnerProductRHS(y, z, alpha=alpha, M=M))


class InnerProductSolver(InnerProduct):
    def __init__(self, y, z, x, alpha=1.0, M=None):
        warnings.warn("InnerProductSolver is deprecated -- "
                      "use InnerProduct instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, z, alpha=alpha, M=M)


class NormSqSolver(InnerProduct):
    def __init__(self, y, x, alpha=1.0, M=None):
        warnings.warn("NormSqSolver is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, y, y, alpha=alpha, M=M)


class SumSolver(LinearEquation):
    def __init__(self, y, x):
        warnings.warn("SumSolver is deprecated",
                      DeprecationWarning, stacklevel=2)

        super().__init__(x, SumRHS(y))


class MatrixActionRHS(RHS):
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
    def __init__(self, x, y, alpha=1.0):
        """
        Represents a dot product of the form, y^T x, with *no* complex
        conjugation.

        Arguments:

        x, y   Dot product arguments. May be the same function.
        alpha  (Optional) Scale the result of the dot product by alpha.
        """

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
    def __init__(self, x, y, alpha=1.0, M=None):
        """
        Represents an inner product, y^* M x.

        Arguments:

        x, y   Inner product arguments. May be the same function.
        alpha  (Optional) Scale the result of the inner product by alpha.
        M      (Optional) Matrix defining the inner product. Must have no
               non-linear dependencies. Defaults to an identity matrix.
        """

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
    def __init__(self, x, alpha=1.0, M=None):
        warnings.warn("NormSqRHS is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(x, x, alpha=alpha, M=M)


class SumRHS(RHS):
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
