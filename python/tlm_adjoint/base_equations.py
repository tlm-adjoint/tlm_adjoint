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

from .interface import function_assign, function_axpy, function_copy, \
    function_get_values, function_global_size, function_id, function_inner, \
    function_is_checkpointed, function_local_indices, function_new, \
    function_replacement, function_set_values, function_space, function_sum, \
    function_update_caches, function_update_state, function_zero, \
    is_function, space_new
from .backend_interface import finalize_adjoint_derivative_action, info, \
    subtract_adjoint_derivative_action

from .alias import WeakAlias, gc_disabled
from .manager import manager as _manager

import copy
import inspect
import numpy as np
import warnings
import weakref

__all__ = \
    [
        "EquationException",

        "AdjointBlockRHS",
        "AdjointEquationRHS",
        "AdjointModelRHS",
        "AdjointRHS",

        "Equation",

        "ControlsMarker",
        "FunctionalMarker",

        "get_tangent_linear",

        "AssignmentSolver",
        "AxpySolver",
        "FixedPointSolver",
        "LinearCombinationSolver",
        "NullSolver",
        "ScaleSolver",

        "LinearEquation",
        "Matrix",
        "RHS",

        "InnerProductRHS",
        "InnerProductSolver",
        "MatrixActionRHS",
        "MatrixActionSolver",
        "NormSqRHS",
        "NormSqSolver",
        "SumRHS",
        "SumSolver",

        "Storage",

        "HDF5Storage",
        "MemoryStorage"
    ]


class EquationException(Exception):
    pass


class AdjointRHS:
    def __init__(self, space):
        self._space = space
        self._b = None

    def b(self, copy=False):
        self.finalize()
        if copy:
            return function_copy(self._b)
        else:
            return self._b

    def initialize(self):
        if self._b is None:
            self._b = space_new(self._space)

    def finalize(self):
        self.initialize()
        finalize_adjoint_derivative_action(self._b)

    def sub(self, b):
        if b is not None:
            self.initialize()
            subtract_adjoint_derivative_action(self._b, b)

    def is_empty(self):
        return self._b is None


class AdjointEquationRHS:
    def __init__(self, eq):
        self._B = tuple(AdjointRHS(function_space(x)) for x in eq.X())

    def __getitem__(self, key):
        return self._B[key]

    def b(self, copy=False):
        if len(self._B) != 1:
            raise EquationException("Right-hand-side does not consist of exactly one function")  # noqa: E501
        return self._B[0].b(copy=copy)

    def B(self, copy=False):
        return tuple(B.b(copy=copy) for B in self._B)

    def finalize(self):
        for b in self._B:
            b.finalize()

    def is_empty(self):
        for b in self._B:
            if not b.is_empty():
                return False
        return True


class AdjointBlockRHS:
    def __init__(self, block):
        self._B = [AdjointEquationRHS(eq) for eq in block]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._B[key]
        else:
            k, m = key
            return self._B[k][m]

    def pop(self):
        return len(self._B) - 1, self._B.pop()

    def finalize(self):
        for B in self._B:
            B.finalize()

    def is_empty(self):
        return len(self._B) == 0


class AdjointModelRHS:
    def __init__(self, blocks):
        self._B = [AdjointBlockRHS(block) for block in blocks]
        self._pop_empty()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._B[key]
        elif len(key) == 2:
            p, k = key
            return self._B[p][k]
        else:
            p, k, m = key
            return self._B[p][k][m]

    def pop(self):
        i, B = self._B[-1].pop()
        n = len(self._B) - 1
        self._pop_empty()
        return (n, i), B

    def _pop_empty(self):
        while len(self._B) > 0 and self._B[-1].is_empty():
            self._B.pop()

    def is_empty(self):
        return len(self._B) == 0


class Referrer:
    _id_counter = [0]

    def __init__(self, referrers=[]):
        self._id = self._id_counter[0]
        self._id_counter[0] += 1
        self._referrers = weakref.WeakValueDictionary()
        self._references_dropped = False

        self.add_referrer(*referrers)

    def id(self):
        return self._id

    @gc_disabled
    def add_referrer(self, *referrers):
        if self._references_dropped:
            raise EquationException("Cannot call add_referrer method after "
                                    "_drop_references method has been called")
        for referrer in referrers:
            referrer_id = referrer.id()
            assert self._referrers.get(referrer_id, referrer) is referrer
            self._referrers[referrer_id] = referrer

    @gc_disabled
    def referrers(self):
        referrers = {}
        remaining_referrers = {self.id(): self}
        while len(remaining_referrers) > 0:
            referrer_id, referrer = remaining_referrers.popitem()
            if referrer_id not in referrers:
                referrers[referrer_id] = referrer
                for child in tuple(referrer._referrers.valuerefs()):
                    child = child()
                    if child is not None:
                        child_id = child.id()
                        if child_id not in referrers and child_id not in remaining_referrers:  # noqa: E501
                            remaining_referrers[child_id] = child
        return tuple(e[1] for e in sorted(tuple(referrers.items()),
                                          key=lambda e: e[0]))

    def _drop_references(self):
        if not self._references_dropped:
            self.drop_references()
            self._references_dropped = True

    def drop_references(self):
        raise EquationException("Method not overridden")


def no_replace_compatibility(function):
    def wrapped_function(*args, **kwargs):
        return function(*args, **kwargs)
    wrapped_function._replace_compatibility = False
    return wrapped_function


class Equation(Referrer):
    def __init__(self, X, deps, nl_deps=None,
                 ic_deps=None, ic=None,
                 adj_ic_deps=None, adj_ic=None):
        """
        An equation. The equation is expressed in the form:
            F ( X, y_0, y_1, ... ) = 0,
        where X is the equation solution and F is a residual function.
        Information regarding F is provided by methods which should be
        overridden as required by derived classes.

        Arguments:

        X        A function, or a list or tuple of functions. The solution to
                 the equation.
        deps     A list or tuple of dependencies, which must include the
                 solution itself.
        nl_deps  (Optional) A list or tuple of non-linear dependencies. Must be
                 a subset of deps. Defaults to deps.
        ic_deps  (Optional) A list or tuple of dependencies whose initial value
                 should be available prior to solving the forward equation.
                 Must be a subset of X.
        ic       (Optional) If true then ic_deps is set equal to X. Defaults to
                 true if ic_deps is None, and false otherwise.
        adj_ic_deps  (Optional) A list or tuple of dependencies whose adjoint
                     value should be available prior to solving the adjoint
                     equation. Must be a subset of X.
        adj_ic       (Optional) If true then adj_ic_deps is set equal to X.
                     Defaults to true if adj_ic_deps is None, and false
                     otherwise.
        """

        if is_function(X):
            X = (X,)
        X_ids = {function_id(x) for x in X}
        for x in X:
            if not is_function(x):
                raise EquationException("Solution must be a function")
            if not function_is_checkpointed(x):
                raise EquationException("Solution must be checkpointed")
            if x not in deps:
                raise EquationException("Solution must be a dependency")

        dep_ids = {function_id(dep): i for i, dep in enumerate(deps)}
        if len(dep_ids) != len(deps):
            raise EquationException("Duplicate dependency")

        if nl_deps is None:
            nl_deps = tuple(deps)
        nl_dep_ids = {function_id(dep) for dep in nl_deps}
        if len(nl_dep_ids) != len(nl_deps):
            raise EquationException("Duplicate non-linear dependency")
        for dep in nl_deps:
            if function_id(dep) not in dep_ids:
                raise EquationException("Non-linear dependency is not a "
                                        "dependency")
        nl_deps_map = tuple(dep_ids[function_id(dep)] for dep in nl_deps)

        if ic_deps is None:
            ic_deps = []
            if ic is None:
                ic = True
        else:
            if ic is None:
                ic = False
        ic_dep_ids = {function_id(dep) for dep in ic_deps}
        if len(ic_dep_ids) != len(ic_deps):
            raise EquationException("Duplicate initial condition dependency")
        for dep in ic_deps:
            if function_id(dep) not in X_ids:
                raise EquationException("Initial condition dependency is not "
                                        "a solution")
        if ic:
            ic_deps = list(X)

        if adj_ic_deps is None:
            adj_ic_deps = []
            if adj_ic is None:
                adj_ic = True
        else:
            if adj_ic is None:
                adj_ic = False
        adj_ic_dep_ids = {function_id(dep) for dep in adj_ic_deps}
        if len(adj_ic_dep_ids) != len(adj_ic_deps):
            raise EquationException("Duplicate adjoint initial condition "
                                    "dependency")
        for dep in adj_ic_deps:
            if function_id(dep) not in X_ids:
                raise EquationException("Adjoint initial condition "
                                        "dependency is not a solution")
        if adj_ic:
            adj_ic_deps = list(X)

        super().__init__()
        self._X = tuple(X)
        self._deps = tuple(deps)
        self._nl_deps = tuple(nl_deps)
        self._nl_deps_map = nl_deps_map
        self._ic_deps = tuple(ic_deps)
        self._adj_ic_deps = tuple(adj_ic_deps)

    _reset_adjoint_warning = True
    _initialize_adjoint_warning = True
    _finalize_adjoint_warning = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if getattr(cls.replace, "_replace_compatibility", True):
            warnings.warn("Equation.replace method is deprecated",
                          DeprecationWarning, stacklevel=2)

            def drop_references(self):
                replace_map = {dep: function_replacement(dep)
                               for dep in self.dependencies()}
                self.replace(replace_map)
            cls.drop_references = drop_references
            cls.replace._replace_compatibility = False

        if hasattr(cls, "reset_adjoint"):
            if cls._reset_adjoint_warning:
                warnings.warn("Equation.reset_adjoint method is deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._reset_adjoint_warning = False
            cls.reset_adjoint = lambda self: None

        if hasattr(cls, "initialize_adjoint"):
            if cls._initialize_adjoint_warning:
                warnings.warn("Equation.initialize_adjoint method is "
                              "deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._initialize_adjoint_warning = False
            cls.initialize_adjoint = lambda self, J, nl_deps: None

        if hasattr(cls, "finalize_adjoint"):
            if cls._finalize_adjoint_warning:
                warnings.warn("Equation.finalize_adjoint method is deprecated",
                              DeprecationWarning, stacklevel=2)
        else:
            cls._finalize_adjoint_warning = False
            cls.finalize_adjoint = lambda self, J: None

        adj_solve_sig = inspect.signature(cls.adjoint_jacobian_solve)
        if tuple(adj_solve_sig.parameters.keys()) in [("self", "nl_deps", "b"),
                                                      ("self", "nl_deps", "B")]:  # noqa: E501
            warnings.warn("Equation.adjoint_jacobian_solve(self, nl_deps, b/B) "  # noqa: E501
                          "method signature is deprecated",
                          DeprecationWarning, stacklevel=2)

            def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
                return adjoint_jacobian_solve_orig(self, nl_deps, B)
            adjoint_jacobian_solve_orig = cls.adjoint_jacobian_solve
            cls.adjoint_jacobian_solve = adjoint_jacobian_solve

    def drop_references(self):
        self._X = tuple(function_replacement(x) for x in self._X)
        self._deps = tuple(function_replacement(dep) for dep in self._deps)
        self._nl_deps = tuple(function_replacement(dep)
                              for dep in self._nl_deps)
        self._ic_deps = tuple(function_replacement(dep)
                              for dep in self._ic_deps)
        self._adj_ic_deps = tuple(function_replacement(dep)
                                  for dep in self._adj_ic_deps)

    @no_replace_compatibility
    def replace(self, replace_map):
        """
        Replace all internal functions using the supplied replace map. Must
        call the base class replace method.
        """

        self._X = tuple(replace_map.get(x, x) for x in self._X)
        self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
        self._nl_deps = tuple(replace_map.get(dep, dep)
                              for dep in self._nl_deps)
        self._ic_deps = tuple(replace_map.get(dep, dep)
                              for dep in self._ic_deps)
        self._adj_ic_deps = tuple(replace_map.get(dep, dep)
                                  for dep in self._adj_ic_deps)

    def x(self):
        """
        If the equation solves for exactly one function, return it. Otherwise
        raise an error.
        """

        if len(self._X) != 1:
            raise EquationException("Equation does not solve for exactly one "
                                    "function")
        return self._X[0]

    def X(self):
        """
        A tuple of functions. The solution to the equation.
        """

        return self._X

    def dependencies(self):
        return self._deps

    def nonlinear_dependencies(self):
        return self._nl_deps

    def nonlinear_dependencies_map(self):
        return self._nl_deps_map

    def initial_condition_dependencies(self):
        return self._ic_deps

    def adjoint_initial_condition_dependencies(self):
        return self._adj_ic_deps

    def _pre_process(self, manager=None, annotate=None):
        if manager is None:
            manager = _manager()
        for dep in self.initial_condition_dependencies():
            manager.add_initial_condition(dep, annotate=annotate)

    def _post_process(self, manager=None, annotate=None, tlm=None,
                      tlm_skip=None):
        if manager is None:
            manager = _manager()
        manager.add_equation(self, annotate=annotate, tlm=tlm,
                             tlm_skip=tlm_skip)

    def solve(self, manager=None, annotate=None, tlm=None, _tlm_skip=None):
        """
        Solve the equation.

        Arguments:

        manager   (Optional) The equation manager.
        annotate  (Optional) Whether the equation should be annotated.
        tlm       (Optional) Whether to derive (and solve) associated
                  tangent-linear equations.
        """

        if manager is None:
            manager = _manager()

        self._pre_process(manager=manager, annotate=annotate)

        annotation_enabled, tlm_enabled = manager.stop()
        self.forward(self.X())
        manager.start(annotation=annotation_enabled, tlm=tlm_enabled)

        self._post_process(manager=manager, annotate=annotate, tlm=tlm,
                           tlm_skip=_tlm_skip)

    def forward(self, X, deps=None):
        """
        Solve the equation. The manager is stopped when this method is called.

        Arguments:

        X     A list or tuple of functions. The solution, which should be set
              by this method.
        deps  (Optional) A list or tuple of functions defining the values of
              dependencies.
        """

        function_update_caches(*self.dependencies(), value=deps)
        self.forward_solve(X[0] if len(X) == 1 else X, deps=deps)
        function_update_state(*X)
        function_update_caches(*self.X(), value=X)

    def forward_solve(self, X, deps=None):
        """
        Solve the equation. The manager is stopped when this method is called.

        The form:
            forward_solve(self, x, deps=None)
        should be used for equations which solve for a single function.

        Arguments:

        x/X   The solution, which should be set by this method.
        deps  (Optional) A list or tuple of functions defining the values of
              dependencies. self.dependencies() should be used if this is not
              supplied.
        """

        raise EquationException("Method not overridden")

    def adjoint(self, J, adj_X, nl_deps, B, dep_Bs):
        """
        Solve the adjoint equation with the given right-hand-side, and subtract
        corresponding adjoint terms from other adjoint equations.

        Arguments:

        J          Adjoint model functional.
        adj_X      A list or tuple of functions defining the initial guess for
                   the adjoint solve, or None if the Equation does not accept
                   an initial guess. May be modified or returned by this
                   method.
        nl_deps    A list or tuple of functions defining the values of
                   non-linear dependencies.
        B          A list or tuple of functions defining the right-hand-side.
                   May be modified or returned by this method.
        dep_Bs     Dictionary of dep_index: dep_B pairs, where each dep_B is an
                   AdjointRHS which should be updated by subtracting derivative
                   information computed by differentiating with respect to
                   self.dependencies()[dep_index].

        Returns the solution of the adjoint equation as a tuple of functions.
        The result will not be modified by calling code.
        """

        function_update_caches(*self.nonlinear_dependencies(), value=nl_deps)

        self.initialize_adjoint(J, nl_deps)

        if adj_X is not None and len(adj_X) == 1:
            adj_X = adj_X[0]
        adj_X = self.adjoint_jacobian_solve(
            adj_X, nl_deps, B[0] if len(B) == 1 else B)
        if adj_X is None:
            warnings.warn("None return from Equation.adjoint_jacobian_solve "
                          "is deprecated",
                          DeprecationWarning, stacklevel=2)
            if len(B) == 1:
                adj_X = function_new(B[0])
            else:
                adj_X = tuple(function_new(b) for b in B)
        self.subtract_adjoint_derivative_actions(adj_X, nl_deps, dep_Bs)

        self.finalize_adjoint(J)

        return (adj_X,) if is_function(adj_X) else tuple(adj_X)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        """
        Return the action of the adjoint of a derivative of the RHS.

        The return value will not be modified by calling code.

        The form:
            adjoint_derivative_action(self, nl_deps, dep_index, adj_x)
        should be used for equations which solve for a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        dep_index    The index of the dependency in self.dependencies() with
                     respect to which a derivative should be taken.
        adj_x/adj_X  The direction of the adjoint derivative action.
        """

        raise EquationException("Method not overridden")

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        """
        Subtract adjoint derivative actions from adjoint right-hand-sides.
        Lower level than adjoint_derivative_action, but can be overridden for
        optimization, and can be defined in place of defining an
        adjoint_derivative_action method.

        The form:
            subtract_adjoint_derivative_actions(self, adj_x, nl_deps, Bs)
        should be used for equations which solve for a single function.

        Arguments:

        adj_x/adj_X  The direction of the adjoint derivative actions.
        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        dep_Bs       Dictionary of dep_index: dep_B pairs, where each dep_B is
                     an AdjointRHS which should be updated by subtracting
                     derivative information computed by differentiating with
                     respect to self.dependencies()[dep_index].
        """

        for dep_index, dep_B in dep_Bs.items():
            dep_B.sub(self.adjoint_derivative_action(nl_deps, dep_index,
                                                     adj_X))

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        """
        Solve an adjoint equation, returning the result. The result will not
        be modified by calling code.

        The form:
            adjoint_jacobian_solve(self, adj_x, nl_deps, b)
        should be used for equations which solve for a single function.

        Arguments:

        adj_x/adj_X    Initial guess for the adjoint solve, or None if the
                       Equation does not accept an initial guess. May be
                       modified or returned by this method.
        nl_deps        A list or tuple of functions defining the values of
                       non-linear dependencies.
        b/B            The right-hand-side. May be modified or returned by this
                       method.
        """

        raise EquationException("Method not overridden")

    def tangent_linear(self, M, dM, tlm_map):
        """
        Return an Equation corresponding to a tangent linear equation,
        computing derivatives with respect to the control M in the direction
        dM.

        Arguments:

        M        A list or tuple of functions defining the control.
        dM       A list or tuple of functions defining the direction.
        tlm_map  The TangentLinearMap.
        """

        raise EquationException("Method not overridden")


class ControlsMarker(Equation):
    def __init__(self, M):
        """
        Represents the equation "controls = inputs".

        Arguments:

        M  A function, or a list or tuple of functions. May be
           non-checkpointed.
        """

        if is_function(M):
            M = (M,)

        super(Equation, self).__init__()
        self._X = tuple(M)
        self._deps = tuple(M)
        self._nl_deps = ()
        self._nl_deps_map = ()
        self._ic_deps = ()
        self._adj_ic_deps = ()

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B


class FunctionalMarker(Equation):
    def __init__(self, J):
        """
        Represents the equation "output = functional".

        Arguments:

        J  A function. The functional.
        """

        J = J.fn()
        # Extra function allocation could be avoided
        J_ = function_new(J)
        super().__init__([J_], [J_, J], nl_deps=[], ic=False, adj_ic=False)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise EquationException("Unexpected dep_index")
        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b


def get_tangent_linear(x, M, dM, tlm_map):
    try:
        return dM[M.index(x)]
    except ValueError:
        return tlm_map[x]


class NullSolver(Equation):
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
            raise EquationException("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        return NullSolver([tlm_map[x] for x in self.X()])


class AssignmentSolver(Equation):
    def __init__(self, y, x):
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
            raise EquationException("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x, y = self.dependencies()
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return NullSolver(tlm_map[x])
        else:
            return AssignmentSolver(tau_y, tlm_map[x])


class LinearCombinationSolver(Equation):
    def __init__(self, x, *args):
        alpha = tuple(float(arg[0]) for arg in args)
        Y = [arg[1] for arg in args]

        super().__init__(x, [x] + Y, nl_deps=[], ic=False, adj_ic=False)
        self._alpha = alpha

    def forward_solve(self, x, deps=None):
        deps = self.dependencies() if deps is None else tuple(deps)
        function_zero(x)
        for alpha, y in zip(self._alpha, deps[1:]):
            function_axpy(x, alpha, y)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        elif dep_index <= len(self._alpha):
            return (-self._alpha[dep_index - 1], adj_x)
        else:
            raise EquationException("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        deps = self.dependencies()
        x, ys = deps[0], deps[1:]
        args = []
        for alpha, y in zip(self._alpha, ys):
            tau_y = get_tangent_linear(y, M, dM, tlm_map)
            if tau_y is not None:
                args.append((alpha, tau_y))
        return LinearCombinationSolver(tlm_map[x], *args)


class ScaleSolver(LinearCombinationSolver):
    def __init__(self, alpha, y, x):
        super().__init__(x, (alpha, y))


class AxpySolver(LinearCombinationSolver):
    def __init__(self, *args):  # self, y_old, alpha, x, y_new
        y_old, alpha, x, y_new = args
        super().__init__(y_new, (1.0, y_old), (alpha, x))


class FixedPointSolver(Equation):
    # Derives tangent-linear and adjoint information using the approach
    # described in
    #   J. G. Gilbert, "Automatic differentiation and iterative processes",
    #     Optimization Methods and Software, 1(1), pp. 13--21, 1992
    #   B. Christianson, "Reverse accumulation and attractive fixed points",
    #     Optimization Methods and Software, 3(4), pp. 311--326, 1994
    def __init__(self, eqs, solver_parameters):
        """
        A fixed point solver.

        Arguments:

        eqs
            A list or tuple of Equation objects. The last equation defines the
            solution of the fixed point iteration. A single function cannot
            appear as the solution to two or more equations.
        solver_parameters
            Solver parameters dictionary. Parameters (based on KrylovSolver
            parameters in FEniCS 2017.2.0):
                absolute_tolerance
                    Absolute tolerance for the solution change 2-norm. Float,
                    required.
                relative_tolerance
                    Relative tolerance for the solution change 2-norm. Float,
                    required.
                maximum_iterations
                    Maximum permitted iterations. Positive integer, optional,
                    default 1000.
                nonzero_initial_guess
                    Whether to use a non-zero initial guess for the forward
                    solve. Logical, optional, default True.
                adjoint_nonzero_initial_guess
                    Whether to use a non-zero initial guess for the adjoint
                    solve. Logical, optional, default True.
                report
                    Whether to display output. Optional, default False.
        """

        X_ids = set()
        for eq in eqs:
            for x in eq.X():
                x_id = function_id(x)
                if x_id in X_ids:
                    raise EquationException("Duplicate solve")
                X_ids.add(x_id)

        solver_parameters = copy.deepcopy(solver_parameters)
        if "nonzero_adjoint_initial_guess" in solver_parameters:
            warnings.warn("'nonzero_adjoint_initial_guess' parameter is "
                          "deprecated -- use 'adjoint_nonzero_initial_guess' "
                          "instead",
                          DeprecationWarning, stacklevel=2)
            if "adjoint_nonzero_initial_guess" in solver_parameters:
                raise EquationException("Cannot supply both "
                                        "'nonzero_adjoint_initial_guess' and "
                                        "'adjoint_nonzero_initial_guess' "
                                        "parameters")
            solver_parameters["adjoint_nonzero_initial_guess"] = \
                solver_parameters.pop("nonzero_adjoint_initial_guess")
        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        for key, default_value in [("maximum_iterations", 1000),
                                   ("nonzero_initial_guess", True),
                                   ("adjoint_nonzero_initial_guess", True),
                                   ("report", False)]:
            if key not in solver_parameters:
                solver_parameters[key] = default_value

        nonzero_initial_guess = solver_parameters["nonzero_initial_guess"]
        adjoint_nonzero_initial_guess = \
            solver_parameters["adjoint_nonzero_initial_guess"]

        X = []
        deps = []
        dep_ids = {}
        nl_deps = []
        nl_dep_ids = {}

        eq_X_indices = tuple([] for eq in eqs)
        eq_dep_indices = tuple([] for eq in eqs)
        eq_nl_dep_indices = tuple([] for eq in eqs)

        for i, eq in enumerate(eqs):
            for x in eq.X():
                X.append(x)
                eq_X_indices[i].append(len(X) - 1)

            for dep in eq.dependencies():
                dep_id = function_id(dep)
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                eq_dep_indices[i].append(dep_ids[dep_id])

            for dep in eq.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                eq_nl_dep_indices[i].append(nl_dep_ids[dep_id])

        del dep_ids, nl_dep_ids

        if nonzero_initial_guess:
            ic_deps = {}
            previous_x_ids = set()
            remaining_x_ids = X_ids.copy()

            for i, eq in enumerate(eqs):
                for x in eq.X():
                    remaining_x_ids.remove(function_id(x))

                for dep in eq.dependencies():
                    dep_id = function_id(dep)
                    if dep_id in remaining_x_ids and dep_id not in ic_deps:
                        ic_deps[dep_id] = dep

                for dep in eq.initial_condition_dependencies():
                    dep_id = function_id(dep)
                    assert dep_id not in previous_x_ids
                    if dep_id not in ic_deps:
                        ic_deps[dep_id] = dep

                for x in eq.X():
                    previous_x_ids.add(function_id(x))

            ic_deps = list(ic_deps.values())
            del previous_x_ids, remaining_x_ids
        else:
            ic_deps = []

        if adjoint_nonzero_initial_guess:
            adj_ic_deps = {}
            previous_x_ids = set()
            remaining_x_ids = X_ids.copy()

            for i in range(len(eqs) - 1, -1, -1):
                i = (i - 1) % len(eqs)
                eq = eqs[i]

                for x in eq.X():
                    remaining_x_ids.remove(function_id(x))

                for dep in eq.dependencies():
                    dep_id = function_id(dep)
                    if dep_id in remaining_x_ids \
                            and dep_id not in adj_ic_deps:
                        adj_ic_deps[dep_id] = dep

                for dep in eq.adjoint_initial_condition_dependencies():
                    dep_id = function_id(dep)
                    assert dep_id not in previous_x_ids
                    if dep_id not in adj_ic_deps:
                        adj_ic_deps[dep_id] = dep

                for x in eq.X():
                    previous_x_ids.add(function_id(x))

            adj_ic_deps = list(adj_ic_deps.values())
            del previous_x_ids, remaining_x_ids
        else:
            adj_ic_deps = []

        eq_dep_index_map = tuple(
            {function_id(dep): i for i, dep in enumerate(eq.dependencies())}
            for eq in eqs)

        dep_eq_index_map = {}
        for i, eq in enumerate(eqs):
            for dep in eq.dependencies():
                dep_id = function_id(dep)
                if dep_id in dep_eq_index_map:
                    dep_eq_index_map[dep_id].append(i)
                else:
                    dep_eq_index_map[dep_id] = [i]

        dep_map = {}
        for k, eq in enumerate(eqs):
            for m, x in enumerate(eq.X()):
                dep_map[function_id(x)] = (k, m)
        dep_B_indices = tuple({} for eq in eqs)
        for i, eq in enumerate(eqs):
            for j, dep in enumerate(eq.dependencies()):
                dep_id = function_id(dep)
                if dep_id in dep_map:
                    k, m = dep_map[dep_id]
                    if k != i:
                        dep_B_indices[i][j] = (k, m)
        del dep_map

        super().__init__(X, deps, nl_deps=nl_deps,
                         ic_deps=ic_deps, adj_ic_deps=adj_ic_deps)
        self._eqs = tuple(eqs)
        self._eq_X_indices = eq_X_indices
        self._eq_dep_indices = eq_dep_indices
        self._eq_nl_dep_indices = eq_nl_dep_indices
        self._eq_dep_index_map = eq_dep_index_map
        self._dep_eq_index_map = dep_eq_index_map
        self._dep_B_indices = dep_B_indices
        self._solver_parameters = solver_parameters

        self.add_referrer(*eqs)

    def drop_references(self):
        super().drop_references()
        self._eqs = tuple(WeakAlias(eq) for eq in self._eqs)

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)

        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        absolute_tolerance = self._solver_parameters["absolute_tolerance"]
        relative_tolerance = self._solver_parameters["relative_tolerance"]
        maximum_iterations = self._solver_parameters["maximum_iterations"]
        nonzero_initial_guess = \
            self._solver_parameters["nonzero_initial_guess"]
        report = self._solver_parameters["report"]

        eq_X = tuple(tuple(X[j] for j in self._eq_X_indices[i])
                     for i in range(len(self._eqs)))
        if deps is None:
            eq_deps = tuple(None for i in range(len(self._eqs)))
        else:
            eq_deps = tuple(tuple(deps[j] for j in self._eq_dep_indices[i])
                            for i in range(len(self._eqs)))

        if not nonzero_initial_guess:
            for x in X:
                function_zero(x)
            function_update_state(*X)
            function_update_caches(*self.X(), value=X)

        it = 0
        X_0 = tuple(function_copy(x) for x in eq_X[-1])
        while True:
            it += 1

            for i, eq in enumerate(self._eqs):
                eq.forward(eq_X[i], deps=eq_deps[i])

            R = X_0
            del X_0
            R_norm_sq = 0.0
            for r, x in zip(R, eq_X[-1]):
                function_axpy(r, -1.0, x)
                R_norm_sq += function_inner(r, r)
            if relative_tolerance == 0.0:
                tolerance_sq = absolute_tolerance ** 2
            else:
                X_norm_sq = 0.0
                for x in eq_X[-1]:
                    X_norm_sq += function_inner(x, x)
                tolerance_sq = max(absolute_tolerance ** 2,
                                   X_norm_sq * (relative_tolerance ** 2))
            if report:
                info(f"Fixed point iteration, forward iteration {it:d}, "
                     f"change norm {np.sqrt(R_norm_sq):.16e} "
                     f"(tolerance {np.sqrt(tolerance_sq):.16e})")
            if np.isnan(R_norm_sq):
                raise EquationException(
                    f"Fixed point iteration, forward iteration {it:d}, "
                    f"NaN encountered")
            if R_norm_sq < tolerance_sq or R_norm_sq == 0.0:
                break
            if it >= maximum_iterations:
                raise EquationException(
                    f"Fixed point iteration, forward iteration {it:d}, "
                    f"failed to converge")

            X_0 = R
            del R
            for x_0, x in zip(X_0, eq_X[-1]):
                function_assign(x_0, x)

    _reset_adjoint_warning = False

    def reset_adjoint(self):
        for eq in self._eqs:
            eq.reset_adjoint()

    _initialize_adjoint_warning = False

    def initialize_adjoint(self, J, nl_deps):
        for i, eq in enumerate(self._eqs):
            eq_nl_deps = tuple(nl_deps[j] for j in self._eq_nl_dep_indices[i])
            eq.initialize_adjoint(J, eq_nl_deps)

    _finalize_adjoint_warning = False

    def finalize_adjoint(self, J):
        for eq in self._eqs:
            eq.finalize_adjoint(J)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        if is_function(B):
            B = (B,)
        if adj_X is None:
            adj_X = [function_new(b) for b in B]
        elif is_function(adj_X):
            adj_X = [adj_X]
        else:
            adj_X = list(adj_X)

        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        absolute_tolerance = self._solver_parameters["absolute_tolerance"]
        relative_tolerance = self._solver_parameters["relative_tolerance"]
        maximum_iterations = self._solver_parameters["maximum_iterations"]
        nonzero_initial_guess = self._solver_parameters["adjoint_nonzero_initial_guess"]  # noqa: E501
        report = self._solver_parameters["report"]

        eq_adj_X = [tuple(adj_X[j] for j in self._eq_X_indices[i])
                    for i in range(len(self._eqs))]
        eq_nl_deps = tuple(tuple(nl_deps[j] for j in nl_dep_indices)
                           for nl_dep_indices in self._eq_nl_dep_indices)
        adj_B = AdjointModelRHS([self._eqs])

        dep_Bs = tuple({} for eq in self._eqs)
        for i, eq in enumerate(self._eqs):
            eq_B = adj_B[0][i].B()
            for j, k in enumerate(self._eq_X_indices[i]):
                function_assign(eq_B[j], B[k])
            for j, (k, m) in self._dep_B_indices[i].items():
                dep_Bs[i][j] = adj_B[0][k][m]

        if nonzero_initial_guess:
            for i, eq in enumerate(self._eqs):
                eq.subtract_adjoint_derivative_actions(
                    eq_adj_X[i][0] if len(eq_adj_X[i]) == 1 else eq_adj_X[i],
                    eq_nl_deps[i], dep_Bs[i])
        else:
            for adj_x in adj_X:
                function_zero(adj_x)

        it = 0
        X_0 = tuple(function_copy(x) for x in eq_adj_X[-1])
        while True:
            it += 1

            for i in range(len(self._eqs) - 1, - 1, -1):
                i = (i - 1) % len(self._eqs)
                # Copy required here, as adjoint_jacobian_solve may return the
                # RHS function itself
                eq_B = adj_B[0][i].B(copy=True)

                eq_adj_X[i] = self._eqs[i].adjoint_jacobian_solve(
                    eq_adj_X[i][0] if len(eq_adj_X[i]) == 1 else eq_adj_X[i],
                    eq_nl_deps[i],
                    eq_B[0] if len(eq_B) == 1 else eq_B)

                if eq_adj_X[i] is None:
                    warnings.warn("None return from "
                                  "Equation.adjoint_jacobian_solve is "
                                  "deprecated",
                                  DeprecationWarning, stacklevel=2)
                    eq_adj_X[i] = tuple(function_new(b) for b in eq_B)
                elif is_function(eq_adj_X[i]):
                    eq_adj_X[i] = (eq_adj_X[i],)
                for j, x in zip(self._eq_X_indices[i], eq_adj_X[i]):
                    adj_X[j] = x

                self._eqs[i].subtract_adjoint_derivative_actions(
                    eq_adj_X[i][0] if len(eq_adj_X[i]) == 1 else eq_adj_X[i],
                    eq_nl_deps[i], dep_Bs[i])

                eq_B = adj_B[0][i].B()
                for j, k in enumerate(self._eq_X_indices[i]):
                    function_assign(eq_B[j], B[k])

            R = X_0
            del X_0
            R_norm_sq = 0.0
            for r, x in zip(R, eq_adj_X[-1]):
                function_axpy(r, -1.0, x)
                R_norm_sq += function_inner(r, r)
            if relative_tolerance == 0.0:
                tolerance_sq = absolute_tolerance ** 2
            else:
                X_norm_sq = 0.0
                for x in eq_adj_X[-1]:
                    X_norm_sq += function_inner(x, x)
                tolerance_sq = max(absolute_tolerance ** 2,
                                   X_norm_sq * (relative_tolerance ** 2))
            if report:
                info(f"Fixed point iteration, adjoint iteration {it:d}, "
                     f"change norm {np.sqrt(R_norm_sq):.16e} "
                     f"(tolerance {np.sqrt(tolerance_sq):.16e})")
            if np.isnan(R_norm_sq):
                raise EquationException(
                    f"Fixed point iteration, adjoint iteration {it:d}, "
                    f"NaN encountered")
            if R_norm_sq < tolerance_sq or R_norm_sq == 0.0:
                break
            if it >= maximum_iterations:
                raise EquationException(
                    f"Fixed point iteration, adjoint iteration {it:d}, "
                    f"failed to converge")

            X_0 = R
            del R
            for x_0, x in zip(X_0, eq_adj_X[-1]):
                function_assign(x_0, x)

        return adj_X

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        if is_function(adj_X):
            adj_X = (adj_X,)

        eq_dep_Bs = tuple({} for eq in self._eqs)
        for dep_index, B in dep_Bs.items():
            dep = self.dependencies()[dep_index]
            dep_id = function_id(dep)
            for i in self._dep_eq_index_map[dep_id]:
                eq_dep_Bs[i][self._eq_dep_index_map[i][dep_id]] = B

        for i, eq in enumerate(self._eqs):
            eq_adj_X = tuple(adj_X[j] for j in self._eq_X_indices[i])
            eq_nl_deps = tuple(nl_deps[j] for j in self._eq_nl_dep_indices[i])
            eq.subtract_adjoint_derivative_actions(
                eq_adj_X[0] if len(eq_adj_X) == 1 else eq_adj_X,
                eq_nl_deps, eq_dep_Bs[i])

    def tangent_linear(self, M, dM, tlm_map):
        tlm_eqs = []
        for eq in self._eqs:
            tlm_eq = eq.tangent_linear(M, dM, tlm_map)
            if tlm_eq is None:
                tlm_eq = NullSolver([tlm_map[x] for x in eq.X()])
            tlm_eqs.append(tlm_eq)

        return FixedPointSolver(tlm_eqs,
                                solver_parameters=self._solver_parameters)


class LinearEquation(Equation):
    def __init__(self, B, X, A=None):
        if isinstance(B, RHS):
            B = (B,)
        if is_function(X):
            X = (X,)

        deps = []
        dep_ids = {}
        nl_deps = []
        nl_dep_ids = {}

        x_ids = set()
        for x in X:
            x_id = function_id(x)
            if x_id in x_ids:
                raise EquationException("Duplicate solve")
            x_ids.add(x_id)
            deps.append(x)
            dep_ids[x_id] = len(deps) - 1

        b_dep_indices = tuple([] for b in B)
        b_nl_dep_indices = tuple([] for b in B)

        for i, b in enumerate(B):
            for dep in b.dependencies():
                dep_id = function_id(dep)
                if dep_id in x_ids:
                    raise EquationException("Invalid dependency in linear Equation")  # noqa: E501
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
            adj_ic=A is not None and A.adjoint_has_initial_condition())
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
            B = tuple(function_new(x) for x in X)

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
            raise EquationException("dep_index out of bounds")
        elif dep_index < len(self.X()):
            if self._A is None:
                return adj_X[dep_index]
            else:
                dep = eq_deps[dep_index]
                F = function_new(dep)
                self._A.adjoint_action([nl_deps[j]
                                        for j in self._A_nl_dep_indices],
                                       adj_X[0] if len(adj_X) == 1 else adj_X,
                                       F, b_index=dep_index, method="assign")
                return F
        else:
            dep = eq_deps[dep_index]
            dep_id = function_id(dep)
            F = function_new(dep)
            for i, (b, b_dep_ids) in enumerate(zip(self._B, self._b_dep_ids)):
                try:
                    b_dep_index = b_dep_ids[dep_id]
                except KeyError:
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
            return NullSolver([tlm_map[x] for x in self.X()])
        else:
            return LinearEquation(tlm_B, [tlm_map[x] for x in self.X()],
                                  A=self._A)


class Matrix(Referrer):
    def __init__(self, nl_deps=[], has_ic_dep=None, ic=None, adj_ic=True):
        if nl_deps is None:
            warnings.warn("'nl_deps=None' is deprecated -- use 'nl_deps=[]' "
                          "instead",
                          DeprecationWarning, stacklevel=2)
            nl_deps = []
        if len({function_id(dep) for dep in nl_deps}) != len(nl_deps):
            raise EquationException("Duplicate non-linear dependency")

        if has_ic_dep is not None:
            warnings.warn("'has_ic_dep' argument is deprecated -- use 'ic' "
                          "instead",
                          DeprecationWarning, stacklevel=2)
            if ic is not None:
                raise EquationException("Cannot pass both 'has_ic_dep' and "
                                        "'ic' arguments")
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

        if getattr(cls.replace, "_replace_compatibility", True):
            warnings.warn("Matrix.replace method is deprecated",
                          DeprecationWarning, stacklevel=2)

            def drop_references(self):
                replace_map = {dep: function_replacement(dep)
                               for dep in self.nonlinear_dependencies()}
                self.replace(replace_map)
            cls.drop_references = drop_references
            cls.replace._replace_compatibility = False

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

    @no_replace_compatibility
    def replace(self, replace_map):
        self._nl_deps = tuple(replace_map.get(dep, dep)
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

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        x/X          The argument of the matrix action.
        b/B          The result of the matrix action.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def adjoint_action(self, nl_deps, adj_X, b, b_index=0, method="assign"):
        """
        Evaluate the adjoint action of the matrix.

        The form:
            adjoint_action(self, nl_deps, adj_x, b, b_index=0, method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        adj_x/adj_X  The argument of the matrix action.
        b            The result of the matrix action.
        b_index      (Optional) The element of the matrix action B to compute.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def forward_solve(self, X, nl_deps, B):
        raise EquationException("Method not overridden")

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

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        nl_dep_index The index of the dependency in
                     self.nonlinear_dependencies() with respect to which a
                     derivative should be taken.
        x/X          The argument of the forward matrix action.
        adj_x/adj_X  The direction of the adjoint derivative action.
        b            The result.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def adjoint_solve(self, adj_X, nl_deps, B):
        raise EquationException("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map, X):
        raise EquationException("Method not overridden")


class RHS(Referrer):
    def __init__(self, deps, nl_deps=None):
        dep_ids = {function_id(dep) for dep in deps}
        if len(dep_ids) != len(deps):
            raise EquationException("Duplicate dependency")

        if nl_deps is None:
            nl_deps = tuple(deps)
        nl_dep_ids = {function_id(dep) for dep in nl_deps}
        if len(nl_dep_ids) != len(nl_deps):
            raise EquationException("Duplicate non-linear dependency")
        if len(dep_ids.intersection(nl_dep_ids)) != len(nl_deps):
            raise EquationException("Non-linear dependency is not a "
                                    "dependency")

        super().__init__()
        self._deps = tuple(deps)
        self._nl_deps = tuple(nl_deps)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if getattr(cls.replace, "_replace_compatibility", True):
            warnings.warn("RHS.replace method is deprecated",
                          DeprecationWarning, stacklevel=2)

            def drop_references(self):
                replace_map = {dep: function_replacement(dep)
                               for dep in self.dependencies()}
                self.replace(replace_map)
            cls.drop_references = drop_references
            cls.replace._replace_compatibility = False

    def drop_references(self):
        self._deps = tuple(function_replacement(dep) for dep in self._deps)
        self._nl_deps = tuple(function_replacement(dep)
                              for dep in self._nl_deps)

    @no_replace_compatibility
    def replace(self, replace_map):
        self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
        self._nl_deps = tuple(replace_map.get(dep, dep)
                              for dep in self._nl_deps)

    def dependencies(self):
        return self._deps

    def nonlinear_dependencies(self):
        return self._nl_deps

    def add_forward(self, B, deps):
        raise EquationException("Method not overridden")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        raise EquationException("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        raise EquationException("Method not overridden")


class MatrixActionSolver(LinearEquation):
    def __init__(self, Y, A, X):
        super().__init__(MatrixActionRHS(A, Y), X)


class InnerProductSolver(LinearEquation):
    def __init__(self, y, z, x, alpha=1.0, M=None):
        super().__init__(InnerProductRHS(y, z, alpha=alpha, M=M), x)


class NormSqSolver(InnerProductSolver):
    def __init__(self, y, x, alpha=1.0, M=None):
        super().__init__(y, y, x, alpha=alpha, M=M)


class SumSolver(LinearEquation):
    def __init__(self, y, x):
        super().__init__(SumRHS(y), x)


class MatrixActionRHS(RHS):
    def __init__(self, A, X):
        if is_function(X):
            X = (X,)
        if len({function_id(x) for x in X}) != len(X):
            raise EquationException("Invalid dependency")

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
            raise EquationException("dep_index out of bounds")
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


class InnerProductRHS(RHS):
    def __init__(self, x, y, alpha=1.0, M=None):
        """
        An equation representing an inner product.

        Arguments:

        x, y   Inner product arguments. May be the same function.
        alpha  (Optional) Scale the result of the inner product by alpha.
        M      (Optional) Matrix defining the inner product. Assumed symmetric,
               and must have no non-linear dependencies. Defaults to an
               identity matrix.
        """

        if M is not None and len(M.nonlinear_dependencies()) > 0:
            raise EquationException("Non-linear matrix dependencies not supported")  # noqa: E501

        norm_sq = x == y
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
            Y = function_new(x)
            self._M.forward_action(M_deps, y, Y, method="assign")

        function_set_values(b,
                            function_get_values(b) + self._alpha
                            * function_inner(x, Y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._norm_sq:
            if dep_index == 0:
                x = nl_deps[0]
                M_deps = nl_deps[1:]

                if self._M is None:
                    X = x
                else:
                    X = function_new(x)
                    self._M.forward_action(M_deps, x, X, method="assign")

                function_axpy(b, -2.0 * self._alpha * function_sum(adj_x), X)
            else:
                raise EquationException("dep_index out of bounds")
        elif dep_index == 0:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                Y = y
            else:
                Y = function_new(x)
                self._M.forward_action(M_deps, y, Y, method="assign")

            function_axpy(b, -self._alpha * function_sum(adj_x), Y)
        elif dep_index == 1:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                X = x
            else:
                X = function_new(y)
                self._M.forward_action(M_deps, x, X, method="assign")

            function_axpy(b, -self._alpha * function_sum(adj_x), X)
        else:
            raise EquationException("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        tlm_B = []

        if self._norm_sq:
            x = self.dependencies()[0]
            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(x, tlm_x, alpha=2.0 * self._alpha,
                                             M=self._M))
        else:
            x, y = self.dependencies()[:2]

            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(tlm_x, y, alpha=self._alpha,
                                             M=self._M))

            tlm_y = get_tangent_linear(y, M, dM, tlm_map)
            if tlm_y is not None:
                tlm_B.append(InnerProductRHS(x, tlm_y, alpha=self._alpha,
                                             M=self._M))

        return tlm_B


class NormSqRHS(InnerProductRHS):
    def __init__(self, x, alpha=1.0, M=None):
        super().__init__(x, x, alpha=alpha, M=M)


class SumRHS(RHS):
    def __init__(self, x):
        super().__init__([x], nl_deps=[])

    def add_forward(self, b, deps):
        y, = deps
        function_set_values(b, function_get_values(b) + function_sum(y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if dep_index == 0:
            function_set_values(b,
                                function_get_values(b) - function_sum(adj_x))
        else:
            raise EquationException("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        y, = self.dependencies()
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return None
        else:
            return SumRHS(tau_y)


class Storage(Equation):
    def __init__(self, x, key, save=False):
        super().__init__(x, [x], nl_deps=[], ic=False, adj_ic=False)
        self._key = key
        self._save = save

    def key(self):
        return self._key

    def is_saved(self):
        raise EquationException("Method not overridden")

    def load(self, x):
        raise EquationException("Method not overridden")

    def save(self, x):
        raise EquationException("Method not overridden")

    def forward_solve(self, x, deps=None):
        if not self._save or self.is_saved():
            self.load(x)
        else:
            self.save(x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        else:
            raise EquationException("dep_index out of bounds")

    def tangent_linear(self, M, dM, tlm_map):
        return NullSolver(tlm_map[self.x()])


class MemoryStorage(Storage):
    def __init__(self, x, d, key, save=False):
        super().__init__(x, key, save=save)
        self._d = d

    def is_saved(self):
        return self.key() in self._d

    def load(self, x):
        function_set_values(x, self._d[self.key()])

    def save(self, x):
        self._d[self.key()] = function_get_values(x)


class HDF5Storage(Storage):
    def __init__(self, x, h, key, save=False):
        super().__init__(x, key, save=save)
        self._h = h

    def is_saved(self):
        return self.key() in self._h

    def load(self, x):
        d = self._h[self.key()]["value"]
        function_set_values(x, d[function_local_indices(x)])

    def save(self, x):
        key = self.key()
        self._h.create_group(key)
        values = function_get_values(x)
        d = self._h[key].create_dataset("value",
                                        shape=(function_global_size(x),),
                                        dtype=values.dtype)
        d[function_local_indices(x)] = values
