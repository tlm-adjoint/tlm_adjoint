#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    check_space_types, is_var, var_id, var_is_alias, var_is_static, var_new,
    var_replacement, var_update_caches, var_update_state, var_zero)

from .alias import gc_disabled
from .manager import manager as _manager
from .manager import paused_manager, restore_manager, set_manager

from collections.abc import Sequence
import inspect
from operator import itemgetter
import warnings
import weakref

__all__ = \
    [
        "Equation",
        "ZeroAssignment",

        "NullSolver"
    ]


class Referrer:
    _id_counter = [0]

    def __init__(self, referrers=None):
        if referrers is None:
            referrers = ()

        self._id, = self._id_counter
        self._id_counter[0] += 1
        self._referrers = weakref.WeakValueDictionary()
        self._references_dropped = False

        self.add_referrer(*referrers)

    @property
    def id(self):
        return self._id

    @gc_disabled
    def add_referrer(self, *referrers):
        if self._references_dropped:
            raise RuntimeError("Cannot call add_referrer method after "
                               "_drop_references method has been called")
        for referrer in referrers:
            referrer_id = referrer.id
            assert self._referrers.get(referrer_id, referrer) is referrer
            self._referrers[referrer_id] = referrer

    @gc_disabled
    def referrers(self):
        referrers = {}
        remaining_referrers = {self.id: self}
        while len(remaining_referrers) > 0:
            referrer_id, referrer = remaining_referrers.popitem()
            if referrer_id not in referrers:
                referrers[referrer_id] = referrer
                for child in tuple(referrer._referrers.valuerefs()):
                    child = child()
                    if child is not None:
                        child_id = child.id
                        if child_id not in referrers and child_id not in remaining_referrers:  # noqa: E501
                            remaining_referrers[child_id] = child
        return tuple(e for _, e in sorted(tuple(referrers.items()),
                                          key=itemgetter(0)))

    def _drop_references(self):
        if not self._references_dropped:
            self.drop_references()
            self._references_dropped = True

    def drop_references(self):
        """Drop references to variables which store values.
        """

        raise NotImplementedError("Method not overridden")


class Equation(Referrer):
    r"""Core equation class. Defines a differentiable operation for use as an
    adjoint tape record.

    The equation is defined via a residual function :math:`\mathcal{F}`. The
    forward solution is defined implicitly as the value :math:`x` for which

    .. math::

        \mathcal{F} \left( x, y_0, y_1, \ldots \right) = 0,

    where :math:`y_i` are dependencies.

    This is an abstract base class. Information required to solve forward
    equations, perform adjoint calculations, and define tangent-linear
    equations, is provided by overloading abstract methods. This class does
    *not* inherit from :class:`abc.ABC`, so that methods may be implemented as
    needed.

    :arg X: A variable, or a :class:`Sequence` of variables, defining the
        forward solution.
    :arg deps: A :class:`Sequence` of variables defining dependencies. Must
        define a superset of `X`.
    :arg nl_deps: A :class:`Sequence` of variables defining non-linear
        dependencies. Must define a subset of `deps`. Defaults to `deps`.
    :arg ic_deps: A :class:`Sequence` of variables whose value must be
        available prior to computing the forward solution. Intended for
        iterative methods with non-zero initial guesses. Must define a subset
        of `X`. Can be overridden by `ic`.
    :arg ic: Whether `ic_deps` should be set equal to `X`. Defaults to `True`
        if `ic_deps` is not supplied, and to `False` otherwise.
    :arg adj_ic_deps: A :class:`Sequence` of variables whose value must be
        available prior to computing the adjoint solution. Intended for
        iterative methods with non-zero initial guesses. Must define a subset
        of `X`. Can be overridden by `adj_ic`.
    :arg adj_ic: Whether `adj_ic_deps` should be set equal to `X`. Defaults to
        `True` if `adj_ic_deps` is not supplied, and to `False` otherwise.
    :arg adj_type: The space type relative to `X` of adjoint variables.
        `'primal'` or `'conjugate_dual'`, or a :class:`Sequence` of these.
    """

    def __init__(self, X, deps, nl_deps=None, *,
                 ic_deps=None, ic=None,
                 adj_ic_deps=None, adj_ic=None,
                 adj_type="conjugate_dual"):
        if is_var(X):
            X = (X,)
        X_ids = set(map(var_id, X))
        dep_ids = {var_id(dep): i for i, dep in enumerate(deps)}
        for x in X:
            if not is_var(x):
                raise ValueError("Solution must be a variable")
            if var_is_static(x):
                raise ValueError("Solution cannot be static")
            if var_is_alias(x):
                raise ValueError("Solution cannot be an alias")
            if var_id(x) not in dep_ids:
                raise ValueError("Solution must be a dependency")

        if len(dep_ids) != len(deps):
            raise ValueError("Duplicate dependency")
        for dep in deps:
            if var_is_alias(dep):
                raise ValueError("Dependency cannot be an alias")

        if nl_deps is None:
            nl_deps = tuple(deps)
        nl_dep_ids = set(map(var_id, nl_deps))
        if len(nl_dep_ids) != len(nl_deps):
            raise ValueError("Duplicate non-linear dependency")
        for dep in nl_deps:
            if var_id(dep) not in dep_ids:
                raise ValueError("Non-linear dependency is not a dependency")

        if ic_deps is None:
            ic_deps = []
            if ic is None:
                ic = True
        else:
            if ic is None:
                ic = False
        ic_dep_ids = set(map(var_id, ic_deps))
        if len(ic_dep_ids) != len(ic_deps):
            raise ValueError("Duplicate initial condition dependency")
        for dep in ic_deps:
            if var_id(dep) not in X_ids:
                raise ValueError("Initial condition dependency is not a "
                                 "solution")
        if ic:
            ic_deps = list(X)

        if adj_ic_deps is None:
            adj_ic_deps = []
            if adj_ic is None:
                adj_ic = True
        else:
            if adj_ic is None:
                adj_ic = False
        adj_ic_dep_ids = set(map(var_id, adj_ic_deps))
        if len(adj_ic_dep_ids) != len(adj_ic_deps):
            raise ValueError("Duplicate adjoint initial condition dependency")
        for dep in adj_ic_deps:
            if var_id(dep) not in X_ids:
                raise ValueError("Adjoint initial condition dependency is not "
                                 "a solution")
        if adj_ic:
            adj_ic_deps = list(X)

        if adj_type in ["primal", "conjugate_dual"]:
            adj_type = tuple(adj_type for _ in X)
        elif isinstance(adj_type, Sequence):
            if len(adj_type) != len(X):
                raise ValueError("Invalid adjoint type")
        else:
            raise ValueError("Invalid adjoint type")
        for adj_x_type in adj_type:
            if adj_x_type not in {"primal", "conjugate_dual"}:
                raise ValueError("Invalid adjoint type")

        super().__init__()
        self._X = tuple(X)
        self._deps = tuple(deps)
        self._nl_deps = tuple(nl_deps)
        self._ic_deps = tuple(ic_deps)
        self._adj_ic_deps = tuple(adj_ic_deps)
        self._adj_X_type = tuple(adj_type)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        adj_solve_sig = inspect.signature(cls.adjoint_jacobian_solve)
        if tuple(adj_solve_sig.parameters.keys()) in {("self", "nl_deps", "b"),
                                                      ("self", "nl_deps", "B")}:  # noqa: E501
            warnings.warn("Equation.adjoint_jacobian_solve(self, nl_deps, b/B) "  # noqa: E501
                          "method signature is deprecated",
                          DeprecationWarning, stacklevel=2)

            def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
                return adjoint_jacobian_solve_orig(self, nl_deps, B)
            adjoint_jacobian_solve_orig = cls.adjoint_jacobian_solve
            cls.adjoint_jacobian_solve = adjoint_jacobian_solve

    def drop_references(self):
        self._X = tuple(map(var_replacement, self._X))
        self._deps = tuple(map(var_replacement, self._deps))
        self._nl_deps = tuple(map(var_replacement, self._nl_deps))
        self._ic_deps = tuple(map(var_replacement, self._ic_deps))
        self._adj_ic_deps = tuple(map(var_replacement, self._adj_ic_deps))

    def x(self):
        """Return the forward solution variable, assuming the forward solution
        has one component.

        :returns: A variable defining the forward solution.
        """

        x, = self._X
        return x

    def X(self, m=None):
        """Return forward solution variables.

        :returns: If `m` is supplied, a variable defining the `m` th component
            of the forward solution. If `m` is not supplied, a :class:`tuple`
            of variables defining the forward solution.
        """

        if m is None:
            return self._X
        else:
            return self._X[m]

    def dependencies(self):
        """Return dependencies.

        :returns: A :class:`tuple` of variables defining dependencies.
        """

        return self._deps

    def nonlinear_dependencies(self):
        """Return non-linear dependencies.

        :returns: A :class:`tuple` of variables defining non-linear
            dependencies.
        """

        return self._nl_deps

    def initial_condition_dependencies(self):
        """Return 'initial condition' dependencies -- dependencies whose value
        is needed prior to computing the forward solution.

        :returns: A :class:`tuple` of variables defining initial condition
            dependencies.
        """

        return self._ic_deps

    def adjoint_initial_condition_dependencies(self):
        """Return adjoint 'initial condition' dependencies -- dependencies
        whose value is needed prior to computing the adjoint solution.

        :returns: A :class:`tuple` of variables defining adjoint initial
            condition dependencies.
        """

        return self._adj_ic_deps

    def adj_x_type(self):
        """Return the space type for the adjoint solution, relative to the
        forward solution, assuming the forward solution has exactly one
        component.

        :returns: One of `'primal'` or `'conjugate_dual'`.
        """

        adj_x_type, = self.adj_X_type()
        return adj_x_type

    def adj_X_type(self, m=None):
        """Return the space type for the adjoint solution, relative to the
        forward solution.

        :returns: If `m` is supplied, one of `'primal'` or `'conjugate_dual'`
            defining the relative space type for the `m` th component of the
            adjoint solution. If `m` is not supplied, a :class:`tuple` whose
            elements are `'primal'` or `'conjugate_dual'`, defining the
            relative space type of the adjoint solution.
        """

        if m is None:
            return self._adj_X_type
        else:
            return self._adj_X_type[m]

    def new_adj_x(self):
        """Return a new variable suitable for storing the adjoint solution,
        assuming the forward solution has exactly one component.

        :returns: A variable suitable for storing the adjoint solution.
        """

        adj_x, = self.new_adj_X()
        return adj_x

    def new_adj_X(self, m=None):
        """Return new variables suitable for storing the adjoint solution.

        :returns: If `m` is supplied, a variable suitable for storing the `m`
            th component of the adjoint solution. If `m` is not supplied, a
            :class:`tuple` of variables suitable for storing the adjoint
            solution.
        """

        if m is None:
            return tuple(self.new_adj_X(m) for m in range(len(self.X())))
        else:
            return var_new(self.X(m), rel_space_type=self.adj_X_type(m))

    def _pre_process(self, manager=None, annotate=None):
        if manager is None:
            manager = _manager()
        for dep in self.initial_condition_dependencies():
            manager.add_initial_condition(dep, annotate=annotate)

    def _post_process(self, manager=None, annotate=None, tlm=None):
        if manager is None:
            manager = _manager()
        manager.add_equation(self, annotate=annotate, tlm=tlm)

    @restore_manager
    def solve(self, *, manager=None, annotate=None, tlm=None):
        """Compute the forward solution.

        :arg manager: The :class:`.EquationManager`. Defaults to `manager()`.
        :arg annotate: Whether the :class:`.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        if manager is not None:
            set_manager(manager)

        self._pre_process(annotate=annotate)

        with paused_manager():
            self.forward(self.X())

        self._post_process(annotate=annotate, tlm=tlm)

    def forward(self, X, deps=None):
        """Wraps :meth:`.Equation.forward_solve` to handle cache invalidation.
        """

        var_update_caches(*self.dependencies(), value=deps)
        self.forward_solve(X[0] if len(X) == 1 else X, deps=deps)
        var_update_state(*X)
        var_update_caches(*self.X(), value=X)

    def forward_solve(self, X, deps=None):
        """Compute the forward solution.

        Can assume that the currently active :class:`.EquationManager` is
        paused.

        :arg X: A variable if the forward solution has a single component,
            otherwise a :class:`Sequence` of variables. May define an initial
            guess, and should be set by this method. Subclasses may replace
            this argument with `x` if the forward solution has a single
            component.
        :arg deps: A :class:`tuple` of variables, defining values for
            dependencies. Only the elements corresponding to `X` may be
            modified. `self.dependencies()` should be used if not supplied.
        """

        raise NotImplementedError("Method not overridden")

    def adjoint(self, adj_X, nl_deps, B, dep_Bs):
        """Compute the adjoint solution, and subtract terms from other adjoint
        right-hand-sides.

        :arg adj_X: Either `None`, or a :class:`Sequence` of variables defining
            the initial guess for an iterative solve. May be modified or
            returned.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg B: A sequence of variables defining the right-hand-side of the
            adjoint equation. May be modified or returned.
        :arg dep_Bs: A :class:`Mapping` whose items are `(dep_index, dep_B)`.
            Each `dep_B` is an :class:`.AdjointRHS` which should be updated by
            subtracting adjoint derivative information computed by
            differentiating with respect to `self.dependencies()[dep_index]`.

        :returns: A :class:`tuple` of variables defining the adjoint solution,
            or `None` to indicate that the solution is zero.
        """

        var_update_caches(*self.nonlinear_dependencies(), value=nl_deps)

        if adj_X is not None and len(adj_X) == 1:
            adj_X = adj_X[0]
        adj_X = self.adjoint_jacobian_solve(
            adj_X, nl_deps, B[0] if len(B) == 1 else B)
        if adj_X is not None:
            if is_var(adj_X):
                adj_X = (adj_X,)
            var_update_state(*adj_X)
            for m, adj_x in enumerate(adj_X):
                check_space_types(adj_x, self.X(m),
                                  rel_space_type=self.adj_X_type(m))

            self.subtract_adjoint_derivative_actions(
                adj_X[0] if len(adj_X) == 1 else adj_X, nl_deps, dep_Bs)

        if adj_X is None:
            return None
        else:
            return tuple(adj_X)

    def adjoint_cached(self, adj_X, nl_deps, dep_Bs):
        """Subtract terms from other adjoint right-hand-sides.

        :arg adj_X: A :class:`Sequence` of variables defining the adjoint
            solution. Should not be modified.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg dep_Bs: A :class:`Mapping` whose items are `(dep_index, dep_B)`.
            Each `dep_B` is an :class:`.AdjointRHS` which should be updated by
            subtracting adjoint derivative information computed by
            differentiating with respect to `self.dependencies()[dep_index]`.
        """

        var_update_caches(*self.nonlinear_dependencies(), value=nl_deps)

        if len(adj_X) == 1:
            adj_X = adj_X[0]
        self.subtract_adjoint_derivative_actions(adj_X, nl_deps, dep_Bs)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        """Return the action of the adjoint of a derivative of the forward
        residual on the adjoint solution. This is the *negative* of an adjoint
        right-hand-side term.

        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg dep_index: An :class:`int`. The derivative is defined by
            differentiation of the forward residual with respect to
            `self.dependencies()[dep_index]`.
        :arg adj_X: The adjoint solution. A variable if the adjoint solution
            has a single component, otherwise a :class:`Sequence` of variables.
            Should not be modified. Subclasses may replace this argument with
            `adj_x` if the adjoint solution has a single component.
        :returns: The action of the adjoint of a derivative on the adjoint
            solution. Will be passed to
            :func:`.subtract_adjoint_derivative_action`, and valid types depend
            upon the adjoint variable type. Typically this will be a variable,
            or a two element :class:`tuple` `(alpha, F)`, where `alpha` is a
            scalar and `F` a variable, with the value defined by the product of
            `alpha` and `F`.
        """

        raise NotImplementedError("Method not overridden")

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        """Subtract terms from other adjoint right-hand-sides.

        Can be overridden for an optimized implementation, but otherwise uses
        :meth:`.Equation.adjoint_derivative_action`.

        :arg adj_X: The adjoint solution. A variable if the adjoint solution
            has a single component, otherwise a :class:`Sequence` of variables.
            Should not be modified. Subclasses may replace this argument with
            `adj_x` if the adjoint solution has a single component.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg dep_Bs: A :class:`Mapping` whose items are `(dep_index, dep_B)`.
            Each `dep_B` is an :class:`.AdjointRHS` which should be updated by
            subtracting adjoint derivative information computed by
            differentiating with respect to `self.dependencies()[dep_index]`.
        """

        for dep_index, dep_B in dep_Bs.items():
            dep_B.sub(self.adjoint_derivative_action(nl_deps, dep_index,
                                                     adj_X))

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        """Compute an adjoint solution.

        :arg adj_X: Either `None`, or a variable (if the adjoint solution has a
            single component) or :class:`Sequence` of variables (otherwise)
            defining the initial guess for an iterative solve. May be modified
            or returned. Subclasses may replace this argument with `adj_x` if
            the adjoint solution has a single component.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg B: The right-hand-side. A variable (if the adjoint solution has a
            single component) or :class:`Sequence` of variables (otherwise)
            storing the value of the right-hand-side. May be modified or
            returned. Subclasses may replace this argument with `b` if the
            adjoint solution has a single component.
        :returns: A variable or :class:`Sequence` of variables storing the
            value of the adjoint solution. May return `None` to indicate a
            value of zero.
        """

        raise NotImplementedError("Method not overridden")

    def tangent_linear(self, M, dM, tlm_map):
        """Derive an :class:`.Equation` corresponding to a tangent-linear
        operation.

        :arg M: A :class:`Sequence` of variables defining the control.
        :arg dM: A :class:`Sequence` of variables defining the derivative
            direction. The tangent-linear computes directional derivatives with
            respect to the control defined by `M` and with direction defined by
            `dM`.
        :arg tlm_map: A :class:`.TangentLinearMap` storing values for
            tangent-linear variables.
        :returns: An :class:`.Equation`, corresponding to the tangent-linear
            operation.
        """

        raise NotImplementedError("Method not overridden")


class ZeroAssignment(Equation):
    r"""Represents an assignment

    .. math::

        x = 0.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x \right) = x.

    :arg X: A variable or a :class:`Sequence` of variables defining the forward
        solution :math:`x`.
    """

    def __init__(self, X):
        if is_var(X):
            X = (X,)
        super().__init__(X, X, nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, X, deps=None):
        if is_var(X):
            X = (X,)
        for x in X:
            var_zero(x)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_var(adj_X):
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
