from .interface import (
    Packed, check_space_types, is_var, packed, var_id, var_is_alias,
    var_is_static, var_locked, var_new, var_replacement, var_update_caches,
    var_update_state, var_zero)

from .alias import WeakAlias, gc_disabled
from .manager import manager as _manager
from .manager import annotation_enabled, paused_manager, tlm_enabled

import functools
import inspect
import itertools
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
    _id_counter = itertools.count()

    def __init__(self, referrers=None):
        if referrers is None:
            referrers = ()

        self._id = next(self._id_counter)
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

    @functools.cached_property
    def _weak_alias(self):
        return WeakAlias(self)


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
        X_packed = Packed(X)
        X = tuple(X_packed)
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
        if len(adj_type) != len(X):
            raise ValueError("Invalid adjoint type")
        for adj_x_type in adj_type:
            if adj_x_type not in {"primal", "conjugate_dual"}:
                raise ValueError("Invalid adjoint type")

        super().__init__()
        self._packed = X_packed.mapped(lambda x: None)
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
                          FutureWarning, stacklevel=2)

            def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
                return adjoint_jacobian_solve_orig(self, nl_deps, B)
            adjoint_jacobian_solve_orig = cls.adjoint_jacobian_solve
            cls.adjoint_jacobian_solve = adjoint_jacobian_solve

        tangent_linear_sig = inspect.signature(cls.tangent_linear)
        if tuple(tangent_linear_sig.parameters.keys()) == ("self", "M", "dM", "tlm_map"):  # noqa: E501
            warnings.warn("Equation.tangent_linear(self, M, dM, tlm_map) "
                          "method signature is deprecated",
                          FutureWarning, stacklevel=2)

            def tangent_linear(self, tlm_map):
                return tangent_linear_orig(self, tlm_map.M, tlm_map.dM,
                                           tlm_map)
            tangent_linear_orig = cls.tangent_linear
            cls.tangent_linear = tangent_linear

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

    def _unpack(self, obj):
        return self._packed.unpack(obj)

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

    @property
    def _pre_process_required(self):
        return len(self.initial_condition_dependencies()) > 0

    def _pre_process(self):
        manager = _manager()
        for dep in self.initial_condition_dependencies():
            manager.add_initial_condition(dep)

    def _post_process(self):
        _manager().add_equation(self)

    def solve(self, *, annotate=None, tlm=None):
        """Compute the forward solution.

        :arg annotate: Whether the :class:`.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        if annotate is None or annotate:
            annotate = annotation_enabled()
        if tlm is None or tlm:
            tlm = tlm_enabled()

        with paused_manager(annotate=not annotate, tlm=not tlm):
            self._pre_process()
        with paused_manager():
            self.forward(self.X())
        with paused_manager(annotate=not annotate, tlm=not tlm):
            self._post_process()

    def forward(self, X, deps=None):
        """Wraps :meth:`.Equation.forward_solve` to handle cache invalidation.
        """

        X_ids = set(map(var_id, X))
        eq_deps = self.dependencies()

        with var_locked(*(dep for dep in (eq_deps if deps is None else deps)
                          if var_id(dep) not in X_ids)):
            var_update_caches(*eq_deps, value=deps)
            self.forward_solve(self._unpack(X), deps=deps)
            var_update_state(*X)
            var_update_caches(*self.X(), value=X)

    def forward_solve(self, X, deps=None):
        """Compute the forward solution.

        Can assume that the currently active :class:`.EquationManager` is
        paused.

        :arg X: A variable or a :class:`Sequence` of variables storing the
            solution. May define an initial guess, and should be set by this
            method.
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
        :arg B: A :class:`Sequence` of variables defining the right-hand-side
            of the adjoint equation. May be modified or returned.
        :arg dep_Bs: A :class:`Mapping` whose items are `(dep_index, dep_B)`.
            Each `dep_B` is an :class:`.AdjointRHS` which should be updated by
            subtracting adjoint derivative information computed by
            differentiating with respect to `self.dependencies()[dep_index]`.

        :returns: A :class:`tuple` of variables defining the adjoint solution,
            or `None` to indicate that the solution is zero.
        """

        with var_locked(*nl_deps):
            var_update_caches(*self.nonlinear_dependencies(), value=nl_deps)

            adj_X = self.adjoint_jacobian_solve(
                None if adj_X is None else self._unpack(adj_X),
                nl_deps, self._unpack(B))
            if adj_X is None:
                return None
            adj_X = packed(adj_X)
            var_update_state(*adj_X)

            for m, adj_x in enumerate(adj_X):
                check_space_types(adj_x, self.X(m),
                                  rel_space_type=self.adj_X_type(m))

            self.subtract_adjoint_derivative_actions(
                self._unpack(adj_X), nl_deps, dep_Bs)

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

        with var_locked(*itertools.chain(adj_X, nl_deps)):
            var_update_caches(*self.nonlinear_dependencies(), value=nl_deps)

            self.subtract_adjoint_derivative_actions(
                self._unpack(adj_X), nl_deps, dep_Bs)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        """Return the action of the adjoint of a derivative of the forward
        residual on the adjoint solution. This is the *negative* of an adjoint
        right-hand-side term.

        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg dep_index: An :class:`int`. The derivative is defined by
            differentiation of the forward residual with respect to
            `self.dependencies()[dep_index]`.
        :arg adj_X: The adjoint solution. A variable or a :class:`Sequence` of
            variables. Should not be modified.
        :returns: The action of the adjoint of a derivative on the adjoint
            solution. Will be passed to
            :func:`.subtract_adjoint_derivative_action`, and valid types depend
            upon the adjoint variable type. Typically this will be a variable,
            or a two element :class:`tuple` `(alpha, F)`, where `alpha` is a
            :class:`numbers.Complex` and `F` a variable, with the value defined
            by the product of `alpha` and `F`.
        """

        raise NotImplementedError("Method not overridden")

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        """Subtract terms from other adjoint right-hand-sides.

        Can be overridden for an optimized implementation, but otherwise uses
        :meth:`.Equation.adjoint_derivative_action`.

        :arg adj_X: The adjoint solution. A variable or a :class:`Sequence` of
            variables. Should not be modified.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg dep_Bs: A :class:`Mapping` whose items are `(dep_index, dep_B)`.
            Each `dep_B` is an :class:`.AdjointRHS` which should be updated by
            subtracting adjoint derivative information computed by
            differentiating with respect to `self.dependencies()[dep_index]`.
        """

        for dep_index, dep_B in dep_Bs.items():
            dep_B.sub(
                self.adjoint_derivative_action(nl_deps, dep_index, adj_X))

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        """Compute an adjoint solution.

        :arg adj_X: Either `None`, or a variable or :class:`Sequence` of
            variables defining the initial guess for an iterative solve. May be
            modified or returned.
        :arg nl_deps: A :class:`Sequence` of variables defining values for
            non-linear dependencies. Should not be modified.
        :arg B: The right-hand-side. A variable or :class:`Sequence` of
            variables storing the value of the right-hand-side. May be modified
            or returned.
        :returns: A variable or :class:`Sequence` of variables storing the
            value of the adjoint solution. May return `None` to indicate a
            value of zero.
        """

        raise NotImplementedError("Method not overridden")

    def tangent_linear(self, tlm_map):
        """Derive an :class:`.Equation` corresponding to a tangent-linear
        operation.

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
        X = packed(X)
        super().__init__(X, X, nl_deps=[], ic=False, adj_ic=False)

    def forward_solve(self, X, deps=None):
        for x in X:
            var_zero(x)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, tlm_map):
        return ZeroAssignment([tlm_map[x] for x in self.X()])


class NullSolver(ZeroAssignment):
    ""

    def __init__(self, X):
        warnings.warn("NullSolver is deprecated -- "
                      "use ZeroAssignment instead",
                      FutureWarning, stacklevel=2)
        super().__init__(X)
