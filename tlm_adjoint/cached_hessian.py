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

from .interface import comm_dup, function_id, function_new, function_state

from .caches import clear_caches
from .functional import Functional
from .hessian import GaussNewton, Hessian
from .manager import manager as _manager
from .tlm_adjoint import AdjointCache, EquationManager

from collections.abc import Sequence
import warnings

__all__ = \
    [
        "CachedGaussNewton",
        "CachedHessian",

        "SingleBlockHessian"
    ]


class HessianOptimization:
    def __init__(self, *, manager=None, cache_adjoint=True):
        if manager is None:
            manager = _manager()
        if manager._alias_eqs:
            raise RuntimeError("Invalid equation manager state")

        comm = comm_dup(manager.comm())

        blocks = list(manager._blocks) + [list(manager._block)]

        ics = dict(manager._cp.initial_conditions(cp=True, refs=True,
                                                  copy=False))

        nl_deps = {}
        for n, block in enumerate(blocks):
            for i, eq in enumerate(block):
                nl_deps[(n, i)] = manager._cp[(n, i)]

        self._comm = comm
        self._blocks = blocks
        self._ics = ics
        self._nl_deps = nl_deps
        self._cache_adjoint = cache_adjoint
        self._adj_cache = AdjointCache()
        if cache_adjoint:
            self._cache_key = None

    def _new_manager(self):
        manager = EquationManager(comm=self._comm,
                                  cp_method="memory",
                                  cp_parameters={"drop_references": False})

        for x_id, value in self._ics.items():
            manager._cp._add_initial_condition(
                x_id=x_id, value=value, refs=False, copy=False)

        return manager

    def _add_forward_equations(self, manager):
        for n, block in enumerate(self._blocks):
            for i, eq in enumerate(block):
                eq_id = eq.id()
                if eq_id not in manager._eqs:
                    manager._eqs[eq_id] = eq
                manager._block.append(eq)
                eq_nl_deps = eq.nonlinear_dependencies()
                nl_deps = self._nl_deps[(n, i)]
                manager._cp.update_keys(
                    len(manager._blocks), len(manager._block) - 1,
                    eq)
                manager._cp._add_equation_data(
                    len(manager._blocks), len(manager._block) - 1,
                    eq_nl_deps, nl_deps, eq_nl_deps, nl_deps,
                    refs=False, copy=False)
                yield n, i, eq

    def _tangent_linear(self, manager, eq, M, dM):
        return manager._tangent_linear(eq, M, dM)

    def _add_tangent_linear_equation(self, manager, n, i, eq, M, dM, tlm_eq,
                                     *, annotate=True, solve=True):
        for tlm_dep in tlm_eq.initial_condition_dependencies():
            manager._cp.add_initial_condition(tlm_dep)

        eq_nl_deps = eq.nonlinear_dependencies()
        cp_deps = self._nl_deps[(n, i)]
        assert len(eq_nl_deps) == len(cp_deps)
        eq_deps = {function_id(eq_dep): cp_dep
                   for eq_dep, cp_dep in zip(eq_nl_deps, cp_deps)}
        del eq_nl_deps, cp_deps

        tlm_deps = list(tlm_eq.dependencies())
        for j, tlm_dep in enumerate(tlm_deps):
            tlm_dep_id = function_id(tlm_dep)
            if tlm_dep_id in eq_deps:
                tlm_deps[j] = eq_deps[tlm_dep_id]
        del eq_deps

        if solve:
            tlm_eq.forward(tlm_eq.X(), deps=tlm_deps)

        if annotate:
            tlm_eq_id = tlm_eq.id()
            if tlm_eq_id not in manager._eqs:
                manager._eqs[tlm_eq_id] = tlm_eq
            manager._block.append(tlm_eq)
            manager._cp.add_equation(
                len(manager._blocks), len(manager._block) - 1, tlm_eq,
                deps=tlm_deps)

    def _setup_manager(self, M, dM, M0=None, *,
                       annotate_tlm=True, solve_tlm=True):
        M = tuple(M)
        dM = tuple(dM)
        # M0 ignored

        clear_caches(*dM)

        manager = self._new_manager()
        manager.configure_tlm((M, dM), annotate=annotate_tlm)

        if self._cache_adjoint:
            cache_key = (set(map(function_id, M)), annotate_tlm)
            if self._cache_key is None or self._cache_key != cache_key:
                self._adj_cache.clear()
                self._cache_key = cache_key
        manager._adj_cache = self._adj_cache

        for n, i, eq in self._add_forward_equations(manager):
            tlm_eq = self._tangent_linear(manager, eq, M, dM)
            if tlm_eq is not None:
                self._add_tangent_linear_equation(
                    manager, n, i, eq, M, dM, tlm_eq,
                    solve=solve_tlm)

        return manager, M, dM


class CachedHessian(Hessian, HessianOptimization):
    """Represents a Hessian associated with a given forward model. Uses a
    cached forward calculation.

    :arg J: The :class:`tlm_adjoint.functional.Functional` defining the
        Hessian.
    :arg manager: The :class:`tlm_adjoint.tlm_adjoint.EquationManager` used to
        record the forward. This must have used `'memory'` checkpointing with
        automatic dropping of function references disabled. `manager()` is used
        if not supplied.
    :arg cache_adjoint: Whether to cache the first order adjoint calculation.
    """

    def __init__(self, J, *, manager=None, cache_adjoint=True):
        HessianOptimization.__init__(self, manager=manager,
                                     cache_adjoint=cache_adjoint)
        Hessian.__init__(self)
        self._J_state = function_state(J.function())
        self._J = Functional(_fn=J.function())

    def compute_gradient(self, M, M0=None):
        if not isinstance(M, Sequence):
            J_val, (dJ,) = self.compute_gradient(
                (M,),
                M0=None if M0 is None else (M0,))
            return J_val, dJ

        if function_state(self._J.function()) != self._J_state:
            raise RuntimeError("State has changed")

        dM = tuple(function_new(m) for m in M)
        manager, M, dM = self._setup_manager(M, dM, M0=M0, solve_tlm=False)

        dJ = self._J.tlm_functional((M, dM), manager=manager)

        J_val = self._J.value()
        dJ = manager.compute_gradient(
            dJ, dM,
            cache_adjoint_degree=1 if self._cache_adjoint else 0,
            store_adjoint=self._cache_adjoint)

        return J_val, dJ

    def action(self, M, dM, M0=None):
        if not isinstance(M, Sequence):
            J_val, dJ_val, (ddJ,) = self.action(
                (M,), (dM,),
                M0=None if M0 is None else (M0,))
            return J_val, dJ_val, ddJ

        if function_state(self._J.function()) != self._J_state:
            raise RuntimeError("State has changed")

        manager, M, dM = self._setup_manager(M, dM, M0=M0, solve_tlm=True)

        dJ = self._J.tlm_functional((M, dM), manager=manager)

        J_val = self._J.value()
        dJ_val = dJ.value()
        ddJ = manager.compute_gradient(
            dJ, M,
            cache_adjoint_degree=1 if self._cache_adjoint else 0,
            store_adjoint=self._cache_adjoint)

        return J_val, dJ_val, ddJ


class SingleBlockHessian(CachedHessian):
    ""

    def __init__(self, *args, **kwargs):
        warnings.warn("SingleBlockHessian class is deprecated -- "
                      "use CachedHessian instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class CachedGaussNewton(GaussNewton, HessianOptimization):
    """Represents a Gauss-Newton approximation to a Hessian associated with a
    given forward model. Uses a cached forward calculation.

    :arg X: A function or a :class:`Sequence` of functions defining the state.
    :arg R_inv_action: See :class:`tlm_adjoint.hessian.GaussNewton`.
    :arg B_inv_action: See :class:`tlm_adjoint.hessian.GaussNewton`.
    :arg J_space: The space for the functional. `FloatSpace()` is used if not
        supplied.
    :arg manager: The :class:`tlm_adjoint.tlm_adjoint.EquationManager` used to
        record the forward. This must have used `'memory'` checkpointing with
        automatic dropping of function references disabled. `manager()` is used
        if not supplied.
    """

    def __init__(self, X, R_inv_action, B_inv_action=None,
                 *, J_space=None, manager=None):
        if not isinstance(X, Sequence):
            X = (X,)

        HessianOptimization.__init__(self, manager=manager,
                                     cache_adjoint=False)
        GaussNewton.__init__(
            self, R_inv_action, B_inv_action=B_inv_action,
            J_space=J_space)
        self._X = tuple(X)
        self._X_state = tuple(function_state(x) for x in X)

    def _setup_manager(self, M, dM, M0=None, *,
                       annotate_tlm=False, solve_tlm=True):
        manager, M, dM = HessianOptimization._setup_manager(
            self, M, dM, M0=M0,
            annotate_tlm=annotate_tlm, solve_tlm=solve_tlm)
        return manager, M, dM, self._X

    def action(self, M, dM, M0=None):
        if tuple(function_state(x) for x in self._X) != self._X_state:
            raise RuntimeError("State has changed")

        return GaussNewton.action(self, M, dM, M0=M0)
