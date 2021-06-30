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

from .interface import function_id, function_state

from .caches import clear_caches
from .functional import Functional
from .hessian import Hessian, HessianException
from .manager import manager as _manager
from .tlm_adjoint import AdjointCache, EquationManager

from collections.abc import Sequence
import warnings

__all__ = \
    [
        "CachedHessian",
        "HessianException",
        "SingleBlockHessian"
    ]


class CachedHessian(Hessian):
    def __init__(self, J, manager=None):
        """
        A Hessian class for the case where memory checkpointing is used,
        without automatic dropping of references to function objects.

        Arguments:

        J        The Functional.
        manager  (Optional) The equation manager used to process the forward.
        """

        if manager is None:
            manager = _manager()
        if manager._cp_method != "memory" \
                or manager._cp_parameters["drop_references"]:
            raise HessianException("Invalid equation manager state")

        blocks = list(manager._blocks) + [list(manager._block)]

        ics = dict(manager._cp.initial_conditions(cp=True, refs=True,
                                                  copy=False))

        nl_deps = {}
        for n, block in enumerate(blocks):
            for i, eq in enumerate(block):
                nl_deps[(n, i)] = manager._cp[(n, i)]

        self._J_state = function_state(J.fn())
        self._J = Functional(fn=J.fn())
        self._manager = manager
        self._blocks = blocks
        self._ics = ics
        self._nl_deps = nl_deps
        self._adj_cache_M = None
        self._adj_cache = None

    def _adj_cache_is_valid(self, M):
        if self._adj_cache_M is None:
            return False

        old_M = self._adj_cache_M
        if len(old_M) != len(M):
            return False

        for old_m, m in zip(old_M, M):
            if old_m is not m:
                return False

        return True

    def _new_manager(self):
        manager = EquationManager(cp_method="memory",
                                  cp_parameters={"drop_references": False})

        for x_id, value in self._ics.items():
            manager._cp._add_initial_condition(
                x_id=x_id, value=value, copy=False)

        return manager

    def _add_forward_equations(self, manager):
        for n, block in enumerate(self._blocks):
            for i, eq in enumerate(block):
                eq_id = eq.id()
                if eq_id not in manager._eqs:
                    manager._eqs[eq_id] = eq
                manager._block.append(eq)
                manager._cp.add_equation(
                    (len(manager._blocks), len(manager._block) - 1), eq,
                    nl_deps=self._nl_deps[(n, i)], copy=lambda x: False)
                yield n, i, eq

    def _tangent_linear(self, manager, eq, M, dM):
        return manager._tangent_linear(eq, M, dM)

    def _add_tangent_linear_equation(self, manager, n, i, eq, M, dM, tlm_eq,
                                     solve=True):
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

        tlm_eq_id = tlm_eq.id()
        if tlm_eq_id not in manager._eqs:
            manager._eqs[tlm_eq_id] = tlm_eq
        manager._block.append(tlm_eq)
        manager._cp.add_equation(
            (len(manager._blocks), len(manager._block) - 1), tlm_eq,
            deps=tlm_deps)

        self._adj_cache.register(
            0, len(manager._blocks), len(manager._block) - 1)

    def _setup_manager(self, M, dM, solve_tlm=True):
        if function_state(self._J.fn()) != self._J_state:
            raise HessianException("Functional state has changed")

        M = tuple(M)
        dM = tuple(dM)

        clear_caches(*dM)
        if not self._adj_cache_is_valid(M):
            self._adj_cache_M = M
            self._adj_cache = AdjointCache()
        assert self._adj_cache is not None

        manager = self._new_manager()
        manager.add_tlm(M, dM)

        for n, i, eq in self._add_forward_equations(manager):
            tlm_eq = self._tangent_linear(manager, eq, M, dM)
            if tlm_eq is not None:
                self._add_tangent_linear_equation(
                    manager, n, i, eq, M, dM, tlm_eq,
                    solve=solve_tlm)

        return manager, M, dM

    def compute_gradient(self, M):
        if not isinstance(M, Sequence):
            J, (dJ,) = self.compute_gradient((M,))
            return J, dJ

        J_val = self._J.value()
        dJ = self._manager.compute_gradient(self._J, M)

        return J_val, dJ

    def action(self, M, dM):
        if not isinstance(M, Sequence):
            J_val, dJ_val, (ddJ,) = self.action((M,), (dM,))
            return J_val, dJ_val, ddJ

        manager, M, dM = self._setup_manager(M, dM, solve_tlm=True)

        dJ = self._J.tlm(M, dM, manager=manager)

        J_val = self._J.value()
        dJ_val = dJ.value()
        ddJ = manager.compute_gradient(dJ, M, adj_cache=self._adj_cache)

        return J_val, dJ_val, ddJ


class SingleBlockHessian(CachedHessian):
    def __init__(self, *args, **kwargs):
        warnings.warn("SingleBlockHessian class is deprecated -- "
                      "use CachedHessian instead",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
