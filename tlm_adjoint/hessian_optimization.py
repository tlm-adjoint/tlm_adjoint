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

from .interface import function_id

from .caches import clear_caches
from .hessian import Hessian, HessianException
from .manager import manager as _manager
from .tlm_adjoint import AdjointCache, EquationManager

from collections.abc import Sequence

__all__ = \
    [
        "HessianException",
        "SingleBlockHessian"
    ]


class SingleBlockHessian(Hessian):
    def __init__(self, J, manager=None):
        """
        A Hessian class for the case where a single block equation with memory
        checkpointing is used. The forward should already have been processed
        by the supplied equation manager.

        Arguments:

        J        The Functional.
        manager  (Optional) The equation manager used to process the forward.
        """

        self._J = J
        self._manager = _manager() if manager is None else manager
        self._M_dM = None
        self._adj_cache = AdjointCache()

    def _adj_cache_is_valid(self, M):
        if self._M_dM is None:
            return False

        old_M, _ = self._M_dM
        if len(old_M) != len(M):
            return False

        for old_m, m in zip(old_M, M):
            if old_m is not m:
                return False

        return True

    def _set_tlm(self, manager, M, dM):
        manager.add_tlm(M, dM)

        (tlm_map, max_depth), = list(manager._tlm.values())
        assert max_depth == 1

        self._M_dM = (M, dM)
        return M, dM, tlm_map

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

        if self._manager._cp_method != "memory" \
           or self._manager._cp_parameters["drop_references"] \
           or not (len(self._manager._blocks) == 0
                   or (len(self._manager._blocks) == 1
                       and len(self._manager._block) == 0)):
            raise HessianException("Invalid equation manager state")
        if len(self._manager._blocks) == 0:
            block = self._manager._block
        else:
            block = self._manager._blocks[0]

        clear_caches(*dM)
        if not self._adj_cache_is_valid(M):
            self._adj_cache = AdjointCache()

        manager = EquationManager(cp_method="memory",
                                  cp_parameters={"drop_references": False})
        M, dM, tlm_map = self._set_tlm(manager, M, dM)

        # Copy (references to) forward model initial condition data
        ics = self._manager._cp.initial_conditions(cp=True, refs=True,
                                                   copy=False).items()
        for x_id, value in ics:
            manager._cp._add_initial_condition(x_id=x_id, value=value,
                                               copy=False)
        del ics

        for i, eq in enumerate(block):
            # Copy annotation of the equation
            manager._eqs[eq.id()] = eq
            manager._block.append(eq)
            manager._cp.add_equation((0, len(manager._block) - 1), eq,
                                     nl_deps=self._manager._cp[(0, i)],
                                     copy=lambda x: False)

            # Generate the associated tangent-linear equation (or extract
            # it from the cache)
            tlm_eq = manager._tangent_linear(eq, M, dM, tlm_map)
            if tlm_eq is not None:
                # Extract the dependency values from storage for use in the
                # solution of the tangent-linear equation
                eq_deps = {function_id(eq_dep): cp_dep
                           for eq_dep, cp_dep in
                           zip(eq.nonlinear_dependencies(),
                               self._manager._cp[(0, i)])}
                tlm_deps = list(tlm_eq.dependencies())
                for j, tlm_dep in enumerate(tlm_deps):
                    tlm_dep_id = function_id(tlm_dep)
                    if tlm_dep_id in eq_deps:
                        tlm_deps[j] = eq_deps[tlm_dep_id]

                tlm_X = tlm_eq.X()
                # Pre-process the tangent-linear equation
                for tlm_dep in tlm_eq.initial_condition_dependencies():
                    manager._cp.add_initial_condition(tlm_dep)
                # Solve the tangent-linear equation
                tlm_eq.forward(tlm_X, deps=tlm_deps)
                # Post-process the tangent-linear equation
                manager._eqs[tlm_eq.id()] = tlm_eq
                manager._block.append(tlm_eq)
                manager._cp.add_equation(
                    (len(manager._blocks), len(manager._block) - 1),
                    tlm_eq, deps=tlm_deps)
                self._adj_cache.register(
                    0, len(manager._blocks), len(manager._block) - 1)

        dJ = self._J.tlm(M, dM, manager=manager)

        J_val = self._J.value()
        dJ_val = dJ.value()
        ddJ = manager.compute_gradient(dJ, M, adj_cache=self._adj_cache)

        return J_val, dJ_val, ddJ
