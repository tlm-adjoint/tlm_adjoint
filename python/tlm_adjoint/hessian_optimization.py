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

from .base_caches import clear_caches
from .hessian import Hessian, HessianException
from .manager import manager as _manager
from .tlm_adjoint import CheckpointStorage

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

    def compute_gradient(self, M):
        if not isinstance(M, (list, tuple)):
            J, (dJ,) = self.compute_gradient((M,))
            return J, dJ

        J_val = self._J.value()
        dJ = self._manager.compute_gradient(self._J, M)

        return J_val, dJ

    def action(self, M, dM):
        if not isinstance(M, (list, tuple)):
            J_val, dJ_val, (ddJ,) = self.action((M,), (dM,))
            return J_val, dJ_val, ddJ

        clear_caches(*dM)

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

        # The following rather nasty piece of code creates a new temporary
        # equation manager with the necessary tangent-linear equations added
        # manually, via direct modification of internal equation manager data

        manager = self._manager.new()
        manager.add_tlm(M, dM)
        manager._annotation_state = "annotating"
        manager._tlm_state = "deriving"
        manager._cp = CheckpointStorage(store_ics=True, store_data=True)

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

            for (M, dM), (tlm_map, max_depth) in manager._tlm.items():
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
                    manager._cp.add_equation((0, len(manager._block) - 1),
                                             tlm_eq, deps=tlm_deps)

        dJ = self._J.tlm(M, dM, manager=manager)

        J_val = self._J.value()
        dJ_val = dJ.value()
        ddJ = manager.compute_gradient(dJ, M)

        return J_val, dJ_val, ddJ
