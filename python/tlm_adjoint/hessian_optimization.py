#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 The University of Edinburgh
#
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

from .hessian import Hessian, HessianException
from .manager import manager as _manager
from .tlm_adjoint import Checkpoint

from collections import OrderedDict

__all__ = \
  [
    "HessianException",
    "SingleBlockHessian"
  ]
  
# Replacing functions makes it much more difficult to derive tangent-linear
# equations, so disable it
_manager().replace = lambda *args, **kwargs : None

class SingleBlockHessian(Hessian):
  def __init__(self, J, manager = None):
    """
    A Hessian class for the case where a single block equation with memory
    checkpointing is used. The forward should already have been processed by
    the supplied equation manager.
    
    Arguments:
    
    J        The Functional.
    manager  (Optional) The equation manager used to process the forward.
    """
  
    self._J = J
    self._manager = _manager() if manager is None else manager
    
  def compute_gradient(self, M):
    if not isinstance(M, (list, tuple)):
      J, (dJ,) = self.compute_gradient([M])
      return J, dJ
      
    if self._manager._cp_method != "memory" \
      or not (len(self._manager._blocks) == 0 or (len(self._manager._blocks) == 1 and len(self._manager._block) == 0)):
      raise HessianException("Invalid equation manager state")    
    
    J_val = self._J.value()
    dJ = self._manager.compute_gradient(self._J, M)
    
    return J_val, dJ
    
  def action(self, M, dM):
    if not isinstance(M, (list, tuple)):
      J_val, dJ_val, (ddJ,) = self.action([M], [dM])
      return J_val, dJ_val, ddJ
      
    if self._manager._cp_method != "memory" \
      or not (len(self._manager._blocks) == 0 or (len(self._manager._blocks) == 1 and len(self._manager._block) == 0)):
      raise HessianException("Invalid equation manager state")    
    block = self._manager._block if len(self._manager._blocks) == 0 else self._manager._blocks[0]

    # The following rather nasty piece of code creates a new temporary equation
    # manager with the necessary tangent-linear equations added manually, via
    # direct modification of internal equation manager data

    manager = self._manager.new()
    manager.add_tlm(M, dM)
    manager._annotation_state = "annotating"
    manager._tlm_state = "deriving"
    manager._cp = Checkpoint(*self._manager._cp.initial_conditions(copy = False))
    
    for i, eq in enumerate(block):
      eq_X = eq.X()
      # Copy annotation of the equation
      manager._block.append(eq)
      manager._cp.configure(checkpoint_ics = False, checkpoint_data = True)
      manager._cp.add_equation((0, len(manager._block) - 1), eq,
        nl_deps = self._manager._cp[(0, i)], copy = False)

      for (M, dM), (tlm_map, tlm_map_next, max_depth) in manager._tlm.items():
        # Generate the associated tangent-linear equation (or extract it from
        # the cache)
        eq_tlm_eqs = manager._tlm_eqs.get(eq, None)
        if eq_tlm_eqs is None:
          eq_tlm_eqs = manager._tlm_eqs[eq] = OrderedDict()
        tlm_eq = eq_tlm_eqs.get((M, dM), None)
        if tlm_eq is None:
          for dep in eq.dependencies():
            if dep in M or dep in tlm_map:
              tlm_eq = eq_tlm_eqs[(M, dM)] = eq.tangent_linear(M, dM, tlm_map)
              break
        if not tlm_eq is None:
          # Extract the dependency values from storage for use in the solution of
          # the tangent-linear equation
          eq_deps = OrderedDict([(eq_dep.id(), cp_dep) for eq_dep, cp_dep in zip(eq.nonlinear_dependencies(), self._manager._cp[(0, i)])])
          tlm_deps = list(tlm_eq.dependencies())
          for j, tlm_dep in enumerate(tlm_deps):
            if tlm_dep.id() in eq_deps:
              tlm_deps[j] = eq_deps[tlm_dep.id()]
            
          tlm_X = tlm_eq.X()
          manager._cp.configure(checkpoint_ics = True, checkpoint_data = True)
          # Pre-process the tangent-linear equation
          for tlm_dep in tlm_eq.initial_condition_dependencies():
            manager._cp.add_initial_condition(tlm_dep)
          # Solve the tangent-linear equation
          tlm_eq.forward_solve(tlm_X[0] if len(tlm_X) == 1 else tlm_X, tlm_deps)
          # Post-process the tangent-linear equation
          manager._block.append(tlm_eq)
          manager._cp.add_equation((0, len(manager._block) - 1), tlm_eq, deps = tlm_deps)
          
          for eq_x in eq_X:
            tlm_map_next[eq_x] = tlm_map[eq_x]
          
    dJ = self._J.tlm(M, dM, manager = manager)
    
    J_val = self._J.value()
    dJ_val = dJ.value()
    ddJ = manager.compute_gradient(dJ, M)
    
    return J_val, dJ_val, ddJ
