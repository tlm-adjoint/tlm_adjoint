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

from .backend import *
from .backend_code_generator_interface import *
from .backend_interface import *

from .base_equations import *
from .caches import Cache, CacheRef, form_dependencies, form_key, parameters_key
from .equations import EquationSolver, alias_assemble, alias_form

import types
import ufl

__all__ = \
  [
    "LocalProjectionSolver",
    
    "LocalSolverCache",
    "local_solver_cache",
    "set_local_solver_cache"
  ]

def local_solver_key(form, bcs, form_compiler_parameters):
  return (form_key(form), tuple(bcs), parameters_key(form_compiler_parameters))

def LocalSolver(form, bcs = [], form_compiler_parameters = {}):
  local_solver = assemble(form, bcs = bcs,
    form_compiler_parameters = form_compiler_parameters, inverse = True)
  local_solver.force_evaluation()
  def solve_local(self, x, b):
    matrix_multiply(self, b, tensor = x)
  local_solver.solve_local = types.MethodType(solve_local, local_solver)
  return local_solver

class LocalSolverCache(Cache):
  def local_solver(self, form, bcs = [], form_compiler_parameters = {},
    replace_map = None):
    key = local_solver_key(form, bcs, form_compiler_parameters)
    value = self.get(key, None)
    if value is None:
      assemble_form = form if replace_map is None else ufl.replace(form, replace_map)
      local_solver = LocalSolver(form, bcs = bcs,
        form_compiler_parameters = form_compiler_parameters)
      value = self.add(key, local_solver, deps = tuple(form_dependencies(form).values()))
    else:
      local_solver = value()

    return value, local_solver

_local_solver_cache = [LocalSolverCache()]
def local_solver_cache():
  return _local_solver_cache[0]
def set_local_solver_cache(local_solver_cache):
  _local_solver_cache[0] = local_solver_cache
  
class LocalProjectionSolver(EquationSolver):
  def __init__(self, rhs, x, bcs = [], form_compiler_parameters = {},
    cache_jacobian = None, cache_rhs_assembly = None, match_quadrature = None,
    defer_adjoint_assembly = None):
    space = x.function_space()
    test, trial = TestFunction(space), TrialFunction(space)
    lhs = ufl.inner(test, trial) * ufl.dx
    if not isinstance(rhs, ufl.classes.Form):
      rhs = ufl.inner(test, rhs) * ufl.dx
    
    EquationSolver.__init__(self, lhs == rhs, x, bcs = bcs,
      form_compiler_parameters = form_compiler_parameters,
      solver_parameters = {},
      cache_jacobian = cache_jacobian,
      cache_rhs_assembly = cache_rhs_assembly,
      match_quadrature = match_quadrature,
      defer_adjoint_assembly = defer_adjoint_assembly)
  
  def forward_solve(self, x, deps = None):
    if self._cache_rhs_assembly:
      b = self._cached_rhs(deps)
    elif deps is None:
      b = assemble(self._rhs,
        form_compiler_parameters = self._form_compiler_parameters)
    else:
      if self._forward_eq is None:
        self._forward_eq = None, None, alias_form(self._rhs, self.dependencies())
      _, _, rhs = self._forward_eq
      b = alias_assemble(rhs, deps,
        form_compiler_parameters = self._form_compiler_parameters)
    
    if self._cache_jacobian:
      local_solver = self._forward_J_solver()
      if local_solver is None:
        self._forward_J_solver, local_solver = local_solver_cache().local_solver(
          self._lhs, bcs = self._bcs,
          form_compiler_parameters = self._form_compiler_parameters)
    else:
      local_solver = LocalSolver(
        self._lhs, bcs = self._bcs,
        form_compiler_parameters = self._form_compiler_parameters)
        
    local_solver.solve_local(x.vector(), b)
    
  def adjoint_jacobian_solve(self, nl_deps, b):
    if self._cache_jacobian:
      local_solver = self._forward_J_solver()
      if local_solver is None:
        self._forward_J_solver, local_solver = local_solver_cache().local_solver(
          self._lhs, bcs = self._hbcs,
          form_compiler_parameters = self._form_compiler_parameters)
    else:
      local_solver = LocalSolver(
        self._lhs, bcs = self._hbcs,
        form_compiler_parameters = self._form_compiler_parameters)
        
    adj_x = function_new(b)
    local_solver.solve_local(adj_x.vector(), b.vector())
    return adj_x
  
  def reset_adjoint_jacobian_solve(self):
    self._forward_J_solver = CacheRef()
  
  #def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
  # A consistent diagonal block adjoint derivative action requires an
  # appropriate quadrature degree to have been selected
  
  def tangent_linear(self, M, dM, tlm_map):
    x = self.x()
    if x in M:
      raise EquationException("Invalid tangent-linear parameter")
  
    tlm_rhs = ufl.classes.Zero()
    for m, dm in zip(M, dM):
      tlm_rhs += ufl.derivative(self._rhs, m, argument = dm)
      
    for dep in self.dependencies():
      if dep != x and not dep in M:
        tau_dep = tlm_map[dep]
        if not tau_dep is None:
          tlm_rhs += ufl.derivative(self._rhs, dep, argument = tau_dep)
    
    if isinstance(tlm_rhs, ufl.classes.Zero):
      return NullSolver(tlm_map[x])
    tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
    if tlm_rhs.empty():
      return NullSolver(tlm_map[x])
    else:    
      return LocalProjectionSolver(tlm_rhs, tlm_map[x],
        form_compiler_parameters = self._form_compiler_parameters,
        cache_jacobian = self._cache_jacobian,
        cache_rhs_assembly = self._cache_rhs_assembly,
        defer_adjoint_assembly = self._defer_adjoint_assembly)
