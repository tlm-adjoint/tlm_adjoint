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

from .backend import *
from .backend_interface import *

from .base_equations import *

from collections import OrderedDict
import copy
import ufl

__all__ = \
  [    
    "AssembleSolver",
    "DirichletBCSolver",
    "EquationSolver"
  ]
  
class AssembleSolver(Equation):
  def __init__(self, rhs, x, form_compiler_parameters = {}):
    rank = len(rhs.arguments())
    if rank != 0:
      raise EquationException("Must be a rank 0 form")
    if not getattr(x.function_space(), "_tlm_adjoint__real_space", False):
      raise EquationException("Rank 0 forms can only be assigned to real functions")
  
    deps = []
    dep_ids = set()
    nl_deps = []
    nl_dep_ids = set()
    for dep in rhs.coefficients():
      if isinstance(dep, backend_Function):
        dep_id = dep.id()
        if not dep_id in dep_ids:
          deps.append(dep)
          dep_ids.add(dep_id)
        if not dep_id in nl_dep_ids:
          n_nl_deps = 0
          for nl_dep in ufl.algorithms.expand_derivatives(ufl.derivative(rhs, dep, argument = TrialFunction(dep.function_space()))).coefficients():
            if isinstance(nl_dep, backend_Function):
              nl_dep_id = nl_dep.id()
              if not nl_dep_id in nl_dep_ids:
                nl_deps.append(nl_dep)
                nl_dep_ids.add(nl_dep_id)
              n_nl_deps += 1
          if not dep_id in nl_dep_ids and n_nl_deps > 0:
            nl_deps.append(dep)
            nl_dep_ids.add(dep_id)
    if x.id() in dep_ids:
      raise EquationException("Invalid non-linear dependency")
    del(dep_ids, nl_dep_ids)
    deps.insert(0, x)
    deps[1:] = sorted(deps[1:], key = lambda dep : dep.id())
    nl_deps = sorted(nl_deps, key = lambda dep : dep.id())
    
    form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
    update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_
    
    Equation.__init__(self, x, deps, nl_deps = nl_deps)
    self._rhs = rhs
    self._form_compiler_parameters = form_compiler_parameters

  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    self._rhs = replace(self._rhs, replace_map)

  def forward_solve(self, x, deps = None):
    if deps is None:
      rhs = self._rhs
    else:
      rhs = replace(self._rhs, OrderedDict(zip(self.dependencies(), deps)))
      
    function_assign(x, assemble(rhs, form_compiler_parameters = self._form_compiler_parameters))
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    else:
      dep = self.dependencies()[dep_index]
      dF = ufl.algorithms.expand_derivatives(ufl.derivative(self._rhs, dep, argument = TestFunction(dep.function_space())))
      if dF.empty():
        return None
      dF = replace(dF, OrderedDict([(eq_dep, dep) for eq_dep, dep in zip(self.nonlinear_dependencies(), nl_deps)]))
      return (-adj_x.vector().max(), assemble(dF, form_compiler_parameters = self._form_compiler_parameters))
  
  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
  
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
      return None
    tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
    if tlm_rhs.empty():
      return None
    else:
      return AssembleSolver(tlm_rhs, tlm_map[x],
        form_compiler_parameters = self._form_compiler_parameters)
  
class EquationSolver(Equation):
  # eq, x, bcs, form_compiler_parameters and solver_parameters argument usage
  # based on the interface for the solve function in FEniCS (see e.g. FEniCS
  # 2017.1.0)
  def __init__(self, eq, x, bcs = [], form_compiler_parameters = {}, solver_parameters = {},
    initial_guess = None):
    if isinstance(bcs, DirichletBC):
      bcs = [bcs]
    
    lhs, rhs = eq.lhs, eq.rhs
    linear = isinstance(lhs, ufl.classes.Form) and isinstance(rhs, ufl.classes.Form)
    if linear:
      if x in lhs.coefficients() or x in rhs.coefficients():
        raise EquationException("Invalid non-linear dependency")
      F = action(lhs, x) - rhs
      J = lhs
    else:
      F = lhs
      if rhs != 0:
        F -= rhs
      J = ufl.algorithms.expand_derivatives(ufl.derivative(F, x, argument = TrialFunction(x.function_space())))
    
    deps = []
    dep_ids = set()
    nl_deps = []
    nl_dep_ids = set()
    for dep in F.coefficients():
      if isinstance(dep, backend_Function):
        dep_id = dep.id()
        if not dep_id in dep_ids:
          deps.append(dep)
          dep_ids.add(dep_id)
        if not dep_id in nl_dep_ids:
          n_nl_deps = 0
          for nl_dep in ufl.algorithms.expand_derivatives(ufl.derivative(F, dep, argument = TrialFunction(dep.function_space()))).coefficients():
            if isinstance(nl_dep, backend_Function):
              nl_dep_id = nl_dep.id()
              if not nl_dep_id in nl_dep_ids:
                nl_deps.append(nl_dep)
                nl_dep_ids.add(nl_dep_id)
              n_nl_deps += 1
          if not dep_id in nl_dep_ids and n_nl_deps > 0:
            nl_deps.append(dep)
            nl_dep_ids.add(dep_id)
          
    if initial_guess == x:
      initial_guess = None
      x_id = x.id()
    if not initial_guess is None:
      initial_guess_id = initial_guess.id()
      if not initial_guess_id in dep_ids:
        deps.append(initial_guess)
        dep_ids.add(initial_guess_id)
    del(dep_ids, nl_dep_ids)
    
    if x in deps:
      deps.remove(x)
    deps.insert(0, x)
      
    deps[1:] = sorted(deps[1:], key = lambda dep : dep.id())
    nl_deps = sorted(nl_deps, key = lambda dep : dep.id())
    
    hbcs = [homogenized_bc(bc) for bc in bcs]
    
    form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
    update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_
    
    Equation.__init__(self, x, deps, nl_deps = nl_deps)    
    self._F = F
    self._lhs, self._rhs = lhs, rhs
    self._bcs = copy.copy(bcs)
    self._hbcs = hbcs
    self._J = J
    self._form_compiler_parameters = copy_parameters_dict(form_compiler_parameters)
    self._solver_parameters = copy_parameters_dict(solver_parameters)
    self._initial_guess_index = None if initial_guess is None else deps.index(initial_guess)
    self._linear = linear

  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    self._F = replace(self._F, replace_map)
    self._lhs = replace(self._lhs, replace_map)
    if self._rhs != 0:
      self._rhs = replace(self._rhs, replace_map)
    self._J = replace(self._J, replace_map)
    
  def forward_solve(self, x, deps = None):  
    eq_deps = self.dependencies()
    if not self._initial_guess_index is None:
      function_assign(x, (eq_deps if deps is None else deps)[self._initial_guess_index])
  
    if deps is None:
      replace_deps = lambda F : F
    else:
      replace_map = OrderedDict([(eq_dep, dep) for eq_dep, dep in zip(eq_deps, deps)])
      replace_deps = lambda F : replace(F, replace_map)
  
    if self._linear:
      solve(replace_deps(self._lhs) == replace_deps(self._rhs), x, self._bcs,
        form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._solver_parameters)
    else:
      solve(replace_deps(self._F) == 0, x, self._bcs, J = replace_deps(self._J),
        form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._solver_parameters)
  
  def adjoint_jacobian_solve(self, nl_deps, b):
    J = replace(adjoint(self._J), OrderedDict([(eq_dep, dep) for eq_dep, dep in zip(self.nonlinear_dependencies(), nl_deps)]))
    J = assemble(J, form_compiler_parameters = self._form_compiler_parameters)
    for bc in self._hbcs:
      bc.apply(J, b)
    x = function_new(b)
    solve(J, x.vector(), b, solver_parameters = self._solver_parameters)
    return x
  
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    dep = self.dependencies()[dep_index]
    dF = ufl.algorithms.expand_derivatives(ufl.derivative(self._F, dep, argument = TrialFunction(dep.function_space())))
    if dF.empty():
      return None
    dF = action(adjoint(dF), adj_x)
    dF = replace(dF, OrderedDict([(eq_dep, dep) for eq_dep, dep in zip(self.nonlinear_dependencies(), nl_deps)]))
    return assemble(dF, form_compiler_parameters = self._form_compiler_parameters)
  
  def tangent_linear(self, M, dM, tlm_map):
    x = self.x()
    if x in M:
      raise EquationException("Invalid tangent-linear parameter")
  
    tlm_rhs = ufl.classes.Zero()
    for m, dm in zip(M, dM):
      tlm_rhs -= ufl.derivative(self._F, m, argument = dm)
      
    for dep in self.dependencies():
      if dep != x and not dep in M:
        tau_dep = tlm_map[dep]
        if not tau_dep is None:
          tlm_rhs -= ufl.derivative(self._F, dep, argument = tau_dep)
    
    if isinstance(tlm_rhs, ufl.classes.Zero):
      return None
    tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
    if tlm_rhs.empty():
      return None
    else:    
      return EquationSolver(self._J == tlm_rhs, tlm_map[x], self._hbcs,
        form_compiler_parameters = self._form_compiler_parameters,
        solver_parameters = self._solver_parameters,
        initial_guess = tlm_map[self.dependencies()[self._initial_guess_index]] if not self._initial_guess_index is None else None)
        
class DirichletBCSolver(Equation):
  def __init__(self, y, x, forward_domain, *bc_args, **bc_kwargs):
    bc_kwargs = copy.copy(bc_kwargs)
    adjoint_domain = bc_kwargs.pop("adjoint_domain", forward_domain)
  
    Equation.__init__(self, x, [x, y], nl_deps = [])
    self._forward_domain = forward_domain
    self._adjoint_domain = adjoint_domain
    self._bc_args = bc_args
    self._bc_kwargs = bc_kwargs

  def forward_solve(self, x, deps = None):
    _, y = self.dependencies() if deps is None else deps
    function_zero(x)
    DirichletBC(x.function_space(), y, self._forward_domain, *self._bc_args, **self._bc_kwargs).apply(x.vector())
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    elif dep_index == 1:
      _, y = self.dependencies()
      F = function_new(y)
      DirichletBC(y.function_space(), adj_x, self._adjoint_domain,
                  *self._bc_args, **self._bc_kwargs).apply(F.vector())
      return (-1.0, F)
    else:
      return None

  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
  
  def tangent_linear(self, M, dM, tlm_map):
    x, y = self.dependencies()
    
    tau_y = None
    for i, m in enumerate(M):
      if m == x:
        raise EquationException("Invalid tangent-linear parameter")
      elif m == y:
        tau_y = dM[i]
    if tau_y is None:
      tau_y = tlm_map[y]
    
    if tau_y is None:
      return None
    else:
      return DirichletBCSolver(tau_y, tlm_map[x], self._forward_domain,
        adjoint_domain = self._adjoint_domain,
        *self._bc_args, **self._bc_kwargs)      
