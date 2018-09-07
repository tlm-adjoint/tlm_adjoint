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
from .caches import CacheIndex, Constant, DirichletBC, Function, is_static, \
  is_static_bcs, linear_solver_cache

from collections import OrderedDict
import copy
import ufl

__all__ = \
  [
    "AssembleSolver",
    "DirichletBCSolver",
    "EquationSolver"
  ]

if not "tlm_adjoint" in parameters:
  parameters["tlm_adjoint"] = {}
if not "AssembleSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"]["AssembleSolver"] = {}
if not "match_quadrature" in parameters["tlm_adjoint"]["AssembleSolver"]:
  parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"] = False
if not "EquationSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"]["EquationSolver"] = {}
if not "enable_jacobian_caching" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] = True
if not "match_quadrature" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"] = False
if not "defer_adjoint_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] = False

def extract_form_compiler_parameters(form, form_compiler_parameters):
  return {"quadrature_degree":ufl.algorithms.estimate_total_polynomial_degree(form)}

class AssembleSolver(Equation):
  def __init__(self, rhs, x, form_compiler_parameters = {},
    match_quadrature = None):
    if match_quadrature is None:
      match_quadrature = parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"]
      
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
    if match_quadrature:
      update_parameters_dict(form_compiler_parameters, extract_form_compiler_parameters(rhs, form_compiler_parameters))
    
    Equation.__init__(self, x, deps, nl_deps = nl_deps)
    self._rhs = rhs
    self._form_compiler_parameters = form_compiler_parameters

  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    self._rhs = ufl.replace(self._rhs, replace_map)

  def forward_solve(self, x, deps = None):
    if deps is None:
      rhs = self._rhs
    else:
      rhs = ufl.replace(self._rhs, OrderedDict(zip(self.dependencies(), deps)))
      
    function_assign(x, assemble(rhs, form_compiler_parameters = self._form_compiler_parameters))
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    # Derived from EquationSolver.derivative_action (see dolfin-adjoint
    # reference below). Code first added 2017-12-07.
    # Re-written 2018-01-28
    # Updated to adjoint only form 2018-01-29
    # Firedrake version first added to tlm_adjoint repository 2018-08-24
    
    eq_deps = self.dependencies()
    if dep_index < 0 or dep_index >= len(eq_deps):
      return None
    elif dep_index == 0:
      return adj_x
    
    dep = eq_deps[dep_index]
    dF = ufl.algorithms.expand_derivatives(ufl.derivative(self._rhs, dep, argument = TestFunction(dep.function_space())))
    if dF.empty():
      return None
    
    dF = ufl.replace(dF, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
    return (-function_max_value(adj_x), assemble(dF, form_compiler_parameters = self._form_compiler_parameters))
  
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
    initial_guess = None, cache_jacobian = None,
    match_quadrature = None, defer_adjoint_assembly = None):
    if isinstance(bcs, DirichletBC):
      bcs = [bcs]
    if cache_jacobian is None:
      if not parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"]:
        cache_jacobian = False
    if match_quadrature is None:
      match_quadrature = parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"]
    if defer_adjoint_assembly is None:
      defer_adjoint_assembly = parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"]
    if match_quadrature and defer_adjoint_assembly:
      raise EquationException("Cannot both match quadrature and defer adjoint assembly")
    
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
    if not initial_guess is None:
      initial_guess_id = initial_guess.id()
      if not initial_guess_id in dep_ids:
        deps.append(initial_guess)
        dep_ids.add(initial_guess_id)
      if not initial_guess_id in nl_dep_ids:
        # This leads to storage of the initial guess, but this is not required
        # e.g. for a linear equation solved with a direct solver
        nl_deps.append(initial_guess)
        nl_dep_ids.add(initial_guess_id)
    
    x_id = x.id()
    if x_id in dep_ids:
      deps.remove(x)
    deps.insert(0, x)
    if initial_guess is None and not x_id in nl_dep_ids:
      # This leads to storage of the initial guess, but this is not required
      # e.g. for a linear equation solved with a direct solver
      nl_deps.append(x)
      nl_dep_ids.add(x_id)
      
    deps[1:] = sorted(deps[1:], key = lambda dep : dep.id())
    nl_deps = sorted(nl_deps, key = lambda dep : dep.id())
    
    hbcs = [homogenized_bc(bc) for bc in bcs]
    
    if cache_jacobian is None:
      cache_jacobian = is_static(J) and is_static_bcs(bcs)
    
    # Note: solver_parameters is not updated to include default parameters
    
    del(dep_ids, nl_dep_ids)
    
    form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
    update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_
    if match_quadrature:
      update_parameters_dict(form_compiler_parameters, extract_form_compiler_parameters(F, form_compiler_parameters))
    
    Equation.__init__(self, x, deps, nl_deps = nl_deps)    
    self._F = F
    self._lhs, self._rhs = lhs, rhs
    self._bcs = copy.copy(bcs)
    self._hbcs = hbcs
    self._J = J
    self._form_compiler_parameters = form_compiler_parameters
    self._solver_parameters = copy_parameters_dict(solver_parameters)
    self._linear_solver_parameters = solver_parameters
    self._initial_guess_index = None if initial_guess is None else deps.index(initial_guess)
    self._linear = linear
    
    self._cache_jacobian = cache_jacobian
    self._defer_adjoint_assembly = defer_adjoint_assembly
    self.reset_forward_solve()
    self.reset_adjoint_jacobian_solve()

  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    self._F = ufl.replace(self._F, replace_map)
    self._lhs = ufl.replace(self._lhs, replace_map)
    if self._rhs != 0:
      self._rhs = ufl.replace(self._rhs, replace_map)
    self._J = ufl.replace(self._J, replace_map)
    
  def forward_solve(self, x, deps = None):
    eq_deps = self.dependencies()
    if not self._initial_guess_index is None:
      function_assign(x, (eq_deps if deps is None else deps)[self._initial_guess_index])
    
    if deps is None:
      replace_deps = lambda F : F
    else:
      replace_map = OrderedDict(zip(eq_deps, deps))
      replace_deps = lambda F : ufl.replace(F, replace_map)
  
    if self._linear:
      if self._cache_jacobian:
        # Case 2: Linear, Jacobian cached, without pre-assembly
        if self._forward_J_solver.index() is None:
          J = replace_deps(self._J)
          # Assemble the Jacobian
          J_mat = assemble(J, bcs = self._bcs, form_compiler_parameters = self._form_compiler_parameters)
          # Construct and cache the linear solver
          self._forward_J_solver, J_solver = linear_solver_cache().linear_solver(J, J_mat, bcs = self._bcs,
            linear_solver_parameters = self._linear_solver_parameters,
            form_compiler_parameters = self._form_compiler_parameters)
        else:
          # Extract the linear solver from the cache
          J_solver = linear_solver_cache()[self._forward_J_solver]
          
        # Assemble the RHS without pre-assembly
        b = assemble(replace_deps(self._rhs))
        
        J_solver.solve(x.vector(), b)
      else:
        # Case 4: Linear, Jacobian not cached, without pre-assembly
        solve(replace_deps(self._lhs) == replace_deps(self._rhs), x, self._bcs,
          form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._linear_solver_parameters)
    else:
      # Case 5: Non-linear
      solve(replace_deps(self._F) == 0, x, self._bcs, J = replace_deps(self._J),
        form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._solver_parameters)
    
  def reset_forward_solve(self):
    self._forward_J_solver = CacheIndex()
  
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    # Similar to 'RHS.derivative_action' and 'RHS.second_derivative_action' in
    # dolfin-adjoint file dolfin_adjoint/adjrhs.py (see e.g. dolfin-adjoint
    # version 2017.1.0)
    # Code first added to JRM personal repository 2016-05-22
    # Code first added to dolfin_adjoint_custom repository 2016-06-02
    # Re-written 2018-01-28
    # Firedrake version first added to tlm_adjoint repository 2018-08-24
    
    eq_deps = self.dependencies()
    if dep_index < 0 or dep_index >= len(eq_deps):
      return None
    elif dep_index == 0:
      return adj_x
    
    dep = eq_deps[dep_index]
    dF = ufl.algorithms.expand_derivatives(ufl.derivative(self._F, dep, argument = TrialFunction(dep.function_space())))
    if dF.empty():
      return None
    dF = adjoint(dF)
    
    dF = ufl.replace(dF, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
    dF = action(dF, adj_x)
    if self._defer_adjoint_assembly:
      return dF
    else:
      return assemble(dF, form_compiler_parameters = self._form_compiler_parameters)
  
  def adjoint_jacobian_solve(self, nl_deps, b):
    if self._cache_jacobian:
      if self._adjoint_J_solver.index() is None:
        J = ufl.replace(adjoint(self._J), OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
        J_mat = assemble(J, form_compiler_parameters = self._form_compiler_parameters)
        for bc in self._hbcs:
          bc.apply(J_mat)
        self._adjoint_J_solver, J_solver = linear_solver_cache().linear_solver(J, J_mat,
          linear_solver_parameters = self._linear_solver_parameters,
          form_compiler_parameters = self._form_compiler_parameters)
      else:
        J_solver = linear_solver_cache()[self._adjoint_J_solver]
      x = function_new(b)
      J_solver.solve(x.vector(), b)
    else:
      J = ufl.replace(adjoint(self._J), OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
      J_mat = assemble(J, form_compiler_parameters = self._form_compiler_parameters)
      for bc in self._hbcs:
        bc.apply(J_mat)
      x = function_new(b)
      solve(J_mat, x.vector(), b, solver_parameters = self._linear_solver_parameters)
    return x
  
  def reset_adjoint_jacobian_solve(self):
    self._adjoint_J_solver = CacheIndex()
  
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
        solver_parameters = self._linear_solver_parameters,
        initial_guess = tlm_map[self.dependencies()[self._initial_guess_index]] if not self._initial_guess_index is None else None,
        cache_jacobian = self._cache_jacobian,
        defer_adjoint_assembly = self._defer_adjoint_assembly)
        
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
