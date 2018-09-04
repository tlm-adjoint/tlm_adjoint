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
from .caches import CacheIndex, Constant, Function, assembly_cache, \
  homogenized, is_static, is_static_bcs, linear_solver_cache, new_id, \
  split_action, split_form
from .manager import manager as _manager

from collections import OrderedDict
import copy
import ffc
import numpy
import ufl

__all__ = \
 [
    "AssembleSolver",
    "DirichletBCSolver",
    "EquationSolver"
  ]

if not "tlm_adjoint" in parameters:
  parameters.add(Parameters("tlm_adjoint"))
if not "AssembleSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"].add(Parameters("AssembleSolver"))
if not "match_quadrature" in parameters["tlm_adjoint"]["AssembleSolver"]:
  parameters["tlm_adjoint"]["AssembleSolver"].add("match_quadrature", False)
if not "EquationSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"].add(Parameters("EquationSolver"))
if not "enable_jacobian_caching" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("enable_jacobian_caching", True)
if not "pre_assemble" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("pre_assemble", True)
if not "match_quadrature" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("match_quadrature", False)
if not "defer_adjoint_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("defer_adjoint_assembly", False)

def extract_form_compiler_parameters(forms, form_compiler_parameters):
  if not isinstance(forms, (list, tuple)):
    forms = [forms]
  (form_data,), _, _, _ = ffc.analysis.analyze_forms(forms, form_compiler_parameters)
  integral_metadata = [integral_data.metadata for integral_data in form_data.integral_data]
  return {"quadrature_rule":ffc.analysis._extract_common_quadrature_rule(integral_metadata),
          "quadrature_degree":ffc.analysis._extract_common_quadrature_degree(integral_metadata)}

class AssembleSolver(Equation):
  def __init__(self, rhs, x, form_compiler_parameters = {},
    match_quadrature = None):
    if match_quadrature is None:
      match_quadrature = parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"]
      
    rank = len(rhs.arguments())
    if rank != 0:
      raise EquationException("Must be a rank 0 form")
    e = x.ufl_element()
    if e.family() != "Real" or e.degree() != 0:
      raise EquationException("Rank 0 forms can only be assigned to R0 functions")
  
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
          for nl_dep in ufl.algorithms.expand_derivatives(derivative(rhs, dep, du = TrialFunction(dep.function_space()))).coefficients():
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
    
    eq_deps = self.dependencies()
    if dep_index < 0 or dep_index >= len(eq_deps):
      return None
    elif dep_index == 0:
      return adj_x
    
    dep = eq_deps[dep_index]
    dF = ufl.algorithms.expand_derivatives(derivative(-self._rhs, dep, du = TestFunction(dep.function_space())))
    if dF.empty():
      return None
    
    dF = ufl.replace(dF, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
    return (function_max_value(adj_x), assemble(dF, form_compiler_parameters = self._form_compiler_parameters))
  
  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
  
  def tangent_linear(self, M, dM, tlm_map):
    x = self.x()
    if x in M:
      raise EquationException("Invalid tangent-linear parameter")
  
    tlm_rhs = ufl.classes.Zero()
    for m, dm in zip(M, dM):
      tlm_rhs += derivative(self._rhs, m, du = dm)
    
    for dep in self.dependencies():
      if dep != x and not dep in M:
        tau_dep = tlm_map[dep]
        if not tau_dep is None:
          tlm_rhs += derivative(self._rhs, dep, du = tau_dep)
    
    if isinstance(tlm_rhs, ufl.classes.Zero):
      return None
    tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
    if tlm_rhs.empty():
      return None
    else:
      return AssembleSolver(tlm_rhs, tlm_map[x],
        form_compiler_parameters = self._form_compiler_parameters)
    
def _linear_solver(A, linear_solver_parameters):
  linear_solver = linear_solver_parameters["linear_solver"]
  if linear_solver in ["direct", "lu"]:
    linear_solver = "default"
  elif linear_solver == "iterative":
    linear_solver = "gmres"
  is_lu_linear_solver = linear_solver == "default" or has_lu_solver_method(linear_solver)
  if is_lu_linear_solver:
    solver = LUSolver(A, linear_solver)
    update_parameters_dict(solver.parameters, linear_solver_parameters["lu_solver"])
  else:
    solver = KrylovSolver(A, linear_solver, linear_solver_parameters["preconditioner"])
    update_parameters_dict(solver.parameters, linear_solver_parameters["krylov_solver"])
  return solver
   
class FunctionAlias(backend_Function):
  def __init__(self, space):
    ufl.classes.Coefficient.__init__(self, space, count = new_id())
  
  def _alias(self, x):
    self._clear()
    if hasattr(x, "_cpp_object"):
      self._cpp_object = x._cpp_object
    if hasattr(x, "this"):
      self.this = x.this
  
  def _clear(self):
    if hasattr(self, "_cpp_object"):
      del(self._cpp_object)
    if hasattr(self, "this"):
      del(self.this)

def alias_form(form, deps):
  adeps = [FunctionAlias(dep.function_space()) for dep in deps]
  return_value = ufl.replace(form, OrderedDict(zip(deps, adeps)))
  assert(not "_tlm_adjoint__adeps" in return_value._cache)
  return_value._cache["_tlm_adjoint__adeps"] = adeps
  return return_value

def alias_replace(form, deps):
  for adep, dep in zip(form._cache["_tlm_adjoint__adeps"], deps):
    adep._alias(dep)

def alias_clear(form):
  for adep in form._cache["_tlm_adjoint__adeps"]:
    adep._clear()
    
def alias_assemble(form, deps, *args, **kwargs):
  alias_replace(form, deps)
  return_value = assemble(form, *args, **kwargs)
  alias_clear(form)
  return return_value
   
class EquationSolver(Equation):
  # eq, x, bcs, form_compiler_parameters and solver_parameters argument usage
  # based on the interface for the solve function in FEniCS (see e.g. FEniCS
  # 2017.1.0)
  def __init__(self, eq, x, bcs = [], form_compiler_parameters = {}, solver_parameters = {},
    initial_guess = None, cache_jacobian = None, pre_assemble = None,
    match_quadrature = None, defer_adjoint_assembly = None):
    if isinstance(bcs, DirichletBC):
      bcs = [bcs]
    if cache_jacobian is None:
      if not parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"]:
        cache_jacobian = False
    if pre_assemble is None:
      pre_assemble = parameters["tlm_adjoint"]["EquationSolver"]["pre_assemble"]
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
      J = ufl.algorithms.expand_derivatives(derivative(F, x, du = TrialFunction(x.function_space())))
    
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
          for nl_dep in ufl.algorithms.expand_derivatives(derivative(F, dep, du = TrialFunction(dep.function_space()))).coefficients():
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
    
    hbcs = [homogenized(bc) for bc in bcs]
    
    if cache_jacobian is None:
      cache_jacobian = is_static(J) and is_static_bcs(bcs)
    
    def update_default_parameters(old_parameters, new_parameters):
      for key in new_parameters:
        value = new_parameters[key]
        if not key in old_parameters:
          if isinstance(value, (Parameters, dict)):
            old_parameters[key] = copy_parameters_dict(value)
          else:
            old_parameters[key] = value
        elif isinstance(old_parameters[key], (Parameters, dict)):
          if not isinstance(value, (Parameters, dict)):
            raise EquationException("Invalid solver parameter: %s" % key)
          update_default_parameters(old_parameters[key], value)
    
    solver_parameters = copy_parameters_dict(solver_parameters)
    if linear:
      linear_solver_parameters = solver_parameters
      update_default_parameters(linear_solver_parameters, {"linear_solver":"default"})
    else:
      update_default_parameters(solver_parameters, {"nonlinear_solver":"newton"})
      nl_solver = solver_parameters["nonlinear_solver"]
      if nl_solver == "newton":
        if hasattr(NewtonSolver, "default_parameters"):
          update_default_parameters(solver_parameters, {"newton_solver":NewtonSolver.default_parameters()})
        else:
          update_default_parameters(solver_parameters, {"newton_solver":NewtonSolver().parameters})
        linear_solver_parameters = solver_parameters["newton_solver"]
      else:
        raise EquationException("Unsupported non-linear solver: %s" % nl_solver)
    
    linear_solver = linear_solver_parameters["linear_solver"]
    is_lu_linear_solver = linear_solver in ["default", "direct", "lu"] or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
      if hasattr(LUSolver, "default_parameters"):
        update_default_parameters(linear_solver_parameters, {"lu_solver":LUSolver.default_parameters()})
      else:
        update_default_parameters(linear_solver_parameters, {"lu_solver":LUSolver().parameters})
    else:
      if hasattr(KrylovSolver, "default_parameters"):
        update_default_parameters(linear_solver_parameters, {"preconditioner":"default",
                                                             "krylov_solver":KrylovSolver.default_parameters()})
      else:
        update_default_parameters(linear_solver_parameters, {"preconditioner":"default",
                                                             "krylov_solver":KrylovSolver().parameters})
      nonzero_initial_guess = linear_solver_parameters["krylov_solver"].get("nonzero_initial_guess", False)
      if nonzero_initial_guess is None:
        nonzero_initial_guess = linear_solver_parameters["krylov_solver"]["nonzero_initial_guess"] = False
    
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
    self._solver_parameters = solver_parameters
    self._linear_solver_parameters = linear_solver_parameters
    self._initial_guess_index = None if initial_guess is None else deps.index(initial_guess)
    self._linear = linear
    
    self._cache_jacobian = cache_jacobian
    self._pre_assemble = pre_assemble
    self._defer_adjoint_assembly = defer_adjoint_assembly
    self.reset_forward_solve()
    self.reset_adjoint_derivative_action()
    self.reset_adjoint_jacobian_solve()

  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    self._F = ufl.replace(self._F, replace_map)
    self._lhs = ufl.replace(self._lhs, replace_map)
    if self._rhs != 0:
      self._rhs = ufl.replace(self._rhs, replace_map)
    self._J = ufl.replace(self._J, replace_map)
    if hasattr(self, "_forward_b_pa") and not self._forward_b_pa is None:
      if not self._forward_b_pa[0] is None:
        self._forward_b_pa[0][0] = ufl.replace(self._forward_b_pa[0][0], replace_map)
      for i, (mat_form, mat_index) in self._forward_b_pa[1].items():
        self._forward_b_pa[1][i][0] = ufl.replace(mat_form, replace_map)
    if self._defer_adjoint_assembly and hasattr(self, "_derivative_mats"):
      for dep_index, mat_cache in self._derivative_mats:
        if isinstance(mat_cache, ufl.classes.Form):
          self._derivative_mats[dep_index] = ufl.replace(mat_cache, replace_map)
    
  def _pre_assembled_rhs(self, deps, b_bc = None):
    eq_deps = self.dependencies()
    
    if self._forward_b_pa is None:
      # Split into static and non-static components
      static_form, non_static_form = split_form(self._rhs)

      mat_forms = OrderedDict()
      if not non_static_form.empty():
        for i, dep in enumerate(eq_deps):
          mat_form, non_static_form = split_action(non_static_form, dep)
          if not mat_form.empty():
            # The non-static part contains a component with can be represented
            # as the action of a static matrix ...
            if is_static(dep):
              # ... on a static dependency. This is part of the static
              # component.
              static_form += action(mat_form, dep)
            else:
              # ... on a non-static dependency.
              mat_forms[i] = [mat_form, CacheIndex()]
          if non_static_form.empty():
            break

      if not non_static_form.empty():
        # Attempt to split the remaining non-static component into static and
        # non-static components
        static_form_term, non_static_form = split_form(non_static_form)
        static_form += static_form_term
        
      if non_static_form.empty():
        non_static_form = None
      else:
        non_static_form = alias_form(non_static_form, eq_deps)

      if static_form.empty():
        static_form = None
      else:
        static_form = [static_form, CacheIndex()]

      self._forward_b_pa = [static_form, mat_forms, non_static_form]
    else:
      static_form, mat_forms, non_static_form = self._forward_b_pa
    
    b = None
     
    for i, (mat_form, mat_index) in mat_forms.items():
      if mat_index.index() is None:
        if not deps is None:
          mat_form = ufl.replace(mat_form, OrderedDict(zip(eq_deps, deps)))
        mat_index, (mat, _) = assembly_cache().assemble(mat_form, form_compiler_parameters = self._form_compiler_parameters)
        mat_forms[i][1] = mat_index
      else:
        mat, _ = assembly_cache()[mat_index]
      if b is None:
        b = mat * (eq_deps if deps is None else deps)[i].vector()
      else:
        b.axpy(1.0, mat * (eq_deps if deps is None else deps)[i].vector())

    if not non_static_form is None:
      b = alias_assemble(non_static_form, eq_deps if deps is None else deps,
        form_compiler_parameters = self._form_compiler_parameters,
        tensor = b, add_values = not b is None)
        
    if not static_form is None:
      if static_form[1].index() is None:
        static_form[1], static_b = assembly_cache().assemble(
          static_form[0] if deps is None else ufl.replace(static_form[0], OrderedDict(zip(eq_deps, deps))),
          form_compiler_parameters = self._form_compiler_parameters)
      else:
        static_b = assembly_cache()[static_form[1]]
      if b is None:
        b = static_b.copy()
      else:
        b.axpy(1.0, static_b)
    
    for bc in self._hbcs:
      bc.apply(b)
    if not b_bc is None:
      b.axpy(1.0, b_bc)
    return b

  def forward_solve(self, x, deps = None):
    eq_deps = self.dependencies()
    if not self._initial_guess_index is None:
      function_assign(x, (eq_deps if deps is None else deps)[self._initial_guess_index])
    
    if self._linear:
      if self._cache_jacobian:
        # Cases 1 and 2: Linear, Jacobian cached, with or without pre-assembly
        
        if self._forward_J_mat.index() is None or \
          self._forward_J_solver.index() is None:
          J = self._J if deps is None else ufl.replace(self._J, OrderedDict(zip(eq_deps, deps)))
          
        if self._forward_J_mat.index() is None:
          # Assemble and cache the Jacobian (and bc RHS terms)
          self._forward_J_mat, (J_mat, b_bc) = assembly_cache().assemble(J, bcs = self._bcs, form_compiler_parameters = self._form_compiler_parameters)
        else:
          # Extract the Jacobian (and bc RHS terms) from the cache
          J_mat, b_bc = assembly_cache()[self._forward_J_mat]
          
        if self._pre_assemble:
          # Assemble the RHS with pre-assembly
          b = self._pre_assembled_rhs(deps, b_bc = b_bc)
        else:
          # Assemble the RHS without pre-assembly
          if deps is None:
            rhs = self._rhs
          else:
            if self._forward_eq is None:
              self._forward_eq = None, None, alias_form(self._rhs, eq_deps)
            _, _, rhs = self._forward_eq
            alias_replace(rhs, deps)
          b = assemble(rhs, form_compiler_parameters = self._form_compiler_parameters)
          if not deps is None:
            alias_clear(rhs)

          # Add bc RHS terms
          for bc in self._hbcs:
            bc.apply(b)
          if not b_bc is None:
            b.axpy(1.0, b_bc)
      
        if self._forward_J_solver.index() is None:
          # Construct and cache the linear solver
          self._forward_J_solver, J_solver = linear_solver_cache().linear_solver(J, J_mat, bcs = self._bcs, linear_solver_parameters = self._linear_solver_parameters)
        else:
          # Extract the linear solver from the cache
          J_solver = linear_solver_cache()[self._forward_J_solver]
      else:
        if self._pre_assemble:
          # Case 3: Linear, Jacobian not cached, with pre-assembly
          
          # Assemble the Jacobian (and bc RHS terms)
          if deps is None:
            J = self._J
          else:
            if self._forward_eq is None:
              self._forward_eq = None, alias_form(self._J, eq_deps), None
            _, J, _ = self._forward_eq
            alias_replace(J, deps)
          test = TestFunction(J.arguments()[0].function_space())
          test_shape = test.ufl_element().value_shape()
          dummy_rhs = inner(test, Constant(0.0 if len(test_shape) == 0 else numpy.zeros(test_shape, dtype = numpy.float64))) * dx
          J_mat, b_bc = assemble_system(J, dummy_rhs, self._bcs, form_compiler_parameters = self._form_compiler_parameters)
          if not deps is None:
            alias_clear(J)

          # Assemble the RHS with pre-assembly
          b = self._pre_assembled_rhs(deps, b_bc = b_bc)
        else:
          # Case 4: Linear, Jacobian not cached, without pre-assembly
          
          # Assemble the Jacobian and RHS
          if deps is None:
            J, rhs = self._J, self._rhs
          else:
            if self._forward_eq is None:
              self._forward_eq = None, alias_form(self._J, eq_deps), alias_form(self._rhs, eq_deps)
            _, J, rhs = self._forward_eq
            alias_replace(J, deps)
            alias_replace(rhs, deps)
          J_mat, b = assemble_system(J, rhs, self._bcs, form_compiler_parameters = self._form_compiler_parameters)
          if not deps is None:
            alias_clear(J)
            alias_clear(rhs)
        
        # Construct the linear solver
        J_solver = _linear_solver(J_mat, self._linear_solver_parameters)
        
#      J_mat_debug, b_debug = assemble_system(self._J if deps is None else ufl.replace(self._J, OrderedDict(zip(eq_deps, deps))),
#                                             self._rhs if deps is None else ufl.replace(self._rhs, OrderedDict(zip(eq_deps, deps))),
#                                             self._bcs,
#                                             form_compiler_parameters = self._form_compiler_parameters)
#      assert((J_mat - J_mat_debug).norm("linf") == 0.0)
#      assert((b - b_debug).norm("linf") <= 1.0e-15 * b.norm("linf"))
        
      J_solver.solve(x.vector(), b)
    else:
      # Case 5: Non-linear, Jacobian not cached, without pre-assembly
      if deps is None:
        lhs, J, rhs = self._lhs, self._J, self._rhs
      else:    
        if self._forward_eq is None:
          self._forward_eq = (alias_form(self._lhs, eq_deps),
                              alias_form(self._J, eq_deps),
                              (0 if self._rhs == 0 else alias_form(self._rhs, eq_deps)))
        lhs, J, rhs = self._forward_eq
        alias_replace(lhs, deps)
        alias_replace(J, deps)
        if rhs != 0:
          alias_replace(rhs, deps)
      solve(lhs == rhs, x, self._bcs, J = J,
        form_compiler_parameters = self._form_compiler_parameters, solver_parameters = self._solver_parameters)
      if not deps is None:
        alias_clear(lhs)
        alias_clear(J)
        if rhs != 0:
          alias_clear(rhs)
    
  def reset_forward_solve(self):
    self._forward_eq = None
    self._forward_J_mat = CacheIndex()
    self._forward_J_solver = CacheIndex()
    self._forward_b_pa = None
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    # Similar to 'RHS.derivative_action' and 'RHS.second_derivative_action' in
    # dolfin-adjoint file dolfin_adjoint/adjrhs.py (see e.g. dolfin-adjoint
    # version 2017.1.0)
    # Code first added to JRM personal repository 2016-05-22
    # Code first added to dolfin_adjoint_custom repository 2016-06-02
    # Re-written 2018-01-28
    
    eq_deps = self.dependencies()
    if dep_index < 0 or dep_index >= len(eq_deps):
      return None
    elif dep_index == 0:
      return adj_x
    
    if dep_index in self._derivative_mats:
      mat_cache = self._derivative_mats[dep_index]
      if mat_cache is None:
        return None
      elif isinstance(mat_cache, CacheIndex):
        if not mat_cache.index() is None:
          mat, _ = assembly_cache()[mat_cache]
          return mat * adj_x.vector()
        #else:
        #  Cache entry cleared
      elif self._defer_adjoint_assembly:
        #assert(isinstance(mat_cache, ufl.classes.Form)
        return action(ufl.replace(mat_cache, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps))), adj_x)
      else:
        #assert(isinstance(mat_cache, ufl.classes.Form)
        return alias_assemble(mat_cache, list(nl_deps) + [adj_x],
          form_compiler_parameters = self._form_compiler_parameters)

    dep = eq_deps[dep_index]
    dF = ufl.algorithms.expand_derivatives(derivative(self._F, dep, du = TrialFunction(dep.function_space())))
    if dF.empty():
      self._derivative_mats[dep_index] = None
      return None
    dF = adjoint(dF)
    
    if self._pre_assemble and is_static(dF):
      dF = ufl.replace(dF, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
      self._derivative_mats[dep_index], (mat, _) = assembly_cache().assemble(dF, form_compiler_parameters = self._form_compiler_parameters)
      return mat * adj_x.vector()
    elif self._defer_adjoint_assembly:
      self._derivative_mats[dep_index] = dF
      dF = ufl.replace(dF, OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
      return action(dF, adj_x)
    else:
      self._derivative_mats[dep_index] = dF = \
        alias_form(action(dF, adj_x), list(self.nonlinear_dependencies()) + [adj_x])
      return alias_assemble(dF, list(nl_deps) + [adj_x],
        form_compiler_parameters = self._form_compiler_parameters)

  def reset_adjoint_derivative_action(self):
    self._derivative_mats = OrderedDict()

  def adjoint_jacobian_solve(self, nl_deps, b):
    if self._cache_jacobian:
      if self._adjoint_J_solver.index() is None:
        J = ufl.replace(adjoint(self._J), OrderedDict(zip(self.nonlinear_dependencies(), nl_deps)))
        _, (J_mat, _) = assembly_cache().assemble(J, bcs = self._hbcs, form_compiler_parameters = self._form_compiler_parameters)
        self._adjoint_J_solver, J_solver = linear_solver_cache().linear_solver(J, J_mat, bcs = self._hbcs, linear_solver_parameters = self._linear_solver_parameters)
      else:
        J_solver = linear_solver_cache()[self._adjoint_J_solver]
    else:
      if self._adjoint_J is None:
        self._adjoint_J = alias_form(adjoint(self._J), self.nonlinear_dependencies())
      alias_replace(self._adjoint_J, nl_deps)
      if len(self._hbcs) > 0:
        test = TestFunction(self._adjoint_J.arguments()[0].function_space())
        test_shape = test.ufl_element().value_shape()
        dummy_rhs = inner(test, Constant(0.0 if len(test_shape) == 0 else numpy.zeros(test_shape, dtype = numpy.float64))) * dx
        J_mat, _ = assemble_system(self._adjoint_J, dummy_rhs, self._hbcs, form_compiler_parameters = self._form_compiler_parameters)
      else:
        J_mat = assemble(self._adjoint_J, form_compiler_parameters = self._form_compiler_parameters)
      alias_clear(self._adjoint_J)
      
      J_solver = _linear_solver(J_mat, self._linear_solver_parameters)
      
    for bc in self._hbcs:
      bc.apply(b.vector())
    adj_x = function_new(b)
    J_solver.solve(adj_x.vector(), b.vector())
    
    return adj_x
    
  def reset_adjoint_jacobian_solve(self):
    self._adjoint_J = None
    self._adjoint_J_solver = CacheIndex()
  
  def tangent_linear(self, M, dM, tlm_map):
    x = self.x()
    if x in M:
      raise EquationException("Invalid tangent-linear parameter")
  
    tlm_rhs = ufl.classes.Zero()
    for m, dm in zip(M, dM):
      tlm_rhs -= derivative(self._F, m, du = dm)
      
    for dep in self.dependencies():
      if dep != x and not dep in M:
        tau_dep = tlm_map[dep]
        if not tau_dep is None:
          tlm_rhs -= derivative(self._F, dep, du = tau_dep)
    
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
        pre_assemble = self._pre_assemble,
        defer_adjoint_assembly = self._defer_adjoint_assembly)
        
class DirichletBCSolver(Equation):
  def __init__(self, y, x, forward_domain, *bc_args, **bc_kwargs):
    bc_kwargs = copy.copy(bc_kwargs)
    adjoint_domain = bc_kwargs.pop("adjoint_domain", forward_domain)
    allow_extrapolation = bc_kwargs.pop("allow_extrapolation", True)
  
    Equation.__init__(self, x, [x, y], nl_deps = [])
    self._forward_domain = forward_domain
    self._adjoint_domain = adjoint_domain
    self._bc_args = bc_args
    self._bc_kwargs = bc_kwargs
    self._allow_extrapolation = allow_extrapolation

  def forward_solve(self, x, deps = None):
    _, y = self.dependencies() if deps is None else deps
    if self._allow_extrapolation:
      allow_extrapolation = y.get_allow_extrapolation()
      y.set_allow_extrapolation(True)
    function_zero(x)
    DirichletBC(x.function_space(), y, self._forward_domain, *self._bc_args, **self._bc_kwargs).apply(x.vector())
    if self._allow_extrapolation:
      y.set_allow_extrapolation(allow_extrapolation)
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    elif dep_index == 1:
      _, y = self.dependencies()
      F = function_new(y)
      if self._allow_extrapolation:
        allow_extrapolation = adj_x.get_allow_extrapolation()
        adj_x.set_allow_extrapolation(True)
      DirichletBC(y.function_space(), adj_x, self._adjoint_domain,
                  *self._bc_args, **self._bc_kwargs).apply(F.vector())
      if self._allow_extrapolation:
        adj_x.set_allow_extrapolation(allow_extrapolation)
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
        adjoint_domain = self._adjoint_domain, allow_extrapolation = self._allow_extrapolation,
        *self._bc_args, **self._bc_kwargs)      
