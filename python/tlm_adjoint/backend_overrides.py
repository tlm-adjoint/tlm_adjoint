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

from .equations import AssignmentSolver, EquationSolver, ProjectionSolver
from .tlm_adjoint import annotation_enabled, tlm_enabled

from collections import OrderedDict
import copy
import ufl

__all__ = \
  [
    "OverrideException",
    
    "KrylovSolver",
    "LUSolver",
    "assemble",
    "assemble_system",
    "project",
    "solve"
  ]

class OverrideException(Exception):
  pass

def parameters_dict_equal(parameters_a, parameters_b):
  for key_a in parameters_a:
    value_a = parameters_a[key_a]
    if not key_a in parameters_b:
      return False
    value_b = parameters_b[key_a]
    if isinstance(value_a, (Parameters, dict)):
      if not isinstance(value_b, (Parameters, dict)):
        return False
      elif not parameters_dict_equal(value_a, value_b):
        return False
    elif value_a != value_b:
      return False
  for key_b in parameters_b:
    if not key_b in parameters_a:
      return False
  return True

# Aim for compatibility with FEniCS 2018.1.0 API

def assemble(form, tensor = None, form_compiler_parameters = None, add_values = False, *args, **kwargs):
  b = backend_assemble(form, tensor = tensor,
    form_compiler_parameters = form_compiler_parameters,
    add_values = add_values, *args, **kwargs)
  if tensor is None:
    tensor = b
      
  if not isinstance(tensor, float):
    form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
    if not form_compiler_parameters is None:
      update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_
  
    if add_values and hasattr(tensor, "_tlm_adjoint__form"):
      if tensor._tlm_adjoint__bcs != []:
        raise OverrideException("Non-matching boundary conditions")
      elif not parameters_dict_equal(tensor._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      tensor._tlm_adjoint__form += form
    else:
      tensor._tlm_adjoint__form = form
      tensor._tlm_adjoint__bcs = []
      tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters
    
  return tensor
  
def assemble_system(A_form, b_form, bcs = None, x0 = None,
  form_compiler_parameters = None, add_values = False,
  finalize_tensor = True, keep_diagonal = False, A_tensor = None, b_tensor = None, *args, **kwargs):
  if not x0 is None:
    raise OverrideException("Non-linear boundary condition case not supported")
    
  A, b = backend_assemble_system(A_form, b_form, bcs = bcs, x0 = x0,
    form_compiler_parameters = form_compiler_parameters,
    add_values = add_values, finalize_tensor = finalize_tensor,
    keep_diagonal = keep_diagonal, A_tensor = A_tensor, b_tensor = b_tensor,
    *args, **kwargs)
  if A_tensor is None:
    A_tensor = A
  if b_tensor is None:
    b_tensor = b
  if bcs is None:
    bcs = []
  elif isinstance(bcs, backend_DirichletBC):
    bcs = [bcs]

  form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
  if not form_compiler_parameters is None:
    update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
  form_compiler_parameters = form_compiler_parameters_
    
  if add_values and hasattr(A_tensor, "_tlm_adjoint__form"):
    if A_tensor._tlm_adjoint__bcs != bcs:
      raise OverrideException("Non-matching boundary conditions")
    elif not parameters_dict_equal(A_tensor._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
      raise OverrideException("Non-matching form compiler parameters")
    A_tensor._tlm_adjoint__form += A_form
  else:
    A_tensor._tlm_adjoint__form = A_form
    A_tensor._tlm_adjoint__bcs = copy.copy(bcs)
    A_tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters
  
  if add_values and hasattr(b_tensor, "_tlm_adjoint__form"):
    if b_tensor._tlm_adjoint__bcs != bcs:
      raise OverrideException("Non-matching boundary conditions")
    elif not parameters_dict_equal(b_tensor._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
      raise OverrideException("Non-matching form compiler parameters")
    b_tensor._tlm_adjoint__form += b_form
  else:
    b_tensor._tlm_adjoint__form = b_form
    b_tensor._tlm_adjoint__bcs = copy.copy(bcs)
    b_tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters
  
  return A_tensor, b_tensor

def solve(*args, **kwargs):
  kwargs = copy.copy(kwargs)
  annotate = kwargs.pop("annotate", None)
  tlm = kwargs.pop("tlm", None)
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    if isinstance(args[0], ufl.classes.Equation):
      eq, x, bcs, J, tol, M, form_compiler_parameters, solver_parameters = extract_args(*args, **kwargs)
      if not J is None:
        raise OverrideException("Custom Jacobians not supported")
      if not tol is None or not M is None:
        raise OverrideException("Adaptive solves not supported")
      lhs, rhs = eq.lhs, eq.rhs
      if isinstance(lhs, ufl.classes.Form) and isinstance(rhs, ufl.classes.Form) and \
        (x in lhs.coefficients() or x in rhs.coefficients()):
        F = function_new(x)
        AssignmentSolver(x, F).solve(annotate = annotate, replace = True, tlm = tlm)
        lhs = ufl.replace(lhs, OrderedDict([(x, F)]))
        rhs = ufl.replace(rhs, OrderedDict([(x, F)]))
        eq = lhs == rhs
      EquationSolver(eq, x, bcs,
        form_compiler_parameters = form_compiler_parameters,
        solver_parameters = solver_parameters, cache_jacobian = False,
        cache_rhs_assembly = False).solve(annotate = annotate, replace = True, tlm = tlm)
    else:
      raise OverrideException("Linear system solves not supported")
  else:
    return backend_solve(*args, **kwargs)

def project(v, V = None, bcs = None, mesh = None, function = None,
  solver_type = "lu", preconditioner_type = "default",
  form_compiler_parameters = None, annotate = None, tlm = None):
  if bcs is None:
    bcs = []
  elif isinstance(bcs, backend_DirichletBC):
    bcs = [bcs]
      
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    if V is None:
      raise OverrideException("Function space required")
    if function is None:
      x = Function(V)
    else:
      x = function
    ProjectionSolver(v, x, bcs,
      solver_parameters = {"linear_solver":solver_type, "preconditioner":preconditioner_type},
      form_compiler_parameters = {} if form_compiler_parameters is None else form_compiler_parameters,
      cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, replace = True, tlm = tlm)
      # ?? Other solver parameters ?
    return x
  else:
    return backend_project(v, V = V, bcs = bcs, mesh = mesh, function = function,
      solver_type = solver_type, preconditioner_type = preconditioner_type,
      form_compiler_parameters = form_compiler_parameters)

_orig_DirichletBC_apply = backend_DirichletBC.apply
def _DirichletBC_apply(self, *args):
  _orig_DirichletBC_apply(self, *args)
  if (len(args) > 1 and not isinstance(args[0], backend_Matrix)) or len(args) > 2:
    return

  if isinstance(args[0], backend_Matrix):
    A = args[0]
    if len(args) > 1:
      b = args[1]
    else:
      b = None
  else:
    A = None
    b = args[0]

  if not A is None and hasattr(A, "_tlm_adjoint__bcs") and not self in A._tlm_adjoint__bcs:
    A._tlm_adjoint__bcs.append(self)
  if not b is None and hasattr(b, "_tlm_adjoint__bcs") and not self in b._tlm_adjoint__bcs:
    b._tlm_adjoint__bcs.append(self)
backend_DirichletBC.apply = _DirichletBC_apply

_orig_Function_assign = backend_Function.assign
def _Function_assign(self, rhs, annotate = None, tlm = None):
  return_value = _orig_Function_assign(self, rhs)
  if not is_function(rhs):
    return
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    eq = AssignmentSolver(rhs, self)
    eq._post_process(annotate = annotate, replace = True, tlm = tlm)
  return return_value
backend_Function.assign = _Function_assign

_orig_Function_vector = backend_Function.vector
def _Function_vector(self):
  return_value = _orig_Function_vector(self)
  return_value._tlm_adjoint__function = self
  return return_value
backend_Function.vector = _Function_vector

_orig_Matrix_mul = backend_Matrix.__mul__
def _Matrix_mul(self, other):
  return_value = _orig_Matrix_mul(self, other)
  if hasattr(self, "_tlm_adjoint__form") and hasattr(other, "_tlm_adjoint__function") and len(self._tlm_adjoint__bcs) == 0:
    return_value._tlm_adjoint__form = ufl.action(self._tlm_adjoint__form, coefficient = other._tlm_adjoint__function)
    return_value._tlm_adjoint__bcs = []
    return_value._tlm_adjoint__form_compiler_parameters = self._tlm_adjoint__form_compiler_parameters
  return return_value
backend_Matrix.__mul__ = _Matrix_mul

class LUSolver(backend_LUSolver):
  def __init__(self, *args):
    backend_LUSolver.__init__(self, *args)
    if len(args) >= 1 and isinstance(args[0], backend_Matrix):
      self._tlm_adjoint__A = args[0]
      self._tlm_adjoint__linear_solver = args[1] if len(args) >= 2 else "default"
    elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
      self._tlm_adjoint__A = args[1]
      self._tlm_adjoint__linear_solver = args[2] if len(args) >= 3 else "default"
    elif len(args) >= 1 and isinstance(args[0], str):
      self._tlm_adjoint__linear_solver = args[0]  # FEniCS < 2018.1.0 compatibility
    else:
      self._tlm_adjoint__linear_solver = args[1] if len(args) >= 2 else "default"
      
  def set_operator(self, A):
    backend_LUSolver.set_operator(self, A)
    self._tlm_adjoint__A = A

  def solve(self, *args, annotate = None, tlm = None):
    backend_LUSolver.solve(self, *args)
    
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      if isinstance(args[0], backend_Matrix):
        A, x, b = args
        self._tlm_adjoint__A = A
      else:
        A = self._tlm_adjoint__A
        x, b = args
        
      bcs = A._tlm_adjoint__bcs
      if bcs != b._tlm_adjoint__bcs:
        raise OverrideException("Non-matching boundary conditions")
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      eq = EquationSolver(A._tlm_adjoint__form == b._tlm_adjoint__form, x._tlm_adjoint__function,
        bcs, solver_parameters = {"linear_solver":self._tlm_adjoint__linear_solver, "lu_solver":self.parameters},
        form_compiler_parameters = form_compiler_parameters, cache_jacobian = False, cache_rhs_assembly = False)
      eq._post_process(annotate = annotate, replace = True, tlm = tlm)

class KrylovSolver(backend_KrylovSolver):
  def __init__(self, *args):
    backend_KrylovSolver.__init__(self, *args)
    if len(args) >= 1 and isinstance(args[0], backend_Matrix):
      self._tlm_adjoint__A = args[0]
      self._tlm_adjoint__linear_solver = args[1] if len(args) >= 2 else "default"
      self._tlm_adjoint__preconditioner = args[2] if len(args) >= 3 else "default"
    elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
      self._tlm_adjoint__A = args[1]
      self._tlm_adjoint__linear_solver = args[2] if len(args) >= 3 else "default"
      self._tlm_adjoint__preconditioner = args[3] if len(args) >= 4 else "default"
    elif len(args) >= 1 and isinstance(args[0], str):
      self._tlm_adjoint__linear_solver = args[0]
      self._tlm_adjoint__preconditioner = args[1] if len(args) >= 2 else "default"
    else:
      self._tlm_adjoint__linear_solver = args[1] if len(args) >= 2 else "default"
      self._tlm_adjoint__preconditioner = args[2] if len(args) >= 3 else "default"
      
  def set_operator(self, A):
    backend_KrylovSolver.set_operator(self, A)
    self._tlm_adjoint__A = A

  def set_operators(self, *args, **kwargs):
    raise OverrideException("Preconditioner matrices not supported")

  def solve(self, *args, annotate = None, tlm = None):
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      if isinstance(args[0], backend_Matrix):
        A, x, b = args
        self._tlm_adjoint__A = None
      else:
        A = self._tlm_adjoint__A
        x, b = args
        
      bcs = A._tlm_adjoint__bcs
      if bcs != b._tlm_adjoint__bcs:
        raise OverrideException("Non-matching boundary conditions")
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      eq = EquationSolver(A._tlm_adjoint__form == b._tlm_adjoint__form, x._tlm_adjoint__function,
        bcs, solver_parameters = {"linear_solver":self._tlm_adjoint__linear_solver, "preconditioner":self._tlm_adjoint__preconditioner, "krylov_solver":self.parameters},
        form_compiler_parameters = form_compiler_parameters, cache_jacobian = False, cache_rhs_assembly = False)

      eq._pre_process(annotate = annotate)
      backend_KrylovSolver.solve(self, *args)
      eq._post_process(annotate = annotate, replace = True, tlm = tlm)
    else:
      backend_KrylovSolver.solve(self, *args)
