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
from .backend_interface import *

from .equations import AssignmentSolver, EquationSolver, ProjectionSolver
from .tlm_adjoint import annotation_enabled, tlm_enabled

import copy
import ufl

__all__ = \
  [
    "OverrideException",
    
    "LinearSolver",
    "LinearVariationalSolver",
    "NonlinearVariationalProblem",
    "NonlinearVariationalSolver",
    "assemble",
    "project",
    "solve"
  ]

class OverrideException(Exception):
  pass

def parameters_dict_equal(parameters_a, parameters_b):
  for key_a, value_a in parameters_a.items():
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

# Aim for compatibility with Firedrake API, git master revision
# 1b6306099f81b89e7eda07209d1a1b99447e063b

def assemble(f, tensor = None, bcs = None, form_compiler_parameters = None, inverse = False, *args, **kwargs):
  if inverse:
    raise OverrideException("Local inverses not supported")

  b = backend_assemble(f, tensor = tensor, bcs = bcs,
    form_compiler_parameters = form_compiler_parameters, inverse = inverse,
    *args, **kwargs)
  if tensor is None:
    tensor = b
      
  if not isinstance(b, float):
    form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])
    if not form_compiler_parameters is None:
      update_parameters_dict(form_compiler_parameters_, form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_
    
    tensor._tlm_adjoint__form = f
    if isinstance(b, backend_Matrix):
      if bcs is None:
        tensor._tlm_adjoint__bcs = []
      elif isinstance(bcs, backend_DirichletBC):
        tensor._tlm_adjoint__bcs = [bcs]
      else:
        tensor._tlm_adjoint__bcs = list(bcs)
    tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters
  
  return tensor
  
def extract_args_linear_solve(A, x, b, bcs = None, solver_parameters = {}):
  return A, x, b, bcs, solver_parameters

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
      eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
        nullspace, transpose_nullspace, near_nullspace, options_prefix = \
        extract_args(*args, **kwargs)
      if not Jp is None:
        raise OverrideException("Preconditioners not supported")
      if not M is None:
        raise OverrideException("Adaptive solves not supported")
      if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
        raise OverrideException("Null spaces not supported")
      solver_parameters = copy.copy(solver_parameters)
      solver_parameters["_tlm_adjoint__options_prefix"] = options_prefix
      lhs, rhs = eq.lhs, eq.rhs
      if isinstance(lhs, ufl.classes.Form) and isinstance(rhs, ufl.classes.Form) and \
        (x in lhs.coefficients() or x in rhs.coefficients()):
        x_old = function_new(x, name = "x_old")
        AssignmentSolver(x, F).solve(annotate = annotate, tlm = tlm)
        lhs = ufl.replace(lhs, {x:x_old})
        rhs = ufl.replace(rhs, {x:x_old})
        eq = lhs == rhs
      EquationSolver(eq, x, bcs, J = J,
        form_compiler_parameters = form_compiler_parameters,
        solver_parameters = solver_parameters, cache_jacobian = False,
        cache_rhs_assembly = False).solve(annotate = annotate, tlm = tlm)
    else:
      A, x, b, bcs, solver_parameters = extract_args_linear_solve(*args, **kwargs)
      
      if bcs is None:
        bcs = A._tlm_adjoint__bcs
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      
      A = A._tlm_adjoint__form
      x = x._tlm_adjoint__function
      b = b._tlm_adjoint__form
      A_x_dep = x in ufl.algorithms.extract_coefficients(A)
      b_x_dep = x in ufl.algorithms.extract_coefficients(b)
      if A_x_dep or b_x_dep:
        x_old = function_new(x, name = "x_old")
        AssignmentSolver(x, x_old).solve(annotate = annotate, tlm = tlm)
        if A_x_dep: A = ufl.replace(A, {x:x_old})
        if b_x_dep: b = ufl.replace(b, {x:x_old})
        
      EquationSolver(A == b, x,
        bcs, solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters,
        cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, tlm = tlm)
  else:
    backend_solve(*args, **kwargs)

def project(v, V, bcs = None, solver_parameters = None,
  form_compiler_parameters = None, use_slate_for_inverse = True, name = None,
  annotate = None, tlm = None):
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    if use_slate_for_inverse:
      raise OverrideException("use_slate_for_inverse argument not supported")
    if is_function(V):
      x = V
    else:
      x = Function(V, name = name)
    ProjectionSolver(v, x, [] if bcs is None else bcs,
      solver_parameters = {} if solver_parameters is None else solver_parameters,
      form_compiler_parameters = {} if form_compiler_parameters is None else form_compiler_parameters,
      cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, tlm = tlm)
    return x
  else:
    return backend_project(v, V, bcs = bcs,
      solver_parameters = solver_parameters,
      form_compiler_parameters = form_compiler_parameters,
      use_slate_for_inverse = use_slate_for_inverse, name = name)

_orig_Function_assign = backend_Function.assign
def _Function_assign(self, expr, subset = None, annotate = None, tlm = None):
  return_value = _orig_Function_assign(self, expr, subset = subset)
  if not is_function(expr) or not subset is None:
    return return_value
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    AssignmentSolver(expr, self).solve(annotate = annotate, tlm = tlm)
  return return_value
backend_Function.assign = _Function_assign

_orig_Function_project = backend_Function.project
def _Function_project(self, b, *args, **kwargs):
  return project(b, self, *args, **kwargs)
backend_Function.project = _Function_project

_orig_Function_vector = backend_Function.vector
def _Function_vector(self):
  return_value = _orig_Function_vector(self)
  return_value._tlm_adjoint__function = self
  return return_value
backend_Function.vector = _Function_vector

class LinearSolver(backend_LinearSolver):
  def __init__(self, A, P = None, solver_parameters = None, nullspace = None,
    transpose_nullspace = None, near_nullspace = None, options_prefix = None):
    if not P is None:
      raise OverrideException("Preconditioners not supported")
    if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
      raise OverrideException("Null spaces not supported")

    backend_LinearSolver.__init__(self, A, P = P,
      solver_parameters = solver_parameters, nullspace = nullspace,
      transpose_nullspace = transpose_nullspace,
      near_nullspace = near_nullspace, options_prefix = options_prefix)
      
    if solver_parameters is None:
      solver_parameters = {}
    else:
      solver_parameters = copy_parameters_dict(solver_parameters)
      
    self.__A = A
    self.__solver_parameters = solver_parameters

  def solve(self, x, b, annotate = None, tlm = None):
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      A = self.__A
      if not is_function(x):
        x = x._tlm_adjoint__function
      if not is_function(b):
        b = b._tlm_adjoint__function
      bcs = A._tlm_adjoint__bcs
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      solver_parameters = self.__solver_parameters
      solver_parameters["_tlm_adjoint__options_prefix"] = self.options_prefix  # Copy not required here
      
      eq = EquationSolver(A._tlm_adjoint__form == b._tlm_adjoint__form, x,
        bcs, solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters,
        cache_jacobian = False, cache_rhs_assembly = False)
      
      del(solver_parameters["_tlm_adjoint__options_prefix"])

      eq._pre_process(annotate = annotate)
      backend_LinearSolver.solve(self, x, b)
      eq._post_process(annotate = annotate, tlm = tlm)
    else:
      backend_LinearSolver.solve(self, x, b)

class LinearVariationalSolver(backend_LinearVariationalSolver):
  def __init__(self, *args, **kwargs):
    problem, = args
    if "nullspace" in kwargs or "transpose_nullspace" in kwargs:
      raise OverrideException("Null spaces not supported")
    if "appctx" in kwargs:
      raise OverrideException("Preconditioners not supported")
  
    backend_LinearVariationalSolver.__init__(self, *args, **kwargs)
    self.__problem = problem
  
  def set_transfer_operators(self, *args, **kwargs):
    raise OverrideException("Transfer operators not supported")
  
  def solve(self, bounds = None, annotate = None, tlm = None):
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      if not bounds is None:
        raise OverrideException("Bounds not supported")        
      if not self.__problem.Jp is None:
        raise OverrideException("Preconditioners not supported")
      
      x = self.__problem.u
      L = ufl.rhs(ufl.replace(self.__problem.F, {x:TrialFunction(x.function_space())}))
      form_compiler_parameters = self.__problem.form_compiler_parameters
      if form_compiler_parameters is None: form_compiler_parameters = {}
      solver_parameters = copy.copy(self.parameters)
      solver_parameters["_tlm_adjoint__options_prefix"] = self.options_prefix
      
      EquationSolver(self.__problem.J == L, x, self.__problem.bcs,
        solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters,
        cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, tlm = tlm)
    else:
      backend_LinearVariationalSolver.solve(self, bounds = bounds)

class NonlinearVariationalProblem(backend_NonlinearVariationalProblem):
  def __init__(self, F, u, bcs = None, J = None, Jp = None,
    form_compiler_parameters = None, is_linear = False):
    if not Jp is None:
      raise OverrideException("Preconditioners not supported")
    
    backend_NonlinearVariationalProblem.__init__(self, F, u, bcs = bcs, J = J,
      Jp = Jp, form_compiler_parameters = form_compiler_parameters,
      is_linear = is_linear)

class NonlinearVariationalSolver(backend_NonlinearVariationalSolver):
  def __init__(self, *args, **kwargs):
    problem, = args
    if "nullspace" in kwargs or "transpose_nullspace" in kwargs or "near_nullspace" in kwargs:
      raise OverrideException("Null spaces not supported")
    if "appctx" in kwargs:
      raise OverrideException("Preconditioners not supported")
    if "pre_jacobian_callback" in kwargs or "pre_function_callback" in kwargs:
      raise OverrideException("Callbacks not supported")
  
    backend_NonlinearVariationalSolver.__init__(self, *args, **kwargs)
    self.__problem = problem
  
  def set_transfer_operators(self, *args, **kwargs):
    raise OverrideException("Transfer operators not supported")
  
  def solve(self, bounds = None, annotate = None, tlm = None):
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      if not bounds is None:
        raise OverrideException("Bounds not supported")        
      if not self.__problem.Jp is None:
        raise OverrideException("Preconditioners not supported")
      
      form_compiler_parameters = self.__problem.form_compiler_parameters
      if form_compiler_parameters is None: form_compiler_parameters = {}
      solver_parameters = copy.copy(self.parameters)
      solver_parameters["_tlm_adjoint__options_prefix"] = self.options_prefix
      
      EquationSolver(self.__problem.F == 0, self.__problem.u,
        self.__problem.bcs, J = self.__problem.J,
        solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters,
        cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, tlm = tlm)
    else:
      backend_NonlinearVariationalSolver.solve(self, bounds = bounds)
