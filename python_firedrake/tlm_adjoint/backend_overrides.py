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
    
    "LinearSolver",
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
# a7e3c6e16728b035b95125321824ca3cc9e40a9f

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
        tensor._tlm_adjoint__bcs = copy.copy(bcs)
    tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters
  
  return tensor

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
      if not J is None or not Jp is None:
        raise OverrideException("Custom Jacobians not supported")
      if not M is None:
        raise OverrideException("Adaptive solves not supported")
      if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
        raise OverrideException("Null spaces not supported")
      if not options_prefix is None:
        raise OverrideException("Options prefixes not supported")
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

def project(v, V, bcs = None, mesh = None, solver_parameters = None,
  form_compiler_parameters = None, name = None, annotate = None, tlm = None):
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    if not mesh is None:
      raise OverrideException("mesh argument not supported")
    x = Function(V, name = name)
    ProjectionSolver(v, x, [] if bcs is None else bcs,
      solver_parameters = {} if solver_parameters is None else solver_parameters,
      form_compiler_parameters = {} if form_compiler_parameters is None else form_compiler_parameters,
      cache_jacobian = False, cache_rhs_assembly = False).solve(annotate = annotate, replace = True, tlm = tlm)
    return x
  else:
    return backend_project(v, V, bcs = bcs, mesh = mesh,
      solver_parameters = solver_parameters,
      form_compiler_parameters = form_compiler_parameters, name = name)

_orig_DirichletBC_apply = backend_DirichletBC.apply
def _DirichletBC_apply(self, r, u = None):
  _orig_DirichletBC_apply(self, r, u = u)
  if not isinstance(r, backend_Matrix):
    return

  if hasattr(r, "_tlm_adjoint__bcs") and not self in r._tlm_adjoint__bcs:
    r._tlm_adjoint__bcs.append(self)
backend_DirichletBC.apply = _DirichletBC_apply

_orig_Function_assign = backend_Function.assign
def _Function_assign(self, expr, subset = None, annotate = None, tlm = None):
  return_value = _orig_Function_assign(self, expr, subset = subset)
  if not is_function(expr) or not subset is None:
    return
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    eq = AssignmentSolver(expr, self)
    eq._post_process(annotate = annotate, replace = True, tlm = tlm)
  return return_value
backend_Function.assign = _Function_assign

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
      raise OverrideException("Preconditioner matrices not supported")
    if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
      raise OverrideException("Null spaces not supported")
    if not options_prefix is None:
      raise OverrideException("Options prefixes not supported")

    backend_LinearSolver.__init__(self, A, P = P,
      solver_parameters = solver_parameters, nullspace = nullspace,
      transpose_nullspace = transpose_nullspace,
      near_nullspace = near_nullspace, options_prefix = options_prefix)
      
    if solver_parameters is None:
      solver_parameters = {}
    else:
      solver_parameters = copy_parameters_dict(solver_parameters)
      
    self._tlm_adjoint__A = A
    self._tlm_adjoint__solver_parameters = solver_parameters

  def solve(self, x, b, annotate = None, tlm = None):
    if annotate is None:
      annotate = annotation_enabled()
    if tlm is None:
      tlm = tlm_enabled()
    if annotate or tlm:
      A = self._tlm_adjoint__A
      if not is_function(x):
        x = x._tlm_adjoint__function
      if not is_function(b):
        b = b._tlm_adjoint__function
      bcs = A._tlm_adjoint__bcs
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise OverrideException("Non-matching form compiler parameters")
      solver_parameters = self._tlm_adjoint__solver_parameters
      
      eq = EquationSolver(A._tlm_adjoint__form == b._tlm_adjoint__form, x,
        bcs, solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters,
        cache_jacobian = False, cache_rhs_assembly = False)

      eq._pre_process(annotate = annotate)
      backend_LinearSolver.solve(self, x, b)
      eq._post_process(annotate = annotate, replace = True, tlm = tlm)
    else:
      backend_LinearSolver.solve(self, x, b)
