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
from .backend_interface import copy_parameters_dict

from .equations import AssignmentSolver, EquationSolver
from .tlm_adjoint import annotation_enabled, tlm_enabled

import copy

__all__ = \
  [
    "LinearSolver",
    "assemble",
    "project",
    "solve"
  ]

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

# Following Firedrake API, git master revision
# efea95b923b9f3319204c6e22f7aaabb792c9abd

def assemble(f, tensor = None, bcs = None, form_compiler_parameters = None, inverse = False, *args, **kwargs):
  if inverse:
    raise ManagerException("Local inverses not supported")

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
    tensor._tlm_adjoint__bcs = []
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
    eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
      nullspace, transpose_nullspace, near_nullspace, options_prefix = \
      extract_args(*args, **kwargs)
    if not J is None or not Jp is None:
      raise ManagerException("Custom Jacobians not supported")
    if not M is None:
      raise ManagerException("Adaptive solves not supported")
    if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
      raise ManagerException("Nullspaces not supported")
    if not options_prefix is None:
      raise ManagerException("Options prefixes not supported")
    eq = EquationSolver(eq, x, bcs,
      form_compiler_parameters = form_compiler_parameters,
      solver_parameters = solver_parameters)
    eq._pre_annotate(annotate = annotate)
    return_value = backend_solve(*args, **kwargs)
    eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
    return return_value
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
      raise ManagerException("'mesh' argument not supported")
    if not form_compiler_parameters is None:
      # Firedrake documentation indicates that this is ignored
      raise ManagerException("'form_compiler_parameters' argument not supported")
    # Since a zero initial guess is used, _pre_annotate is not needed
    return_value = backend_project(v, V, bcs = bcs, mesh = mesh,
      solver_parameters = solver_parameters,
      form_compiler_parameters = form_compiler_parameters, name = name)
    test, trial = TestFunction(V), TrialFunction(V)
    eq = EquationSolver(inner(test, trial) * dx == inner(test, v) * dx,
      return_value, [] if bcs is None else bcs,
      solver_parameters = {} if solver_parameters is None else solver_parameters)
    eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
    return return_value
  else:
    return backend_project(v, V, bcs = bcs, mesh = mesh,
      solver_parameters = solver_parameters,
      form_compiler_parameters = form_compiler_parameters, name = name)

_orig_DirichletBC_apply = DirichletBC.apply
def _DirichletBC_apply(self, r, u = None):
  _orig_DirichletBC_apply(self, r, u = u)
  
  if hasattr(r, "_tlm_adjoint__bcs"):
    r._tlm_adjoint__bcs.append(self)
  if not u is None and hasattr(u, "_tlm_adjoint__bcs"):
    u._tlm_adjoint__bcs.append(self)
DirichletBC.apply = _DirichletBC_apply

_orig_Function_assign = backend_Function.assign
def _Function_assign(self, expr, subset = None, annotate = None, tlm = None):
  return_value = _orig_Function_assign(self, expr, subset = subset)
  if not isinstance(expr, backend_Function) or not subset is None:
    # Only assignment to a Function annotated
    return
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    eq = AssignmentSolver(expr, self)
    eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
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
      raise ManagerException("Preconditioner matrices not supported")
    if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
      raise ManagerException("Nullspaces not supported")
    if not options_prefix is None:
      raise ManagerException("Options prefixes not supported")
  
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
      if not isinstance(x, backend_Function):
        x = x._tlm_adjoint__function
      bcs = A._tlm_adjoint__bcs
      if bcs != b._tlm_adjoint__bcs:
        raise ManagerException("Non-matching boundary conditions")
      form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
      if not parameters_dict_equal(b._tlm_adjoint__form_compiler_parameters, form_compiler_parameters):
        raise ManagerException("Non-matching form compiler parameters")
      solver_parameters = self._tlm_adjoint__solver_parameters
      
      eq = EquationSolver(A._tlm_adjoint__form == b._tlm_adjoint__form, x,
        bcs = bcs, solver_parameters = solver_parameters,
        form_compiler_parameters = form_compiler_parameters)

      eq._pre_annotate(annotate = annotate)
      backend_LinearSolver.solve(self, x, b)
      eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
    else:
      backend_LinearSolver.solve(self, x, b)
