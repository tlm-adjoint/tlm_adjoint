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

from .base import Constant, FunctionSpace, UnitIntervalMesh, base_Function, \
  firedrake, homogenize

import copy
import numpy
import ufl
import sys

__all__ = \
  [  
    "Function",
    "FunctionSpace",
    "RealFunctionSpace",
    "ReplacementFunction",
    "apply_bcs",
    "clear_caches",
    "copy_parameters_dict",
    "finalise_adjoint_derivative_action",
    "function_assign",
    "function_axpy",
    "function_comm",
    "function_copy",
    "function_get_values",
    "function_global_size",
    "function_inner",
    "function_is_static",
    "function_local_indices",
    "function_local_size",
    "function_max_value",
    "function_min_value",
    "function_new",
    "function_set_values",
    "function_zero",
    "homogenized_bc",
    "info",
    "is_function",
    "replaced_function",
    "subtract_adjoint_derivative_action",
    "warning"
  ]
  
def clear_caches():
  pass

def info(message):
  sys.stdout.write("%s\n" % message)
  sys.stdout.flush()

def warning(message):
  sys.stderr.write("%s\n" % message)
  sys.stderr.flush()

def copy_parameters_dict(parameters):
  return copy.deepcopy(parameters)

firedrake.functionspaceimpl.WithGeometry.id = lambda self : id(self)

def RealFunctionSpace():
  return FunctionSpace(UnitIntervalMesh(1), "Discontinuous Lagrange", 0)

base_Function.id = lambda self : id(self)  
class Function(base_Function):
  def __init__(self, space, name = None, static = False):
    base_Function.__init__(self, space, name = name)
    self.__static = static
    
  def is_static(self):
    return self.__static

def new_id():
  return Constant(0).count()
  
class ReplacementFunction(ufl.classes.Coefficient):
  def __init__(self, x):
    ufl.classes.Coefficient.__init__(self, x.function_space(), count = new_id())
    self.__space = x.function_space()
    self.__id = x.id()
    self.__name = x.name()
    self.__static = function_is_static(x)
    if hasattr(x, "_tlm_adjoint__tlm_basename"):
      self._tlm_adjoint__tlm_basename = x._tlm_adjoint__tlm_basename
    if hasattr(x, "_tlm_adjoint__tlm_depth"):
      self._tlm_adjoint__tlm_depth = x._tlm_adjoint__tlm_depth
  
  def function_space(self):
    return self.__space
  
  def id(self):
    return self.__id
  
  def name(self):
    return self.__name
  
  def is_static(self):
    return self.__static

def replaced_function(x):
  if isinstance(x, ReplacementFunction):
    return x
  if not hasattr(x, "_tlm_adjoint__ReplacementFunction"):
    x._tlm_adjoint__ReplacementFunction = ReplacementFunction(x)
  return x._tlm_adjoint__ReplacementFunction

def is_function(x):
  return isinstance(x, base_Function)

def function_is_static(x):
  return isinstance(x, base_Function) and hasattr(x, "is_static") and x.is_static()
  
def function_copy(x, name = None, static = False):
  y = Function(x.function_space(), name = name, static = static)
  function_assign(y, x)
  return y

def function_assign(x, y):
  if isinstance(y, (int, float)):
    x.vector()[:] = float(y)
  else:
    function_set_values(x, function_get_values(y))

def function_axpy(x, alpha, y):
  function_set_values(x, function_get_values(x) + alpha * function_get_values(y))

def function_comm(x):
  import petsc4py
  return petsc4py.PETSc.Comm(x.comm)

def function_inner(x, y):
  return function_get_values(x).dot(function_get_values(y))

def function_local_size(x):
  return x.vector().local_size()

def function_get_values(x):
  return x.vector().get_local()

def function_set_values(x, values):
  x.vector().set_local(values)
  x.vector().apply("insert")

def function_max_value(x):
  return x.vector().max()

def function_min_value(x):
  return function_get_values(x).min()
  
def function_new(x, name = None, static = False):
  return Function(x.function_space(), name = name, static = static)

def function_global_size(x):
  return x.function_space().dim()

def function_zero(x):
  x.vector()[:] = 0.0

def function_local_indices(x):
  return slice(*x.vector().local_range())

def subtract_adjoint_derivative_action(x, y):
  if y is None:
    return
  if isinstance(y, tuple):
    alpha, y = y
    if isinstance(y, base_Function):
      y = y.vector()
    function_axpy(x, -alpha, y)
  else:
    if isinstance(y, base_Function):
      y = y.vector()
    function_axpy(x, -1.0, y)
  return
    
def finalise_adjoint_derivative_action(x):
  pass

def apply_bcs(x, bcs):
  for bc in bcs:
    bc.apply(x.vector())

def homogenized_bc(bc):
  return homogenize(bc)
