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

from .backend import Constant, FunctionSpace, Parameters, UnitIntervalMesh, \
  as_backend_type, backend_Function, firedrake, homogenize

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
    "default_comm",
    "finalise_adjoint_derivative_action",
    "function_alias",
    "function_assign",
    "function_axpy",
    "function_comm",
    "function_copy",
    "function_get_values",
    "function_global_size",
    "function_inner",
    "function_is_static",
    "function_linf_norm",
    "function_local_indices",
    "function_local_size",
    "function_max_value",
    "function_new",
    "function_set_values",
    "function_zero",
    "homogenized_bc",
    "info",
    "is_function",
    "replaced_function",
    "subtract_adjoint_derivative_action",
    "update_parameters_dict",
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
  parameters_copy = parameters.copy()
  for key, value in parameters.items():
    if isinstance(value, (Parameters, dict)):
      parameters_copy[key] = copy_parameters_dict(value)
  return parameters_copy

def update_parameters_dict(parameters, new_parameters):
  for key, value in new_parameters.items():
    if key in parameters \
      and isinstance(parameters[key], (Parameters, dict)) \
      and isinstance(value, (Parameters, dict)):
      update_parameters_dict(parameters[key], value)
    elif isinstance(value, (Parameters, dict)):
      parameters[key] = copy_parameters_dict(value)
    else:
      parameters[key] = value

ufl.classes.FunctionSpace.id = lambda self : id(self)
firedrake.functionspaceimpl.FunctionSpace.id = lambda self : id(self)
firedrake.functionspaceimpl.MixedFunctionSpace.id = lambda self : id(self)

def RealFunctionSpace(comm = None):
  if comm is None:
    comm = default_comm()
  space = FunctionSpace(UnitIntervalMesh(comm.size, comm = comm), "Discontinuous Lagrange", 0)
  space._tlm_adjoint__real_space = True
  return space

backend_Function.id = lambda self : self.count()
class Function(backend_Function):
  def __init__(self, space, name = None, static = False, val = None):
    backend_Function.__init__(self, space, name = name, val = val)
    self.__static = static
    
  def is_static(self):
    return self.__static

def new_count():
  return Constant(0).count()
  
class ReplacementFunction(ufl.classes.Coefficient):
  def __init__(self, x):
    ufl.classes.Coefficient.__init__(self, x.function_space(), count = new_count())
    self.__space = x.function_space()
    self.__id = x.id()
    self.__name = x.name()
    self.__static = function_is_static(x)
  
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
  return isinstance(x, backend_Function)

def function_is_static(x):
  return hasattr(x, "is_static") and x.is_static()
  
def function_copy(x, name = None, static = False):
  y = Function(x.function_space(), name = name, static = static)
  function_assign(y, x)
  return y

def function_assign(x, y):
  if isinstance(y, (int, float, Constant)):
    x.vector()[:] = float(y)
  else:
    function_set_values(x, function_get_values(y))

def function_axpy(x, alpha, y):
  function_set_values(x, function_get_values(x) + alpha * function_get_values(y))

def default_comm():
  import mpi4py.MPI
  return mpi4py.MPI.COMM_WORLD

def function_comm(x):
  return x.comm

def function_inner(x, y):
  x_v = as_backend_type(x.vector()).vec()
  y_v = as_backend_type(y.vector()).vec()
  return x_v.dot(y_v)

def function_local_size(x):
  return x.vector().local_size()

def function_get_values(x):
  return x.vector().get_local()

def function_set_values(x, values):
  x.vector().set_local(values)

def function_max_value(x):
  x_v = as_backend_type(x.vector()).vec()
  return x_v.max()[1]

def function_linf_norm(x):
  x_v = as_backend_type(x.vector()).vec()
  import petsc4py.PETSc
  return x_v.norm(norm_type = petsc4py.PETSc.NormType.NORM_INFINITY)
  
def function_new(x, name = None, static = False):
  return Function(x.function_space(), name = name, static = static)

def function_alias(x):
  return Function(x.function_space(), name = x.name(), static = function_is_static(x), val = x.dat)

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
    if isinstance(y, backend_Function):
      y = y.vector()
    function_axpy(x, -alpha, y)
  else:
    if isinstance(y, backend_Function):
      y = y.vector()
    function_axpy(x, -1.0, y)
    
def finalise_adjoint_derivative_action(x):
  pass

def apply_bcs(x, bcs):
  for bc in bcs:
    bc.apply(x.vector())

def homogenized_bc(bc):
  return homogenize(bc)
