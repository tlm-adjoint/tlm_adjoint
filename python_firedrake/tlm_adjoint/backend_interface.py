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
from .backend_code_generator_interface import copy_parameters_dict

from .caches import Function, ReplacementFunction, assembly_cache, is_static, \
  is_static_bcs, linear_solver_cache, replaced_function

import numpy
import ufl
import sys

__all__ = \
  [
    "Function",
    "FunctionSpace",
    "RealFunctionSpace",
    "ReplacementFunction",
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
    "function_space_id",
    "function_zero",
    "info",
    "is_function",
    "replaced_function",
    "subtract_adjoint_derivative_action",
    "warning"
  ]
  
def clear_caches():
  assembly_cache().clear()
  linear_solver_cache().clear()

def info(message):
  sys.stdout.write("%s\n" % message)
  sys.stdout.flush()

def warning(message):
  sys.stderr.write("%s\n" % message)
  sys.stderr.flush()

#def copy_parameters_dict(parameters):

def function_space_id(space):
  return id(space)

def RealFunctionSpace(comm = None):
  if comm is None:
    comm = default_comm()
  space = FunctionSpace(UnitIntervalMesh(comm.size, comm = comm), "Discontinuous Lagrange", 0)
  space._tlm_adjoint__real_space = True
  return space

backend_Function.id = lambda self : self.count()
#class Function:
#  def __init__(self, space, name = None, static = False):
#  def function_space(self):
#  def id(self):
#  def name(self):

# class ReplacementFunction:
#  def __init__(self, x):
#  def function_space(self):
#  def id(self):
#  def name(self):

#def replaced_function(x):

def is_function(x):
  return isinstance(x, backend_Function)

def function_is_static(x):
  return is_static(x)
  
def function_copy(x, name = None, static = False):
  y = Function(x.function_space(), name = name, static = static)
  function_assign(y, x)
  return y

def function_assign(x, y):
  if isinstance(y, (int, float, backend_Constant)):
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
  local_range = x.vector().local_range()
  return local_range[1] - local_range[0]

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

def function_zero(x):
  x.vector()[:] = 0.0

def function_global_size(x):
  return x.function_space().dim()

def function_local_indices(x):
  return slice(*x.vector().local_range())

def subtract_adjoint_derivative_action(x, y):
  if y is None:
    return
  if isinstance(y, tuple):
    alpha, y = y
    if isinstance(y, backend_Function):
      y = y.vector()
    function_axpy(x, -alpha, y)  # Works even if x or y are Vector objects
  elif isinstance(y, ufl.classes.Form):
    if hasattr(x, "_tlm_adjoint__adj_b"):
      x._tlm_adjoint__adj_b -= y
    else:
      x._tlm_adjoint__adj_b = -y
  else:
    if isinstance(y, backend_Function):
      y = y.vector()
    function_axpy(x, -1.0, y)  # Works even if x or y are Vector objects
    
def finalise_adjoint_derivative_action(x):
  if hasattr(x, "_tlm_adjoint__adj_b"):
    function_axpy(x, 1.0, assemble(x._tlm_adjoint__adj_b))
    delattr(x, "_tlm_adjoint__adj_b")
