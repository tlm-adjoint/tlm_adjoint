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

import copy
import numpy
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
  pass

def info(message):
  sys.stdout.write("%s\n" % message)
  sys.stdout.flush()

def warning(message):
  sys.stderr.write("%s\n" % message)
  sys.stderr.flush()

def copy_parameters_dict(parameters):
  return copy.deepcopy(parameters)

class FunctionSpace:
  def __init__(self, dim):
    self._dim = dim
  
  def dim(self):
    return self._dim

def function_space_id(space):
  return space.dim()

def RealFunctionSpace():
  return FunctionSpace(1)

Function_id_counter = [0]
class Function:
  def __init__(self, space, name = None, static = False, _data = None):
    id = Function_id_counter[0]
    Function_id_counter[0] += 1
    if name is None:
      name = "f_%i" % id  # Following FEniCS 2017.2.0 behaviour
    
    self._space = space
    self._name = name
    self._static = static
    self._id = id
    self._data = numpy.zeros(space.dim(), dtype = numpy.float64) if _data is None else _data
    
  def function_space(self):
    return self._space
  
  def id(self):
    return self._id
  
  def name(self):
    return self._name
  
  def is_static(self):
    return self._static
  
  def vector(self):
    return self._data

class ReplacementFunction:
  def __init__(self, x):
    self._space = x.function_space()
    self._name = x.name()
    self._static = x.is_static()
    self._id = x.id()
    
  def function_space(self):
    return self._space
  
  def id(self):
    return self._id
  
  def name(self):
    return self._name
  
  def is_static(self):
    return self._static
    
def replaced_function(x):
  if isinstance(x, ReplacementFunction):
    return x
  if not hasattr(x, "_tlm_adjoint__ReplacementFunction"):
    x._tlm_adjoint__ReplacementFunction = ReplacementFunction(x)
  return x._tlm_adjoint__ReplacementFunction

def is_function(x):
  return isinstance(x, Function)

def function_is_static(x):
  return x.is_static()

def function_copy(x, name = None, static = False):
  return Function(x.function_space(), name = name, static = static,
    _data = x.vector().copy())

def function_assign(x, y):
  if isinstance(y, (int, float)):
    x.vector()[:] = y
  else:
    x.vector()[:] = y.vector()

def function_axpy(x, alpha, y):
  x.vector()[:] += alpha * y.vector()

class SerialComm:
  def allgather(self, v):
    w = v.view()
    w.setflags(write = False)
    return (w,)
  
  # Interface as in mpi4py 3.0.1
  def bcast(self, obj, root = 0):
    return copy.deepcopy(obj)

  @property
  def rank(self):
    return 0
  
  @property
  def size(self):
    return 1

def default_comm():
  return SerialComm()

def function_comm(x):
  return SerialComm()

def function_inner(x, y):
  return x.vector().dot(y.vector())

def function_local_size(x):
  return x.vector().shape[0]

def function_get_values(x):
  values = x.vector().view()
  values.setflags(write = False)
  return values

def function_set_values(x, values):
  x.vector()[:] = values

def function_max_value(x):
  return x.vector().max()

def function_linf_norm(x):
  return abs(x.vector()).max()
  
def function_new(x, name = None, static = False):
  return Function(x.function_space(), name = name, static = static)

def function_alias(x):
  return Function(x.function_space(), name = x.name(), static = x.is_static(), _data = x.vector())

def function_zero(x):
  x.vector()[:] = 0.0

def function_global_size(x):
  return x.vector().shape[0]

def function_local_indices(x):
  return slice(0, x.vector().shape[0])

def subtract_adjoint_derivative_action(x, y):
  if y is None:
    return
  if isinstance(y, tuple):
    alpha, y = y
    if isinstance(y, Function):
      y = y.vector()
    if alpha == 1.0:
      x.vector()[:] -= y
    else:
      x.vector()[:] -= alpha * y
  else:
    if isinstance(y, Function):
      y = y.vector()
    x.vector()[:] -= y
    
def finalise_adjoint_derivative_action(x):
  pass
