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
from .backend_code_generator_interface import copy_parameters_dict

from .caches import Function, ReplacementFunction, assembly_cache, \
  clear_caches, form_neg, function_caches, function_is_cached, \
  function_is_checkpointed, function_is_static, function_state, \
  function_tlm_depth, function_update_state, is_function, linear_solver_cache, \
  replaced_function, update_caches

import numpy
import ufl
import sys
import weakref

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
    "function_is_cached",
    "function_is_checkpointed",
    "function_is_static",
    "function_linf_norm",
    "function_local_indices",
    "function_local_size",
    "function_max_value",
    "function_new",
    "function_new_tlm",
    "function_set_values",
    "function_space_id",
    "function_state",
    "function_tlm_depth",
    "function_update_state",
    "function_zero",
    "info",
    "is_function",
    "replaced_function",
    "subtract_adjoint_derivative_action",
    "update_caches",
    "warning"
  ]
  
#def clear_caches(*deps):

#def update_caches(eq_deps, deps = None):

def info(message):
  sys.stdout.write("%s\n" % message)
  sys.stdout.flush()

def warning(message):
  sys.stderr.write("%s\n" % message)
  sys.stderr.flush()

#def copy_parameters_dict(parameters):

#class FunctionSpace:

def function_space_id(space):
  return id(space)

_real_spaces = weakref.WeakValueDictionary()
def RealFunctionSpace(comm = None):
  if comm is None:
    comm = default_comm()
  comm_f = comm.py2f()
  
  try:
    space = _real_spaces[comm_f]
  except KeyError:
    space = FunctionSpace(UnitIntervalMesh(comm.size, comm = comm), "R", 0)
    _real_spaces[comm_f] = space
  return space

#class Function:
#  def __init__(self, space, name = None, static = False, cache = None,
#  checkpoint = None, tlm_depth = 0):
#  def function_space(self):
#  def id(self):
#  def name(self):
backend_Function.id = lambda self : self.count()

#class ReplacementFunction:
#  def __init__(self, x):
#  def function_space(self):
#  def id(self):
#  def name(self):

#def replaced_function(x):

#def is_function(x):

#def function_state(x):

#def function_update_state(*X):

#def function_is_static(x):

#def function_is_cached(x):

#def function_is_checkpointed(x):

#def function_tlm_depth(x):

def function_copy(x, name = None, static = False, cache = None,
  checkpoint = None, tlm_depth = 0):
  y = x.copy(deepcopy = True)
  if not name is None: y.rename(name, "a Function")
  y.is_static = lambda : static
  if cache is None:
    cache = static
  y.is_cached = lambda : cache
  if checkpoint is None:
    checkpoint = not static
  y.is_checkpointed = lambda : checkpoint
  y.tlm_depth = lambda : tlm_depth
  return y

def function_assign(x, y):
  if isinstance(y, (int, float)):
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
  
def function_new(x, name = None, static = False, cache = None,
  checkpoint = None, tlm_depth = 0):
  return Function(x.function_space(), name = name, static = static,
    cache = cache, checkpoint = checkpoint, tlm_depth = tlm_depth)

def function_new_tlm(x, name = None):
  if hasattr(x, "new_tlm"):
    return x.new_tlm(name = name)
  elif function_is_static(x):
    return None
  else:
    return function_new(x, name = name, static = False,
      cache = function_is_cached(x), checkpoint = function_is_checkpointed(x),
      tlm_depth = function_tlm_depth(x) + 1)

def function_alias(x):
  return Function(x.function_space(), name = x.name(),
    static = function_is_static(x), cache = function_is_cached(x),
    checkpoint = function_is_checkpointed(x), tlm_depth = function_tlm_depth(x),
    val = x.dat)

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
      x._tlm_adjoint__adj_b += form_neg(y)
    else:
      x._tlm_adjoint__adj_b = form_neg(y)
  else:
    if isinstance(y, backend_Function):
      y = y.vector()
    function_axpy(x, -1.0, y)  # Works even if x or y are Vector objects
    
def finalise_adjoint_derivative_action(x):
  if hasattr(x, "_tlm_adjoint__adj_b"):
    function_axpy(x, 1.0, assemble(x._tlm_adjoint__adj_b))
    delattr(x, "_tlm_adjoint__adj_b")
