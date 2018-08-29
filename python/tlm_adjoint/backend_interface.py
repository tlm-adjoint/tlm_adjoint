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

from .backend import FunctionSpace, UnitIntervalMesh, assemble, assign_vector, \
  backend_Constant, backend_Function, clear_backend_caches, \
  copy_parameters_dict, info, warning

from .caches import Function, ReplacementFunction, assembly_cache, \
  homogenized as homogenized_bc, is_static, linear_solver_cache, \
  replaced_function

import numpy
import ufl

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
    "warning"
  ]
  
def clear_caches():
  assembly_cache().clear()
  linear_solver_cache().clear()
  #clear_backend_caches()

#def info(message):

#def warning(message):

#def copy_parameters_dict(parameters):

#class FunctionSpace:
#  def id(self):

class RealFunctionSpace(FunctionSpace):
  def __init__(self, comm = None):
    if comm is None:
      import petsc4py.PETSc
      comm = petsc4py.PETSc.COMM_WORLD
    FunctionSpace.__init__(self, UnitIntervalMesh(comm, comm.size), "R", 0)

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
  return isinstance(x, backend_Function) and is_static(x)
  
def function_copy(x, name = None, static = False):
  y = x.copy(deepcopy = True)
  if not name is None:
    y.rename(name, y.label())
  y.is_static = lambda : static
  return y

def function_assign(x, y):
  if isinstance(y, (int, float, backend_Constant)):
    function_set_values(x, numpy.ones(x.vector().local_size(), dtype = numpy.float64) * float(y))
  else:
    assign_vector(x.vector(), y.vector())

def function_axpy(x, alpha, y):
  x.vector().axpy(alpha, y.vector())

def function_comm(x):
  return x.function_space().mesh().mpi_comm()

def function_inner(x, y):
  return x.vector().inner(y.vector())

def function_local_size(x):
  return x.vector().local_size()

def function_get_values(x):
  return x.vector().get_local()

def function_set_values(x, values):
  x.vector().set_local(values)
  x.vector().apply("insert")

def function_max_value(x):
  return x.vector().max()

def function_linf_norm(x):
  return x.vector().norm("linf")
  
def function_new(x, name = None, static = False):
  if isinstance(x, backend_Function):
    y = function_copy(x, name = name, static = static)
    function_zero(y)
    return y
  else:
    return Function(x.function_space(), name = name, static = static)

def function_zero(x):
  x.vector().zero()

def function_global_size(x):
  return x.function_space().dofmap().global_dimension()

def function_local_indices(x):
  return slice(*x.function_space().dofmap().ownership_range())

def subtract_adjoint_derivative_action(x, y):
  if y is None:
    return
  if isinstance(y, tuple):
    alpha, y = y
    if isinstance(y, backend_Function):
      y = y.vector()
    x.vector().axpy(-alpha, y)
  elif isinstance(y, ufl.classes.Form):
    if hasattr(x, "_tlm_adjoint__adj_b"):
      x._tlm_adjoint__adj_b -= y
    else:
      x._tlm_adjoint__adj_b = -y
  else:
    if isinstance(y, backend_Function):
      y = y.vector()
    x.vector().axpy(-1.0, y)
  return

def finalise_adjoint_derivative_action(x):
  if hasattr(x, "_tlm_adjoint__adj_b"):
    assemble(x._tlm_adjoint__adj_b, tensor = x.vector(), add_values = True)
    delattr(x, "_tlm_adjoint__adj_b")

def apply_bcs(x, bcs):
  for bc in bcs:
    bc.apply(x.vector())

#def homogenized_bc(bc):
