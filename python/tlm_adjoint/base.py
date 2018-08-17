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

# 'vector' follows Function::init_vector in dolfin/function/Function.cpp,
# DOLFIN 2017.2.0.post0
# Code first added 2018-08-03
#
# Copyright notice from dolfin/function/Function.cpp, DOLFIN 2017.2.0.post0
#
# Copyright (C) 2003-2012 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Garth N. Wells 2005-2010
# Modified by Martin Sandve Alnes 2008-2014
# Modified by Andre Massing 2009

base = "FEniCS"

from fenics import *

import fenics

base_Matrix = fenics.GenericMatrix
base_Vector = fenics.GenericVector
extract_args = fenics.fem.solving._extract_args

base_Constant = fenics.Constant
base_DirichletBC = fenics.DirichletBC
base_Function = fenics.Function
base_KrylovSolver = fenics.KrylovSolver
base_LUSolver = LUSolver
base_assemble = fenics.assemble
base_assemble_system = fenics.assemble_system
base_project = project
base_solve = fenics.solve
  
def copy_vector(x):
  return x.copy()
  
def assign_vector(x, y):
  if isinstance(y, float):
    x[:] = y
  else:
    x.zero()
    x.axpy(1.0, y)
  return

def copy_parameters_dict(parameters):
  new_parameters = {}
  for key, value in parameters.items():
    if isinstance(value, (Parameters, dict)):
      new_parameters[key] = copy_parameters_dict(value)
    else:
      new_parameters[key] = value
  return new_parameters
  
__all__ = \
  [  
    "base",
    
    "base_Constant",
    "base_DirichletBC",
    "base_Function",
    "base_KrylovSolver",
    "base_LUSolver",
    "base_Matrix",
    "base_Vector",
    "base_assemble",
    "base_assemble_system",
    "base_project",
    "base_solve",
    
    "Constant",
    "DirichletBC",
    "Function",
    "FunctionSpace",
    "KrylovSolver",
    "LUSolver",
    "NewtonSolver",
    "ParameterValue",
    "Parameters",
    "TestFunction",
    "TrialFunction",
    "UnitIntervalMesh",
    "Variable",
    "action",
    "adjoint",
    "assemble",
    "assemble_system",
    "derivative",
    "dx",
    "extract_args",
    "has_lu_solver_method",
    "info",
    "inner",
    "parameters",
    "replace",
    "solve",
    "system",
    "warning",
  
    "assign_vector",
    "copy_parameters_dict",
    "copy_vector",
    
    "clear_base_caches"
  ]

from collections import OrderedDict
layout_cache = OrderedDict()
matrix_cache = OrderedDict()
def clear_base_caches():
  layout_cache.clear()
  matrix_cache.clear()
      
# Following Function::init_vector in dolfin/function/Function.cpp,
# DOLFIN 2017.2.0.post0 (see copyright information above)
# Code first added 2018-08-03
def vector(space):
  from fenics import DefaultFactory, TensorLayout
  factory = DefaultFactory()
  comm = space.mesh().mpi_comm()
  if space.id() in layout_cache:
    layout = layout_cache[space.id()]
  else:
    layout_cache[space.id()] = layout = factory.create_layout(comm, 1)
    layout.init([space.dofmap().index_map()], TensorLayout.Ghosts_GHOSTED)
  v = factory.create_vector(comm)
  v.init(layout)
  v.zero()
  return v
# End of code following Function::init_vector in dolfin/function/Function.cpp,
# DOLFIN 2017.2.0.post0

def matrix(space_0, space_1):
  key = (space_0.id(), space_1.id())
  if not key in matrix_cache:
    factory = DefaultFactory()
    comm = space_0.mesh().mpi_comm()
    matrix_cache[key] = factory.create_matrix(comm)
  return matrix_cache[key].copy()

# The following workaround various FEniCS 2017.2.0 memory leaks
# Following FEniCS 2017.2.0 API

_orig_Function__init__ = base_Function.__init__
def _Function__init__(self, *args, **kwargs):
  if len(args) == 1 and isinstance(args[0], FunctionSpace):
    _orig_Function__init__(self, args[0], vector(args[0]), **kwargs)
  else:
    _orig_Function__init__(self, *args, **kwargs)
base_Function.__init__ = _Function__init__

_orig_assemble = fenics.assemble
def assemble(form, tensor = None, *args, **kwargs):
  if tensor is None:
    arguments = form.arguments()
    rank = len(form.arguments()) 
    if rank == 1:
      tensor = vector(arguments[0].function_space())
    elif rank == 2:
      tensor = matrix(arguments[0].function_space(), arguments[1].function_space())
  return _orig_assemble(form, tensor = tensor, *args, **kwargs)
base_assemble = fenics.assemble = assemble

_orig_assemble_system = fenics.assemble_system
def assemble_system(A_form, b_form, bcs = None, x0 = None,
  form_compiler_parameters = None, add_values = False,
  finalize_tensor = True, keep_diagonal = False, A_tensor = None, b_tensor = None, *args, **kwargs):
  if A_tensor is None:
    arguments = A_form.arguments()
    A_tensor = matrix(arguments[0].function_space(), arguments[1].function_space())
  if b_tensor is None:
    arguments = b_form.arguments()
    tensor = vector(arguments[0].function_space())
  return _orig_assemble_system(A_form, b_form, bcs = bcs, x0 = x0,
    form_compiler_parameters = form_compiler_parameters, add_values = add_values,
    finalize_tensor = finalize_tensor, keep_diagonal = keep_diagonal, A_tensor = A_tensor, b_tensor = b_tensor, *args, **kwargs)
base_assemble_system = fenics.assemble_system = assemble_system
