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

# This file previously included a 'vector' function, which followed
# Function::init_vector in dolfin/function/Function.cpp, DOLFIN 2017.2.0.post0
# Code first added 2018-08-03, removed 2018-09-04
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

backend = "FEniCS"

from fenics import *

import fenics

backend_Matrix = fenics.cpp.la.GenericMatrix
backend_Vector = fenics.GenericVector
extract_args = fenics.fem.solving._extract_args

backend_Constant = fenics.Constant
backend_DirichletBC = fenics.DirichletBC
backend_Function = fenics.Function
backend_KrylovSolver = fenics.KrylovSolver
backend_LUSolver = LUSolver
backend_assemble = fenics.assemble
backend_assemble_system = fenics.assemble_system
backend_project = project
backend_solve = fenics.solve

def copy_parameters_dict(parameters):
  new_parameters = {}
  for key in parameters:
    value = parameters[key]
    if isinstance(value, (Parameters, dict)):
      new_parameters[key] = copy_parameters_dict(value)
    else:
      new_parameters[key] = value
  return new_parameters

def update_parameters_dict(parameters, new_parameters):
  for key in new_parameters:
    value = new_parameters[key]
    if key in parameters \
      and isinstance(parameters[key], (Parameters, dict)) \
      and isinstance(value, (Parameters, dict)):
      update_parameters_dict(parameters[key], value)
    elif isinstance(value, (Parameters, dict)):
      parameters[key] = copy_parameters_dict(value)
    else:
      parameters[key] = value
  
__all__ = \
  [  
    "backend",
    
    "backend_Constant",
    "backend_DirichletBC",
    "backend_Function",
    "backend_KrylovSolver",
    "backend_LUSolver",
    "backend_Matrix",
    "backend_Vector",
    "backend_assemble",
    "backend_assemble_system",
    "backend_project",
    "backend_solve",
    
    "Constant",
    "DirichletBC",
    "Function",
    "FunctionSpace",
    "KrylovSolver",
    "LUSolver",
    "NewtonSolver",
    "Parameters",
    "TestFunction",
    "TrialFunction",
    "UnitIntervalMesh",
    "action",
    "adjoint",
    "assemble",
    "assemble_system",
    "dx",
    "extract_args",
    "has_lu_solver_method",
    "info",
    "inner",
    "parameters",
    "project",
    "solve",
    "system",
  
    "copy_parameters_dict",
    "update_parameters_dict"
  ]
