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

backend = "Firedrake"

from firedrake import *

import firedrake

extract_args = firedrake.solving._extract_args

backend_Matrix = firedrake.matrix.Matrix

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_LinearSolver = LinearSolver
backend_assemble = assemble
backend_project = project
backend_solve = solve

def copy_parameters_dict(parameters):
  parameters_copy = parameters.copy()
  for key, value in parameters.items():
    if isinstance(value, (Parameters, dict)):
      parameters_copy[key] = copy_parameters_dict(value)
  return parameters_copy

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
    "backend_LinearSolver",
    "backend_Matrix",
    "backend_assemble",
    "backend_project",
    "backend_solve",
    
    "Constant",
    "DirichletBC",
    "Function",
    "FunctionSpace",
    "LinearSolver",
    "Parameters",
    "TestFunction",
    "TrialFunction",
    "UnitIntervalMesh",
    "action",
    "adjoint",
    "as_backend_type",
    "assemble",
    "dx",
    "extract_args",
    "firedrake",
    "homogenize",
    "inner",
    "parameters",
    "project",
    "solve",
    "system",
    
    "copy_parameters_dict",
    "update_parameters_dict"
  ]
