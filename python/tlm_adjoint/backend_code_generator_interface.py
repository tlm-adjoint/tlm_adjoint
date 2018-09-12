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

import numpy

__all__ = \
  [
    "assemble_matrix",
    "copy_parameters_dict",
    "linear_solver",
    "update_parameters_dict"
  ]

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
      
def assemble_matrix(form, bcs, form_compiler_parameters):
  if len(bcs) > 0:
    test = TestFunction(form.arguments()[0].function_space())
    test_shape = test.ufl_element().value_shape()
    dummy_rhs = inner(test, Constant(0.0 if len(test_shape) == 0 else numpy.zeros(test_shape, dtype = numpy.float64))) * dx
    b, b_bc = assemble_system(form, dummy_rhs, bcs, form_compiler_parameters = form_compiler_parameters)
    if b_bc.norm("linf") == 0.0:
      b_bc = None
  else:
    b = assemble(form, form_compiler_parameters = form_compiler_parameters)
    b_bc = None
  return b, b_bc

def linear_solver(A, linear_solver_parameters):
  linear_solver = linear_solver_parameters.get("linear_solver", "default")
  if linear_solver in ["direct", "lu"]:
    linear_solver = "default"
  elif linear_solver == "iterative":
    linear_solver = "gmres"
  is_lu_linear_solver = linear_solver == "default" or has_lu_solver_method(linear_solver)
  if is_lu_linear_solver:
    solver = LUSolver(A, linear_solver)
    update_parameters_dict(solver.parameters, linear_solver_parameters.get("lu_solver", {}))
  else:
    solver = KrylovSolver(A, linear_solver, linear_solver_parameters.get("preconditioner", "default"))
    update_parameters_dict(solver.parameters, linear_solver_parameters.get("krylov_solver", {}))
  return solver
