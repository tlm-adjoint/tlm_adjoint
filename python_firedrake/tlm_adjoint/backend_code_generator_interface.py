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

__all__ = \
  [
    "assemble_matrix",
    "copy_parameters_dict",
    "linear_solver",
    "update_parameters_dict"
  ]

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
      
def assemble_matrix(form, bcs, form_compiler_parameters):
  b = assemble(form, form_compiler_parameters = form_compiler_parameters)
  for bc in bcs:
    bc.apply(b)
  b.force_evaluation()
  return b#, None

def linear_solver(A, linear_solver_parameters):
  return LinearSolver(A, solver_parameters = linear_solver_parameters)
