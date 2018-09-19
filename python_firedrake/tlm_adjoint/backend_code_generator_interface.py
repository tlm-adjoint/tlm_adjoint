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

import ufl

__all__ = \
  [
    "InterfaceException",
  
    "apply_rhs_bcs",
    "assemble_matrix",
    "assemble_system",
    "copy_parameters_dict",
    "form_form_compiler_parameters",
    "homogenize",
    "is_real_function",
    "linear_solver",
    "matrix_multiply",
    "process_solver_parameters",
    "rhs_addto",
    "rhs_copy",
    "update_parameters_dict"
  ]
  
class InterfaceException(Exception):
  pass

if not "tlm_adjoint" in parameters:
  parameters["tlm_adjoint"] = {}
if not "AssembleSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"]["AssembleSolver"] = {}
if not "match_quadrature" in parameters["tlm_adjoint"]["AssembleSolver"]:
  parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"] = False
if not "EquationSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"]["EquationSolver"] = {}
if not "enable_jacobian_caching" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"] = True
if not "pre_assemble" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["pre_assemble"] = True
if not "match_quadrature" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"] = False
if not "defer_adjoint_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"] = False

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
      
def process_solver_parameters(linear, solver_parameters):
  solver_parameters = copy_parameters_dict(solver_parameters)
  return solver_parameters, solver_parameters, True

def assemble_matrix(form, bcs, form_compiler_parameters, force_evaluation = True):
  A = assemble(form, bcs = bcs, form_compiler_parameters = form_compiler_parameters)
  if force_evaluation:
    A.force_evaluation()
  return A, None

# Similar interface to assemble_system in FEniCS 2018.1.0
def assemble_system(A_form, b_form, bcs = [], form_compiler_parameters = {}):
  return (assemble(A_form, bcs = bcs, form_compiler_parameter = form_compiler_parameters),
          assemble(b_form, form_compiler_parameters = form_compiler_parameter))

def linear_solver(A, linear_solver_parameters):
  return LinearSolver(A, solver_parameters = linear_solver_parameters)

def form_form_compiler_parameters(form, form_compiler_parameters):
  return {"quadrature_degree":ufl.algorithms.estimate_total_polynomial_degree(form)}

#def homogenize(bc):

def apply_rhs_bcs(b, hbcs, b_bc = None):
  if not b_bc is None:
    raise InterfaceException("Unexpected RHS terms")

def matrix_multiply(A, x, addto = None, space_fn = None):
  if space_fn is None:
    if addto is None:
      raise InterfaceException("Unable to create Function")
    else:
      b = backend_Function(addto.function_space()).vector()  # function_new
  else:
    b = backend_Function(space_fn.function_space()).vector()  # function_new
  as_backend_type(A).mat().mult(as_backend_type(x).vec(), as_backend_type(b).vec())
  if addto is None:
    return b
  else:
    addto.set_local(addto.get_local() + b.get_local())  # function_axpy
    return addto

def is_real_function(x):
  return getattr(x.function_space(), "_tlm_adjoint__real_space", False)

def rhs_copy(x):
  return x.copy(deepcopy = True)

def rhs_addto(x, y):
  x.vector().set_local(x.vector().get_local() + y.vector().get_local())  # function_axpy
