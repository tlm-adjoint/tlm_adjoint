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

import copy
import ufl

__all__ = \
  [
    "InterfaceException",
  
    "apply_rhs_bcs",
    "assemble_arguments",
    "assemble_matrix",
    "assemble_system",
    "copy_parameters_dict",
    "form_form_compiler_parameters",
    "homogenize",
    "is_real_function",
    "linear_solver",
    "matrix_multiply",
    "process_adjoint_solver_parameters",
    "process_solver_parameters",
    "rhs_addto",
    "rhs_copy",
    "update_parameters_dict",
    
    "solve"
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
if not "cache_rhs_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"] = True
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
      
def process_solver_parameters(solver_parameters, J, linear):
  solver_parameters = copy_parameters_dict(solver_parameters)
  if "tlm_adjoint" in solver_parameters:
    tlm_adjoint_parameters = solver_parameters["tlm_adjoint"]
  else:
    tlm_adjoint_parameters = solver_parameters["tlm_adjoint"] = {}
  
  if not "options_prefix" in tlm_adjoint_parameters:
    tlm_adjoint_parameters["options_prefix"] = None

  if "nullspace" in tlm_adjoint_parameters:
    nullspace = tlm_adjoint_parameters["nullspace"]
    if not nullspace is None:
      for fn in nullspace._vecs:
        if not function_is_static(fn) or not function_is_cached(fn) or function_is_checkpointed(fn):
          raise InterfaceExecption("Invalid basis function")
  else:
    tlm_adjoint_parameters["nullspace"] = None

  if "transpose_nullspace" in tlm_adjoint_parameters:
    transpose_nullspace = tlm_adjoint_parameters["transpose_nullspace"]
    if not transpose_nullspace is None:
      for fn in transpose_nullspace._vecs:
        if not function_is_static(fn) or not function_is_cached(fn) or function_is_checkpointed(fn):
          raise InterfaceExecption("Invalid basis function")
  else:
    tlm_adjoint_parameters["transpose_nullspace"] = None

  if "near_nullspace" in tlm_adjoint_parameters:
    near_nullspace = tlm_adjoint_parameters["near_nullspace"]
    if not near_nullspace is None:
      for fn in near_nullspace._vecs:
        if not function_is_static(fn) or not function_is_cached(fn) or function_is_checkpointed(fn):
          raise InterfaceExecption("Invalid basis function")
  else:
    tlm_adjoint_parameters["near_nullspace"] = None

  return solver_parameters, solver_parameters, True

def process_adjoint_solver_parameters(linear_solver_parameters):
  if "tlm_adjoint" in linear_solver_parameters:
    adjoint_solver_parameters = copy.copy(linear_solver_parameters)
    tlm_adjoint_parameters = adjoint_solver_parameters["tlm_adjoint"] = copy.copy(linear_solver_parameters["tlm_adjoint"])
    
    tlm_adjoint_parameters["nullspace"] = linear_solver_parameters["tlm_adjoint"]["transpose_nullspace"]
    tlm_adjoint_parameters["transpose_nullspace"] = linear_solver_parameters["tlm_adjoint"]["nullspace"]

    return adjoint_solver_parameters
  else:
    return linear_solver_parameters  # Copy not required

def assemble_arguments(rank, form_compiler_parameters, solver_parameters):
  kwargs = {"form_compiler_parameters":form_compiler_parameters}
  if rank == 2 and "mat_type" in solver_parameters:
    kwargs["mat_type"] = solver_parameters["mat_type"]
  return kwargs

def assemble_matrix(form, bcs, force_evaluation = True, **assemble_kwargs):
  A = assemble(form, bcs = bcs, **assemble_kwargs)
  if force_evaluation:
    A.force_evaluation()
  return A, None

# Similar interface to assemble_system in FEniCS 2018.1.0
def assemble_system(A_form, b_form, bcs = [], form_compiler_parameters = {}):
  return (assemble(A_form, bcs = bcs, form_compiler_parameters = form_compiler_parameters),
          assemble(b_form, form_compiler_parameters = form_compiler_parameters))

def linear_solver(A, linear_solver_parameters):
  if "tlm_adjoint" in linear_solver_parameters:
    linear_solver_parameters = copy.copy(linear_solver_parameters)
    tlm_adjoint_parameters = linear_solver_parameters.pop("tlm_adjoint")
    options_prefix = tlm_adjoint_parameters.get("options_prefix", None)
    nullspace = tlm_adjoint_parameters.get("nullspace", None)
    transpose_nullspace = tlm_adjoint_parameters.get("transpose_nullspace", None)
    near_nullspace = tlm_adjoint_parameters.get("near_nullspace", None)
  else:
    options_prefix = None
    nullspace = None
    transpose_nullspace = None
    near_nullspace = None
  return LinearSolver(A, solver_parameters = linear_solver_parameters,
    options_prefix = options_prefix, nullspace = nullspace,
    transpose_nullspace = transpose_nullspace, near_nullspace = near_nullspace)

def form_form_compiler_parameters(form, form_compiler_parameters):
  return {"quadrature_degree":ufl.algorithms.estimate_total_polynomial_degree(form)}

#def homogenize(bc):

def apply_rhs_bcs(b, hbcs, b_bc = None):
  if not b_bc is None:
    raise InterfaceException("Unexpected RHS terms")

def matrix_multiply(A, x, addto = None, space = None):
  if space is None:
    if addto is None:
      raise InterfaceException("Unable to create Function")
    else:
      b = backend_Function(addto.function_space()).vector()  # function_new
  else:
    b = backend_Function(space).vector()
  with x.dat.vec_ro as x_v, b.dat.vec_wo as b_v:
    A.petscmat.mult(x_v, b_v)
  if addto is None:
    return b
  else:
    addto.set_local(addto.get_local() + b.get_local())  # function_axpy
    return addto

def is_real_function(x):
  e = x.ufl_element()
  return e.family() == "Real" and e.degree() == 0

def rhs_copy(x):
  return x.copy(deepcopy = True)

def rhs_addto(x, y):
  x.vector().set_local(x.vector().get_local() + y.vector().get_local())  # function_axpy

def solve(*args, **kwargs):
  if not isinstance(args[0], ufl.classes.Equation):
    return backend_solve(*args, **kwargs)
  
  eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
    nullspace, transpose_nullspace, near_nullspace, options_prefix = \
    extract_args(*args, **kwargs)

  if "tlm_adjoint" in solver_parameters:
    solver_parameters = copy.copy(solver_parameters)
    tlm_adjoint_parameters = solver_parameters.pop("tlm_adjoint")

    if "options_prefix" in tlm_adjoint_parameters:
      if not options_prefix is None:
        raise InterfaceException("Cannot pass both options_prefix argument and solver parameter")
      options_prefix = tlm_adjoint_parameters["options_prefix"]

    if "nullspace" in tlm_adjoint_parameters:
      if not nullspace is None:
        raise InterfaceException("Cannot pass both nullspace argument and solver parameter")
      nullspace = tlm_adjoint_parameters["nullspace"]

    if "transpose_nullspace" in tlm_adjoint_parameters:
      if not transpose_nullspace is None:
        raise InterfaceException("Cannot pass both transpose_nullspace argument and solver parameter")
      transpose_nullspace = tlm_adjoint_parameters["transpose_nullspace"]

    if "near_nullspace" in tlm_adjoint_parameters:
      if not near_nullspace is None:
        raise InterfaceException("Cannot pass both near_nullspace argument and solver parameter")
      near_nullspace = tlm_adjoint_parameters["near_nullspace"]

  return backend_solve(eq, x, bcs, J = J, Jp = Jp, M = M,
    form_compiler_parameters = form_compiler_parameters, 
    solver_parameters = solver_parameters, nullspace = nullspace,
    transpose_nullspace = transpose_nullspace, near_nullspace = near_nullspace,
    options_prefix = options_prefix)
