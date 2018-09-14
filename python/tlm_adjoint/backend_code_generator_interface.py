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

import ffc
import numpy

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
    "update_parameters_dict"
  ]
  
class InterfaceException(Exception):
  pass

if not "tlm_adjoint" in parameters:
  parameters.add(Parameters("tlm_adjoint"))
if not "AssembleSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"].add(Parameters("AssembleSolver"))
if not "match_quadrature" in parameters["tlm_adjoint"]["AssembleSolver"]:
  parameters["tlm_adjoint"]["AssembleSolver"].add("match_quadrature", False)
if not "EquationSolver" in parameters["tlm_adjoint"]:
  parameters["tlm_adjoint"].add(Parameters("EquationSolver"))
if not "enable_jacobian_caching" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("enable_jacobian_caching", True)
if not "pre_assemble" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("pre_assemble", True)
if not "match_quadrature" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("match_quadrature", False)
if not "defer_adjoint_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("defer_adjoint_assembly", False)

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
      
def process_solver_parameters(linear, solver_parameters):
  solver_parameters = copy_parameters_dict(solver_parameters)
  if linear:
    linear_solver_parameters = solver_parameters
  else:
    nl_solver = solver_parameters["nonlinear_solver"] = solver_parameters.get("nonlinear_solver", "newton")
    if nl_solver == "newton":
      linear_solver_parameters = solver_parameters["newton_solver"] = solver_parameters.get("newton_solver", {})
    elif nl_solver == "snes":
      linear_solver_parameters = solver_parameters["snes_solver"] = solver_parameters.get("snes_solver", {})
    else:
      raise InterfaceException("Unsupported non-linear solver: %s" % nl_solver)
  
  linear_solver = linear_solver_parameters["linear_solver"] = linear_solver_parameters.get("linear_solver", "default")
  is_lu_linear_solver = linear_solver in ["default", "direct", "lu"] or has_lu_solver_method(linear_solver)
  if is_lu_linear_solver:
    checkpoint_ic = not linear
  else:
    krylov_solver_parameters = linear_solver_parameters["krylov_solver"] = linear_solver_parameters.get("krylov_solver", {})
    nonzero_initial_guess = krylov_solver_parameters.get("nonzero_initial_guess", False)
    if nonzero_initial_guess is None:
      nonzero_initial_guess = False
    krylov_solver_parameters["nonzero_initial_guess"] = nonzero_initial_guess
    checkpoint_ic = not linear or nonzero_initial_guess
  
  return solver_parameters, linear_solver_parameters, checkpoint_ic

def assemble_matrix(form, bcs, form_compiler_parameters, force_evaluation = True):
  if len(bcs) > 0:
    test = TestFunction(form.arguments()[0].function_space())
    test_shape = test.ufl_element().value_shape()
    dummy_rhs = inner(test, Constant(0.0 if len(test_shape) == 0 else numpy.zeros(test_shape, dtype = numpy.float64))) * dx
    A, b_bc = assemble_system(form, dummy_rhs, bcs, form_compiler_parameters = form_compiler_parameters)
    if b_bc.norm("linf") == 0.0:
      b_bc = None
  else:
    A = assemble(form, form_compiler_parameters = form_compiler_parameters)
    b_bc = None
  return A, b_bc

#def assemble_system(A, b, bcs = [], form_compiler_parameters = {}):

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

def form_form_compiler_parameters(form, form_compiler_parameters):
  (form_data,), _, _, _ = ffc.analysis.analyze_forms((form,), form_compiler_parameters)
  integral_metadata = [integral_data.metadata for integral_data in form_data.integral_data]
  return {"quadrature_rule":ffc.analysis._extract_common_quadrature_rule(integral_metadata),
          "quadrature_degree":ffc.analysis._extract_common_quadrature_degree(integral_metadata)}

def homogenize(bc):
  hbc = DirichletBC(bc)
  hbc.homogenize()
  return hbc

def apply_rhs_bcs(b, hbcs, b_bc = None):
  for bc in hbcs:
    bc.apply(b)
  if not b_bc is None:
    b.axpy(1.0, b_bc)

def matrix_multiply(A, x, addto = None, space_fn = None):
  b = A * x
  if addto is None:
    return b
  else:
    addto.axpy(1.0, b)
    return addto

def is_real_function(x):
  e = x.ufl_element()
  return e.family() == "Real" and e.degree() == 0

def rhs_addto(x, y):
  x.axpy(1.0, y)
