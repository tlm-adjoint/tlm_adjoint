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

import ffc
import numpy
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
    
    "dolfin_form",
    "clear_dolfin_form",
    
    "assemble",
    "assemble_system",
    "solve"
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
if not "cache_rhs_assembly" in parameters["tlm_adjoint"]["EquationSolver"]:
  parameters["tlm_adjoint"]["EquationSolver"].add("cache_rhs_assembly", True)
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
      
def process_solver_parameters(solver_parameters, J, linear):
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
    lu_solver_parameters = linear_solver_parameters["lu_solver"] = linear_solver_parameters.get("lu_solver", {})
    if not "symmetric" in lu_solver_parameters and J == adjoint(J):
      lu_solver_parameters["symmetric"] = True
    checkpoint_ic = not linear
  else:
    krylov_solver_parameters = linear_solver_parameters["krylov_solver"] = linear_solver_parameters.get("krylov_solver", {})
    nonzero_initial_guess = krylov_solver_parameters.get("nonzero_initial_guess", False)
    if nonzero_initial_guess is None:
      nonzero_initial_guess = False
    krylov_solver_parameters["nonzero_initial_guess"] = nonzero_initial_guess
    checkpoint_ic = not linear or nonzero_initial_guess
  
  return solver_parameters, linear_solver_parameters, checkpoint_ic

def process_adjoint_solver_parameters(linear_solver_parameters):
  return linear_solver_parameters  # Copy not required

def assemble_arguments(rank, form_compiler_parameters, solver_parameters):
  return {"form_compiler_parameters":form_compiler_parameters}

def assemble_matrix(form, bcs, force_evaluation = True, **assemble_kwargs):
  if len(bcs) > 0:
    test = TestFunction(form.arguments()[0].function_space())
    test_shape = test.ufl_element().value_shape()
    dummy_rhs = ufl.inner(test, backend_Constant(0.0 if len(test_shape) == 0 else numpy.zeros(test_shape, dtype = numpy.float64))) * ufl.dx
    A, b_bc = assemble_system(form, dummy_rhs, bcs, **assemble_kwargs)
    if b_bc.norm("linf") == 0.0:
      b_bc = None
  else:
    A = assemble(form, **assemble_kwargs)
    b_bc = None
  return A, b_bc

# Similar interface to assemble_system in FEniCS 2018.1.0
#def assemble_system(A_form, b_form, bcs = [], form_compiler_parameters = {}):

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
  hbc = backend_DirichletBC(bc)
  hbc.homogenize()
  return hbc

def apply_rhs_bcs(b, hbcs, b_bc = None):
  for bc in hbcs:
    bc.apply(b)
  if not b_bc is None:
    b.axpy(1.0, b_bc)

def matrix_multiply(A, x, tensor = None, addto = False):
  if tensor is None:
    return A *x
  else:
    x_v = as_backend_type(x).vec()
    tensor_v = as_backend_type(tensor).vec()
    if addto:
      as_backend_type(A).mat().multAdd(x_v, tensor_v, tensor_v)
    else:
      as_backend_type(A).mat().mult(x_v, tensor_v)
    return tensor

def is_real_function(x):
  e = x.ufl_element()
  return e.family() == "Real" and e.degree() == 0

def rhs_copy(x):
  return x.copy()

def rhs_addto(x, y):
  x.axpy(1.0, y)
  
# The following override assemble, assemble_system, and solve so that DOLFIN
# Form objects are cached on UFL form objects. The first call to assemble,
# assemble_system, or (for supported cases) solve defines the
# form_compiler_parameters used to build the DOLFIN form -- subsequent
# form_compiler_parameters arguments are *ignored*.
  
def dolfin_form(form, form_compiler_parameters):
  if "_tlm_adjoint__form" in form._cache:
    dolfin_form = form._cache["_tlm_adjoint__form"]
    deps = form.coefficients()
    for i, j in enumerate(form._cache["_tlm_adjoint__deps_map"]):
      dolfin_form.set_coefficient(i, deps[j].this if hasattr(deps[j], "this") else deps[j]._cpp_object)
  else:
    dolfin_form = form._cache["_tlm_adjoint__form"] = Form(form, form_compiler_parameters = form_compiler_parameters)
    if not hasattr(dolfin_form, "_compiled_form"): dolfin_form._compiled_form = None  # Work around DOLFIN 2018.1.0 bug
    form._cache["_tlm_adjoint__deps_map"] = tuple(map(dolfin_form.original_coefficient_position, range(dolfin_form.num_coefficients())))
  return dolfin_form

def clear_dolfin_form(form):
  for i in range(form.num_coefficients()):
    form.set_coefficient(i, None)

# Aim for compatibility with FEniCS 2018.1.0 API
   
def assemble(form, tensor = None, form_compiler_parameters = None, *args, **kwargs):
  is_dolfin_form = isinstance(form, Form)
  if not is_dolfin_form: form = dolfin_form(form, form_compiler_parameters)
  return_value = backend_assemble(form, tensor = tensor, *args, **kwargs)
  if not is_dolfin_form: clear_dolfin_form(form)
  return return_value
  
def assemble_system(A_form, b_form, bcs = None, x0 = None,
  form_compiler_parameters = None, *args, **kwargs):
  A_is_dolfin_form = isinstance(A_form, Form)
  b_is_dolfin_form = isinstance(b_form, Form)
  if not A_is_dolfin_form: A_form = dolfin_form(A_form, form_compiler_parameters)
  if not b_is_dolfin_form: b_form = dolfin_form(b_form, form_compiler_parameters)
  return_value = backend_assemble_system(A_form, b_form, bcs = bcs, x0 = x0, *args, **kwargs)
  if not A_is_dolfin_form: clear_dolfin_form(A_form)
  if not b_is_dolfin_form: clear_dolfin_form(b_form)
  return return_value

def solve(*args, **kwargs):
  if not isinstance(args[0], ufl.classes.Equation):
    return backend_solve(*args, **kwargs)
    
  eq, x, bcs, J, tol, M, form_compiler_parameters, solver_parameters = extract_args(*args, **kwargs)
  if not tol is None or not M is None:
    return backend_solve(*args, **kwargs)

  lhs, rhs = eq.lhs, eq.rhs
  linear = isinstance(lhs, ufl.classes.Form) and isinstance(rhs, ufl.classes.Form)
  if linear:
    lhs = dolfin_form(lhs, form_compiler_parameters)
    rhs = dolfin_form(rhs, form_compiler_parameters)
    problem = cpp_LinearVariationalProblem(lhs, rhs, x.this if hasattr(x, "this") else x._cpp_object, bcs)
    solver = LinearVariationalSolver(problem)
    solver.parameters.update(solver_parameters)
    return_value = solver.solve()
    clear_dolfin_form(lhs)
    clear_dolfin_form(rhs)
    return return_value
  else:
    F = lhs
    assert(rhs == 0)
    if J is None:
      if not "_tlm_adjoint__J" in F._cache:
        F._cache["_tlm_adjoint__J"] = ufl.algorithms.expand_derivatives(ufl.derivative(F, x, argument = TrialFunction(x.function_space())))
      J = F._cache["_tlm_adjoint__J"]
    
    F = dolfin_form(F, form_compiler_parameters)
    J = dolfin_form(J, form_compiler_parameters)
    problem = cpp_NonlinearVariationalProblem(F, x.this if hasattr(x, "this") else x._cpp_object, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters.update(solver_parameters)
    return_value = solver.solve()
    clear_dolfin_form(F)
    clear_dolfin_form(J)    
    return return_value
