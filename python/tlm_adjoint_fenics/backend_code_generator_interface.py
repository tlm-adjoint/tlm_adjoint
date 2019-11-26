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
from .functions import ConstantInterface, ConstantSpaceInterface, \
    eliminate_zeros, new_count
from .interface import InterfaceException, add_interface

import ffc
import mpi4py.MPI as MPI
import numpy as np
import ufl

__all__ = \
    [
        "InterfaceException",

        "assemble_arguments",
        "assemble_linear_solver",
        "assemble_matrix",
        "copy_parameters_dict",
        "form_form_compiler_parameters",
        "function_vector",
        "homogenize",
        "linear_solver",
        "matrix_multiply",
        "parameters_key",
        "process_adjoint_solver_parameters",
        "process_solver_parameters",
        "r0_space",
        "rhs_addto",
        "rhs_copy",
        "update_parameters_dict",
        "verify_assembly",

        "dolfin_form",
        "clear_dolfin_form",

        "assemble",
        "assemble_system",
        "solve"
    ]


if "tlm_adjoint" not in parameters:
    parameters.add(Parameters("tlm_adjoint"))
_parameters = parameters["tlm_adjoint"]
if "AssembleSolver" not in _parameters:
    _parameters.add(Parameters("AssembleSolver"))
if "match_quadrature" not in _parameters["AssembleSolver"]:
    _parameters["AssembleSolver"].add("match_quadrature", False)
if "EquationSolver" not in _parameters:
    _parameters.add(Parameters("EquationSolver"))
if "enable_jacobian_caching" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("enable_jacobian_caching", True)
if "cache_rhs_assembly" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("cache_rhs_assembly", True)
if "match_quadrature" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("match_quadrature", False)
if "defer_adjoint_assembly" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("defer_adjoint_assembly", False)
if "assembly_verification" not in _parameters:
    _parameters.add(Parameters("assembly_verification"))
if "jacobian_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"].add("jacobian_tolerance", np.inf)
if "rhs_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"].add("rhs_tolerance", np.inf)
del _parameters


def copy_parameters_dict(parameters):
    new_parameters = {}
    for key in parameters:
        value = parameters[key]
        if isinstance(value, (Parameters, dict)):
            new_parameters[key] = copy_parameters_dict(value)
        elif isinstance(value, list):
            new_parameters[key] = list(value)
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
        if "nonlinear_solver" not in solver_parameters:
            solver_parameters["nonlinear_solver"] = "newton"
        nl_solver = solver_parameters["nonlinear_solver"]
        if nl_solver == "newton":
            if "newton_solver" not in solver_parameters:
                solver_parameters["newton_solver"] = {}
            linear_solver_parameters = solver_parameters["newton_solver"]
        elif nl_solver == "snes":
            if "snes_solver" not in solver_parameters:
                solver_parameters["snes_solver"] = {}
            linear_solver_parameters = solver_parameters["snes_solver"]
        else:
            raise InterfaceException(f"Unsupported non-linear solver: {nl_solver}")  # noqa: E501

    if "linear_solver" not in linear_solver_parameters:
        linear_solver_parameters["linear_solver"] = "default"
    linear_solver = linear_solver_parameters["linear_solver"]
    is_lu_linear_solver = linear_solver in ["default", "direct", "lu"] \
        or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
        if "lu_solver" not in linear_solver_parameters:
            linear_solver_parameters["lu_solver"] = {}
        lu_parameters = linear_solver_parameters["lu_solver"]
        if "symmetric" not in lu_parameters and J == adjoint(J):
            lu_parameters["symmetric"] = True
        checkpoint_ic = not linear
    else:
        if "krylov_solver" not in linear_solver_parameters:
            linear_solver_parameters["krylov_solver"] = {}
        ks_parameters = linear_solver_parameters["krylov_solver"]
        if "nonzero_initial_guess" not in ks_parameters:
            ks_parameters["nonzero_initial_guess"] = False
        nonzero_initial_guess = ks_parameters["nonzero_initial_guess"]
        checkpoint_ic = not linear or nonzero_initial_guess

    return solver_parameters, linear_solver_parameters, checkpoint_ic


def process_adjoint_solver_parameters(linear_solver_parameters):
    # Copy not required
    return linear_solver_parameters


def assemble_arguments(rank, form_compiler_parameters, solver_parameters):
    return {"form_compiler_parameters": form_compiler_parameters}


def assemble_matrix(form, bcs=[], form_compiler_parameters={},
                    *args, **kwargs):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)

    if len(bcs) > 0:
        test = TestFunction(form.arguments()[0].function_space())
        if len(test.ufl_shape) == 0:
            zero = backend_Constant(0.0)
        else:
            zero = backend_Constant(np.zeros(test.ufl_shape, dtype=np.float64))
        dummy_rhs = ufl.inner(test, zero) * ufl.dx
        A, b_bc = assemble_system(
            form, dummy_rhs, bcs,
            form_compiler_parameters=form_compiler_parameters, *args, **kwargs)
        if b_bc.norm("linf") == 0.0:
            b_bc = None
    else:
        A = assemble(
            form, form_compiler_parameters=form_compiler_parameters,
            *args, **kwargs)
        b_bc = None

    return A, b_bc


# def assemble(form, tensor=None, form_compiler_parameters={}, *args,
#              **kwargs):
#     # Similar interface to assemble in FEniCS 2019.1.0


# def assemble_system(A_form, b_form, bcs=[], form_compiler_parameters={},
#                     *args, **kwargs):
#     # Similar interface to assemble_system in FEniCS 2019.1.0


def assemble_linear_solver(A_form, b_form=None, bcs=[],
                           form_compiler_parameters={},
                           linear_solver_parameters={}):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)

    if b_form is None:
        A, b = assemble_matrix(
            A_form, bcs, form_compiler_parameters=form_compiler_parameters)
        solver = linear_solver(A, linear_solver_parameters)
    else:
        A, b = assemble_system(
            A_form, b_form, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    linear_solver = linear_solver_parameters.get("linear_solver", "default")
    if linear_solver in ["direct", "lu"]:
        linear_solver = "default"
    elif linear_solver == "iterative":
        linear_solver = "gmres"
    is_lu_linear_solver = linear_solver == "default" \
        or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
        solver = backend_LUSolver(A, linear_solver)
        lu_parameters = linear_solver_parameters.get("lu_solver", {})
        update_parameters_dict(solver.parameters, lu_parameters)
    else:
        pc = linear_solver_parameters.get("preconditioner", "default")
        ks_parameters = linear_solver_parameters.get("krylov_solver", {})
        solver = backend_KrylovSolver(A, linear_solver, pc)
        update_parameters_dict(solver.parameters, ks_parameters)
    return solver


def form_form_compiler_parameters(form, form_compiler_parameters):
    (form_data,), _, _, _ \
        = ffc.analysis.analyze_forms((form,), form_compiler_parameters)
    integral_metadata = tuple(integral_data.metadata
                              for integral_data in form_data.integral_data)
    qr = ffc.analysis._extract_common_quadrature_rule(integral_metadata)
    qd = ffc.analysis._extract_common_quadrature_degree(integral_metadata)
    return {"quadrature_rule": qr, "quadrature_degree": qd}


def homogenize(bc):
    hbc = backend_DirichletBC(bc)
    hbc.homogenize()
    return hbc


def matrix_multiply(A, x, tensor=None, addto=False):
    if tensor is None:
        return A * x
    else:
        x_v = as_backend_type(x).vec()
        tensor_v = as_backend_type(tensor).vec()
        if addto:
            as_backend_type(A).mat().multAdd(x_v, tensor_v, tensor_v)
        else:
            as_backend_type(A).mat().mult(x_v, tensor_v)
        return tensor


def r0_space(x):
    if not hasattr(x, "_tlm_adjoint__r0_space"):
        x_domains = x.ufl_domains()
        if len(x_domains) == 0:
            raise InterfaceException("Domain not defined")
        domain, = x_domains
        domain = domain.ufl_cargo()
        if len(x.ufl_shape) == 0:
            space = FunctionSpace(domain, "R", 0)
        else:
            space = TensorFunctionSpace(domain, "R", degree=0,
                                        shape=x.ufl_shape)
        x._tlm_adjoint__r0_space = space
    return x._tlm_adjoint__r0_space


def _Constant__init__(self, *args, name=None, domain=None, space=None,
                      comm=MPI.COMM_WORLD, **kwargs):
    if domain is not None and hasattr(domain, "ufl_domain"):
        domain = domain.ufl_domain()

    _Constant__init__._tlm_adjoint__orig(self, *args, name=name, **kwargs)

    self.ufl_domain = lambda: domain
    if domain is None:
        self.ufl_domains = lambda: ()
    else:
        self.ufl_domains = lambda: (domain,)

    if space is None:
        space = self.ufl_function_space()
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm, "domain": domain, "id": new_count()})
    add_interface(self, ConstantInterface,
                  {"space": space})


_Constant__init__._tlm_adjoint__orig = backend_Constant.__init__
backend_Constant.__init__ = _Constant__init__


def function_vector(x):
    return x.vector()


def rhs_copy(x):
    return x.copy()


def rhs_addto(x, y):
    x.axpy(1.0, y)


def parameters_key(parameters):
    key = []
    for name in sorted(parameters.keys()):
        sub_parameters = parameters[name]
        if isinstance(sub_parameters, (Parameters, dict)):
            key.append((name, parameters_key(sub_parameters)))
        elif isinstance(sub_parameters, list):
            key.append((name, tuple(sub_parameters)))
        else:
            key.append((name, sub_parameters))
    return tuple(key)


def verify_assembly(J, rhs, J_mat, b, bcs, form_compiler_parameters,
                    linear_solver_parameters, J_tolerance, b_tolerance):
    if np.isposinf(J_tolerance) and np.isposinf(b_tolerance):
        return

    J_mat_debug, b_debug = backend_assemble_system(
        J, rhs, bcs, form_compiler_parameters=form_compiler_parameters)

    if not np.isposinf(J_tolerance):
        assert (J_mat - J_mat_debug).norm("linf") \
            <= J_tolerance * J_mat.norm("linf")

    if not np.isposinf(b_tolerance):
        assert (b - b_debug).norm("linf") <= b_tolerance * b.norm("linf")

# The following override assemble, assemble_system, and solve so that DOLFIN
# Form objects are cached on UFL form objects


def dolfin_form(form, form_compiler_parameters=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = parameters["form_compiler"]

    if "_tlm_adjoint__form" in form._cache and \
       parameters_key(form_compiler_parameters) != \
       form._cache["_tlm_adjoint__form_compiler_parameters_key"]:
        del form._cache["_tlm_adjoint__form"]
        del form._cache["_tlm_adjoint__deps_map"]
        del form._cache["_tlm_adjoint__form_compiler_parameters_key"]

    if "_tlm_adjoint__form" in form._cache:
        dolfin_form = form._cache["_tlm_adjoint__form"]
        bindings = form._cache.get("_tlm_adjoint__bindings", None)
        if bindings is None:
            deps = form.coefficients()
        else:
            deps = tuple(bindings.get(c, c) for c in form.coefficients())
        for i, j in enumerate(form._cache["_tlm_adjoint__deps_map"]):
            # FEniCS backwards compatibility
            if hasattr(deps[j], "this"):
                cpp_object = deps[j].this
            else:
                cpp_object = deps[j]._cpp_object
            dolfin_form.set_coefficient(i, cpp_object)
    else:
        bindings = form._cache.get("_tlm_adjoint__bindings", None)
        if bindings is not None:
            # FEniCS backwards compatibility
            dep_this = {}
            dep_cpp_object = {}
            for dep in form.coefficients():
                if dep in bindings:
                    dep_binding = bindings[dep]
                    # FEniCS backwards compatibility
                    if hasattr(dep_binding, "this"):
                        dep_this[dep] = (dep.__class__,
                                         getattr(dep, "this", None))
                        dep.__class__ = dep_binding.__class__
                        dep.this = dep_binding.this
                    else:
                        dep_cpp_object[dep] = getattr(dep, "_cpp_object", None)
                        dep._cpp_object = dep_binding._cpp_object

        simplified_form = eliminate_zeros(form, non_empty_form=True)
        dolfin_form = Form(
            simplified_form, form_compiler_parameters=form_compiler_parameters)
        if not hasattr(dolfin_form, "_compiled_form"):
            dolfin_form._compiled_form = None

        if bindings is not None:
            # FEniCS backwards compatibility
            for dep, (cls, this) in dep_this.items():
                dep.__class__ = cls
                if this is None:
                    del dep.this
                else:
                    dep.this = this
            for dep, cpp_object in dep_cpp_object.items():
                if cpp_object is None:
                    del dep._cpp_object
                else:
                    dep._cpp_object = cpp_object

        form._cache["_tlm_adjoint__form"] = dolfin_form
        form._cache["_tlm_adjoint__deps_map"] = \
            tuple(map(dolfin_form.original_coefficient_position,
                      range(dolfin_form.num_coefficients())))
        form._cache["_tlm_adjoint__form_compiler_parameters_key"] = \
            parameters_key(form_compiler_parameters)
    return dolfin_form


def clear_dolfin_form(form):
    for i in range(form.num_coefficients()):
        form.set_coefficient(i, None)

# Aim for compatibility with FEniCS 2019.1.0 API


def assemble(form, tensor=None, form_compiler_parameters=None,
             *args, **kwargs):
    is_dolfin_form = isinstance(form, Form)
    if not is_dolfin_form:
        form = dolfin_form(form, form_compiler_parameters)
    return_value = backend_assemble(form, tensor=tensor, *args, **kwargs)
    if not is_dolfin_form:
        clear_dolfin_form(form)
    return return_value


def assemble_system(A_form, b_form, bcs=None, x0=None,
                    form_compiler_parameters=None, *args, **kwargs):
    A_is_dolfin_form = isinstance(A_form, Form)
    b_is_dolfin_form = isinstance(b_form, Form)
    if not A_is_dolfin_form:
        A_form = dolfin_form(A_form, form_compiler_parameters)
    if not b_is_dolfin_form:
        b_form = dolfin_form(b_form, form_compiler_parameters)
    return_value = backend_assemble_system(A_form, b_form, bcs=bcs, x0=x0,
                                           *args, **kwargs)
    if not A_is_dolfin_form:
        clear_dolfin_form(A_form)
    if not b_is_dolfin_form:
        clear_dolfin_form(b_form)
    return return_value


def solve(*args, **kwargs):
    if not isinstance(args[0], ufl.classes.Equation):
        return backend_solve(*args, **kwargs)

    eq, x, bcs, J, tol, M, form_compiler_parameters, solver_parameters \
        = extract_args(*args, **kwargs)
    # FEniCS backwards compatibility
    bcs = list(bcs)
    if tol is not None or M is not None:
        return backend_solve(*args, **kwargs)

    lhs, rhs = eq.lhs, eq.rhs
    linear = isinstance(rhs, ufl.classes.Form)
    if linear:
        lhs = dolfin_form(lhs, form_compiler_parameters)
        rhs = dolfin_form(rhs, form_compiler_parameters)
        # FEniCS backwards compatibility
        if hasattr(x, "this"):
            cpp_object = x.this
        else:
            cpp_object = x._cpp_object
        problem = cpp_LinearVariationalProblem(lhs, rhs, cpp_object, bcs)
        solver = backend_LinearVariationalSolver(problem)
        solver.parameters.update(solver_parameters)
        return_value = solver.solve()
        clear_dolfin_form(lhs)
        clear_dolfin_form(rhs)
        return return_value
    else:
        F = lhs
        assert rhs == 0
        if J is None:
            if "_tlm_adjoint__J" in F._cache:
                J = F._cache["_tlm_adjoint__J"]
            else:
                J = ufl.derivative(F, x,
                                   argument=TrialFunction(x.function_space()))
                J = ufl.algorithms.expand_derivatives(J)
                F._cache["_tlm_adjoint__J"] = J

        F = dolfin_form(F, form_compiler_parameters)
        J = dolfin_form(J, form_compiler_parameters)
        # FEniCS backwards compatibility
        if hasattr(x, "this"):
            cpp_object = x.this
        else:
            cpp_object = x._cpp_object
        problem = cpp_NonlinearVariationalProblem(F, cpp_object, bcs, J)
        solver = backend_NonlinearVariationalSolver(problem)
        solver.parameters.update(solver_parameters)
        return_value = solver.solve()
        clear_dolfin_form(F)
        clear_dolfin_form(J)
        return return_value
