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
from .interface import InterfaceException

import copy
import numpy as np
import petsc4py.PETSc as PETSc
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
        "is_real_function",
        "linear_solver",
        "matrix_multiply",
        "parameters_key",
        "process_adjoint_solver_parameters",
        "process_solver_parameters",
        "rhs_addto",
        "rhs_copy",
        "update_parameters_dict",
        "verify_assembly",

        "assemble",
        "assemble_system",
        "solve"
    ]


if "tlm_adjoint" not in parameters:
    parameters["tlm_adjoint"] = {}
_parameters = parameters["tlm_adjoint"]
if "AssembleSolver" not in _parameters:
    _parameters["AssembleSolver"] = {}
if "match_quadrature" not in _parameters["AssembleSolver"]:
    _parameters["AssembleSolver"]["match_quadrature"] = False
if "EquationSolver" not in _parameters:
    _parameters["EquationSolver"] = {}
if "enable_jacobian_caching" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"]["enable_jacobian_caching"] = True
if "cache_rhs_assembly" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"]["cache_rhs_assembly"] = True
if "match_quadrature" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"]["match_quadrature"] = False
if "defer_adjoint_assembly" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"]["defer_adjoint_assembly"] = False
if "assembly_verification" not in _parameters:
    _parameters["assembly_verification"] = {}
if "jacobian_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"]["jacobian_tolerance"] = np.inf
if "rhs_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"]["rhs_tolerance"] = np.inf
del(_parameters)


def copy_parameters_dict(parameters):
    parameters_copy = parameters.copy()
    for key, value in parameters.items():
        if isinstance(value, (Parameters, dict)):
            parameters_copy[key] = copy_parameters_dict(value)
        elif isinstance(value, list):
            parameters_copy[key] = list(value)
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

    if "options_prefix" not in tlm_adjoint_parameters:
        tlm_adjoint_parameters["options_prefix"] = None

    if "nullspace" not in tlm_adjoint_parameters:
        tlm_adjoint_parameters["nullspace"] = None

    if "transpose_nullspace" not in tlm_adjoint_parameters:
        tlm_adjoint_parameters["transpose_nullspace"] = None

    if "near_nullspace" not in tlm_adjoint_parameters:
        tlm_adjoint_parameters["near_nullspace"] = None

    if linear:
        if "ksp_initial_guess_nonzero" not in solver_parameters:
            solver_parameters["ksp_initial_guess_nonzero"] = False
        checkpoint_ic = solver_parameters["ksp_initial_guess_nonzero"]
    else:
        checkpoint_ic = True

    return solver_parameters, solver_parameters, checkpoint_ic


def process_adjoint_solver_parameters(linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        adjoint_solver_parameters = copy.copy(linear_solver_parameters)
        tlm_adjoint_parameters = adjoint_solver_parameters["tlm_adjoint"] \
            = copy.copy(linear_solver_parameters["tlm_adjoint"])

        tlm_adjoint_parameters["nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["transpose_nullspace"]
        tlm_adjoint_parameters["transpose_nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["nullspace"]

        return adjoint_solver_parameters
    else:
        # Copy not required
        return linear_solver_parameters


def assemble_arguments(rank, form_compiler_parameters, solver_parameters):
    kwargs = {"form_compiler_parameters": form_compiler_parameters}
    if rank == 2 and "mat_type" in solver_parameters:
        kwargs["mat_type"] = solver_parameters["mat_type"]
    return kwargs


_form_binding_names = ("dat",
                       "split")


def form_bindings(*forms):
    if len(forms) == 1:
        if "_tlm_adjoint__bindings" in forms[0]._cache:
            for dep, binding in forms[0]._cache["_tlm_adjoint__bindings"].items():  # noqa: E501
                yield dep, binding
    else:
        seen = set()
        for form in forms:
            if "_tlm_adjoint__bindings" in form._cache:
                for dep, binding in form._cache["_tlm_adjoint__bindings"].items():  # noqa: E501
                    if dep not in seen:
                        seen.add(dep)
                        yield dep, binding


def bind_forms(*forms):
    for dep, binding in form_bindings(*forms):
        for name in _form_binding_names:
            assert not hasattr(dep, name)
            setattr(dep, name, getattr(binding, name))


def unbind_forms(*forms):
    for dep, binding in form_bindings(*forms):
        for name in _form_binding_names:
            delattr(dep, name)


def _assemble_system(A_form, b_form=None, bcs=[], form_compiler_parameters={},
                     *args, **kwargs):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if b_form is None:
        bind_forms(A_form)
    else:
        bind_forms(A_form, b_form)

    A = backend_assemble(
        A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
        *args, **kwargs)

    if len(bcs) > 0:
        F = backend_Function(A_form.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)

        if b_form is None:
            b = backend_assemble(
                -ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)

            with b.dat.vec_ro as b_v:
                if b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) == 0.0:
                    b = None
        else:
            b = backend_assemble(
                b_form - ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)
    else:
        if b_form is None:
            b = None
        else:
            b = backend_assemble(
                b_form,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)

    A._tlm_adjoint__lift_bcs = False

    if b_form is None:
        unbind_forms(A_form)
    else:
        unbind_forms(A_form, b_form)
    return A, b


_orig_LinearSolver_lifted = backend_LinearSolver._lifted


def _LinearSolver_lifted(self, b):
    if getattr(self.A, "_tlm_adjoint__lift_bcs", True):
        return _orig_LinearSolver_lifted(self, b)
    else:
        return b


backend_LinearSolver._lifted = _LinearSolver_lifted


def assemble_matrix(form, bcs=[], form_compiler_parameters={},
                    *args, **kwargs):
    return _assemble_system(form, bcs=bcs,
                            form_compiler_parameters=form_compiler_parameters,
                            *args, **kwargs)


def assemble(form, tensor=None, form_compiler_parameters={}, *args,
             **kwargs):
    # Similar interface to assemble in FEniCS 2019.1.0
    bind_forms(form)
    tensor = backend_assemble(
        form, tensor=tensor, form_compiler_parameters=form_compiler_parameters,
        *args, **kwargs)
    unbind_forms(form)
    return tensor


def assemble_system(A_form, b_form, bcs=[], form_compiler_parameters={},
                    *args, **kwargs):
    # Similar interface to assemble_system in FEniCS 2019.1.0
    return _assemble_system(
        A_form, b_form=b_form, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters, *args, **kwargs)


def assemble_linear_solver(A_form, b_form=None, bcs=[],
                           form_compiler_parameters={},
                           linear_solver_parameters={}):
    A, b = _assemble_system(
        A_form, b_form=b_form, bcs=bcs,
        **assemble_arguments(2, form_compiler_parameters,
                             linear_solver_parameters))

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        linear_solver_parameters = copy.copy(linear_solver_parameters)
        tlm_adjoint_parameters = linear_solver_parameters.pop("tlm_adjoint")
        options_prefix = tlm_adjoint_parameters.get("options_prefix", None)
        nullspace = tlm_adjoint_parameters.get("nullspace", None)
        transpose_nullspace = tlm_adjoint_parameters.get("transpose_nullspace",
                                                         None)
        near_nullspace = tlm_adjoint_parameters.get("near_nullspace", None)
    else:
        options_prefix = None
        nullspace = None
        transpose_nullspace = None
        near_nullspace = None
    return backend_LinearSolver(A, solver_parameters=linear_solver_parameters,
                                options_prefix=options_prefix,
                                nullspace=nullspace,
                                transpose_nullspace=transpose_nullspace,
                                near_nullspace=near_nullspace)


def form_form_compiler_parameters(form, form_compiler_parameters):
    qd = ufl.algorithms.estimate_total_polynomial_degree(form)
    return {"quadrature_degree": qd}


# def homogenize(bc):


def matrix_multiply(A, x, tensor=None, addto=False):
    if tensor is None:
        tensor = backend_Function(A.a.arguments()[0].function_space())
    if addto:
        with x.dat.vec_ro as x_v, tensor.dat.vec as tensor_v:
            A.petscmat.multAdd(x_v, tensor_v, tensor_v)
    else:
        with x.dat.vec_ro as x_v, tensor.dat.vec_wo as tensor_v:
            A.petscmat.mult(x_v, tensor_v)
    return tensor


def is_real_function(x):
    e = x.ufl_element()
    return e.family() == "Real" and e.degree() == 0


def function_vector(x):
    return x


def rhs_copy(x):
    return x.copy(deepcopy=True)


def rhs_addto(x, y):
    if is_real_function(x):
        # Work around Firedrake bug (related to issue #1459?)
        x.dat.data[:] += y.dat.data_ro
    else:
        with x.dat.vec as x_v, y.dat.vec_ro as y_v:
            x_v.axpy(1.0, y_v)


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
    if not np.isposinf(J_tolerance):
        J_mat_debug = backend_assemble(
            J, bcs=bcs, **assemble_arguments(2,
                                             form_compiler_parameters,
                                             linear_solver_parameters))
        J_error = J_mat.petscmat.copy()
        J_error.axpy(-1.0, J_mat_debug.petscmat)
        assert J_error.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
            <= J_tolerance * J_mat.petscmat.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501

    if not np.isposinf(b_tolerance):
        b_debug = backend_assemble(
            rhs, form_compiler_parameters=form_compiler_parameters)
        b_error = b.copy(deepcopy=True)
        with b_error.dat.vec as b_error_v, b_debug.dat.vec_ro as b_debug_v:
            b_error_v.axpy(-1.0, b_debug_v)
        with b_error.dat.vec_ro as b_error_v, b.dat.vec_ro as b_v:
            assert b_error_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
                <= b_tolerance * b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501


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
            if options_prefix is not None:
                raise InterfaceException("Cannot pass both options_prefix argument and solver parameter")  # noqa: E501
            options_prefix = tlm_adjoint_parameters["options_prefix"]

        if "nullspace" in tlm_adjoint_parameters:
            if nullspace is not None:
                raise InterfaceException("Cannot pass both nullspace argument and solver parameter")  # noqa: E501
            nullspace = tlm_adjoint_parameters["nullspace"]

        if "transpose_nullspace" in tlm_adjoint_parameters:
            if transpose_nullspace is not None:
                raise InterfaceException("Cannot pass both transpose_nullspace argument and solver parameter")  # noqa: E501
            transpose_nullspace = tlm_adjoint_parameters["transpose_nullspace"]

        if "near_nullspace" in tlm_adjoint_parameters:
            if near_nullspace is not None:
                raise InterfaceException("Cannot pass both near_nullspace argument and solver parameter")  # noqa: E501
            near_nullspace = tlm_adjoint_parameters["near_nullspace"]

    return backend_solve(eq, x, bcs, J=J, Jp=Jp, M=M,
                         form_compiler_parameters=form_compiler_parameters,
                         solver_parameters=solver_parameters,
                         nullspace=nullspace,
                         transpose_nullspace=transpose_nullspace,
                         near_nullspace=near_nullspace,
                         options_prefix=options_prefix)
