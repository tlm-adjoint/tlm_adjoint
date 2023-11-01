#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    LinearSolver, Interpolator, Parameters, backend_Cofunction,
    backend_Constant, backend_DirichletBC, backend_Function, backend_Matrix,
    backend_assemble, backend_solve, complex_mode, extract_args, homogenize,
    parameters)
from ..interface import (
    check_space_type, check_space_types, space_new, var_assign, var_axpy,
    var_copy, var_inner, var_new_conjugate_dual, var_space, var_space_type)

from ..manager import manager_disabled
from ..override import override_method

from .functions import eliminate_zeros, extract_coefficients

from collections.abc import Sequence
import numpy as np
import petsc4py.PETSc as PETSc
import ufl

__all__ = \
    [
        "complex_mode",

        "assemble_linear_solver",
        "assemble_matrix",
        "linear_solver",
        "matrix_multiply",

        "homogenize",

        "interpolate_expression",

        "assemble",
        "solve"
    ]


_parameters = parameters.setdefault("tlm_adjoint", {})
_parameters.setdefault("Assembly", {})
_parameters["Assembly"].setdefault("match_quadrature", False)
_parameters.setdefault("EquationSolver", {})
_parameters["EquationSolver"].setdefault("enable_jacobian_caching", True)
_parameters["EquationSolver"].setdefault("cache_rhs_assembly", True)
_parameters["EquationSolver"].setdefault("match_quadrature", False)
_parameters["EquationSolver"].setdefault("defer_adjoint_assembly", False)
_parameters.setdefault("assembly_verification", {})
_parameters["assembly_verification"].setdefault("jacobian_tolerance", np.inf)
_parameters["assembly_verification"].setdefault("rhs_tolerance", np.inf)
del _parameters


def copy_parameters_dict(parameters):
    new_parameters = dict(parameters)
    for key, value in parameters.items():
        if isinstance(value, (Parameters, dict)):
            value = copy_parameters_dict(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
        new_parameters[key] = value
    return new_parameters


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


def process_solver_parameters(solver_parameters, linear):
    solver_parameters = copy_parameters_dict(solver_parameters)
    tlm_adjoint_parameters = solver_parameters.setdefault("tlm_adjoint", {})

    tlm_adjoint_parameters.setdefault("options_prefix", None)
    tlm_adjoint_parameters.setdefault("nullspace", None)
    tlm_adjoint_parameters.setdefault("transpose_nullspace", None)
    tlm_adjoint_parameters.setdefault("near_nullspace", None)

    linear_solver_ic = solver_parameters.setdefault("ksp_initial_guess_nonzero", False)  # noqa: E501

    return (solver_parameters, solver_parameters,
            not linear or linear_solver_ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        adjoint_solver_parameters = dict(linear_solver_parameters)
        tlm_adjoint_parameters = adjoint_solver_parameters["tlm_adjoint"] \
            = dict(linear_solver_parameters["tlm_adjoint"])

        tlm_adjoint_parameters["nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["transpose_nullspace"]
        tlm_adjoint_parameters["transpose_nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["nullspace"]

        return adjoint_solver_parameters
    else:
        # Copy not required
        return linear_solver_parameters


def assemble_arguments(arity, form_compiler_parameters, solver_parameters):
    kwargs = {"form_compiler_parameters": form_compiler_parameters}
    if arity == 2 and "mat_type" in solver_parameters:
        kwargs["mat_type"] = solver_parameters["mat_type"]
    return kwargs


def _assemble(form, tensor=None, bcs=None, *,
              form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    form = eliminate_zeros(form, force_non_empty_form=True)
    b = backend_assemble(
        form, tensor=tensor, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters, mat_type=mat_type)

    return b


def _assemble_system(A_form, b_form=None, bcs=None, *,
                     form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    A = _assemble(
        A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
        mat_type=mat_type)

    if len(bcs) > 0:
        F = backend_Function(A_form.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)

        if b_form is None:
            b = _assemble(
                -ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

            with b.dat.vec_ro as b_v:
                if b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) == 0.0:
                    b = None
        else:
            b = _assemble(
                b_form - ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)
    else:
        if b_form is None:
            b = None
        else:
            b = _assemble(
                b_form,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

    A._tlm_adjoint__lift_bcs = False

    return A, b


@override_method(LinearSolver, "_lifted")
def LinearSolver_lifted(self, orig, orig_args, b):
    if getattr(self.A, "_tlm_adjoint__lift_bcs", True):
        return orig_args()
    else:
        return b


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    return _assemble_system(form, bcs=bcs,
                            form_compiler_parameters=form_compiler_parameters,
                            mat_type=mat_type)


def assemble(form, tensor=None, *,
             form_compiler_parameters=None, mat_type=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    b = _assemble(
        form, tensor=tensor, form_compiler_parameters=form_compiler_parameters,
        mat_type=mat_type)

    return b


def assemble_linear_solver(A_form, b_form=None, bcs=None, *,
                           form_compiler_parameters=None,
                           linear_solver_parameters=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if linear_solver_parameters is None:
        linear_solver_parameters = {}

    A, b = _assemble_system(
        A_form, b_form=b_form, bcs=bcs,
        **assemble_arguments(2, form_compiler_parameters,
                             linear_solver_parameters))

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        linear_solver_parameters = dict(linear_solver_parameters)
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
    return LinearSolver(A, solver_parameters=linear_solver_parameters,
                        options_prefix=options_prefix,
                        nullspace=nullspace,
                        transpose_nullspace=transpose_nullspace,
                        near_nullspace=near_nullspace)


def form_compiler_quadrature_parameters(form, form_compiler_parameters):
    qd = form_compiler_parameters.get("quadrature_degree", "auto")
    if qd in {None, "auto", -1}:
        qd = ufl.algorithms.estimate_total_polynomial_degree(form)
    return {"quadrature_degree": qd}


# def homogenize(bc):


def matrix_copy(A):
    if not isinstance(A, backend_Matrix):
        raise TypeError("Unexpected matrix type")

    options_prefix = A.petscmat.getOptionsPrefix()
    A_copy = backend_Matrix(A.a, A.bcs, A.mat_type,
                            A.M.sparsity, A.M.dtype,
                            options_prefix=options_prefix)

    assert A.petscmat.assembled
    A_copy.petscmat.axpy(1.0, A.petscmat)
    assert A_copy.petscmat.assembled

    # MatAXPY does not propagate the options prefix
    A_copy.petscmat.setOptionsPrefix(options_prefix)

    if hasattr(A, "_tlm_adjoint__lift_bcs"):
        A_copy._tlm_adjoint__lift_bcs = A._tlm_adjoint__lift_bcs

    return A_copy


def matrix_multiply(A, x, *,
                    tensor=None, addto=False, action_type="conjugate_dual"):
    if tensor is None:
        tensor = space_new(
            A.a.arguments()[0].function_space(),
            space_type=var_space_type(x, rel_space_type=action_type))
    else:
        check_space_types(tensor, x, rel_space_type=action_type)

    if addto:
        with x.dat.vec_ro as x_v, tensor.dat.vec as tensor_v:
            A.petscmat.multAdd(x_v, tensor_v, tensor_v)
    else:
        with x.dat.vec_ro as x_v, tensor.dat.vec_wo as tensor_v:
            A.petscmat.mult(x_v, tensor_v)

    return tensor


def r0_space(x):
    raise NotImplementedError("r0_space not implemented")


def rhs_copy(x):
    check_space_type(x, "conjugate_dual")
    return var_copy(x)


def rhs_addto(x, y):
    check_space_type(x, "conjugate_dual")
    check_space_type(y, "conjugate_dual")
    var_axpy(x, 1.0, y)


def parameters_key(parameters):
    key = []
    for name in sorted(parameters.keys()):
        sub_parameters = parameters[name]
        if isinstance(sub_parameters, (Parameters, dict)):
            key.append((name, parameters_key(sub_parameters)))
        elif isinstance(sub_parameters, Sequence) \
                and not isinstance(sub_parameters, str):
            key.append((name, tuple(sub_parameters)))
        else:
            key.append((name, sub_parameters))
    return tuple(key)


def verify_assembly(J, rhs, J_mat, b, bcs, form_compiler_parameters,
                    linear_solver_parameters, J_tolerance, b_tolerance):
    if J_mat is not None and not np.isposinf(J_tolerance):
        J_mat_debug = backend_assemble(
            J, bcs=bcs, **assemble_arguments(2,
                                             form_compiler_parameters,
                                             linear_solver_parameters))
        assert J_mat.petscmat.assembled
        J_error = J_mat.petscmat.copy()
        J_error.axpy(-1.0, J_mat_debug.petscmat)
        assert J_error.assembled
        assert J_error.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
            <= J_tolerance * J_mat_debug.petscmat.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501

    if b is not None and not np.isposinf(b_tolerance):
        F = backend_Function(rhs.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)
        b_debug = backend_assemble(
            rhs - ufl.action(J, F), bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        b_error = b.copy(deepcopy=True)
        with b_error.dat.vec as b_error_v, b_debug.dat.vec_ro as b_debug_v:
            b_error_v.axpy(-1.0, b_debug_v)
        with b_error.dat.vec_ro as b_error_v, b_debug.dat.vec_ro as b_v:
            assert b_error_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
                <= b_tolerance * b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501


@manager_disabled()
def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is None:
        check_space_type(x, "primal")
    else:
        check_space_type(x, "conjugate_dual")
        check_space_type(adj_x, "conjugate_dual")
    for dep in extract_coefficients(expr):
        check_space_type(dep, "primal")

    expr = eliminate_zeros(expr)

    if adj_x is None:
        if isinstance(x, backend_Constant):
            x.assign(expr)
        elif isinstance(x, backend_Function):
            x.interpolate(expr)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
    elif isinstance(x, backend_Constant):
        if len(x.ufl_shape) > 0:
            raise ValueError("Scalar Constant required")
        expr_val = var_new_conjugate_dual(adj_x)
        interpolate_expression(expr_val, expr)
        var_assign(x, var_inner(adj_x, expr_val))
    elif isinstance(x, backend_Cofunction):
        adj_x_space = var_space(adj_x)
        interp = Interpolator(expr, adj_x_space)
        adj_x = var_copy(adj_x)
        adj_x.dat.data[:] = adj_x.dat.data_ro.conjugate()
        interp.interpolate(adj_x, transpose=True, output=x)
        x.dat.data[:] = x.dat.data_ro.conjugate()
    else:
        raise TypeError(f"Unexpected type: {type(x)}")


def solve(*args, **kwargs):
    if not isinstance(args[0], ufl.classes.Equation):
        return backend_solve(*args, **kwargs)

    eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
        nullspace, transpose_nullspace, near_nullspace, options_prefix = \
        extract_args(*args, **kwargs)
    check_space_type(x, "primal")
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if solver_parameters is None:
        solver_parameters = {}

    if "tlm_adjoint" in solver_parameters:
        solver_parameters = dict(solver_parameters)
        tlm_adjoint_parameters = solver_parameters.pop("tlm_adjoint")

        if "options_prefix" in tlm_adjoint_parameters:
            if options_prefix is not None:
                raise TypeError("Cannot pass both options_prefix argument and "
                                "solver parameter")
            options_prefix = tlm_adjoint_parameters["options_prefix"]

        if "nullspace" in tlm_adjoint_parameters:
            if nullspace is not None:
                raise TypeError("Cannot pass both nullspace argument and "
                                "solver parameter")
            nullspace = tlm_adjoint_parameters["nullspace"]

        if "transpose_nullspace" in tlm_adjoint_parameters:
            if transpose_nullspace is not None:
                raise TypeError("Cannot pass both transpose_nullspace "
                                "argument and solver parameter")
            transpose_nullspace = tlm_adjoint_parameters["transpose_nullspace"]

        if "near_nullspace" in tlm_adjoint_parameters:
            if near_nullspace is not None:
                raise TypeError("Cannot pass both near_nullspace argument and "
                                "solver parameter")
            near_nullspace = tlm_adjoint_parameters["near_nullspace"]

    return backend_solve(eq, x, bcs, J=J, Jp=Jp, M=M,
                         form_compiler_parameters=form_compiler_parameters,
                         solver_parameters=solver_parameters,
                         nullspace=nullspace,
                         transpose_nullspace=transpose_nullspace,
                         near_nullspace=near_nullspace,
                         options_prefix=options_prefix)
