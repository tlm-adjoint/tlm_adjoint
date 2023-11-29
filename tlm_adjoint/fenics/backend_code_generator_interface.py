#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    LUSolver, KrylovSolver, Parameters, TestFunction, UserExpression,
    as_backend_type, backend_Constant, backend_DirichletBC, backend_Function,
    backend_ScalarType, backend_Vector, backend_assemble,
    backend_assemble_system, backend_solve as solve, complex_mode,
    has_lu_solver_method, parameters)
from ..interface import (
    check_space_type, check_space_types, is_var, space_new, var_assign,
    var_get_values, var_inner, var_new_conjugate_dual, var_set_values,
    var_space, var_space_type)

from ..manager import manager_disabled

from .functions import eliminate_zeros, extract_coefficients

from collections.abc import Sequence
import ffc
import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
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


if "tlm_adjoint" not in parameters:
    parameters.add(Parameters("tlm_adjoint"))
_parameters = parameters["tlm_adjoint"]
if "Assembly" not in _parameters:
    _parameters.add(Parameters("Assembly"))
if "match_quadrature" not in _parameters["Assembly"]:
    _parameters["Assembly"].add("match_quadrature", False)
if "EquationSolver" not in _parameters:
    _parameters.add(Parameters("EquationSolver"))
if "enable_jacobian_caching" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("enable_jacobian_caching", True)
if "cache_rhs_assembly" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("cache_rhs_assembly", True)
if "match_quadrature" not in _parameters["EquationSolver"]:
    _parameters["EquationSolver"].add("match_quadrature", False)
if "assembly_verification" not in _parameters:
    _parameters.add(Parameters("assembly_verification"))
if "jacobian_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"].add("jacobian_tolerance", np.inf)
if "rhs_tolerance" not in _parameters["assembly_verification"]:
    _parameters["assembly_verification"].add("rhs_tolerance", np.inf)
del _parameters


def copy_parameters_dict(parameters):
    if isinstance(parameters, Parameters):
        parameters = dict(parameters)
    new_parameters = {}
    for key in parameters:
        value = parameters[key]
        if isinstance(value, (Parameters, dict)):
            value = copy_parameters_dict(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
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


def process_solver_parameters(solver_parameters, linear):
    solver_parameters = copy_parameters_dict(solver_parameters)
    if linear:
        linear_solver_parameters = solver_parameters
    else:
        nl_solver = solver_parameters.setdefault("nonlinear_solver", "newton")
        if nl_solver == "newton":
            linear_solver_parameters = solver_parameters.setdefault("newton_solver", {})  # noqa: E501
        elif nl_solver == "snes":
            linear_solver_parameters = solver_parameters.setdefault("snes_solver", {})  # noqa: E501
        else:
            raise ValueError(f"Unsupported non-linear solver: {nl_solver}")

    linear_solver = linear_solver_parameters.setdefault("linear_solver", "default")  # noqa: E501
    is_lu_linear_solver = linear_solver in {"default", "direct", "lu"} \
        or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
        linear_solver_parameters.setdefault("lu_solver", {})
        linear_solver_ic = False
    else:
        ks_parameters = linear_solver_parameters.setdefault("krylov_solver", {})  # noqa: E501
        linear_solver_ic = ks_parameters.setdefault("nonzero_initial_guess", False)  # noqa: E501

    return (solver_parameters, linear_solver_parameters,
            not linear or linear_solver_ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    # Copy not required
    return linear_solver_parameters


def assemble_arguments(arity, form_compiler_parameters, solver_parameters):
    return {"form_compiler_parameters": form_compiler_parameters}


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    if len(bcs) > 0:
        test = TestFunction(form.arguments()[0].function_space())
        if len(test.ufl_shape) == 0:
            zero = backend_Constant(0.0)
        else:
            zero = backend_Constant(np.zeros(test.ufl_shape,
                                             dtype=backend_ScalarType))
        dummy_rhs = ufl.inner(zero, test) * ufl.dx
        A, b_bc = assemble_system(
            form, dummy_rhs, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        if b_bc.norm("linf") == 0.0:
            b_bc = None
    else:
        A = assemble(
            form, form_compiler_parameters=form_compiler_parameters)
        b_bc = None

    return A, b_bc


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

    if b_form is None:
        A, b = assemble_matrix(
            A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
    else:
        A, b = assemble_system(
            A_form, b_form, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    linear_solver = linear_solver_parameters.get("linear_solver", "default")
    if linear_solver in {"direct", "lu"}:
        linear_solver = "default"
    elif linear_solver == "iterative":
        linear_solver = "gmres"
    is_lu_linear_solver = linear_solver == "default" \
        or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
        solver = LUSolver(A, linear_solver)
        lu_parameters = linear_solver_parameters.get("lu_solver", {})
        update_parameters_dict(solver.parameters, lu_parameters)
    else:
        pc = linear_solver_parameters.get("preconditioner", "default")
        ks_parameters = linear_solver_parameters.get("krylov_solver", {})
        solver = KrylovSolver(A, linear_solver, pc)
        update_parameters_dict(solver.parameters, ks_parameters)
    return solver


def form_compiler_quadrature_parameters(form, form_compiler_parameters):
    (form_data,), _, _, _ \
        = ffc.analysis.analyze_forms((form,), form_compiler_parameters)
    integral_metadata = tuple(integral_data.metadata
                              for integral_data in form_data.integral_data)
    qr = form_compiler_parameters.get("quadrature_rule", "auto")
    if qr in {None, "auto"}:
        qr = ffc.analysis._extract_common_quadrature_rule(integral_metadata)
    qd = form_compiler_parameters.get("quadrature_degree", "auto")
    if qd in {None, "auto", -1}:
        qd = ffc.analysis._extract_common_quadrature_degree(integral_metadata)
    return {"quadrature_rule": qr, "quadrature_degree": qd}


def homogenize(bc):
    hbc = backend_DirichletBC(bc)
    hbc.homogenize()
    return hbc


def matrix_copy(A):
    return A.copy()


def matrix_multiply(A, x, *,
                    tensor=None, addto=False, action_type="conjugate_dual"):
    if isinstance(x, backend_Function):
        x = x.vector()
    if tensor is not None and isinstance(tensor, backend_Function):
        tensor = tensor.vector()
    if tensor is None:
        if hasattr(A, "_tlm_adjoint__form") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
            tensor = space_new(
                A._tlm_adjoint__form.arguments()[0].function_space(),
                space_type=var_space_type(x._tlm_adjoint__function,
                                          rel_space_type=action_type))
            tensor = tensor.vector()
        else:
            return A * x
    elif hasattr(tensor, "_tlm_adjoint__function") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
        check_space_types(tensor._tlm_adjoint__function,
                          x._tlm_adjoint__function,
                          rel_space_type=action_type)

    x_v = as_backend_type(x).vec()
    tensor_v = as_backend_type(tensor).vec()
    if addto:
        as_backend_type(A).mat().multAdd(x_v, tensor_v, tensor_v)
    else:
        as_backend_type(A).mat().mult(x_v, tensor_v)
    tensor.apply("insert")

    return tensor


def rhs_copy(x):
    if not isinstance(x, backend_Vector):
        raise TypeError("Invalid RHS")
    if hasattr(x, "_tlm_adjoint__function"):
        check_space_type(x._tlm_adjoint__function, "conjugate_dual")
    return x.copy()


def rhs_addto(x, y):
    if not isinstance(x, backend_Vector):
        raise TypeError("Invalid RHS")
    if not isinstance(y, backend_Vector):
        raise TypeError("Invalid RHS")
    if hasattr(x, "_tlm_adjoint__function"):
        check_space_type(x._tlm_adjoint__function, "conjugate_dual")
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_type(y._tlm_adjoint__function, "conjugate_dual")
    x.axpy(1.0, y)


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
    if np.isposinf(J_tolerance) and np.isposinf(b_tolerance):
        return

    J_mat_debug, b_debug = backend_assemble_system(
        J, rhs, bcs=bcs, form_compiler_parameters=form_compiler_parameters)

    if J_mat is not None and not np.isposinf(J_tolerance):
        assert (J_mat - J_mat_debug).norm("linf") \
            <= J_tolerance * J_mat_debug.norm("linf")

    if b is not None and not np.isposinf(b_tolerance):
        assert (b - b_debug).norm("linf") <= b_tolerance * b_debug.norm("linf")


@manager_disabled()
def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is None:
        check_space_type(x, "primal")
    else:
        check_space_type(x, "conjugate_dual")
        check_space_type(adj_x, "conjugate_dual")
    for dep in extract_coefficients(expr):
        if is_var(dep):
            check_space_type(dep, "primal")

    expr = eliminate_zeros(expr)

    class Expr(UserExpression):
        def eval(self, value, x):
            value[:] = expr(tuple(x))

        def value_shape(self):
            return x.ufl_shape

    if adj_x is None:
        if isinstance(x, backend_Constant):
            if isinstance(expr, backend_Constant):
                value = expr
            else:
                if len(x.ufl_shape) > 0:
                    raise ValueError("Scalar Constant required")
                value = x.values()
                Expr().eval(value, ())
                value, = value
            var_assign(x, value)
        elif isinstance(x, backend_Function):
            try:
                x.assign(expr)
            except RuntimeError:
                x.interpolate(Expr())
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
    else:
        expr_val = var_new_conjugate_dual(adj_x)
        expr_arguments = ufl.algorithms.extract_arguments(expr)
        if len(expr_arguments) > 0:
            test, = expr_arguments
            if len(test.ufl_shape) > 0:
                raise NotImplementedError("Case not implemented")
            expr = ufl.replace(expr, {test: ufl.classes.IntValue(1)})
        interpolate_expression(expr_val, expr)

        if isinstance(x, backend_Constant):
            if len(x.ufl_shape) > 0:
                raise ValueError("Scalar Constant required")
            var_assign(x, var_inner(adj_x, expr_val))
        elif isinstance(x, backend_Function):
            x_space = var_space(x)
            adj_x_space = var_space(adj_x)
            if x_space.ufl_domains() != adj_x_space.ufl_domains() \
                    or x_space.ufl_element() != adj_x_space.ufl_element():
                raise ValueError("Unable to perform transpose interpolation")
            var_set_values(
                x, var_get_values(expr_val).conjugate() * var_get_values(adj_x))  # noqa: E501
        else:
            raise TypeError(f"Unexpected type: {type(x)}")


def assemble(form, tensor=None, *,
             form_compiler_parameters=None):
    if tensor is not None and hasattr(tensor, "_tlm_adjoint__function"):
        check_space_type(tensor._tlm_adjoint__function, "conjugate_dual")

    return backend_assemble(form, tensor,
                            form_compiler_parameters=form_compiler_parameters)


def assemble_system(A_form, b_form, bcs=None, *,
                    form_compiler_parameters=None):
    return backend_assemble_system(
        A_form, b_form, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters)


# def solve(*args, **kwargs):
