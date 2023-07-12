#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    FormAssembler, Interpolator, LinearSolver, NonlinearVariationalSolver,
    Parameters, Projector, ProjectorBase, backend_Constant,
    backend_DirichletBC, backend_Function, backend_Vector, backend_assemble,
    backend_interpolate, backend_project, backend_solve, parameters)
from ..interface import (
    function_comm, function_new, function_space, function_update_state,
    is_function, space_id, space_new)
from .backend_code_generator_interface import (
    copy_parameters_dict, update_parameters_dict)

from ..equations import Assignment
from ..override import (
    add_manager_controls, manager_method, override_function, override_method,
    override_property)

from .equations import (
    Assembly, EquationSolver, ExprEvaluation, Projection, expr_new_x,
    linear_equation_new_x)
from .functions import Constant, define_function_alias
from .firedrake_equations import LocalProjection

import numpy as np
import ufl

__all__ = \
    [
        "assemble",
        "interpolate",
        "project",
        "solve"
    ]


def parameters_dict_equal(parameters_a, parameters_b):
    if set(parameters_a) != set(parameters_b):
        return False
    for key_a, value_a in parameters_a.items():
        value_b = parameters_b[key_a]
        if isinstance(value_a, (Parameters, dict)):
            if not isinstance(value_b, (Parameters, dict)):
                return False
            elif not parameters_dict_equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False
    return True


def packed_solver_parameters(solver_parameters, *, options_prefix=None,
                             nullspace=None, transpose_nullspace=None,
                             near_nullspace=None):
    solver_parameters = dict(solver_parameters)
    tlm_adjoint_parameters = solver_parameters["tlm_adjoint"] = \
        dict(solver_parameters.get("tlm_adjoint", {}))

    def set_parameter(key, value):
        if value is not None:
            if key in tlm_adjoint_parameters:
                raise TypeError(f"Cannot pass both {key:s} argument and "
                                f"solver parameter")
            tlm_adjoint_parameters[key] = value

    set_parameter("options_prefix", options_prefix)
    set_parameter("nullspace", nullspace)
    set_parameter("transpose_nullspace", transpose_nullspace)
    set_parameter("near_nullspace", near_nullspace)

    return solver_parameters


# Aim for compatibility with Firedrake API, git master revision
# a94b01c4b3361db9c73056d92fdbd01a5bc6d1aa, Jun 16 2023


def FormAssembler_assemble_post_call(self, return_value, *args, **kwargs):
    if is_function(return_value):
        function_update_state(return_value)

    if len(self._form.arguments()) > 0:
        form_compiler_parameters = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        update_parameters_dict(form_compiler_parameters,
                               self._form_compiler_params)

        return_value._tlm_adjoint__form = self._form
        return_value._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


@manager_method(FormAssembler, "assemble",
                post_call=FormAssembler_assemble_post_call)
def FormAssembler_assemble(self, orig, orig_args, *args,
                           annotate, tlm, **kwargs):
    return_value = orig_args()

    if len(self._form.arguments()) == 1:
        eq = Assembly(return_value, return_value._tlm_adjoint__form,
                      form_compiler_parameters=return_value._tlm_adjoint__form_compiler_parameters)  # noqa: E501
        assert len(eq.initial_condition_dependencies()) == 0
        eq._post_process(annotate=annotate, tlm=tlm)

    return return_value


def function_update_state_post_call(self, return_value, *args, **kwargs):
    function_update_state(self)
    return return_value


@manager_method(backend_Constant, "assign",
                post_call=function_update_state_post_call)
def Constant_assign(self, orig, orig_args, value, *, annotate, tlm):
    if isinstance(value, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)):
        eq = Assignment(self, Constant(value, comm=function_comm(self)))
    elif isinstance(value, backend_Constant):
        if value is not self:
            eq = Assignment(self, value)
        else:
            eq = None
    elif isinstance(value, ufl.classes.Expr):
        eq = ExprEvaluation(
            self, expr_new_x(value, self, annotate=annotate, tlm=tlm))
    else:
        raise TypeError(f"Unexpected type: {type(value)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    return_value = orig_args()
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@manager_method(backend_Function, "assign",
                post_call=function_update_state_post_call)
def Function_assign(self, orig, orig_args, expr, subset=None, *,
                    annotate, tlm):
    if subset is not None:
        raise NotImplementedError("subset not supported")

    if isinstance(expr, (int, np.integer,
                         float, np.floating,
                         complex, np.complexfloating)):
        eq = Assignment(self, Constant(expr, comm=function_comm(self)))
    elif isinstance(expr, backend_Function) \
            and space_id(function_space(expr)) == space_id(function_space(self)):  # noqa: E501
        if expr is not self:
            eq = Assignment(self, expr)
        else:
            eq = None
    elif isinstance(expr, ufl.classes.Expr):
        eq = ExprEvaluation(
            self, expr_new_x(expr, self, annotate=annotate, tlm=tlm))
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    return_value = orig_args()
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@manager_method(backend_Function, "project",
                post_call=function_update_state_post_call)
def Function_project(self, orig, orig_args, b, bcs=None,
                     solver_parameters=None, form_compiler_parameters=None,
                     use_slate_for_inverse=True, name=None, ad_block_tag=None,
                     *, annotate, tlm):
    if use_slate_for_inverse:
        # Is a local solver actually used?
        projector = Projector(
            b, function_space(self), bcs=bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            use_slate_for_inverse=True)
        use_slate_for_inverse = getattr(projector,
                                        "use_slate_for_inverse", False)

    if use_slate_for_inverse:
        if bcs is not None and (isinstance(bcs, backend_DirichletBC)
                                or len(bcs) > 0):
            raise NotImplementedError("Boundary conditions not supported")
        eq = LocalProjection(
            self, expr_new_x(b, self, annotate=annotate, tlm=tlm),
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)
    else:
        eq = Projection(
            self, expr_new_x(b, self, annotate=annotate, tlm=tlm), bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@manager_method(backend_Function, "copy", override_without_manager=True)
def Function_copy(self, orig, orig_args, deepcopy=False, *, annotate, tlm):
    if deepcopy:
        F = function_new(self)
        F.assign(self, annotate=annotate, tlm=tlm)
    else:
        F = orig_args()
        define_function_alias(F, self, key="copy")
    return F


@manager_method(backend_Function, "interpolate")
def Function_interpolate(self, orig, orig_args, expression, subset=None,
                         ad_block_tag=None, *, annotate, tlm):
    return interpolate(
        expression, self, subset=subset, ad_block_tag=ad_block_tag,
        annotate=annotate, tlm=tlm)


def LinearSolver_solve_post_call(self, return_value, x, b):
    if isinstance(x, backend_Vector):
        x = x.function
    if isinstance(b, backend_Vector):
        b = b.function
    function_update_state(x)
    function_update_state(b)
    return return_value


@manager_method(LinearSolver, "solve",
                post_call=LinearSolver_solve_post_call)
def LinearSolver_solve(self, orig, orig_args, x, b, *, annotate, tlm):
    if self.P is not self.A:
        raise NotImplementedError("Preconditioners not supported")

    if isinstance(x, backend_Vector):
        x = x.function
    if isinstance(b, backend_Vector):
        b = b.function

    A = self.A
    bcs = A.bcs
    solver_parameters = packed_solver_parameters(
        self.parameters, options_prefix=self.options_prefix,
        nullspace=self.nullspace, transpose_nullspace=self.transpose_nullspace,
        near_nullspace=self.near_nullspace)
    form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
    if not parameters_dict_equal(
            b._tlm_adjoint__form_compiler_parameters,
            form_compiler_parameters):
        raise ValueError("Non-matching form compiler parameters")

    eq = EquationSolver(
        linear_equation_new_x(A.a == b._tlm_adjoint__form, x,
                              annotate=annotate, tlm=tlm),
        x, bcs, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@override_method(NonlinearVariationalSolver, "__init__")
def NonlinearVariationalSolver__init__(
        self, orig, orig_args, problem, *, appctx=None,
        pre_jacobian_callback=None, post_jacobian_callback=None,
        pre_function_callback=None, post_function_callback=None, **kwargs):
    orig_args()
    self._tlm_adjoint__appctx = {} if appctx is None else appctx
    self._tlm_adjoint__pre_jacobian_callback = pre_jacobian_callback
    self._tlm_adjoint__post_jacobian_callback = post_jacobian_callback
    self._tlm_adjoint__pre_function_callback = pre_function_callback
    self._tlm_adjoint__post_function_callback = post_function_callback
    self._tlm_adjoint__transfer_manager = None


@override_method(NonlinearVariationalSolver, "set_transfer_manager")
def NonlinearVariationalSolver_set_transfer_manager(
        self, orig, orig_args, manager):
    orig_args()
    self._tlm_adjoint__transfer_manager = manager


def NonlinearVariationalSolver_solve_post_call(
        self, return_value, *args, **kwargs):
    function_update_state(self._problem.u)
    return return_value


@manager_method(NonlinearVariationalSolver, "solve",
                post_call=NonlinearVariationalSolver_solve_post_call)
def NonlinearVariationalSolver_solve(
        self, orig, orig_args, bounds=None, *, annotate, tlm):
    if len(set(self._tlm_adjoint__appctx).difference({"state", "form_compiler_parameters"})) > 0:  # noqa: E501
        raise NotImplementedError("appctx not supported")
    if self._tlm_adjoint__pre_jacobian_callback is not None \
            or self._tlm_adjoint__post_jacobian_callback is not None \
            or self._tlm_adjoint__pre_function_callback is not None \
            or self._tlm_adjoint__post_function_callback is not None:
        raise NotImplementedError("Callbacks not supported")
    if self._tlm_adjoint__transfer_manager is not None:
        raise NotImplementedError("Transfer managers not supported")
    if bounds is not None:
        raise NotImplementedError("Bounds not supported")
    if self._problem.Jp is not None:
        raise NotImplementedError("Preconditioners not supported")

    solver_parameters = packed_solver_parameters(
        self.parameters, options_prefix=self.options_prefix,
        nullspace=self._ctx._nullspace,
        transpose_nullspace=self._ctx._nullspace_T,
        near_nullspace=self._ctx._near_nullspace)
    form_compiler_parameters = self._problem.form_compiler_parameters

    eq = EquationSolver(
        self._problem.F == 0, self._problem.u, self._problem.bcs,
        J=self._problem.J, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@override_property(ProjectorBase, "residual", cached=True)
def ProjectorBase_residual(self, orig):
    residual = orig()
    residual._tlm_adjoint__function_interface_attrs.d_setitem("space_type", "conjugate_dual")  # noqa: E501
    return residual


def Interpolator_interpolate_post_call(self, return_value, *args, **kwargs):
    function_update_state(return_value)
    return return_value


@manager_method(Interpolator, "interpolate",
                post_call=Interpolator_interpolate_post_call)
def Interpolator_interpolate(
        self, orig, orig_args, *function, output=None, transpose=False,
        annotate, tlm, **kwargs):
    if transpose:
        raise NotImplementedError("transpose not supported")

    return_value = orig_args()

    args = ufl.algorithms.extract_arguments(self.expr)
    if len(args) != len(function):
        raise TypeError("Unexpected number of functions")
    expr = ufl.replace(self.expr, dict(zip(args, function)))
    eq = ExprEvaluation(return_value, expr)

    assert len(eq.initial_condition_dependencies()) == 0
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@add_manager_controls
@override_function(backend_assemble)
def assemble(orig, orig_args, expr, *args, **kwargs):
    if isinstance(expr, ufl.classes.Form):
        def assemble_form(form, tensor=None, *args, **kwargs):
            if len(form.arguments()) == 1 and tensor is None:
                test, = form.arguments()
                tensor = space_new(test.function_space(),
                                   space_type="conjugate_dual")
            return orig(form, tensor, *args, **kwargs)

        return assemble_form(expr, *args, **kwargs)
    else:
        return orig_args()


solve = add_manager_controls(backend_solve)
project = add_manager_controls(backend_project)
interpolate = add_manager_controls(backend_interpolate)
