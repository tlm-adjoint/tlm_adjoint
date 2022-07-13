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

from .backend import Parameters, Projector, backend_Constant, \
    backend_DirichletBC, backend_Function, backend_LinearSolver, \
    backend_LinearVariationalProblem, backend_LinearVariationalSolver, \
    backend_NonlinearVariationalSolver, backend_Vector, backend_assemble, \
    backend_project, backend_solve, extract_args, extract_linear_solver_args, \
    parameters
from ..interface import check_space_type, function_new, function_space, \
    function_update_state, space_id, space_new
from .backend_code_generator_interface import copy_parameters_dict, \
    update_parameters_dict

from ..manager import annotation_enabled, tlm_enabled

from .equations import AssignmentSolver, EquationSolver, \
    ExprEvaluationSolver, ProjectionSolver, linear_equation_new_x
from .firedrake_equations import LocalProjectionSolver

import copy
import numpy as np
import ufl

__all__ = \
    [
        "LinearSolver",
        "LinearVariationalProblem",
        "LinearVariationalSolver",
        "NonlinearVariationalSolver",
        "assemble",
        "project",
        "solve"
    ]


def parameters_dict_equal(parameters_a, parameters_b):
    for key_a, value_a in parameters_a.items():
        if key_a not in parameters_b:
            return False
        value_b = parameters_b[key_a]
        if isinstance(value_a, (Parameters, dict)):
            if not isinstance(value_b, (Parameters, dict)):
                return False
            elif not parameters_dict_equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False
    for key_b in parameters_b:
        if key_b not in parameters_a:
            return False
    return True


def packed_solver_parameters(solver_parameters, *, options_prefix=None,
                             nullspace=None, transpose_nullspace=None,
                             near_nullspace=None):
    if options_prefix is not None or nullspace is not None \
       or transpose_nullspace is not None or near_nullspace is not None:
        solver_parameters = copy.copy(solver_parameters)
        if "tlm_adjoint" in solver_parameters:
            tlm_adjoint_parameters = solver_parameters["tlm_adjoint"] = \
                copy.copy(solver_parameters["tlm_adjoint"])
        else:
            tlm_adjoint_parameters = solver_parameters["tlm_adjoint"] = {}

        if options_prefix is not None:
            if "options_prefix" in tlm_adjoint_parameters:
                raise TypeError("Cannot pass both options_prefix argument and "
                                "solver parameter")
            tlm_adjoint_parameters["options_prefix"] = options_prefix

        if nullspace is not None:
            if "nullspace" in tlm_adjoint_parameters:
                raise TypeError("Cannot pass both nullspace argument and "
                                "solver parameter")
            tlm_adjoint_parameters["nullspace"] = nullspace

        if transpose_nullspace is not None:
            if "transpose_nullspace" in tlm_adjoint_parameters:
                raise TypeError("Cannot pass both transpose_nullspace "
                                "argument and solver parameter")
            tlm_adjoint_parameters["transpose_nullspace"] = transpose_nullspace

        if near_nullspace is not None:
            if "near_nullspace" in tlm_adjoint_parameters:
                raise TypeError("Cannot pass both near_nullspace argument and "
                                "solver parameter")
            tlm_adjoint_parameters["near_nullspace"] = near_nullspace

    return solver_parameters


def extract_args_assemble_form(form, tensor=None, bcs=None, *,
                               form_compiler_parameters=None, **kwargs):
    return form, tensor, bcs, form_compiler_parameters, kwargs


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def assemble(expr, *args, **kwargs):
    if not isinstance(expr, ufl.classes.Form):
        return backend_assemble(expr, *args, **kwargs)

    form, tensor, bcs, form_compiler_parameters, kwargs = \
        extract_args_assemble_form(expr, *args, **kwargs)

    if tensor is not None and isinstance(tensor, backend_Function):
        check_space_type(tensor, "conjugate_dual")

    b = backend_assemble(
        form, tensor=tensor, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters,
        **kwargs)

    if tensor is not None and isinstance(tensor, backend_Function):
        function_update_state(tensor)
    if tensor is None and isinstance(b, backend_Function):
        b._tlm_adjoint__function_interface_attrs.d_setitem("space_type", "conjugate_dual")  # noqa: E501

    rank = len(form.arguments())
    if rank != 0:
        form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        if form_compiler_parameters is not None:
            update_parameters_dict(form_compiler_parameters_,
                                   form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_

        if rank == 1:
            b._tlm_adjoint__form = form
        b._tlm_adjoint__form_compiler_parameters = form_compiler_parameters

    return b


def extract_args_linear_solve(A, x, b, *args, **kwargs):
    (bcs,
     solver_parameters,
     nullspace, transpose_nullspace, near_nullspace,
     options_prefix) = extract_linear_solver_args(A, x, b, *args, **kwargs)

    if isinstance(x, backend_Vector):
        x = x.function
    if isinstance(b, backend_Vector):
        b = b.function
    if bcs is not None:
        raise TypeError("Unexpected boundary conditions")

    return (A, x, b,
            solver_parameters,
            nullspace, transpose_nullspace, near_nullspace,
            options_prefix)


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def solve(*args, annotate=None, tlm=None, **kwargs):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(args[0], ufl.classes.Equation):
            (eq_arg, x, bcs,
             J, Jp,
             M,
             form_compiler_parameters, solver_parameters,
             nullspace, transpose_nullspace, near_nullspace,
             options_prefix) = extract_args(*args, **kwargs)
            if bcs is None:
                bcs = ()
            elif isinstance(bcs, backend_DirichletBC):
                bcs = (bcs,)
            if form_compiler_parameters is None:
                form_compiler_parameters = {}
            if solver_parameters is None:
                solver_parameters = {}

            if Jp is not None:
                raise NotImplementedError("Preconditioners not supported")
            if M is not None:
                raise NotImplementedError("Adaptive solves not supported")

            solver_parameters = packed_solver_parameters(
                solver_parameters, options_prefix=options_prefix,
                nullspace=nullspace, transpose_nullspace=transpose_nullspace,
                near_nullspace=near_nullspace)

            if isinstance(eq_arg.rhs, ufl.classes.Form):
                eq_arg = linear_equation_new_x(eq_arg, x,
                                               annotate=annotate, tlm=tlm)
            eq = EquationSolver(
                eq_arg, x, bcs, J=J,
                form_compiler_parameters=form_compiler_parameters,
                solver_parameters=solver_parameters, cache_jacobian=False,
                cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            (A, x, b,
             solver_parameters,
             nullspace, transpose_nullspace, near_nullspace,
             options_prefix) = extract_args_linear_solve(*args, **kwargs)
            if solver_parameters is None:
                solver_parameters = {}

            A = A.a
            b = b._tlm_adjoint__form

            bcs = A.bcs

            solver_parameters = packed_solver_parameters(
                solver_parameters, options_prefix=options_prefix,
                nullspace=nullspace, transpose_nullspace=transpose_nullspace,
                near_nullspace=near_nullspace)

            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise ValueError("Non-matching form compiler parameters")

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
    else:
        backend_solve(*args, **kwargs)
        if isinstance(args[0], ufl.classes.Equation):
            x = extract_args(*args, **kwargs)[1]
            function_update_state(x)
        else:
            (_, x, b,
             _,
             _, _, _,
             _) = extract_args_linear_solve(*args, **kwargs)
            function_update_state(x)
            function_update_state(b)


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def project(v, V, bcs=None, solver_parameters=None,
            form_compiler_parameters=None, use_slate_for_inverse=True,
            name=None, ad_block_tag=None, *, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if use_slate_for_inverse:
            # Is a local solver actually used?
            projector = Projector(
                v, V, bcs=bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                use_slate_for_inverse=True)
            use_slate_for_inverse = getattr(projector,
                                            "use_slate_for_inverse", False)
        if isinstance(V, backend_Function):
            x = V
        else:
            x = space_new(V, name=name)
        if bcs is None:
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        if solver_parameters is None:
            solver_parameters = {}
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if x in ufl.algorithms.extract_coefficients(v):
            x_old = function_new(x)
            AssignmentSolver(x, x_old).solve(annotate=annotate, tlm=tlm)
            v = ufl.replace(v, {x: x_old})
        if use_slate_for_inverse:
            if len(bcs) > 0:
                raise NotImplementedError("Boundary conditions not supported")
            eq = LocalProjectionSolver(
                v, x, form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            eq = ProjectionSolver(
                v, x, bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
    else:
        x = backend_project(
            v, V, bcs=bcs, solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            use_slate_for_inverse=use_slate_for_inverse, name=name,
            ad_block_tag=ad_block_tag)
        function_update_state(x)
    return x


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def _Constant_assign(self, value, *, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(value, (int, np.integer,
                              float, np.floating,
                              complex, np.complexfloating)):
            AssignmentSolver(backend_Constant(value), self).solve(
                annotate=annotate, tlm=tlm)
            return
        elif isinstance(value, backend_Constant):
            if value is not self:
                AssignmentSolver(value, self).solve(annotate=annotate, tlm=tlm)
                return
        elif isinstance(value, ufl.classes.Expr):
            if self in ufl.algorithms.extract_coefficients(value):
                self_old = function_new(self)
                AssignmentSolver(self, self_old).solve(
                    annotate=annotate, tlm=tlm)
                value = ufl.replace(value, {self: self_old})
            ExprEvaluationSolver(value, self).solve(annotate=annotate, tlm=tlm)
            return

    return_value = backend_Constant._tlm_adjoint__orig_assign(
        self, value)
    function_update_state(self)
    return return_value


assert not hasattr(backend_Constant, "_tlm_adjoint__orig_assign")
backend_Constant._tlm_adjoint__orig_assign = backend_Constant.assign
backend_Constant.assign = _Constant_assign


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def _Function_assign(self, expr, subset=None, *, annotate=None, tlm=None):
    if subset is None:
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(expr, (int, np.integer,
                                 float, np.floating,
                                 complex, np.complexfloating)):
                expr = backend_Constant(expr)
            if isinstance(expr, backend_Function) \
                    and space_id(function_space(expr)) == space_id(function_space(self)):  # noqa: E501
                if expr is not self:
                    AssignmentSolver(expr, self).solve(
                        annotate=annotate, tlm=tlm)
                    return self
            elif isinstance(expr, ufl.classes.Expr):
                if self in ufl.algorithms.extract_coefficients(expr):
                    self_old = function_new(self)
                    AssignmentSolver(self, self_old).solve(
                        annotate=annotate, tlm=tlm)
                    expr = ufl.replace(expr, {self: self_old})
                ExprEvaluationSolver(expr, self).solve(
                    annotate=annotate, tlm=tlm)
                return self

    return_value = backend_Function._tlm_adjoint__orig_assign(
        self, expr, subset=subset)
    function_update_state(self)
    return return_value


assert not hasattr(backend_Function, "_tlm_adjoint__orig_assign")
backend_Function._tlm_adjoint__orig_assign = backend_Function.assign
backend_Function.assign = _Function_assign


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
def _Function_project(self, b, *args, **kwargs):
    return project(b, self, *args, **kwargs)


assert not hasattr(backend_Function, "_tlm_adjoint__orig_project")
backend_Function._tlm_adjoint__orig_project = backend_Function.project
backend_Function.project = _Function_project


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
class LinearSolver(backend_LinearSolver):
    def __init__(self, A, *, P=None, **kwargs):
        if P is not None:
            raise NotImplementedError("Preconditioners not supported")

        super().__init__(A, P=P, **kwargs)

    def solve(self, x, b, *, annotate=None, tlm=None):
        if isinstance(x, backend_Vector):
            x = x.function
        if isinstance(b, backend_Vector):
            b = b.function

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            A = self.A
            bcs = A.bcs
            solver_parameters = packed_solver_parameters(
                self.parameters, options_prefix=self.options_prefix,
                nullspace=self.nullspace,
                transpose_nullspace=self.transpose_nullspace,
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
            super().solve(x, b)
            function_update_state(x)
            function_update_state(b)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            super().solve(x, b)
            function_update_state(x)
            function_update_state(b)


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
class LinearVariationalProblem(backend_LinearVariationalProblem):
    def __init__(self, a, L, *args, **kwargs):
        super().__init__(a, L, *args, **kwargs)
        self._tlm_adjoint__b = L


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
class LinearVariationalSolver(backend_LinearVariationalSolver):
    def __init__(self, problem, *, appctx=None, pre_jacobian_callback=None,
                 post_jacobian_callback=None, pre_function_callback=None,
                 post_function_callback=None, **kwargs):
        if appctx is not None:
            raise NotImplementedError("Preconditioners not supported")
        if pre_jacobian_callback is not None \
                or post_jacobian_callback is not None \
                or pre_function_callback is not None \
                or post_function_callback is not None:
            raise NotImplementedError("Callbacks not supported")

        super().__init__(
            problem, appctx=appctx,
            pre_jacobian_callback=pre_jacobian_callback,
            post_jacobian_callback=post_jacobian_callback,
            pre_function_callback=pre_function_callback,
            post_function_callback=post_function_callback, **kwargs)

    def set_transfer_manager(self, *args, **kwargs):
        raise NotImplementedError("Transfer managers not supported")

    def solve(self, bounds=None, *, annotate=None, tlm=None):
        x = self._problem.u

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
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
            if form_compiler_parameters is None:
                form_compiler_parameters = {}

            A = self._problem.J
            b = self._problem._tlm_adjoint__b

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
                                      annotate=annotate, tlm=tlm),
                x, self._problem.bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=self._problem._constant_jacobian,
                cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            super().solve(bounds=bounds)
            function_update_state(x)


# Aim for compatibility with Firedrake API, git master revision
# bc79502544ca78c06d60532c2d674b7808aef0af, Mar 30 2022
class NonlinearVariationalSolver(backend_NonlinearVariationalSolver):
    def __init__(self, problem, *, appctx=None, pre_jacobian_callback=None,
                 post_jacobian_callback=None, pre_function_callback=None,
                 post_function_callback=None, **kwargs):
        if appctx is not None:
            raise NotImplementedError("Preconditioners not supported")
        if pre_jacobian_callback is not None \
                or post_jacobian_callback is not None \
                or pre_function_callback is not None \
                or post_function_callback is not None:
            raise NotImplementedError("Callbacks not supported")

        super().__init__(
            problem, appctx=appctx,
            pre_jacobian_callback=pre_jacobian_callback,
            post_jacobian_callback=post_jacobian_callback,
            pre_function_callback=pre_function_callback,
            post_function_callback=post_function_callback, **kwargs)

    def set_transfer_manager(self, *args, **kwargs):
        raise NotImplementedError("Transfer managers not supported")

    def solve(self, bounds=None, *, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
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
            if form_compiler_parameters is None:
                form_compiler_parameters = {}

            eq = EquationSolver(
                self._problem.F == 0, self._problem.u, self._problem.bcs,
                J=self._problem.J, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            super().solve(bounds=bounds)
            function_update_state(self._problem.u)
