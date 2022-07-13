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

from .backend import Parameters, backend_Constant, backend_DirichletBC, \
    backend_Function, backend_KrylovSolver, backend_LUSolver, \
    backend_LinearVariationalSolver, backend_Matrix, \
    backend_NonlinearVariationalProblem, backend_NonlinearVariationalSolver, \
    backend_assemble, backend_assemble_system, backend_project, \
    backend_solve, extract_args, parameters
from ..interface import check_space_type, function_assign, function_new, \
    function_space, function_update_state, space_id, space_new
from .backend_code_generator_interface import copy_parameters_dict, \
    update_parameters_dict

from ..manager import annotation_enabled, tlm_enabled

from .equations import AssignmentSolver, EquationSolver, \
    ExprEvaluationSolver, ProjectionSolver, linear_equation_new_x

import numpy as np
import ufl

__all__ = \
    [
        "LinearVariationalSolver",
        "NonlinearVariationalProblem",
        "NonlinearVariationalSolver",
        "KrylovSolver",
        "LUSolver",
        "assemble",
        "assemble_system",
        "project",
        "solve"
    ]


def parameters_dict_equal(parameters_a, parameters_b):
    for key_a in parameters_a:
        if key_a not in parameters_b:
            return False
        value_a = parameters_a[key_a]
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


# Aim for compatibility with FEniCS 2019.1.0 API


def assemble(form, tensor=None, form_compiler_parameters=None,
             add_values=False, *args, **kwargs):
    if not isinstance(form, ufl.classes.Form):
        raise TypeError("form must be a UFL form")

    if tensor is not None and hasattr(tensor, "_tlm_adjoint__function"):
        check_space_type(tensor._tlm_adjoint__function, "conjugate_dual")

    b = backend_assemble(form, tensor=tensor,
                         form_compiler_parameters=form_compiler_parameters,
                         add_values=add_values, *args, **kwargs)

    if tensor is not None and hasattr(tensor, "_tlm_adjoint__function"):
        function_update_state(tensor._tlm_adjoint__function)

    if not isinstance(b, (float, np.floating)):
        form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        if form_compiler_parameters is not None:
            update_parameters_dict(form_compiler_parameters_,
                                   form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_

        if add_values and hasattr(b, "_tlm_adjoint__form"):
            if b._tlm_adjoint__bcs != []:
                raise ValueError("Non-matching boundary conditions")
            elif not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise ValueError("Non-matching form compiler parameters")
            b._tlm_adjoint__form += form
        else:
            b._tlm_adjoint__form = form
            b._tlm_adjoint__bcs = []
            b._tlm_adjoint__form_compiler_parameters = form_compiler_parameters

    return b


def assemble_system(A_form, b_form, bcs=None, x0=None,
                    form_compiler_parameters=None, add_values=False,
                    finalize_tensor=True, keep_diagonal=False, A_tensor=None,
                    b_tensor=None, *args, **kwargs):
    if not isinstance(A_form, ufl.classes.Form):
        raise TypeError("A_form must be a UFL form")
    if not isinstance(b_form, ufl.classes.Form):
        raise TypeError("b_form must be a UFL form")
    if x0 is not None:
        raise NotImplementedError("Non-linear boundary condition case not "
                                  "supported")

    if b_tensor is not None and hasattr(b_tensor, "_tlm_adjoint__function"):
        check_space_type(b_tensor._tlm_adjoint__function, "conjugate_dual")

    A, b = backend_assemble_system(
        A_form, b_form, bcs=bcs, x0=x0,
        form_compiler_parameters=form_compiler_parameters,
        add_values=add_values, finalize_tensor=finalize_tensor,
        keep_diagonal=keep_diagonal, A_tensor=A_tensor, b_tensor=b_tensor,
        *args, **kwargs)

    if b_tensor is not None and hasattr(b_tensor, "_tlm_adjoint__function"):
        function_update_state(b_tensor._tlm_adjoint__function)

    if bcs is None:
        bcs = []
    elif isinstance(bcs, backend_DirichletBC):
        bcs = [bcs]

    form_compiler_parameters_ = parameters["form_compiler"]
    form_compiler_parameters_ = copy_parameters_dict(form_compiler_parameters_)
    if form_compiler_parameters is not None:
        update_parameters_dict(form_compiler_parameters_,
                               form_compiler_parameters)
    form_compiler_parameters = form_compiler_parameters_

    if add_values and hasattr(A, "_tlm_adjoint__form"):
        if A._tlm_adjoint__bcs != bcs:
            raise ValueError("Non-matching boundary conditions")
        elif not parameters_dict_equal(
                A._tlm_adjoint__form_compiler_parameters,
                form_compiler_parameters):
            raise ValueError("Non-matching form compiler parameters")
        A._tlm_adjoint__form += A_form
    else:
        A._tlm_adjoint__form = A_form
        A._tlm_adjoint__bcs = list(bcs)
        A._tlm_adjoint__form_compiler_parameters = form_compiler_parameters

    if add_values and hasattr(b, "_tlm_adjoint__form"):
        if b._tlm_adjoint__bcs != bcs:
            raise ValueError("Non-matching boundary conditions")
        elif not parameters_dict_equal(
                b._tlm_adjoint__form_compiler_parameters,
                form_compiler_parameters):
            raise ValueError("Non-matching form compiler parameters")
        b._tlm_adjoint__form += b_form
    else:
        b._tlm_adjoint__form = b_form
        b._tlm_adjoint__bcs = list(bcs)
        b._tlm_adjoint__form_compiler_parameters = form_compiler_parameters

    return A, b


def extract_args_linear_solve(A, x, b,
                              linear_solver="default",
                              preconditioner="default", /):  # noqa: E225
    solver_parameters = {"linear_solver": linear_solver,
                         "preconditioner": preconditioner}
    return A, x, b, solver_parameters


def solve(*args, annotate=None, tlm=None, **kwargs):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(args[0], ufl.classes.Equation):
            extracted_args = extract_args(*args, **kwargs)
            if len(extracted_args) == 8:
                (eq_arg, x, bcs, J,
                 tol, M,
                 form_compiler_parameters,
                 solver_parameters) = extracted_args
                preconditioner = None
            else:
                (eq_arg, x, bcs, J,
                 tol, M,
                 preconditioner,
                 form_compiler_parameters,
                 solver_parameters) = extracted_args
            del extracted_args

            if bcs is None:
                bcs = ()
            elif isinstance(bcs, backend_DirichletBC):
                bcs = (bcs,)
            if form_compiler_parameters is None:
                form_compiler_parameters = {}
            if solver_parameters is None:
                solver_parameters = {}

            if tol is not None or M is not None:
                raise NotImplementedError("Adaptive solves not supported")
            if preconditioner is not None:
                raise NotImplementedError("Preconditioners not supported")

            if isinstance(eq_arg.rhs, ufl.classes.Form):
                eq_arg = linear_equation_new_x(eq_arg, x,
                                               annotate=annotate, tlm=tlm)
            eq = EquationSolver(
                eq_arg, x, bcs, J=J,
                form_compiler_parameters=form_compiler_parameters,
                solver_parameters=solver_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            A, x, b, solver_parameters = \
                extract_args_linear_solve(*args, **kwargs)
            # if solver_parameters is None:
            #     solver_parameters = {}

            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise ValueError("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise ValueError("Non-matching form compiler parameters")

            A = A._tlm_adjoint__form
            x = x._tlm_adjoint__function
            b = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)

            eq._pre_process(annotate=annotate)
            return_value = backend_solve(*args, **kwargs)
            function_update_state(x)
            function_update_state(b)
            eq._post_process(annotate=annotate, tlm=tlm)

            return return_value
    else:
        return_value = backend_solve(*args, **kwargs)

        if isinstance(args[0], ufl.classes.Equation):
            x = extract_args(*args, **kwargs)[1]
            function_update_state(x)
        else:
            _, x, b, _ = extract_args_linear_solve(*args, **kwargs)
            if hasattr(x, "_tlm_adjoint__function"):
                function_update_state(x._tlm_adjoint__function)
            if hasattr(b, "_tlm_adjoint__function"):
                function_update_state(b._tlm_adjoint__function)
        return return_value


def project(v, V=None, bcs=None, mesh=None, function=None, solver_type="lu",
            preconditioner_type="default", form_compiler_parameters=None, *,
            solver_parameters=None, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm or solver_parameters is not None:
        if function is None:
            if V is None:
                raise TypeError("V or function required")
            x = space_new(V)
        else:
            x = function

        if bcs is None:
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)

        solver_parameters_ = {"linear_solver": solver_type,
                              "preconditioner": preconditioner_type}
        if solver_parameters is not None:
            solver_parameters_.update(solver_parameters)
        solver_parameters = solver_parameters_

        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        if x in ufl.algorithms.extract_coefficients(v):
            x_old = function_new(x)
            AssignmentSolver(x, x_old).solve(annotate=annotate, tlm=tlm)
            v = ufl.replace(v, {x: x_old})

        eq = ProjectionSolver(
            v, x, bcs, solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)
        eq.solve(annotate=annotate, tlm=tlm)
    else:
        x = backend_project(
            v, V=V, bcs=bcs, mesh=mesh, function=function,
            solver_type=solver_type,
            preconditioner_type=preconditioner_type,
            form_compiler_parameters=form_compiler_parameters)
        function_update_state(x)
    return x


def _DirichletBC_apply(self, *args):
    if len(args) == 1:
        if isinstance(args[0], backend_Matrix):
            (A,), b, x = args, None, None
        else:
            (b,), A, x = args, None, None
    elif len(args) == 2:
        if isinstance(args[0], backend_Matrix):
            (A, b), x = args, None
        else:
            (b, x), A = args, None
    else:
        A, b, x = args

    backend_DirichletBC._tlm_adjoint__orig_apply(self, *args)

    if b is not None:
        if hasattr(b, "_tlm_adjoint__function"):
            function_update_state(b._tlm_adjoint__function)
    if x is None:
        if A is not None and hasattr(A, "_tlm_adjoint__bcs") \
                and self not in A._tlm_adjoint__bcs:
            A._tlm_adjoint__bcs.append(self)
        if b is not None and hasattr(b, "_tlm_adjoint__bcs") \
                and self not in b._tlm_adjoint__bcs:
            b._tlm_adjoint__bcs.append(self)
    else:
        if hasattr(x, "_tlm_adjoint__function"):
            function_update_state(x._tlm_adjoint__function)


assert not hasattr(backend_DirichletBC, "_tlm_adjoint__orig_apply")
backend_DirichletBC._tlm_adjoint__orig_apply = backend_DirichletBC.apply
backend_DirichletBC.apply = _DirichletBC_apply


def _Constant_assign(self, x, *, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(x, (int, np.integer,
                          float, np.floating)):
            AssignmentSolver(backend_Constant(x), self).solve(
                annotate=annotate, tlm=tlm)
            return
        elif isinstance(x, backend_Constant):
            if x is not self:
                AssignmentSolver(x, self).solve(annotate=annotate, tlm=tlm)
                return
        elif isinstance(x, ufl.classes.Expr):
            if self in ufl.algorithms.extract_coefficients(x):
                self_old = function_new(self)
                AssignmentSolver(self, self_old).solve(
                    annotate=annotate, tlm=tlm)
                x = ufl.replace(x, {self: self_old})
            ExprEvaluationSolver(x, self).solve(annotate=annotate, tlm=tlm)
            return

    return_value = backend_Constant._tlm_adjoint__orig_assign(
        self, x)
    function_update_state(self)
    return return_value


assert not hasattr(backend_Constant, "_tlm_adjoint__orig_assign")
backend_Constant._tlm_adjoint__orig_assign = backend_Constant.assign
backend_Constant.assign = _Constant_assign


def _Function_assign(self, rhs, *, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(rhs, backend_Function) \
                and space_id(function_space(rhs)) == space_id(function_space(self)):  # noqa: E501
            if rhs is not self:
                AssignmentSolver(rhs, self).solve(
                    annotate=annotate, tlm=tlm)
                return
        elif isinstance(rhs, ufl.classes.Expr):
            x_space = function_space(self)
            deps = ufl.algorithms.extract_coefficients(rhs)
            for dep in deps:
                if isinstance(dep, backend_Function):
                    dep_space = function_space(dep)
                    if dep_space.ufl_domains() != x_space.ufl_domains() \
                            or dep_space.ufl_element() != x_space.ufl_element():  # noqa: E501
                        break
            else:
                if self in deps:
                    self_old = function_new(self)
                    AssignmentSolver(self, self_old).solve(
                        annotate=annotate, tlm=tlm)
                    rhs = ufl.replace(rhs, {self: self_old})
                ExprEvaluationSolver(rhs, self).solve(
                    annotate=annotate, tlm=tlm)
                return
    elif isinstance(rhs, backend_Function) and rhs is not self:
        if space_id(function_space(rhs)) == space_id(function_space(self)):
            value = rhs
        else:
            value = function_new(self)
            backend_Function._tlm_adjoint__orig_assign(value, rhs)
        function_assign(self, value)
        return

    return_value = backend_Function._tlm_adjoint__orig_assign(
        self, rhs)
    function_update_state(self)
    return return_value


assert not hasattr(backend_Function, "_tlm_adjoint__orig_assign")
backend_Function._tlm_adjoint__orig_assign = backend_Function.assign
backend_Function.assign = _Function_assign


def _Function_vector(self):
    vector = backend_Function._tlm_adjoint__orig_vector(self)
    vector._tlm_adjoint__function = self
    if vector is not self._tlm_adjoint__vector:
        raise RuntimeError("Vector has changed")
    return vector


assert not hasattr(backend_Function, "_tlm_adjoint__orig_vector")
backend_Function._tlm_adjoint__orig_vector = backend_Function.vector
backend_Function.vector = _Function_vector


def _Matrix_mul(self, other):
    return_value = backend_Matrix._tlm_adjoint__orig___mul__(self, other)
    if hasattr(self, "_tlm_adjoint__form") \
       and hasattr(other, "_tlm_adjoint__function") \
       and len(self._tlm_adjoint__bcs) == 0:
        return_value._tlm_adjoint__form = ufl.action(
            self._tlm_adjoint__form,
            coefficient=other._tlm_adjoint__function)
        return_value._tlm_adjoint__bcs = []
        return_value._tlm_adjoint__form_compiler_parameters \
            = self._tlm_adjoint__form_compiler_parameters
    return return_value


assert not hasattr(backend_Matrix, "_tlm_adjoint__orig___mul__")
backend_Matrix._tlm_adjoint__orig___mul__ = backend_Matrix.__mul__
backend_Matrix.__mul__ = _Matrix_mul


class LUSolver(backend_LUSolver):
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) >= 1 and isinstance(args[0], backend_Matrix):
            self.__A = args[0]
            self.__linear_solver = args[1] if len(args) >= 2 else "default"
        elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
            self.__A = args[1]
            self.__linear_solver = args[2] if len(args) >= 3 else "default"
        else:
            self.__linear_solver = args[0] if len(args) >= 1 else "default"

    def set_operator(self, A):
        super().set_operator(A)
        self.__A = A

    def solve(self, *args, annotate=None, tlm=None):
        if isinstance(args[0], backend_Matrix):
            A, x, b = args
            self.__A = A
        else:
            A = self.__A
            x, b = args

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise ValueError("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise ValueError("Non-matching form compiler parameters")

            A_form = A._tlm_adjoint__form
            x = x._tlm_adjoint__function
            b_form = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A_form == b_form, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs,
                solver_parameters={"linear_solver": self.__linear_solver,
                                   "lu_solver": self.parameters},
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq._pre_process(annotate=annotate)
            return_value = super().solve(*args)
            function_update_state(x)
            if hasattr(b, "_tlm_adjoint__function"):
                function_update_state(b._tlm_adjoint__function)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            return_value = super().solve(*args)
            if hasattr(x, "_tlm_adjoint__function"):
                function_update_state(x._tlm_adjoint__function)
            if hasattr(b, "_tlm_adjoint__function"):
                function_update_state(b._tlm_adjoint__function)

        return return_value


class KrylovSolver(backend_KrylovSolver):
    def __init__(self, *args):
        super().__init__(*args)
        if len(args) >= 1 and isinstance(args[0], backend_Matrix):
            self.__A = args[0]
            self.__linear_solver = args[1] if len(args) >= 2 else "default"
            self.__preconditioner = args[2] if len(args) >= 3 else "default"
        elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
            self.__A = args[1]
            self.__linear_solver = args[2] if len(args) >= 3 else "default"
            self.__preconditioner = args[3] if len(args) >= 4 else "default"
        else:
            self.__linear_solver = args[0] if len(args) >= 1 else "default"
            self.__preconditioner = args[1] if len(args) >= 2 else "default"

    def set_operator(self, A):
        super().set_operator(A)
        self.__A = A

    def set_operators(self, *args, **kwargs):
        raise NotImplementedError("Preconditioners not supported")

    def solve(self, *args, annotate=None, tlm=None):
        if isinstance(args[0], backend_Matrix):
            A, x, b = args
            self.__A = None
        else:
            A = self.__A
            x, b = args

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise ValueError("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise ValueError("Non-matching form compiler parameters")

            A_form = A._tlm_adjoint__form
            x = x._tlm_adjoint__function
            b_form = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A_form == b_form, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs,
                solver_parameters={"linear_solver": self.__linear_solver,
                                   "preconditioner": self.__preconditioner,
                                   "krylov_solver": self.parameters},
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)

            eq._pre_process(annotate=annotate)
            return_value = super().solve(*args)
            function_update_state(x)
            if hasattr(b, "_tlm_adjoint__function"):
                function_update_state(b._tlm_adjoint__function)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            return_value = super().solve(*args)
            if hasattr(x, "_tlm_adjoint__function"):
                function_update_state(x._tlm_adjoint__function)
            if hasattr(b, "_tlm_adjoint__function"):
                function_update_state(b._tlm_adjoint__function)

        return return_value


class LinearVariationalSolver(backend_LinearVariationalSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.__problem = problem

    def solve(self, *, annotate=None, tlm=None):
        x = self.__problem.u_ufl

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            lhs, rhs = self.__problem.a_ufl, self.__problem.L_ufl
            eq = EquationSolver(
                linear_equation_new_x(lhs == rhs, x,
                                      annotate=annotate, tlm=tlm),
                x, self.__problem.bcs(),
                solver_parameters=self.parameters,
                form_compiler_parameters=self.__problem.form_compiler_parameters,  # noqa: E501
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
        else:
            super().solve()
            function_update_state(x)


class NonlinearVariationalProblem(backend_NonlinearVariationalProblem):
    def __init__(self, F, u, bcs=None, J=None, form_compiler_parameters=None):
        super().__init__(F, u, bcs=bcs, J=J,
                         form_compiler_parameters=form_compiler_parameters)
        if bcs is None:
            self._tlm_adjoint__bcs = []
        elif isinstance(bcs, backend_DirichletBC):
            self._tlm_adjoint__bcs = [bcs]
        else:
            self._tlm_adjoint__bcs = list(bcs)

    def set_bounds(self, *args, **kwargs):
        raise NotImplementedError("Bounds not supported")


class NonlinearVariationalSolver(backend_NonlinearVariationalSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.__problem = problem

    def solve(self, *, annotate=None, tlm=None):
        x = self.__problem.u_ufl

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            eq = EquationSolver(
                self.__problem.F_ufl == 0, x,
                self.__problem._tlm_adjoint__bcs, J=self.__problem.J_ufl,
                solver_parameters=self.parameters,
                form_compiler_parameters=self.__problem.form_compiler_parameters,  # noqa: E501
                cache_jacobian=False, cache_rhs_assembly=False)

            eq._pre_process(annotate=annotate)
            return_value = super().solve()
            function_update_state(x)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            return_value = super().solve()
            function_update_state(x)

        return return_value
