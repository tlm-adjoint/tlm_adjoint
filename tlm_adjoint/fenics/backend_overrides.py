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

from .backend import Parameters, backend_DirichletBC, backend_Function, \
    backend_KrylovSolver, backend_LUSolver, backend_LinearVariationalSolver, \
    backend_Matrix, backend_NonlinearVariationalProblem, \
    backend_NonlinearVariationalSolver, backend_assemble, \
    backend_assemble_system, backend_project, backend_solve, extract_args, \
    parameters
from ..interface import function_new, function_update_state, space_new
from .backend_code_generator_interface import copy_parameters_dict, \
    update_parameters_dict

from ..tlm_adjoint import annotation_enabled, tlm_enabled

from .equations import AssignmentSolver, EquationSolver, ProjectionSolver, \
    linear_equation_new_x

import ufl

__all__ = \
    [
        "OverrideException",

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


class OverrideException(Exception):
    pass


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
    b = backend_assemble(form, tensor=tensor,
                         form_compiler_parameters=form_compiler_parameters,
                         add_values=add_values, *args, **kwargs)
    if tensor is None:
        tensor = b

    if not isinstance(tensor, float):
        if not isinstance(form, ufl.classes.Form):
            raise OverrideException("form must be a UFL form")

        form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        if form_compiler_parameters is not None:
            update_parameters_dict(form_compiler_parameters_,
                                   form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_

        if add_values and hasattr(tensor, "_tlm_adjoint__form"):
            if tensor._tlm_adjoint__bcs != []:
                raise OverrideException("Non-matching boundary conditions")
            elif not parameters_dict_equal(
                    tensor._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501
            tensor._tlm_adjoint__form += form
        else:
            tensor._tlm_adjoint__form = form
            tensor._tlm_adjoint__bcs = []
            tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return tensor


def assemble_system(A_form, b_form, bcs=None, x0=None,
                    form_compiler_parameters=None, add_values=False,
                    finalize_tensor=True, keep_diagonal=False, A_tensor=None,
                    b_tensor=None, *args, **kwargs):
    if not isinstance(A_form, ufl.classes.Form):
        raise OverrideException("A_form must be a UFL form")
    if not isinstance(b_form, ufl.classes.Form):
        raise OverrideException("b_form must be a UFL form")
    if x0 is not None:
        raise OverrideException("Non-linear boundary condition case not supported")  # noqa: E501

    A, b = backend_assemble_system(
        A_form, b_form, bcs=bcs, x0=x0,
        form_compiler_parameters=form_compiler_parameters,
        add_values=add_values, finalize_tensor=finalize_tensor,
        keep_diagonal=keep_diagonal, A_tensor=A_tensor, b_tensor=b_tensor,
        *args, **kwargs)
    if A_tensor is None:
        A_tensor = A
    if b_tensor is None:
        b_tensor = b
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

    if add_values and hasattr(A_tensor, "_tlm_adjoint__form"):
        if A_tensor._tlm_adjoint__bcs != bcs:
            raise OverrideException("Non-matching boundary conditions")
        elif not parameters_dict_equal(
                A_tensor._tlm_adjoint__form_compiler_parameters,
                form_compiler_parameters):
            raise OverrideException("Non-matching form compiler parameters")
        A_tensor._tlm_adjoint__form += A_form
    else:
        A_tensor._tlm_adjoint__form = A_form
        A_tensor._tlm_adjoint__bcs = list(bcs)
        A_tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    if add_values and hasattr(b_tensor, "_tlm_adjoint__form"):
        if b_tensor._tlm_adjoint__bcs != bcs:
            raise OverrideException("Non-matching boundary conditions")
        elif not parameters_dict_equal(
                b_tensor._tlm_adjoint__form_compiler_parameters,
                form_compiler_parameters):
            raise OverrideException("Non-matching form compiler parameters")
        b_tensor._tlm_adjoint__form += b_form
    else:
        b_tensor._tlm_adjoint__form = b_form
        b_tensor._tlm_adjoint__bcs = list(bcs)
        b_tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return A_tensor, b_tensor


def extract_args_linear_solve(A, x, b, linear_solver="default",
                              preconditioner="default"):
    return A, x, b, linear_solver, preconditioner


def solve(*args, annotate=None, tlm=None, **kwargs):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(args[0], ufl.classes.Equation):
            (eq_arg, x, bcs, J,
             tol, M,
             form_compiler_parameters,
             solver_parameters) = extract_args(*args, **kwargs)
            if tol is not None or M is not None:
                raise OverrideException("Adaptive solves not supported")
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
            if len(kwargs) > 0:
                raise OverrideException("Unexpected keyword arguments")
            A, x, b, linear_solver, preconditioner = \
                extract_args_linear_solve(*args)
            solver_parameters = {"linear_solver": linear_solver,
                                 "preconditioner": preconditioner}

            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise OverrideException("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501

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
            eq._post_process(annotate=annotate, tlm=tlm)

            return return_value
    else:
        return backend_solve(*args, **kwargs)


def project(v, V=None, bcs=None, mesh=None, function=None, solver_type="lu",
            preconditioner_type="default", form_compiler_parameters=None,
            solver_parameters=None, annotate=None, tlm=None):
    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm or solver_parameters is not None:
        if function is None:
            if V is None:
                raise OverrideException("V or function required")
            x = space_new(V)
        else:
            x = function

        if bcs is None:
            bcs = []
        elif isinstance(bcs, backend_DirichletBC):
            bcs = [bcs]

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

        return x
    else:
        return backend_project(
            v, V=V, bcs=bcs, mesh=mesh, function=function,
            solver_type=solver_type,
            preconditioner_type=preconditioner_type,
            form_compiler_parameters=form_compiler_parameters)


def _DirichletBC_apply(self, *args):
    backend_DirichletBC._tlm_adjoint__orig_apply(self, *args)
    if (len(args) > 1 and not isinstance(args[0], backend_Matrix)) \
       or len(args) > 2:
        return

    if isinstance(args[0], backend_Matrix):
        A = args[0]
        if len(args) > 1:
            b = args[1]
        else:
            b = None
    else:
        A = None
        b = args[0]

    if A is not None and hasattr(A, "_tlm_adjoint__bcs") \
       and self not in A._tlm_adjoint__bcs:
        A._tlm_adjoint__bcs.append(self)
    if b is not None and hasattr(b, "_tlm_adjoint__bcs") \
       and self not in b._tlm_adjoint__bcs:
        b._tlm_adjoint__bcs.append(self)


backend_DirichletBC._tlm_adjoint__orig_apply = backend_DirichletBC.apply
backend_DirichletBC.apply = _DirichletBC_apply


def _Function_assign(self, rhs, annotate=None, tlm=None):
    return_value = backend_Function._tlm_adjoint__orig_assign(self, rhs)
    if not isinstance(rhs, backend_Function):
        return return_value

    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        AssignmentSolver(rhs, self).solve(annotate=annotate, tlm=tlm)
    return return_value


backend_Function._tlm_adjoint__orig_assign = backend_Function.assign
backend_Function.assign = _Function_assign


def _Function_vector(self):
    return_value = backend_Function._tlm_adjoint__orig_vector(self)
    return_value._tlm_adjoint__function = self
    return return_value


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
        return_value = super().solve(*args)

        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(args[0], backend_Matrix):
                A, x, b = args
                self.__A = A
            else:
                A = self.__A
                x, b = args

            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise OverrideException("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501

            A = A._tlm_adjoint__form
            x = x._tlm_adjoint__function
            b = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs,
                solver_parameters={"linear_solver": self.__linear_solver,
                                   "lu_solver": self.parameters},
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            function_update_state(x)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            if isinstance(args[0], backend_Matrix):
                A, x, b = args
                self.__A = A

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
        raise OverrideException("Preconditioners not supported")

    def solve(self, *args, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(args[0], backend_Matrix):
                A, x, b = args
                self.__A = None
            else:
                A = self.__A
                x, b = args

            bcs = A._tlm_adjoint__bcs
            if bcs != b._tlm_adjoint__bcs:
                raise OverrideException("Non-matching boundary conditions")
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501

            A = A._tlm_adjoint__form
            x = x._tlm_adjoint__function
            b = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
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
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            return_value = super().solve(*args)
            if isinstance(args[0], backend_Matrix):
                self.__A = None

        return return_value


class LinearVariationalSolver(backend_LinearVariationalSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.__problem = problem

    def solve(self, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            lhs, rhs = self.__problem.a_ufl, self.__problem.L_ufl
            x = self.__problem.u_ufl
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
        raise OverrideException("Bounds not supported")


class NonlinearVariationalSolver(backend_NonlinearVariationalSolver):
    def __init__(self, problem):
        super().__init__(problem)
        self.__problem = problem

    def solve(self, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            x = self.__problem.u_ufl
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

        return return_value
