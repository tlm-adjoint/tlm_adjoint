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
from .backend_interface import *
from .backend_code_generator_interface import copy_parameters_dict, \
    update_parameters_dict

from .equations import AssignmentSolver, EquationSolver, ProjectionSolver, \
    linear_equation_new_x
from .firedrake_equations import LocalProjectionSolver
from .tlm_adjoint import annotation_enabled, tlm_enabled

import copy
import ufl

__all__ = \
    [
        "OverrideException",

        "LinearSolver",
        "LinearVariationalProblem",
        "LinearVariationalSolver",
        "NonlinearVariationalSolver",
        "assemble",
        "project",
        "solve"
    ]


class OverrideException(Exception):
    pass


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


def packed_solver_parameters(solver_parameters, options_prefix=None,
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
                raise InterfaceException("Cannot pass both options_prefix argument and solver parameter")  # noqa: E501
            tlm_adjoint_parameters["options_prefix"] = options_prefix

        if nullspace is not None:
            if "nullspace" in tlm_adjoint_parameters:
                raise InterfaceException("Cannot pass both nullspace argument and solver parameter")  # noqa: E501
            tlm_adjoint_parameters["nullspace"] = nullspace

        if transpose_nullspace is not None:
            if "transpose_nullspace" in tlm_adjoint_parameters:
                raise InterfaceException("Cannot pass both transpose_nullspace argument and solver parameter")  # noqa: E501
            tlm_adjoint_parameters["transpose_nullspace"] = transpose_nullspace

        if near_nullspace is not None:
            if "near_nullspace" in tlm_adjoint_parameters:
                raise InterfaceException("Cannot pass both near_nullspace argument and solver parameter")  # noqa: E501
            tlm_adjoint_parameters["near_nullspace"] = near_nullspace

    return solver_parameters


# Aim for compatibility with Firedrake API, git master revision
# cf7b18cddacae582fd1e92e6d2148d9a538d131a


def assemble(f, tensor=None, bcs=None, form_compiler_parameters=None,
             inverse=False, *args, **kwargs):
    b = backend_assemble(
        f, tensor=tensor, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters, inverse=inverse,
        *args, **kwargs)
    if tensor is None:
        tensor = b

    rank = len(f.arguments())
    if rank != 0 and not inverse:
        form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        if form_compiler_parameters is not None:
            update_parameters_dict(form_compiler_parameters_,
                                   form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_

        if rank != 2:
            tensor._tlm_adjoint__form = f
        tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return tensor


def extract_args_linear_solve(A, x, b, bcs=None, solver_parameters={}):
    return A, x, b, bcs, solver_parameters


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
            if Jp is not None:
                raise OverrideException("Preconditioners not supported")
            if M is not None:
                raise OverrideException("Adaptive solves not supported")
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
             bcs,
             solver_parameters) = extract_args_linear_solve(*args, **kwargs)

            if bcs is None:
                bcs = A.bcs
            form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
            if not parameters_dict_equal(
                    b._tlm_adjoint__form_compiler_parameters,
                    form_compiler_parameters):
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501

            A = A.a
            if isinstance(x, backend_Vector):
                x = x.function
            b = b._tlm_adjoint__form

            eq = EquationSolver(
                linear_equation_new_x(A == b, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)
            eq.solve(annotate=annotate, tlm=tlm)
    else:
        backend_solve(*args, **kwargs)


def project(v, V, bcs=None, solver_parameters=None,
            form_compiler_parameters=None, use_slate_for_inverse=True,
            name=None, annotate=None, tlm=None):
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
            bcs = []
        elif isinstance(bcs, backend_DirichletBC):
            bcs = [bcs]
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
                raise OverrideException("Boundary conditions not supported")
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
        return x
    else:
        return backend_project(
            v, V, bcs=bcs, solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            use_slate_for_inverse=use_slate_for_inverse, name=name)


_orig_Function_assign = backend_Function.assign


def _Function_assign(self, expr, subset=None, annotate=None, tlm=None):
    return_value = _orig_Function_assign(self, expr, subset=subset)
    if not isinstance(expr, backend_Function) or subset is not None:
        return return_value

    if annotate is None:
        annotate = annotation_enabled()
    if tlm is None:
        tlm = tlm_enabled()
    if annotate or tlm:
        AssignmentSolver(expr, self).solve(annotate=annotate, tlm=tlm)
    return return_value


backend_Function.assign = _Function_assign

_orig_Function_project = backend_Function.project


def _Function_project(self, b, *args, **kwargs):
    return project(b, self, *args, **kwargs)


backend_Function.project = _Function_project


class LinearSolver(backend_LinearSolver):
    def __init__(self, A, P=None, solver_parameters=None, nullspace=None,
                 transpose_nullspace=None, near_nullspace=None,
                 options_prefix=None):
        if P is not None:
            raise OverrideException("Preconditioners not supported")

        backend_LinearSolver.__init__(
            self, A, P=P, solver_parameters=solver_parameters,
            nullspace=nullspace, transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace, options_prefix=options_prefix)

    def solve(self, x, b, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            A = self.A
            if isinstance(x, backend_Vector):
                x = x.function
            if isinstance(b, backend_Vector):
                b = b.function
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
                raise OverrideException("Non-matching form compiler parameters")  # noqa: E501

            eq = EquationSolver(
                linear_equation_new_x(A.a == b._tlm_adjoint__form, x,
                                      annotate=annotate, tlm=tlm),
                x, bcs, solver_parameters=solver_parameters,
                form_compiler_parameters=form_compiler_parameters,
                cache_jacobian=False, cache_rhs_assembly=False)

            eq._pre_process(annotate=annotate)
            backend_LinearSolver.solve(self, x, b)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            backend_LinearSolver.solve(self, x, b)


class LinearVariationalProblem(backend_LinearVariationalProblem):
    def __init__(self, a, L, *args, **kwargs):
        backend_LinearVariationalProblem.__init__(self, a, L, *args, **kwargs)
        self._tlm_adjoint__b = L


class LinearVariationalSolver(backend_LinearVariationalSolver):
    def __init__(self, *args, **kwargs):
        problem, = args
        if "appctx" in kwargs:
            raise OverrideException("Preconditioners not supported")

        backend_LinearVariationalSolver.__init__(self, *args, **kwargs)

    def set_transfer_operators(self, *args, **kwargs):
        raise OverrideException("Transfer operators not supported")

    def solve(self, bounds=None, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            if bounds is not None:
                raise OverrideException("Bounds not supported")
            if self._problem.Jp is not None:
                raise OverrideException("Preconditioners not supported")

            solver_parameters = packed_solver_parameters(
                self.parameters, options_prefix=self.options_prefix,
                nullspace=self._ctx._nullspace,
                transpose_nullspace=self._ctx._nullspace_T,
                near_nullspace=self._ctx._near_nullspace)
            form_compiler_parameters = self._problem.form_compiler_parameters
            if form_compiler_parameters is None:
                form_compiler_parameters = {}

            A = self._problem.J
            x = self._problem.u
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
            backend_LinearVariationalSolver.solve(self, bounds=bounds)


class NonlinearVariationalSolver(backend_NonlinearVariationalSolver):
    def __init__(self, *args, **kwargs):
        problem, = args
        if "appctx" in kwargs:
            raise OverrideException("Preconditioners not supported")
        if "pre_jacobian_callback" in kwargs \
           or "pre_function_callback" in kwargs:
            raise OverrideException("Callbacks not supported")

        backend_NonlinearVariationalSolver.__init__(self, *args, **kwargs)

    def set_transfer_operators(self, *args, **kwargs):
        raise OverrideException("Transfer operators not supported")

    def solve(self, bounds=None, annotate=None, tlm=None):
        if annotate is None:
            annotate = annotation_enabled()
        if tlm is None:
            tlm = tlm_enabled()
        if annotate or tlm:
            if bounds is not None:
                raise OverrideException("Bounds not supported")
            if self._problem.Jp is not None:
                raise OverrideException("Preconditioners not supported")

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
            backend_NonlinearVariationalSolver.solve(self, bounds=bounds)
