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

from .backend import TestFunction, TrialFunction, adjoint, backend_Constant, \
    backend_DirichletBC, backend_Function, backend_FunctionSpace, parameters
from ..interface import check_space_type, check_space_types, function_assign, \
    function_get_values, function_id, function_inner, function_is_scalar, \
    function_new, function_new_conjugate_dual, function_replacement, \
    function_scalar_value, function_set_values, function_space, \
    function_update_caches, function_update_state, function_zero, \
    is_function, space_id
from .backend_code_generator_interface import assemble, \
    assemble_linear_solver, copy_parameters_dict, \
    form_form_compiler_parameters, function_vector, homogenize, \
    interpolate_expression, matrix_multiply, \
    process_adjoint_solver_parameters, process_solver_parameters, r0_space, \
    rhs_addto, rhs_copy, solve, update_parameters_dict, verify_assembly

from ..caches import CacheRef
from ..equations import AssignmentSolver, Equation, EquationException, \
    NullSolver, get_tangent_linear

from .caches import assembly_cache, form_neg, is_cached, linear_solver_cache, \
    split_form
from .functions import bcs_is_cached, bcs_is_homogeneous, bcs_is_static, \
    eliminate_zeros, extract_coefficients

import copy
import numpy as np
import ufl
import warnings

__all__ = \
    [
        "AssembleSolver",
        "DirichletBCSolver",
        "EquationSolver",
        "ExprEvaluationSolver",
        "ProjectionSolver",
        "linear_equation_new_x"
    ]


def derivative_space(x):
    space = function_space(x)
    if not isinstance(space, backend_FunctionSpace):
        e = x.ufl_element()
        assert e.family() == "Real" and e.degree() == 0
        space = r0_space(x)
    return space


def derivative(form, x, argument=None):
    if argument is None:
        space = derivative_space(x)
        rank = len(form.arguments())
        Argument = {0: TestFunction, 1: TrialFunction}[rank]
        argument = Argument(space)

    return ufl.derivative(form, x, argument=argument)


def derivative_dependencies(expr, dep):
    dexpr = ufl.derivative(expr, dep)
    dexpr = ufl.algorithms.expand_derivatives(dexpr)
    return extract_coefficients(dexpr)


def extract_dependencies(expr):
    deps = {}
    nl_deps = {}
    for dep in extract_coefficients(expr):
        if is_function(dep):
            dep_id = function_id(dep)
            if dep_id not in deps:
                deps[dep_id] = dep
            if dep_id not in nl_deps:
                n_nl_deps = 0
                for nl_dep in derivative_dependencies(expr, dep):
                    if is_function(nl_dep):
                        nl_dep_id = function_id(nl_dep)
                        if nl_dep_id not in nl_deps:
                            nl_deps[nl_dep_id] = nl_dep
                        n_nl_deps += 1
                if n_nl_deps > 0 and dep_id not in nl_deps:
                    nl_deps[dep_id] = dep

    deps = {dep_id: deps[dep_id]
            for dep_id in sorted(deps.keys())}
    nl_deps = {nl_dep_id: nl_deps[nl_dep_id]
               for nl_dep_id in sorted(nl_deps.keys())}

    assert len(set(nl_deps.keys()).difference(set(deps.keys()))) == 0
    for dep in deps.values():
        check_space_type(dep, "primal")

    return deps, nl_deps


def apply_rhs_bcs(b, hbcs, b_bc=None):
    for bc in hbcs:
        bc.apply(b)
    if b_bc is not None:
        rhs_addto(b, b_bc)


class ExprEquation(Equation):
    def _replace_map(self, deps):
        eq_deps = self.dependencies()
        assert len(eq_deps) == len(deps)
        return dict(zip(eq_deps, deps))

    def _replace(self, expr, deps):
        return ufl.replace(expr, self._replace_map(deps))

    def _nonlinear_replace_map(self, nl_deps):
        eq_nl_deps = self.nonlinear_dependencies()
        assert len(eq_nl_deps) == len(nl_deps)
        return dict(zip(eq_nl_deps, nl_deps))

    def _nonlinear_replace(self, expr, nl_deps):
        return ufl.replace(expr, self._nonlinear_replace_map(nl_deps))


class AssembleSolver(ExprEquation):
    def __init__(self, rhs, x, form_compiler_parameters=None,
                 match_quadrature=None):
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if match_quadrature is None:
            match_quadrature = parameters["tlm_adjoint"]["AssembleSolver"]["match_quadrature"]  # noqa: E501

        rhs = ufl.classes.Form(rhs.integrals())

        rank = len(rhs.arguments())
        if rank == 0:
            check_space_type(x, "primal")
            if not function_is_scalar(x):
                raise EquationException("Rank 0 forms can only be assigned to "
                                        "scalars")
        else:
            raise EquationException("Must be a rank 0 form")

        deps, nl_deps = extract_dependencies(rhs)
        if function_id(x) in deps:
            raise EquationException("Invalid non-linear dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        form_compiler_parameters_ = \
            copy_parameters_dict(parameters["form_compiler"])
        update_parameters_dict(form_compiler_parameters_,
                               form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_
        if match_quadrature:
            update_parameters_dict(
                form_compiler_parameters,
                form_form_compiler_parameters(rhs, form_compiler_parameters))

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs
        self._form_compiler_parameters = form_compiler_parameters

    def drop_references(self):
        replace_map = {dep: function_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        if deps is None:
            rhs = self._rhs
        else:
            rhs = self._replace(self._rhs, deps)

        function_assign(
            x,
            assemble(rhs,
                     form_compiler_parameters=self._form_compiler_parameters))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        # Derived from EquationSolver.derivative_action (see dolfin-adjoint
        # reference below). Code first added 2017-12-07.
        # Re-written 2018-01-28
        # Updated to adjoint only form 2018-01-29

        eq_deps = self.dependencies()
        if dep_index < 0 or dep_index >= len(eq_deps):
            raise EquationException("dep_index out of bounds")
        elif dep_index == 0:
            return adj_x

        dep = eq_deps[dep_index]
        dF = derivative(self._rhs, dep)
        dF = ufl.algorithms.expand_derivatives(dF)
        dF = eliminate_zeros(dF)
        if dF.empty():
            return None

        dF = self._nonlinear_replace(dF, nl_deps)
        dF = ufl.Form([integral.reconstruct(integrand=ufl.conj(integral.integrand()))  # noqa: E501
                       for integral in dF.integrals()])  # dF = adjoint(dF)
        dF = assemble(
            dF, form_compiler_parameters=self._form_compiler_parameters)
        return (-function_scalar_value(adj_x), dF)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.Form([])
        for dep in self.dependencies():
            if dep != x:
                tau_dep = get_tangent_linear(dep, M, dM, tlm_map)
                if tau_dep is not None:
                    tlm_rhs += derivative(self._rhs, dep, argument=tau_dep)

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if tlm_rhs.empty():
            return NullSolver(tlm_map[x])
        else:
            return AssembleSolver(
                tlm_rhs, tlm_map[x],
                form_compiler_parameters=self._form_compiler_parameters)


def unbound_form(form, deps):
    replacement_deps = tuple(function_replacement(dep) for dep in deps)
    assert len(deps) == len(replacement_deps)
    return_value = ufl.replace(form, dict(zip(deps, replacement_deps)))
    return_value._cache["_tlm_adjoint__replacement_deps"] = replacement_deps
    return return_value


def bind_form(form, deps):
    replacement_deps = form._cache["_tlm_adjoint__replacement_deps"]
    assert len(replacement_deps) == len(deps)
    form._cache["_tlm_adjoint__bindings"] = dict(zip(replacement_deps, deps))


def unbind_form(form):
    if "_tlm_adjoint__bindings" in form._cache:
        del form._cache["_tlm_adjoint__bindings"]


def homogenized_bc(bc):
    if bcs_is_homogeneous([bc]):
        return bc
    else:
        hbc = homogenize(bc)
        static = bcs_is_static([bc])
        hbc.is_static = lambda: static
        cache = bcs_is_cached([bc])
        hbc.is_cached = lambda: cache
        hbc.is_homogeneous = lambda: True
        hbc.homogenize = lambda: hbc
        return hbc


class EquationSolver(ExprEquation):
    # eq, x, bcs, J, form_compiler_parameters and solver_parameters argument
    # usage based on the interface for the solve function in FEniCS (see e.g.
    # FEniCS 2017.1.0)
    def __init__(self, eq, x, bcs=None, J=None, form_compiler_parameters=None,
                 solver_parameters=None, adjoint_solver_parameters=None,
                 tlm_solver_parameters=None, initial_guess=None,
                 cache_jacobian=None, cache_adjoint_jacobian=None,
                 cache_tlm_jacobian=None, cache_rhs_assembly=None,
                 match_quadrature=None, defer_adjoint_assembly=None):
        if bcs is None:
            bcs = []
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if solver_parameters is None:
            solver_parameters = {}

        if isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        else:
            bcs = tuple(bcs)
        if cache_jacobian is None:
            if not parameters["tlm_adjoint"]["EquationSolver"]["enable_jacobian_caching"]:  # noqa: E501
                cache_jacobian = False
        if cache_rhs_assembly is None:
            cache_rhs_assembly = parameters["tlm_adjoint"]["EquationSolver"]["cache_rhs_assembly"]  # noqa: E501
        if match_quadrature is None:
            match_quadrature = parameters["tlm_adjoint"]["EquationSolver"]["match_quadrature"]  # noqa: E501
        if defer_adjoint_assembly is None:
            defer_adjoint_assembly = parameters["tlm_adjoint"]["EquationSolver"]["defer_adjoint_assembly"]  # noqa: E501
        if match_quadrature and defer_adjoint_assembly:
            raise EquationException("Cannot both match quadrature and defer adjoint assembly")  # noqa: E501

        check_space_type(x, "primal")

        lhs, rhs = eq.lhs, eq.rhs
        del eq
        lhs = ufl.classes.Form(lhs.integrals())
        linear = isinstance(rhs, ufl.classes.Form)
        if linear:
            rhs = ufl.classes.Form(rhs.integrals())
        if J is not None:
            J = ufl.classes.Form(J.integrals())

        if linear:
            if x in lhs.coefficients() or x in rhs.coefficients():
                raise EquationException("Invalid non-linear dependency")
            F = ufl.action(lhs, coefficient=x) + form_neg(rhs)
            nl_solve_J = None
            J = lhs
        else:
            F = lhs
            if rhs != 0:
                raise EquationException("Invalid right-hand-side")
            nl_solve_J = J
            J = derivative(F, x)
            J = ufl.algorithms.expand_derivatives(J)

        deps, nl_deps = extract_dependencies(F)

        if nl_solve_J is not None:
            for dep in nl_solve_J.coefficients():
                if is_function(dep):
                    dep_id = function_id(dep)
                    if dep_id not in deps:
                        deps[dep_id] = dep

        if initial_guess is not None:
            warnings.warn("'initial_guess' argument is deprecated",
                          DeprecationWarning, stacklevel=2)
            if initial_guess == x:
                initial_guess = None
            else:
                initial_guess_id = function_id(initial_guess)
                if initial_guess_id not in deps:
                    deps[initial_guess_id] = initial_guess

        deps = list(deps.values())
        if x in deps:
            deps.remove(x)
        deps.insert(0, x)
        nl_deps = tuple(nl_deps.values())

        hbcs = tuple(homogenized_bc(bc) for bc in bcs)

        if cache_jacobian is None:
            cache_jacobian = is_cached(J) and bcs_is_cached(bcs)
        if cache_adjoint_jacobian is None:
            cache_adjoint_jacobian = cache_jacobian
        if cache_tlm_jacobian is None:
            cache_tlm_jacobian = cache_jacobian

        (solver_parameters, linear_solver_parameters,
         ic, J_ic) = process_solver_parameters(solver_parameters, linear)

        if adjoint_solver_parameters is None:
            adjoint_solver_parameters = process_adjoint_solver_parameters(linear_solver_parameters)  # noqa: E501
            adj_ic = J_ic
        else:
            (_, adjoint_solver_parameters,
             adj_ic, _) = process_solver_parameters(adjoint_solver_parameters, linear=True)  # noqa: E501

        if tlm_solver_parameters is not None:
            (_, tlm_solver_parameters,
             _, _) = process_solver_parameters(tlm_solver_parameters, linear=True)  # noqa: E501

        form_compiler_parameters_ = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        update_parameters_dict(form_compiler_parameters_,
                               form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_
        if match_quadrature:
            update_parameters_dict(
                form_compiler_parameters,
                form_form_compiler_parameters(F, form_compiler_parameters))

        super().__init__(x, deps, nl_deps=nl_deps,
                         ic=initial_guess is None and ic,
                         adj_ic=adj_ic, adj_type="primal")
        self._F = F
        self._lhs, self._rhs = lhs, rhs
        self._bcs = bcs
        self._hbcs = hbcs
        self._J = J
        self._nl_solve_J = nl_solve_J
        self._form_compiler_parameters = form_compiler_parameters
        self._solver_parameters = solver_parameters
        self._linear_solver_parameters = linear_solver_parameters
        self._adjoint_solver_parameters = adjoint_solver_parameters
        self._tlm_solver_parameters = tlm_solver_parameters
        if initial_guess is None:
            self._initial_guess_index = None
        else:
            self._initial_guess_index = deps.index(initial_guess)
        self._linear = linear

        self._cache_jacobian = cache_jacobian
        self._cache_adjoint_jacobian = cache_adjoint_jacobian
        self._cache_tlm_jacobian = cache_tlm_jacobian
        self._cache_rhs_assembly = cache_rhs_assembly
        self._defer_adjoint_assembly = defer_adjoint_assembly

        self._forward_eq = None
        self._forward_J_solver = CacheRef()
        self._forward_b_pa = None

        self._adjoint_dF_cache = {}
        self._adjoint_action_cache = {}

        self._adjoint_J_solver = CacheRef()
        self._adjoint_J = None

    def drop_references(self):
        replace_map = {dep: function_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._F = ufl.replace(self._F, replace_map)
        self._lhs = ufl.replace(self._lhs, replace_map)
        if self._rhs != 0:
            self._rhs = ufl.replace(self._rhs, replace_map)
        self._J = ufl.replace(self._J, replace_map)
        if self._nl_solve_J is not None:
            self._nl_solve_J = ufl.replace(self._nl_solve_J, replace_map)

        if self._forward_b_pa is not None:
            cached_form, mat_forms, non_cached_form = self._forward_b_pa

            if cached_form is not None:
                cached_form[0] = ufl.replace(cached_form[0], replace_map)
            for dep_index, (mat_form, mat_cache) in mat_forms.items():
                mat_forms[dep_index][0] = ufl.replace(mat_form, replace_map)

            # self._forward_b_pa = (cached_form, mat_forms, non_cached_form)

        for dep_index, dF in self._adjoint_dF_cache.items():
            if dF is not None:
                self._adjoint_dF_cache[dep_index] = ufl.replace(dF, replace_map)  # noqa: E501

    def _cached_rhs(self, deps, b_bc=None):
        eq_deps = self.dependencies()

        if self._forward_b_pa is None:
            rhs = eliminate_zeros(self._rhs, force_non_empty_form=True)
            cached_form, mat_forms_, non_cached_form = split_form(rhs)
            mat_forms = {}
            for dep_index, dep in enumerate(eq_deps):
                dep_id = function_id(dep)
                if dep_id in mat_forms_:
                    mat_forms[dep_index] = [mat_forms_[dep_id], CacheRef()]
            del mat_forms_

            if non_cached_form.empty():
                non_cached_form = None
            else:
                non_cached_form = unbound_form(non_cached_form, eq_deps)

            if cached_form.empty():
                cached_form = None
            else:
                cached_form = [cached_form, CacheRef()]

            self._forward_b_pa = (cached_form, mat_forms, non_cached_form)
        else:
            cached_form, mat_forms, non_cached_form = self._forward_b_pa

        b = None

        if non_cached_form is not None:
            bind_form(non_cached_form, eq_deps if deps is None else deps)
            b = assemble(
                non_cached_form,
                form_compiler_parameters=self._form_compiler_parameters)
            unbind_form(non_cached_form)

        for dep_index, (mat_form, mat_cache) in mat_forms.items():
            mat_bc = mat_cache()
            if mat_bc is None:
                mat_forms[dep_index][1], mat_bc = assembly_cache().assemble(
                    mat_form,
                    form_compiler_parameters=self._form_compiler_parameters,
                    linear_solver_parameters=self._linear_solver_parameters,
                    replace_map=None if deps is None else self._replace_map(deps))  # noqa: E501
            mat, _ = mat_bc
            dep = (eq_deps if deps is None else deps)[dep_index]
            if b is None:
                b = function_vector(function_new_conjugate_dual(self.x()))
                matrix_multiply(mat, function_vector(dep), tensor=b)
            else:
                matrix_multiply(mat, function_vector(dep), tensor=b,
                                addto=True)

        if cached_form is not None:
            cached_b = cached_form[1]()
            if cached_b is None:
                cached_form[1], cached_b = assembly_cache().assemble(
                    cached_form[0],
                    form_compiler_parameters=self._form_compiler_parameters,
                    replace_map=None if deps is None else self._replace_map(deps))  # noqa: E501
            if b is None:
                b = rhs_copy(cached_b)
            else:
                rhs_addto(b, cached_b)

        if b is None:
            raise EquationException("Empty right-hand-side")

        apply_rhs_bcs(b, self._hbcs, b_bc=b_bc)
        return b

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()

        if self._initial_guess_index is not None:
            if deps is None:
                initial_guess = eq_deps[self._initial_guess_index]
            else:
                initial_guess = deps[self._initial_guess_index]
            function_assign(x, initial_guess)
            function_update_state(x)
            function_update_caches(self.x(), value=x)

        if self._linear:
            if self._cache_jacobian:
                # Cases 1 and 2: Linear, Jacobian cached, with or without RHS
                # assembly caching

                J_solver_mat_bc = self._forward_J_solver()
                if J_solver_mat_bc is None:
                    # Assemble and cache the Jacobian, construct and cache the
                    # linear solver
                    self._forward_J_solver, J_solver_mat_bc = \
                        linear_solver_cache().linear_solver(
                            self._J, bcs=self._bcs,
                            form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                            linear_solver_parameters=self._linear_solver_parameters,  # noqa: E501
                            replace_map=None if deps is None else self._replace_map(deps))  # noqa: E501
                J_solver, J_mat, b_bc = J_solver_mat_bc

                if self._cache_rhs_assembly:
                    # Assemble the RHS with RHS assembly caching
                    b = self._cached_rhs(deps, b_bc=b_bc)
                else:
                    # Assemble the RHS without RHS assembly caching
                    if deps is None:
                        rhs = self._rhs
                    else:
                        if self._forward_eq is None:
                            self._forward_eq = \
                                (None,
                                 None,
                                 unbound_form(self._rhs, eq_deps))
                        _, _, rhs = self._forward_eq
                        bind_form(rhs, deps)
                    b = assemble(
                        rhs,
                        form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
                    if deps is not None:
                        unbind_form(rhs)

                    # Add bc RHS terms
                    apply_rhs_bcs(b, self._hbcs, b_bc=b_bc)
            else:
                if self._cache_rhs_assembly:
                    # Case 3: Linear, Jacobian not cached, with RHS assembly
                    # caching

                    # Construct the linear solver, assemble the Jacobian
                    if deps is None:
                        J = self._J
                    else:
                        if self._forward_eq is None:
                            self._forward_eq = \
                                (None,
                                 unbound_form(self._J, eq_deps),
                                 None)
                        _, J, _ = self._forward_eq
                        bind_form(J, deps)
                    J_solver, J_mat, b_bc = assemble_linear_solver(
                        J, bcs=self._bcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._linear_solver_parameters)  # noqa: E501
                    if deps is not None:
                        unbind_form(J)

                    # Assemble the RHS with RHS assembly caching
                    b = self._cached_rhs(deps, b_bc=b_bc)
                else:
                    # Case 4: Linear, Jacobian not cached, without RHS assembly
                    # caching

                    # Construct the linear solver, assemble the Jacobian and
                    # RHS
                    if deps is None:
                        J, rhs = self._J, self._rhs
                    else:
                        if self._forward_eq is None:
                            self._forward_eq = \
                                (None,
                                 unbound_form(self._J, eq_deps),
                                 unbound_form(self._rhs, eq_deps))
                        _, J, rhs = self._forward_eq
                        bind_form(J, deps)
                        bind_form(rhs, deps)
                    J_solver, J_mat, b = assemble_linear_solver(
                        J, b_form=rhs, bcs=self._bcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._linear_solver_parameters)  # noqa: E501
                    if deps is not None:
                        unbind_form(J)
                        unbind_form(rhs)

            J_tolerance = parameters["tlm_adjoint"]["assembly_verification"]["jacobian_tolerance"]  # noqa: E501
            b_tolerance = parameters["tlm_adjoint"]["assembly_verification"]["rhs_tolerance"]  # noqa: E501
            if not np.isposinf(J_tolerance) or not np.isposinf(b_tolerance):
                verify_assembly(
                    self._J if deps is None
                    else self._replace(self._J, deps),
                    self._rhs if deps is None
                    else self._replace(self._rhs, deps),
                    J_mat, b, self._bcs, self._form_compiler_parameters,
                    self._linear_solver_parameters, J_tolerance, b_tolerance)

            J_solver.solve(function_vector(x), b)
        else:
            # Case 5: Non-linear
            assert self._rhs == 0
            lhs = self._lhs
            if self._nl_solve_J is None:
                J = self._J
            else:
                J = self._nl_solve_J
            if deps is not None:
                lhs = self._replace(lhs, deps)
                J = self._replace(J, deps)
            solve(lhs == 0, x, self._bcs, J=J,
                  form_compiler_parameters=self._form_compiler_parameters,
                  solver_parameters=self._solver_parameters)

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        for dep_index, dep_B in dep_Bs.items():
            if dep_index not in self._adjoint_dF_cache:
                dep = self.dependencies()[dep_index]
                dF = derivative(self._F, dep)
                dF = ufl.algorithms.expand_derivatives(dF)
                dF = eliminate_zeros(dF)
                if dF.empty():
                    dF = None
                else:
                    dF = adjoint(dF)
                self._adjoint_dF_cache[dep_index] = dF
            dF = self._adjoint_dF_cache[dep_index]

            if dF is not None:
                if dep_index not in self._adjoint_action_cache:
                    if self._cache_rhs_assembly \
                            and isinstance(adj_x, backend_Function) \
                            and is_cached(dF):
                        # Cached matrix action
                        self._adjoint_action_cache[dep_index] = CacheRef()
                    elif self._defer_adjoint_assembly:
                        # Cached form, deferred assembly
                        self._adjoint_action_cache[dep_index] = None
                    else:
                        # Cached form, immediate assembly
                        self._adjoint_action_cache[dep_index] = unbound_form(
                            ufl.action(dF, coefficient=adj_x),
                            list(self.nonlinear_dependencies()) + [adj_x])
                cache = self._adjoint_action_cache[dep_index]

                if cache is None:
                    # Cached form, deferred assembly
                    dep_B.sub(ufl.action(
                        self._nonlinear_replace(dF, nl_deps),
                        coefficient=adj_x))
                elif isinstance(cache, CacheRef):
                    # Cached matrix action
                    mat_bc = cache()
                    if mat_bc is None:
                        self._adjoint_action_cache[dep_index], (mat, _) = \
                            assembly_cache().assemble(
                                dF,
                                form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                                replace_map=self._nonlinear_replace_map(nl_deps))  # noqa: E501
                    else:
                        mat, _ = mat_bc
                    F = function_vector(function_new_conjugate_dual(self.dependencies()[dep_index]))  # noqa: E501
                    matrix_multiply(mat, function_vector(adj_x), tensor=F)
                    dep_B.sub(F)
                else:
                    # Cached form, immediate assembly
                    assert isinstance(cache, ufl.classes.Form)
                    bind_form(cache, list(nl_deps) + [adj_x])
                    dep_B.sub(assemble(
                        cache,
                        form_compiler_parameters=self._form_compiler_parameters))  # noqa: E501
                    unbind_form(cache)

    # def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    #     # Similar to 'RHS.derivative_action' and
    #     # 'RHS.second_derivative_action' in dolfin-adjoint file
    #     # dolfin_adjoint/adjrhs.py (see e.g. dolfin-adjoint version 2017.1.0)
    #     # Code first added to JRM personal repository 2016-05-22
    #     # Code first added to dolfin_adjoint_custom repository 2016-06-02
    #     # Re-written 2018-01-28

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        if adj_x is None:
            adj_x = self.new_adj_x()

        if self._cache_adjoint_jacobian:
            J_solver_mat_bc = self._adjoint_J_solver()
            if J_solver_mat_bc is None:
                J = adjoint(self._J)
                self._adjoint_J_solver, J_solver_mat_bc = \
                    linear_solver_cache().linear_solver(
                        J, bcs=self._hbcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._adjoint_solver_parameters,  # noqa: E501
                        replace_map=self._nonlinear_replace_map(nl_deps))
            J_solver, _, _ = J_solver_mat_bc

            apply_rhs_bcs(function_vector(b), self._hbcs)
            J_solver.solve(function_vector(adj_x), function_vector(b))

            return adj_x
        else:
            if self._adjoint_J is None:
                self._adjoint_J = unbound_form(
                    adjoint(self._J), self.nonlinear_dependencies())
            bind_form(self._adjoint_J, nl_deps)
            J_solver, _, _ = assemble_linear_solver(
                self._adjoint_J, bcs=self._hbcs,
                form_compiler_parameters=self._form_compiler_parameters,
                linear_solver_parameters=self._adjoint_solver_parameters)
            unbind_form(self._adjoint_J)

            apply_rhs_bcs(function_vector(b), self._hbcs)
            J_solver.solve(function_vector(adj_x), function_vector(b))

            return adj_x

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.Form([])
        for dep in self.dependencies():
            if dep != x:
                tau_dep = get_tangent_linear(dep, M, dM, tlm_map)
                if tau_dep is not None:
                    tlm_rhs += form_neg(derivative(self._F, dep,
                                                   argument=tau_dep))

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if tlm_rhs.empty():
            return NullSolver(tlm_map[x])
        else:
            if self._tlm_solver_parameters is None:
                tlm_solver_parameters = self._linear_solver_parameters
            else:
                tlm_solver_parameters = self._tlm_solver_parameters
            if self._initial_guess_index is None:
                tlm_initial_guess = None
            else:
                initial_guess = self.dependencies()[self._initial_guess_index]
                tlm_initial_guess = tlm_map[initial_guess]
            return EquationSolver(
                self._J == tlm_rhs, tlm_map[x], self._hbcs,
                form_compiler_parameters=self._form_compiler_parameters,
                solver_parameters=tlm_solver_parameters,
                adjoint_solver_parameters=self._adjoint_solver_parameters,
                tlm_solver_parameters=tlm_solver_parameters,
                initial_guess=tlm_initial_guess,
                cache_jacobian=self._cache_tlm_jacobian,
                cache_adjoint_jacobian=self._cache_adjoint_jacobian,
                cache_tlm_jacobian=self._cache_tlm_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly,
                defer_adjoint_assembly=self._defer_adjoint_assembly)


def linear_equation_new_x(eq, x, manager=None, annotate=None, tlm=None):
    lhs, rhs = eq.lhs, eq.rhs
    lhs_x_dep = x in lhs.coefficients()
    rhs_x_dep = x in rhs.coefficients()
    if lhs_x_dep or rhs_x_dep:
        x_old = function_new(x)
        AssignmentSolver(x, x_old).solve(manager=manager, annotate=annotate,
                                         tlm=tlm)
        if lhs_x_dep:
            lhs = ufl.replace(lhs, {x: x_old})
        if rhs_x_dep:
            rhs = ufl.replace(rhs, {x: x_old})
        return lhs == rhs
    else:
        return eq


class ProjectionSolver(EquationSolver):
    def __init__(self, rhs, x, *args, **kwargs):
        space = function_space(x)
        test, trial = TestFunction(space), TrialFunction(space)
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(rhs, test) * ufl.dx
        super().__init__(ufl.inner(trial, test) * ufl.dx == rhs, x,
                         *args, **kwargs)


class DirichletBCSolver(Equation):
    def __init__(self, y, x, *args, **kwargs):
        check_space_type(x, "primal")
        check_space_type(y, "primal")

        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)
        self._bc_args = copy.copy(args)
        self._bc_kwargs = copy.copy(kwargs)

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        function_zero(x)
        backend_DirichletBC(
            function_space(x), y,
            *self._bc_args, **self._bc_kwargs).apply(function_vector(x))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        elif dep_index == 1:
            _, y = self.dependencies()
            F = function_new_conjugate_dual(y)
            backend_DirichletBC(
                function_space(y), adj_x,
                *self._bc_args, **self._bc_kwargs).apply(function_vector(F))
            return (-1.0, F)
        else:
            raise EquationException("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x, y = self.dependencies()

        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return NullSolver(tlm_map[x])
        else:
            return DirichletBCSolver(tau_y, tlm_map[x],
                                     *self._bc_args, **self._bc_kwargs)


class ExprEvaluationSolver(ExprEquation):
    def __init__(self, rhs, x):
        if isinstance(rhs, ufl.classes.Form):
            raise EquationException("rhs should not be a Form")

        deps, nl_deps = extract_dependencies(rhs)
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        for dep in deps:
            if dep == x:
                raise EquationException("Invalid non-linear dependency")
            check_space_types(x, dep)
            if isinstance(dep, backend_Function) \
                    and space_id(function_space(dep)) != space_id(function_space(x)):  # noqa: E501
                raise EquationException("Invalid dependency")
        deps.insert(0, x)

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs

    def drop_references(self):
        replace_map = {dep: function_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        if deps is None:
            interpolate_expression(x, self._rhs)
        else:
            interpolate_expression(x, self._replace(self._rhs, deps))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        eq_deps = self.dependencies()
        if dep_index < 0 or dep_index >= len(eq_deps):
            raise EquationException("dep_index out of bounds")
        elif dep_index == 0:
            return adj_x

        dep = eq_deps[dep_index]
        dF = ufl.diff(self._rhs, dep)
        dF = ufl.algorithms.expand_derivatives(dF)
        dF = eliminate_zeros(dF)
        dF = self._nonlinear_replace(dF, nl_deps)

        dF_val = function_new_conjugate_dual(adj_x)
        interpolate_expression(dF_val, dF)

        F = function_new_conjugate_dual(dep)
        if isinstance(F, backend_Constant):
            function_assign(F, function_inner(adj_x, dF_val))
        else:
            assert isinstance(F, backend_Function)
            function_set_values(
                F, function_get_values(dF_val).conjugate() * function_get_values(adj_x))  # noqa: E501
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.Zero()
        for dep in self.dependencies():
            if dep != x:
                tau_dep = get_tangent_linear(dep, M, dM, tlm_map)
                if tau_dep is not None:
                    tlm_rhs += derivative(self._rhs, dep, argument=tau_dep)

        if isinstance(tlm_rhs, ufl.classes.Zero):
            return NullSolver(tlm_map[x])
        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.Zero):
            return NullSolver(tlm_map[x])
        else:
            return ExprEvaluationSolver(tlm_rhs, tlm_map[x])
