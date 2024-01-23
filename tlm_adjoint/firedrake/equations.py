"""This module implements finite element calculations. In particular the
:class:`.EquationSolver` class implements the solution of finite element
variational problems.
"""

from .backend import (
    TestFunction, TrialFunction, adjoint, backend_Constant,
    backend_DirichletBC, backend_Function, complex_mode, parameters)
from ..interface import (
    check_space_type, is_var, var_assign, var_axpy, var_id,
    var_increment_state_lock, var_is_scalar, var_new, var_new_conjugate_dual,
    var_replacement, var_scalar_value, var_space, var_update_caches, var_zero)
from .backend_code_generator_interface import (
    assemble, assemble_linear_solver, copy_parameters_dict,
    form_compiler_quadrature_parameters, homogenize, interpolate_expression,
    matrix_multiply, process_adjoint_solver_parameters,
    process_solver_parameters, solve, update_parameters_dict, verify_assembly)

from ..caches import CacheRef
from ..equation import Equation, ZeroAssignment
from ..equations import Assignment

from .caches import assembly_cache, is_cached, linear_solver_cache, split_form
from .functions import (
    ReplacementConstant, bcs_is_cached, bcs_is_homogeneous, bcs_is_static,
    derivative, eliminate_zeros, expr_zero, extract_coefficients, iter_expr)

import itertools
import numpy as np
import ufl

__all__ = \
    [
        "Assembly",
        "DirichletBCApplication",
        "EquationSolver",
        "ExprInterpolation",
        "Projection",
        "expr_new_x",
        "linear_equation_new_x"
    ]


def extract_derivative_coefficients(expr, dep):
    dexpr = derivative(expr, dep, enable_automatic_argument=False)
    dexpr = ufl.algorithms.expand_derivatives(dexpr)
    return extract_coefficients(dexpr)


def extract_dependencies(expr, *, space_type=None):
    deps = {}
    nl_deps = {}
    for dep in extract_coefficients(expr):
        if is_var(dep):
            deps.setdefault(var_id(dep), dep)
            for nl_dep in extract_derivative_coefficients(expr, dep):
                if is_var(nl_dep):
                    nl_deps.setdefault(var_id(dep), dep)
                    nl_deps.setdefault(var_id(nl_dep), nl_dep)

    deps = {dep_id: deps[dep_id]
            for dep_id in sorted(deps.keys())}
    nl_deps = {nl_dep_id: nl_deps[nl_dep_id]
               for nl_dep_id in sorted(nl_deps.keys())}

    assert len(set(nl_deps.keys()).difference(set(deps.keys()))) == 0
    if space_type is not None:
        for dep in deps.values():
            check_space_type(dep, space_type)

    return deps, nl_deps


def apply_rhs_bcs(b, hbcs, *, b_bc=None):
    for bc in hbcs:
        bc.apply(b)
    if b_bc is not None:
        var_axpy(b, 1.0, b_bc)


class ExprEquation(Equation):
    def _replace_map(self, deps):
        if deps is None:
            return None
        else:
            eq_deps = self.dependencies()
            assert len(eq_deps) == len(deps)
            return {eq_dep: dep
                    for eq_dep, dep in zip(eq_deps, deps)
                    if isinstance(eq_dep, (ufl.classes.Expr,
                                           ufl.classes.Cofunction))}

    def _replace(self, expr, deps):
        if deps is None:
            return expr
        else:
            replace_map = self._replace_map(deps)
            return ufl.replace(expr, replace_map)

    def _nonlinear_replace_map(self, nl_deps):
        eq_nl_deps = self.nonlinear_dependencies()
        assert len(eq_nl_deps) == len(nl_deps)
        return {eq_nl_dep: nl_dep
                for eq_nl_dep, nl_dep in zip(eq_nl_deps, nl_deps)
                if isinstance(eq_nl_dep, (ufl.classes.Expr,
                                          ufl.classes.Cofunction))}

    def _nonlinear_replace(self, expr, nl_deps):
        replace_map = self._nonlinear_replace_map(nl_deps)
        return ufl.replace(expr, replace_map)


class Assembly(ExprEquation):
    r"""Represents assignment to the result of finite element assembly:

    .. code-block:: python

        x = assemble(rhs)

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: A variable defining the forward solution.
    :arg rhs: A :class:`ufl.form.BaseForm`` to assemble. Should have arity 0 or
        1, and should not depend on `x`.
    :arg form_compiler_parameters: Form compiler parameters.
    :arg match_quadrature: Whether to set quadrature parameters consistently in
        the forward, adjoint, and tangent-linears. Defaults to
        `parameters['tlm_adjoint']['Assembly']['match_quadrature']`.
    """

    def __init__(self, x, rhs, *,
                 form_compiler_parameters=None, match_quadrature=None):
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if match_quadrature is None:
            match_quadrature = parameters["tlm_adjoint"]["Assembly"]["match_quadrature"]  # noqa: E501

        for weight, _ in iter_expr(rhs):
            if len(tuple(c for c in extract_coefficients(weight)
                         if is_var(c))) > 0:
                # See Firedrake issue #3292
                raise NotImplementedError("FormSum weights cannot depend on "
                                          "variables")

        arity = len(rhs.arguments())
        if arity == 0:
            check_space_type(x, "primal")
            if not var_is_scalar(x):
                raise ValueError("Arity 0 forms can only be assigned to "
                                 "scalar variables")
        elif arity == 1:
            check_space_type(x, "conjugate_dual")
        else:
            raise ValueError("Must be an arity 0 or arity 1 form")

        deps, nl_deps = extract_dependencies(rhs)
        if var_id(x) in deps:
            raise ValueError("Invalid dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        form_compiler_parameters_ = \
            copy_parameters_dict(parameters["form_compiler"])
        update_parameters_dict(form_compiler_parameters_,
                               form_compiler_parameters)
        form_compiler_parameters = form_compiler_parameters_
        del form_compiler_parameters_
        if match_quadrature:
            update_parameters_dict(
                form_compiler_parameters,
                form_compiler_quadrature_parameters(rhs, form_compiler_parameters))  # noqa: E501

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs
        self._form_compiler_parameters = form_compiler_parameters
        self._arity = arity

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()
                       if isinstance(dep, (ufl.classes.Expr,
                                           ufl.classes.Cofunction))}

        super().drop_references()
        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        rhs = self._replace(self._rhs, deps)

        if self._arity == 0:
            var_assign(
                x,
                assemble(rhs, form_compiler_parameters=self._form_compiler_parameters))  # noqa: E501
        elif self._arity == 1:
            assemble(
                rhs, form_compiler_parameters=self._form_compiler_parameters,
                tensor=x)
        else:
            raise ValueError("Must be an arity 0 or arity 1 form")

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        eq_deps = self.dependencies()
        if self._arity == 0:
            for dep_index, dep_B in dep_Bs.items():
                if dep_index <= 0 or dep_index >= len(eq_deps):
                    raise ValueError("Unexpected dep_index")
                dep = eq_deps[dep_index]

                for weight, comp in iter_expr(self._rhs):
                    if isinstance(comp, ufl.classes.Form):
                        dF = derivative(weight * comp, dep)
                        dF = ufl.algorithms.expand_derivatives(dF)
                        dF = eliminate_zeros(dF)
                        if not isinstance(dF, ufl.classes.ZeroBaseForm):
                            dF = ufl.classes.Form(
                                [integral.reconstruct(integrand=ufl.conj(integral.integrand()))  # noqa: E501
                                 for integral in dF.integrals()])
                            dF = self._nonlinear_replace(dF, nl_deps)
                            dF = assemble(
                                dF, form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
                            dep_B.sub((-var_scalar_value(adj_x), dF))
                    elif isinstance(comp, ufl.classes.Action):
                        if complex_mode:
                            # See Firedrake issue #3346
                            raise NotImplementedError("Complex case not "
                                                      "implemented")
                        dF = derivative(weight * comp, dep)
                        dF = ufl.algorithms.expand_derivatives(dF)
                        dF = eliminate_zeros(dF)
                        for dF_weight, dF_comp in iter_expr(dF, evaluate_weights=True):  # noqa: E501
                            dF_comp = self._nonlinear_replace(dF_comp, nl_deps)
                            dF_comp = var_new_conjugate_dual(dep).assign(dF_comp)  # noqa: E501
                            dep_B.sub((-var_scalar_value(adj_x) * dF_weight.conjugate(), dF_comp))  # noqa: E501
                    else:
                        raise TypeError(f"Unexpected type: {type(comp)}")
        elif self._arity == 1:
            for dep_index, dep_B in dep_Bs.items():
                if dep_index <= 0 or dep_index >= len(eq_deps):
                    raise ValueError("Unexpected dep_index")
                dep = eq_deps[dep_index]

                # Note: Ignores weight dependencies
                for weight, comp in iter_expr(self._rhs,
                                              evaluate_weights=True):
                    if isinstance(comp, ufl.classes.Form):
                        dF = derivative(comp, dep)
                        dF = ufl.algorithms.expand_derivatives(dF)
                        dF = eliminate_zeros(dF)
                        if not isinstance(dF, ufl.classes.ZeroBaseForm):
                            dF = adjoint(dF)
                            dF = ufl.action(dF, coefficient=adj_x)
                            dF = self._nonlinear_replace(dF, nl_deps)
                            dF = assemble(
                                dF, form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
                            dep_B.sub((-weight.conjugate(), dF))
                    elif isinstance(comp, ufl.classes.Cofunction):
                        dF = derivative(comp, dep)
                        if not isinstance(dF, ufl.classes.ZeroBaseForm):
                            dep_B.sub((-weight.conjugate(), adj_x))
                    else:
                        raise TypeError(f"Unexpected type: {type(comp)}")
        else:
            raise ValueError("Must be an arity 0 or arity 1 form")

    # def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    #     # Derived from EquationSolver.derivative_action (see dolfin-adjoint
    #     # reference below). Code first added 2017-12-07.
    #     # Re-written 2018-01-28
    #     # Updated to adjoint only form 2018-01-29

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(self._rhs)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    for weight, comp in iter_expr(self._rhs):
                        # Note: Ignores weight dependencies
                        tlm_rhs = (tlm_rhs
                                   + weight * derivative(comp, dep,
                                                         argument=tau_dep))

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.ZeroBaseForm):
            return ZeroAssignment(tlm_map[x])
        else:
            return Assembly(
                tlm_map[x], tlm_rhs,
                form_compiler_parameters=self._form_compiler_parameters)


def homogenized_bc(bc):
    if bcs_is_homogeneous(bc):
        return bc
    else:
        hbc = homogenize(bc)
        hbc._tlm_adjoint__static = bcs_is_static(bc)
        hbc._tlm_adjoint__cache = bcs_is_cached(bc)
        hbc._tlm_adjoint__homogeneous = True
        return hbc


class EquationSolver(ExprEquation):
    """Represents the solution of a finite element variational problem.

    Caching is based on the approach described in

        - J. R. Maddison and P. E. Farrell, 'Rapid development and adjoining of
          transient finite element models', Computer Methods in Applied
          Mechanics and Engineering, 276, 95--121, 2014, doi:
          10.1016/j.cma.2014.03.010

    The arguments `eq`, `x`, `bcs`, `J`, `form_compiler_parameters`, and
    `solver_parameters` are based on the interface for the DOLFIN
    `dolfin.solve` function (see e.g. FEniCS 2017.1.0).

    :arg eq: A :class:`ufl.equation.Equation` defining the finite element
        variational problem.
    :arg x: A :class:`firedrake.function.Function` defining the forward
        solution.
    :arg bcs: Dirichlet boundary conditions.
    :arg J: A :class:`ufl.Form` defining a Jacobian matrix approximation to use
        in a non-linear forward solve.
    :arg form_compiler_parameters: Form compiler parameters.
    :arg solver_parameters: Linear or non-linear solver parameters.
    :arg adjoint_solver_parameters: Linear solver parameters to use in an
        adjoint solve.
    :arg tlm_solver_parameters: Linear solver parameters to use when solving
        tangent-linear problems.
    :arg cache_jacobian: Whether to cache the forward Jacobian matrix and
        linear solver data. Defaults to
        `parameters['tlm_adjoint']['EquationSolver]['cache_jacobian']`. If
        `None` then caching is autodetected.
    :arg cache_adjoint_jacobian: Whether to cache the adjoint Jacobian matrix
        and linear solver data. Defaults to `cache_jacobian`.
    :arg cache_tlm_jacobian: Whether to cache the Jacobian matrix and linear
        solver data when solving tangent-linear problems. Defaults to
        `cache_jacobian`.
    :arg cache_rhs_assembly: Whether to enable right-hand-side caching. If
        enabled then right-hand-side terms are divided into terms which are
        cached, terms which are converted into matrix multiplication by a
        cached matrix, and terms which are not cached. Defaults to
        `parameters['tlm_adjoint']['EquationSolver']['cache_rhs_assembly']`.
    :arg match_quadrature: Whether to set quadrature parameters consistently in
        the forward, adjoint, and tangent-linears. Defaults to
        `parameters['tlm_adjoint']['EquationSolver']['match_quadrature']`.
    """

    def __init__(self, eq, x, bcs=None, *,
                 J=None, form_compiler_parameters=None, solver_parameters=None,
                 adjoint_solver_parameters=None, tlm_solver_parameters=None,
                 cache_jacobian=None, cache_adjoint_jacobian=None,
                 cache_tlm_jacobian=None, cache_rhs_assembly=None,
                 match_quadrature=None):
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

        check_space_type(x, "primal")

        lhs, rhs = eq.lhs, eq.rhs
        del eq
        linear = isinstance(rhs, ufl.classes.BaseForm)

        if linear:
            if len(lhs.arguments()) != 2:
                raise ValueError("Invalid left-hand-side arguments")
            if rhs.arguments() != (lhs.arguments()[0],):
                raise ValueError("Invalid right-hand-side arguments")
            if x in extract_coefficients(lhs) \
                    or x in extract_coefficients(rhs):
                raise ValueError("Invalid dependency")

            F = ufl.action(lhs, coefficient=x) - rhs
            nl_solve_J = None
            J = lhs
        else:
            if len(lhs.arguments()) != 1:
                raise ValueError("Invalid left-hand-side arguments")
            if not isinstance(rhs, int) or rhs != 0:
                raise ValueError("Invalid right-hand-side")

            F = lhs
            nl_solve_J = J
            J = derivative(F, x)
            J = ufl.algorithms.expand_derivatives(J)

        for weight, _ in iter_expr(F):
            if len(tuple(c for c in extract_coefficients(weight)
                         if is_var(c))) > 0:
                # See Firedrake issue #3292
                raise NotImplementedError("FormSum weights cannot depend on "
                                          "variables")

        deps, nl_deps = extract_dependencies(F)
        if nl_solve_J is not None:
            for dep in extract_coefficients(nl_solve_J):
                if is_var(dep):
                    deps.setdefault(var_id(dep), dep)

        deps = list(deps.values())
        if x in deps:
            deps.remove(x)
        deps.insert(0, x)
        nl_deps = tuple(nl_deps.values())

        hbcs = tuple(map(homogenized_bc, bcs))

        class DirichletBCLock:
            pass

        bc_lock = DirichletBCLock()
        for bc in itertools.chain(bcs, hbcs):
            bc_value = bc.function_arg
            if is_var(bc_value):
                var_increment_state_lock(bc_value, bc_lock)

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
        del form_compiler_parameters_
        if match_quadrature:
            update_parameters_dict(
                form_compiler_parameters,
                form_compiler_quadrature_parameters(F, form_compiler_parameters))  # noqa: E501

        super().__init__(x, deps, nl_deps=nl_deps,
                         ic=ic, adj_ic=adj_ic, adj_type="primal")
        self._F = F
        self._lhs = lhs
        self._rhs = rhs
        self._bcs = bcs
        self._hbcs = hbcs
        self._bc_lock = bc_lock
        self._J = J
        self._nl_solve_J = nl_solve_J
        self._form_compiler_parameters = form_compiler_parameters
        self._solver_parameters = solver_parameters
        self._linear_solver_parameters = linear_solver_parameters
        self._adjoint_solver_parameters = adjoint_solver_parameters
        self._tlm_solver_parameters = tlm_solver_parameters
        self._linear = linear

        self._cache_jacobian = cache_jacobian
        self._cache_adjoint_jacobian = cache_adjoint_jacobian
        self._cache_tlm_jacobian = cache_tlm_jacobian
        self._cache_rhs_assembly = cache_rhs_assembly

        self._forward_J_solver = CacheRef()
        self._forward_b_pa = None

        self._adjoint_dF_cache = {}
        self._adjoint_action_cache = {}

        self._adjoint_J_solver = CacheRef()

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._F = ufl.replace(self._F, replace_map)
        self._lhs = ufl.replace(self._lhs, replace_map)
        if isinstance(self._rhs, ufl.classes.BaseForm):
            self._rhs = ufl.replace(self._rhs, replace_map)
        self._J = ufl.replace(self._J, replace_map)
        if self._nl_solve_J is not None:
            self._nl_solve_J = ufl.replace(self._nl_solve_J, replace_map)

        if self._forward_b_pa is not None:
            cached_form, mat_forms, non_cached_form = self._forward_b_pa

            if cached_form is not None:
                cached_form[0] = ufl.replace(cached_form[0], replace_map)
            for dep_index, (mat_form, _) in mat_forms.items():
                mat_forms[dep_index][0] = ufl.replace(mat_form, replace_map)
            if non_cached_form is not None:
                non_cached_form = ufl.replace(non_cached_form, replace_map)

            self._forward_b_pa = (cached_form, mat_forms, non_cached_form)

        for dep_index, (dF_forms, dF_cofunctions) in self._adjoint_dF_cache.items():  # noqa: E501
            self._adjoint_dF_cache[dep_index] = \
                (ufl.replace(dF_forms, replace_map),
                 ufl.replace(dF_cofunctions, replace_map))

    def _cached_rhs(self, deps, *, b_bc=None):
        eq_deps = self.dependencies()

        if self._forward_b_pa is None:
            rhs = eliminate_zeros(self._rhs)
            cached_form, mat_forms_, non_cached_form = split_form(rhs)

            dep_indices = {var_id(dep): dep_index
                           for dep_index, dep in enumerate(eq_deps)}
            mat_forms = {dep_indices[dep_id]: [mat_forms_[dep_id], CacheRef()]
                         for dep_id in mat_forms_}
            del mat_forms_, dep_indices

            if isinstance(non_cached_form, ufl.classes.ZeroBaseForm):
                non_cached_form = None

            if cached_form.empty():
                cached_form = None
            else:
                cached_form = [cached_form, CacheRef()]

            self._forward_b_pa = (cached_form, mat_forms, non_cached_form)
        else:
            cached_form, mat_forms, non_cached_form = self._forward_b_pa

        b = None

        if non_cached_form is not None:
            b = assemble(
                self._replace(non_cached_form, deps),
                form_compiler_parameters=self._form_compiler_parameters)

        for dep_index, (mat_form, mat_cache) in mat_forms.items():
            var_update_caches(*eq_deps, value=deps)
            mat_bc = mat_cache()
            if mat_bc is None:
                mat_forms[dep_index][1], mat_bc = assembly_cache().assemble(
                    mat_form,
                    form_compiler_parameters=self._form_compiler_parameters,
                    linear_solver_parameters=self._linear_solver_parameters,
                    replace_map=self._replace_map(deps))
            mat, _ = mat_bc
            dep = (eq_deps if deps is None else deps)[dep_index]
            if b is None:
                b = matrix_multiply(mat, dep)
            else:
                matrix_multiply(mat, dep, tensor=b, addto=True)

        if cached_form is not None:
            var_update_caches(*eq_deps, value=deps)
            cached_b = cached_form[1]()
            if cached_b is None:
                cached_form[1], cached_b = assembly_cache().assemble(
                    cached_form[0],
                    form_compiler_parameters=self._form_compiler_parameters,
                    replace_map=self._replace_map(deps))
            if b is None:
                b = cached_b.copy(deepcopy=True)
            else:
                var_axpy(b, 1.0, cached_b)

        if b is None:
            b = var_new_conjugate_dual(self.x())

        apply_rhs_bcs(b, self._hbcs, b_bc=b_bc)
        return b

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()

        if self._linear:
            if self._cache_jacobian:
                # Cases 1 and 2: Linear, Jacobian cached, with or without RHS
                # assembly caching

                var_update_caches(*eq_deps, value=deps)
                J_solver_mat_bc = self._forward_J_solver()
                if J_solver_mat_bc is None:
                    # Assemble and cache the Jacobian, construct and cache the
                    # linear solver
                    self._forward_J_solver, J_solver_mat_bc = \
                        linear_solver_cache().linear_solver(
                            self._J, bcs=self._bcs,
                            form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                            linear_solver_parameters=self._linear_solver_parameters,  # noqa: E501
                            replace_map=self._replace_map(deps))
                J_solver, J_mat, b_bc = J_solver_mat_bc

                if self._cache_rhs_assembly:
                    # Assemble the RHS with RHS assembly caching
                    b = self._cached_rhs(deps, b_bc=b_bc)
                else:
                    # Assemble the RHS without RHS assembly caching
                    b = assemble(
                        self._replace(self._rhs, deps),
                        form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501

                    # Add bc RHS terms
                    apply_rhs_bcs(b, self._hbcs, b_bc=b_bc)
            else:
                if self._cache_rhs_assembly:
                    # Case 3: Linear, Jacobian not cached, with RHS assembly
                    # caching

                    # Construct the linear solver, assemble the Jacobian
                    J_solver, J_mat, b_bc = assemble_linear_solver(
                        self._replace(self._J, deps), bcs=self._bcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._linear_solver_parameters)  # noqa: E501

                    # Assemble the RHS with RHS assembly caching
                    b = self._cached_rhs(deps, b_bc=b_bc)
                else:
                    # Case 4: Linear, Jacobian not cached, without RHS assembly
                    # caching

                    # Construct the linear solver, assemble the Jacobian and
                    # RHS
                    J_solver, J_mat, b = assemble_linear_solver(
                        self._replace(self._J, deps),
                        b_form=self._replace(self._rhs, deps), bcs=self._bcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._linear_solver_parameters)  # noqa: E501

            J_tolerance = parameters["tlm_adjoint"]["assembly_verification"]["jacobian_tolerance"]  # noqa: E501
            b_tolerance = parameters["tlm_adjoint"]["assembly_verification"]["rhs_tolerance"]  # noqa: E501
            if not np.isposinf(J_tolerance) or not np.isposinf(b_tolerance):
                verify_assembly(
                    self._replace(self._J, deps),
                    self._replace(self._rhs, deps),
                    J_mat, b, self._bcs, self._form_compiler_parameters,
                    self._linear_solver_parameters, J_tolerance, b_tolerance)

            J_solver.solve(x, b)
        else:
            # Case 5: Non-linear
            lhs = self._lhs
            assert isinstance(self._rhs, int) and self._rhs == 0
            if self._nl_solve_J is None:
                J = self._J
            else:
                J = self._nl_solve_J
            solve(self._replace(lhs, deps) == 0, x, self._bcs,
                  J=self._replace(J, deps),
                  form_compiler_parameters=self._form_compiler_parameters,
                  solver_parameters=self._solver_parameters)

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        eq_nl_deps = self.nonlinear_dependencies()

        for dep_index, dep_B in dep_Bs.items():
            if dep_index not in self._adjoint_dF_cache:
                dep = self.dependencies()[dep_index]
                dF_forms = ufl.classes.Form([])
                dF_cofunctions = ufl.classes.ZeroBaseForm((TestFunction(var_space(self.x()).dual()),))  # noqa: E501
                for weight, comp in iter_expr(self._F):
                    if isinstance(comp, ufl.classes.Form):
                        dF_term = derivative(weight * comp, dep)
                        dF_term = ufl.algorithms.expand_derivatives(dF_term)
                        dF_term = eliminate_zeros(dF_term)
                        if not isinstance(dF_term, ufl.classes.ZeroBaseForm):
                            dF_forms = dF_forms + adjoint(dF_term)
                    elif isinstance(comp, ufl.classes.Cofunction):
                        # Note: Ignores weight dependencies
                        dF_term = ufl.conj(weight) * derivative(comp, dep)
                        if not isinstance(dF_term, ufl.classes.ZeroBaseForm):
                            dF_cofunctions = dF_cofunctions + dF_term
                    else:
                        raise TypeError(f"Unexpected type: {type(comp)}")
                self._adjoint_dF_cache[dep_index] = (dF_forms, dF_cofunctions)
            dF_forms, dF_cofunctions = self._adjoint_dF_cache[dep_index]

            if not dF_forms.empty():
                if dep_index not in self._adjoint_action_cache:
                    if self._cache_rhs_assembly \
                            and isinstance(adj_x, backend_Function) \
                            and is_cached(dF_forms):
                        self._adjoint_action_cache[dep_index] = CacheRef()
                    else:
                        self._adjoint_action_cache[dep_index] = None

                if self._adjoint_action_cache[dep_index] is not None:
                    # Cached matrix action
                    var_update_caches(*eq_nl_deps, value=nl_deps)
                    mat_bc = self._adjoint_action_cache[dep_index]()
                    if mat_bc is None:
                        self._adjoint_action_cache[dep_index], mat_bc = \
                            assembly_cache().assemble(
                                dF_forms,
                                form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                                replace_map=self._nonlinear_replace_map(nl_deps))  # noqa: E501
                    mat, _ = mat_bc
                    dep_B.sub(matrix_multiply(mat, adj_x))
                else:
                    # Cached form
                    dF_forms = ufl.action(self._nonlinear_replace(dF_forms, nl_deps),  # noqa: E501
                                          coefficient=adj_x)
                    dF_forms = assemble(dF_forms, form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
                    dep_B.sub(dF_forms)

            if not isinstance(dF_cofunctions, ufl.classes.ZeroBaseForm):
                for weight, _ in iter_expr(dF_cofunctions,
                                           evaluate_weights=True):
                    dep_B.sub((weight, adj_x))

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
        eq_nl_deps = self.nonlinear_dependencies()

        if self._cache_adjoint_jacobian:
            var_update_caches(*eq_nl_deps, value=nl_deps)
            J_solver_mat_bc = self._adjoint_J_solver()
            if J_solver_mat_bc is None:
                self._adjoint_J_solver, J_solver_mat_bc = \
                    linear_solver_cache().linear_solver(
                        adjoint(self._J), bcs=self._hbcs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        linear_solver_parameters=self._adjoint_solver_parameters,  # noqa: E501
                        replace_map=self._nonlinear_replace_map(nl_deps))
        else:
            J_solver_mat_bc = assemble_linear_solver(
                self._nonlinear_replace(adjoint(self._J), nl_deps),
                bcs=self._hbcs,
                form_compiler_parameters=self._form_compiler_parameters,
                linear_solver_parameters=self._adjoint_solver_parameters)
        J_solver, _, _ = J_solver_mat_bc

        apply_rhs_bcs(b, self._hbcs)
        J_solver.solve(adj_x, b)
        return adj_x

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(self._F)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    for weight, comp in iter_expr(self._F):
                        # Note: Ignores weight dependencies
                        tlm_rhs = (tlm_rhs
                                   - weight * derivative(comp, dep, argument=tau_dep))  # noqa: E501

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.ZeroBaseForm):
            return ZeroAssignment(tlm_map[x])
        else:
            if self._tlm_solver_parameters is None:
                tlm_solver_parameters = self._linear_solver_parameters
            else:
                tlm_solver_parameters = self._tlm_solver_parameters
            return EquationSolver(
                self._J == tlm_rhs, tlm_map[x], self._hbcs,
                form_compiler_parameters=self._form_compiler_parameters,
                solver_parameters=tlm_solver_parameters,
                adjoint_solver_parameters=self._adjoint_solver_parameters,
                tlm_solver_parameters=tlm_solver_parameters,
                cache_jacobian=self._cache_tlm_jacobian,
                cache_adjoint_jacobian=self._cache_adjoint_jacobian,
                cache_tlm_jacobian=self._cache_tlm_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly)


def expr_new_x(expr, x, *,
               annotate=None, tlm=None):
    """If an expression depends on `x`, then record the assignment `x_old =
    x`, and replace `x` with `x_old` in the expression.

    :arg expr: A :class:`ufl.core.expr.Expr`.
    :arg x: Defines `x`.
    :arg annotate: Whether the :class:`.EquationManager` should record the
        solution of equations.
    :arg tlm: Whether tangent-linear equations should be solved.
    :returns: A :class:`ufl.core.expr.Expr` with `x` replaced with `x_old`, or
        `expr` if the expression does not depend on `x`.
    """

    if x in extract_coefficients(expr):
        x_old = var_new(x)
        Assignment(x_old, x).solve(annotate=annotate, tlm=tlm)
        return ufl.replace(expr, {x: x_old})
    else:
        return expr


def linear_equation_new_x(eq, x, *,
                          annotate=None, tlm=None):
    """If a symbolic expression for a linear finite element variational problem
    depends on the symbolic variable representing the problem solution `x`,
    then record the assignment `x_old = x`, and replace `x` with `x_old` in the
    symbolic expression.

    :arg eq: A :class:`ufl.equation.Equation` defining the finite element
        variational problem.
    :arg x: A :class:`firedrake.function.Function` defining the solution to the
        finite element variational problem.
    :arg annotate: Whether the :class:`.EquationManager` should record the
        solution of equations.
    :arg tlm: Whether tangent-linear equations should be solved.
    :returns: A :class:`ufl.equation.Equation` with `x` replaced with `x_old`,
        or `eq` if the symbolic expression does not depend on `x`.
    """

    lhs, rhs = eq.lhs, eq.rhs
    lhs_x_dep = x in extract_coefficients(lhs)
    rhs_x_dep = x in extract_coefficients(rhs)
    if lhs_x_dep or rhs_x_dep:
        x_old = var_new(x)
        Assignment(x_old, x).solve(annotate=annotate, tlm=tlm)
        if lhs_x_dep:
            lhs = ufl.replace(lhs, {x: x_old})
        if rhs_x_dep:
            rhs = ufl.replace(rhs, {x: x_old})
        return lhs == rhs
    else:
        return eq


class Projection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`.

    :arg x: A :class:`firedrake.function.Function` defining the forward
        solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`, or a :class:`ufl.form.BaseForm` defining the
        right-hand-side of the finite element variational problem. Should not
        depend on `x`.

    Remaining arguments are passed to the :class:`.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *args, **kwargs):
        space = var_space(x)
        test, trial = TestFunction(space), TrialFunction(space)
        if not isinstance(rhs, ufl.classes.BaseForm):
            rhs = ufl.inner(rhs, test) * ufl.dx
        super().__init__(ufl.inner(trial, test) * ufl.dx == rhs, x,
                         *args, **kwargs)


class DirichletBCApplication(Equation):
    r"""Represents the application of a Dirichlet boundary condition to a zero
    valued :class:`firedrake.function.Function`. Specifically this represents:

    .. code-block:: python

        x.zero()
        DirichletBC(x.function_space(), y, *args, **kwargs).apply(x)

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: A :class:`firedrake.function.Function`, updated by the above
        operations.
    :arg y: A :class:`firedrake.function.Function`, defines the Dirichet
        boundary condition.

    Remaining arguments are passed to the :class:`firedrake.bcs.DirichletBC`
    constructor.
    """

    def __init__(self, x, y, *args, **kwargs):
        check_space_type(x, "primal")
        check_space_type(y, "primal")

        super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)
        self._bc_args = args
        self._bc_kwargs = kwargs

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        var_zero(x)
        backend_DirichletBC(
            var_space(x), y,
            *self._bc_args, **self._bc_kwargs).apply(x)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise ValueError("Unexpected dep_index")

        _, y = self.dependencies()
        F = var_new_conjugate_dual(y)
        backend_DirichletBC(
            var_space(y), adj_x,
            *self._bc_args, **self._bc_kwargs).apply(F)
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x, y = self.dependencies()

        tau_y = tlm_map[y]
        if tau_y is None:
            return ZeroAssignment(tlm_map[x])
        else:
            return DirichletBCApplication(
                tlm_map[x], tau_y,
                *self._bc_args, **self._bc_kwargs)


class ExprInterpolation(ExprEquation):
    r"""Represents interpolation of `rhs` onto the space for `x`.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: The forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to
        interpolate onto the space for `x`. Should not depend on `x`.
    """

    def __init__(self, x, rhs):
        deps, nl_deps = extract_dependencies(rhs, space_type="primal")
        if var_id(x) in deps:
            raise ValueError("Invalid dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()
        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        interpolate_expression(x, self._replace(self._rhs, deps))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        eq_deps = self.dependencies()
        if dep_index <= 0 or dep_index >= len(eq_deps):
            raise ValueError("Unexpected dep_index")

        dep = eq_deps[dep_index]

        if isinstance(dep, (backend_Constant, ReplacementConstant)):
            if len(dep.ufl_shape) > 0:
                raise NotImplementedError("Case not implemented")
            dF = derivative(self._rhs, dep, argument=ufl.classes.IntValue(1))
        else:
            dF = derivative(self._rhs, dep)
        dF = ufl.algorithms.expand_derivatives(dF)
        dF = eliminate_zeros(dF)
        dF = self._nonlinear_replace(dF, nl_deps)

        F = var_new_conjugate_dual(dep)
        interpolate_expression(F, dF, adj_x=adj_x)
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(x)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    # Cannot use += as Firedrake might add to the *values* for
                    # tlm_rhs
                    tlm_rhs = (tlm_rhs
                               + derivative(self._rhs, dep, argument=tau_dep))

        if isinstance(tlm_rhs, ufl.classes.Zero):
            return ZeroAssignment(tlm_map[x])
        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.Zero):
            return ZeroAssignment(tlm_map[x])
        else:
            return ExprInterpolation(tlm_map[x], tlm_rhs)
