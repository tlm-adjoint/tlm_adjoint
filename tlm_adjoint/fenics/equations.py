"""This module implements finite element calculations. In particular the
:class:`.EquationSolver` class implements the solution of finite element
variational problems.
"""

from .backend import (
    TestFunction, TrialFunction, adjoint, backend_DirichletBC,
    has_lu_solver_method, parameters)
from ..interface import (
    check_space_type, is_var, var_assign, var_axpy, var_copy, var_id,
    var_is_scalar, var_new_conjugate_dual, var_replacement, var_scalar_value,
    var_space, var_update_caches, var_zero)
from .backend_code_generator_interface import (
    assemble, assemble_linear_solver, copy_parameters_dict,
    form_compiler_quadrature_parameters, matrix_multiply, solve,
    update_parameters_dict)

from ..caches import CacheRef
from ..equation import Equation, ZeroAssignment

from .caches import assembly_cache, is_cached, linear_solver_cache, split_form
from .functions import (
    derivative, eliminate_zeros, expr_zero, extract_coefficients)

from collections import defaultdict
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "Assembly",
        "DirichletBCApplication",
        "EquationSolver",
        "Projection"
    ]


def process_form_compiler_parameters(form_compiler_parameters):
    params = copy_parameters_dict(parameters["form_compiler"])
    update_parameters_dict(params, form_compiler_parameters)
    return params


def process_solver_parameters(solver_parameters, *, linear):
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
            raise ValueError(f"Unexpected non-linear solver: {nl_solver}")

    linear_solver = linear_solver_parameters.setdefault("linear_solver", "default")  # noqa: E501
    if linear_solver in {"default", "direct", "lu"} \
            or has_lu_solver_method(linear_solver):
        linear_solver_parameters.setdefault("lu_solver", {})
        linear_solver_ic = False
    else:
        ks_parameters = linear_solver_parameters.setdefault("krylov_solver", {})  # noqa: E501
        linear_solver_ic = ks_parameters.setdefault("nonzero_initial_guess", False)  # noqa: E501
    ic = not linear or linear_solver_ic

    return (solver_parameters, linear_solver_parameters, ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    return copy_parameters_dict(linear_solver_parameters)


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


class ExprEquation(Equation):
    def _replace_map(self, deps):
        if deps is None:
            return None
        else:
            eq_deps = self.dependencies()
            assert len(eq_deps) == len(deps)
            return {eq_dep: dep
                    for eq_dep, dep in zip(eq_deps, deps)
                    if isinstance(eq_dep, ufl.classes.Expr)}

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
                if isinstance(eq_nl_dep, ufl.classes.Expr)}

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
    :arg rhs: A :class:`ufl.Form` to assemble. Should have arity 0 or 1, and
        should not depend on `x`.
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

        deps, nl_deps = extract_dependencies(rhs, space_type="primal")
        if var_id(x) in deps:
            raise ValueError("Invalid dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        form_compiler_parameters = \
            process_form_compiler_parameters(form_compiler_parameters)
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
                       if isinstance(dep, ufl.classes.Expr)}

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

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        # Derived from EquationSolver.derivative_action (see dolfin-adjoint
        # reference below). Code first added 2017-12-07.
        # Re-written 2018-01-28
        # Updated to adjoint only form 2018-01-29

        eq_deps = self.dependencies()
        if dep_index <= 0 or dep_index >= len(eq_deps):
            raise ValueError("Unexpected dep_index")

        dep = eq_deps[dep_index]
        dF = derivative(self._rhs, dep)
        dF = ufl.algorithms.expand_derivatives(dF)
        dF = eliminate_zeros(dF)
        if dF.empty():
            return None

        dF = self._nonlinear_replace(dF, nl_deps)
        if self._arity == 0:
            dF = ufl.classes.Form(
                [integral.reconstruct(integrand=ufl.conj(integral.integrand()))
                 for integral in dF.integrals()])  # dF = adjoint(dF)
            dF = assemble(
                dF, form_compiler_parameters=self._form_compiler_parameters)
            return (-var_scalar_value(adj_x), dF)
        elif self._arity == 1:
            dF = ufl.action(adjoint(dF), coefficient=adj_x)
            dF = assemble(
                dF, form_compiler_parameters=self._form_compiler_parameters)
            return (-1.0, dF)
        else:
            raise ValueError("Must be an arity 0 or arity 1 form")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(self._rhs)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = tlm_rhs + derivative(self._rhs, dep, argument=tau_dep)  # noqa: E501

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if tlm_rhs.empty():
            return ZeroAssignment(tlm_map[x])
        else:
            return Assembly(
                tlm_map[x], tlm_rhs,
                form_compiler_parameters=self._form_compiler_parameters)


def homogenized_bc(bc):
    if hasattr(bc, "_tlm_adjoint__hbc"):
        hbc = bc._tlm_adjoint__hbc
    else:
        hbc = backend_DirichletBC(bc)
        hbc.homogenize()
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
    :arg x: A DOLFIN `Function` defining the forward solution.
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
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        else:
            bcs = tuple(bcs)
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if solver_parameters is None:
            solver_parameters = {}

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
        linear = isinstance(rhs, ufl.classes.Form)

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

        deps, nl_deps = extract_dependencies(F, space_type="primal")
        if nl_solve_J is not None:
            for dep in extract_coefficients(nl_solve_J):
                if is_var(dep):
                    deps.setdefault(var_id(dep), dep)
        deps = list(deps.values())
        if x in deps:
            deps.remove(x)
        deps.insert(0, x)
        nl_deps = tuple(nl_deps.values())

        if len(bcs) == 0:
            rhs_bc = expr_zero(F)
        else:
            rhs_bc = -ufl.action(J, coefficient=x)
        bcs = tuple(map(backend_DirichletBC, bcs))
        hbcs = tuple(map(homogenized_bc, bcs))

        if cache_adjoint_jacobian is None:
            cache_adjoint_jacobian = cache_jacobian if cache_jacobian is not None else is_cached(J)  # noqa: E501
        if cache_tlm_jacobian is None:
            cache_tlm_jacobian = cache_jacobian
        if cache_jacobian is None:
            cache_jacobian = is_cached(J) and all(getattr(bc, "_tlm_adjoint__cache", False) for bc in bcs)  # noqa: E501

        (solver_parameters, linear_solver_parameters,
         ic, J_ic) = process_solver_parameters(solver_parameters, linear=linear)  # noqa: E501
        if adjoint_solver_parameters is None:
            adjoint_solver_parameters = process_adjoint_solver_parameters(linear_solver_parameters)  # noqa: E501
            adj_ic = J_ic
        else:
            (_, adjoint_solver_parameters,
             adj_ic, _) = process_solver_parameters(adjoint_solver_parameters, linear=True)  # noqa: E501
        if tlm_solver_parameters is None:
            tlm_solver_parameters = linear_solver_parameters
        else:
            (_, tlm_solver_parameters,
             _, _) = process_solver_parameters(tlm_solver_parameters, linear=True)  # noqa: E501

        form_compiler_parameters = \
            process_form_compiler_parameters(form_compiler_parameters)
        if match_quadrature:
            update_parameters_dict(
                form_compiler_parameters,
                form_compiler_quadrature_parameters(F, form_compiler_parameters))  # noqa: E501

        super().__init__(x, deps, nl_deps=nl_deps,
                         ic=ic, adj_ic=adj_ic, adj_type="primal")
        self._linear = linear
        self._F = F
        self._lhs = lhs
        self._rhs = rhs
        self._J = J
        self._nl_solve_J = nl_solve_J
        self._bcs = bcs
        self._hbcs = hbcs
        self._rhs_bc = rhs_bc
        self._form_compiler_parameters = form_compiler_parameters
        self._solver_parameters = solver_parameters
        self._linear_solver_parameters = linear_solver_parameters
        self._adjoint_solver_parameters = adjoint_solver_parameters
        self._tlm_solver_parameters = tlm_solver_parameters

        self._cache_jacobian = cache_jacobian
        self._cache_adjoint_jacobian = cache_adjoint_jacobian
        self._cache_tlm_jacobian = cache_tlm_jacobian
        self._cache_rhs_assembly = cache_rhs_assembly

        self._forward_b_cache = None
        self._adjoint_b_cache = {}
        self._assembly_cache = defaultdict(CacheRef)
        self._solver_cache = defaultdict(CacheRef)

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._F = ufl.replace(self._F, replace_map)
        if isinstance(self._rhs, ufl.classes.Form):
            self._rhs = ufl.replace(self._rhs, replace_map)
        self._rhs_bc = ufl.replace(self._rhs_bc, replace_map)
        self._J = ufl.replace(self._J, replace_map)
        if self._nl_solve_J is not None:
            self._nl_solve_J = ufl.replace(self._nl_solve_J, replace_map)
        self._lhs = self._J if self._linear else self._F

        if self._forward_b_cache is not None:
            cached_form, mat_forms, non_cached_form = self._forward_b_cache
            cached_form = ufl.replace(cached_form, replace_map)
            mat_forms = {dep_id: ufl.replace(mat_form, replace_map)
                         for dep_id, mat_form in mat_forms.items()}
            non_cached_form = ufl.replace(non_cached_form, replace_map)
            self._forward_b_cache = (cached_form, mat_forms, non_cached_form)

        for dep_index, dF in self._adjoint_b_cache.items():
            self._adjoint_b_cache[dep_index] = ufl.replace(dF, replace_map)

    def _assemble(self, form, bcs=None, *,
                  eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            form = ufl.replace(form, dict(zip(eq_deps, deps)))

        return assemble(
            form, bcs=bcs,
            form_compiler_parameters=self._form_compiler_parameters)

    def _assemble_cached(self, key, form, bcs=None, *,
                         eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            replace_map = dict(zip(eq_deps, deps))
        else:
            replace_map = None

        var_update_caches(*eq_deps, value=deps)
        value = self._assembly_cache[key]()
        if value is None:
            self._assembly_cache[key], value = assembly_cache().assemble(
                form, bcs=bcs,
                form_compiler_parameters=self._form_compiler_parameters,
                linear_solver_parameters=self._linear_solver_parameters,
                replace_map=replace_map)

        return value

    def _assemble_rhs(self, x, bcs, *, deps):
        eq_deps = self.dependencies()
        dep_indices = {var_id(dep): dep_index
                       for dep_index, dep in enumerate(eq_deps)}

        if self._rhs_bc.empty():
            x_0 = x
        else:
            x_0 = var_copy(x)
            var_zero(x)
            for bc in bcs:
                bc.apply(x)
        for bc in self._hbcs:
            bc.apply(x_0)

        if self._forward_b_cache is None:
            rhs = self._rhs + self._rhs_bc
            rhs = eliminate_zeros(rhs)
            if self._cache_rhs_assembly:
                self._forward_b_cache = split_form(rhs)
            else:
                self._forward_b_cache = (ufl.classes.Form([]), {}, rhs)
        cached_form, mat_forms, non_cached_form = self._forward_b_cache

        # Non-cached
        if non_cached_form.empty():
            b = var_new_conjugate_dual(self.x()).vector()
        else:
            b = self._assemble(non_cached_form,
                               eq_deps=eq_deps, deps=deps)

        # Cached matrix action
        for dep_id, mat_form in mat_forms.items():
            dep_index = dep_indices[dep_id]
            mat, _ = self._assemble_cached(
                ("cached_rhs_mat", dep_index), mat_form,
                eq_deps=eq_deps, deps=deps)
            dep = (eq_deps if deps is None else deps)[dep_index]
            matrix_multiply(mat, dep, tensor=b, addto=True)

        # Cached
        if not cached_form.empty():
            cached_b = self._assemble_cached(
                "cached_rhs_b", cached_form,
                eq_deps=eq_deps, deps=deps)
            b.axpy(1.0, cached_b)

        # Boundary conditions
        for bc in self._hbcs:
            bc.apply(b)

        return x_0, b

    def _linear_solver(self, *args, cache_key=None, **kwargs):
        if cache_key is None:
            return self._assemble_linear_solver(*args, **kwargs)
        else:
            return self._assemble_linear_solver_cached(
                cache_key, *args, **kwargs)

    def _assemble_linear_solver(self, form, bcs, *,
                                linear_solver_parameters,
                                eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            form = ufl.replace(form, dict(zip(eq_deps, deps)))

        return assemble_linear_solver(
            form, bcs=bcs,
            form_compiler_parameters=self._form_compiler_parameters,
            linear_solver_parameters=linear_solver_parameters)

    def _assemble_linear_solver_cached(self, key, form, bcs, *,
                                       linear_solver_parameters,
                                       eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            replace_map = dict(zip(eq_deps, deps))
        else:
            replace_map = None

        var_update_caches(*eq_deps, value=deps)
        value = self._solver_cache[key]()
        if value is None:
            self._solver_cache[key], value = \
                linear_solver_cache().linear_solver(
                    form, bcs=bcs,
                    form_compiler_parameters=self._form_compiler_parameters,
                    linear_solver_parameters=linear_solver_parameters,
                    replace_map=replace_map)
        return value

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()

        if self._linear:
            x_0, b = self._assemble_rhs(x, self._bcs, deps=deps)

            J_solver, _, _ = self._linear_solver(
                self._J, bcs=self._hbcs,
                linear_solver_parameters=self._linear_solver_parameters,
                eq_deps=eq_deps, deps=deps,
                cache_key="J_solver" if self._cache_jacobian else None)

            J_solver.solve(x_0, b)
            if x_0 is not x:
                var_axpy(x, 1.0, x_0)
        else:
            if self._nl_solve_J is None:
                J = self._J
            else:
                J = self._nl_solve_J
            solve(self._replace(self._F, deps) == 0, x, self._bcs,
                  J=self._replace(J, deps),
                  form_compiler_parameters=self._form_compiler_parameters,
                  solver_parameters=self._solver_parameters)

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        eq_nl_deps = self.nonlinear_dependencies()

        for dep_index, dep_B in dep_Bs.items():
            if dep_index not in self._adjoint_b_cache:
                dep = self.dependencies()[dep_index]
                dF = derivative(self._F, dep)
                dF = ufl.algorithms.expand_derivatives(dF)
                dF = eliminate_zeros(dF)
                if not dF.empty():
                    dF = adjoint(dF)
                self._adjoint_b_cache[dep_index] = dF
            dF = self._adjoint_b_cache[dep_index]

            if not dF.empty():
                if self._cache_rhs_assembly and is_cached(dF):
                    mat, _ = self._assemble_cached(
                        ("cached_adjoint_rhs_mat", dep_index), dF,
                        eq_deps=eq_nl_deps, deps=nl_deps)
                    dep_B.sub(matrix_multiply(mat, adj_x))
                else:
                    # Cached form
                    dF = ufl.action(dF, coefficient=adj_x)
                    dF = self._assemble(dF,
                                        eq_deps=eq_nl_deps, deps=nl_deps)
                    dep_B.sub(dF)

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

        J_solver, _, _ = self._linear_solver(
            adjoint(self._J), bcs=self._hbcs,
            linear_solver_parameters=self._adjoint_solver_parameters,
            eq_deps=eq_nl_deps, deps=nl_deps,
            cache_key="adjoint_J_solver" if self._cache_adjoint_jacobian else None)  # noqa: E501

        for bc in self._hbcs:
            bc.apply(adj_x)
            bc.apply(b)
        J_solver.solve(adj_x, b)
        return adj_x

    def _tangent_linear_rhs(self, tlm_map):
        x = self.x()
        tlm_rhs = expr_zero(self._F)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = (tlm_rhs
                               - derivative(self._F, dep, argument=tau_dep))
        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        return tlm_rhs

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()
        tlm_rhs = self._tangent_linear_rhs(tlm_map)
        tlm_bcs = self._hbcs
        if tlm_rhs.empty():
            return ZeroAssignment(tlm_map[x])
        else:
            return EquationSolver(
                self._J == tlm_rhs, tlm_map[x], tlm_bcs,
                form_compiler_parameters=self._form_compiler_parameters,
                solver_parameters=self._tlm_solver_parameters,
                adjoint_solver_parameters=self._adjoint_solver_parameters,
                tlm_solver_parameters=self._tlm_solver_parameters,
                cache_jacobian=self._cache_tlm_jacobian,
                cache_adjoint_jacobian=self._cache_adjoint_jacobian,
                cache_tlm_jacobian=self._cache_tlm_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly)


class Projection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`.

    :arg x: A DOLFIN `Function` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`, or a :class:`ufl.Form` defining the
        right-hand-side of the finite element variational problem. Should not
        depend on `x`.

    Remaining arguments are passed to the :class:`.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *args, **kwargs):
        space = var_space(x)
        test, trial = TestFunction(space), TrialFunction(space)
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(rhs, test) * ufl.dx
        super().__init__(ufl.inner(trial, test) * ufl.dx == rhs, x,
                         *args, **kwargs)


class DirichletBCApplication(Equation):
    r"""Represents the application of a Dirichlet boundary condition to a zero
    valued DOLFIN `Function`. Specifically this represents:

    .. code-block:: python

        x.vector().zero()
        DirichletBC(x.function_space(), y,
                    *args, **kwargs).apply(x.vector())

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: A DOLFIN `Function`, updated by the above operations.
    :arg y: A DOLFIN `Function`, defines the Dirichet boundary condition.

    Remaining arguments are passed to the DOLFIN `DirichletBC` constructor.
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
