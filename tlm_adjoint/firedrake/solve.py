"""Finite element variational problem solution operations with Firedrake.
"""

from .backend import adjoint, backend_Matrix, parameters
from ..interface import (
    check_space_type, is_var, packed, var_axpy, var_copy, var_id, var_new,
    var_new_conjugate_dual, var_replacement, var_update_caches, var_zero)

from ..caches import CacheRef
from ..equation import ZeroAssignment

from .backend_interface import (
    LocalSolver, assemble, assemble_linear_solver, matrix_multiply, solve)
from .caches import (
    assembly_cache, is_cached, linear_solver_cache, local_solver_cache,
    split_form)
from .expr import (
    ExprEquation, action, derivative, eliminate_zeros, expr_zero,
    extract_dependencies, extract_variables, iter_expr)
from .parameters import (
    form_compiler_quadrature_parameters, process_adjoint_solver_parameters,
    process_form_compiler_parameters, process_solver_parameters,
    update_parameters)

from collections import defaultdict
import ufl

__all__ = \
    [
        "EquationSolver",
        "LocalEquationSolver"
    ]


def homogenized_bc(bc):
    return bc._tlm_adjoint__hbc


class BCIndex(int):
    pass


def unpack_bcs(bcs, *, deps=None):
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if deps is None:
        deps = []
    dep_ids = set(map(var_id, deps))

    bc_deps = {}
    bc_map = {}
    bc_gs = []
    for i, bc in enumerate(bcs):
        g = bc._function_arg
        for g_previous, subset in bc_deps.items():
            bc_deps[g_previous] = subset.difference(bc.node_set)

        if is_var(g):
            if var_id(g) in dep_ids:
                raise ValueError("Invalid dependency")

            if g in bc_deps:
                bc_deps[g] = bc_deps[g].union(bc.node_set)
            else:
                bc_deps[g] = bc.node_set
                bc_map[g] = len(bc_map)

            bc_gs.append(BCIndex(bc_map[g]))
        elif isinstance(g, (ufl.classes.Zero,
                            ufl.classes.ScalarValue)):
            bc_gs.append(g)
        else:
            raise TypeError(f"Unexpected type: {type(g)}")

    n_deps = len(deps)
    deps = tuple(list(deps) + list(bc_deps.keys()))
    bc_nodes = {i + n_deps: subset
                for i, subset in enumerate(bc_deps.values())}
    bc_gs = tuple(BCIndex(g + n_deps) if isinstance(g, BCIndex) else g
                  for g in bc_gs)

    return deps, bc_nodes, bc_gs


_parameters = parameters.setdefault("tlm_adjoint", {})
_parameters.setdefault("EquationSolver", {})
_parameters["EquationSolver"].setdefault("enable_jacobian_caching", True)
_parameters["EquationSolver"].setdefault("cache_rhs_assembly", True)
_parameters["EquationSolver"].setdefault("match_quadrature", False)
del _parameters


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
            bcs = ()
        else:
            bcs = packed(bcs)
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
        linear = isinstance(rhs, ufl.classes.BaseForm)

        if linear:
            if len(lhs.arguments()) != 2:
                raise ValueError("Invalid left-hand-side arguments")
            if rhs.arguments() != (lhs.arguments()[0],):
                raise ValueError("Invalid right-hand-side arguments")
            if x in extract_variables(lhs) \
                    or x in extract_variables(rhs):
                raise ValueError("Invalid dependency")

            if isinstance(lhs, backend_Matrix):
                lhs = lhs.a
            F = action(lhs, x) - rhs
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

        for weight, _ in iter_expr(F):
            if len(extract_variables(weight)) > 0:
                # See Firedrake issue #3292
                raise NotImplementedError("FormSum weights cannot depend on "
                                          "variables")

        deps, nl_deps = extract_dependencies(F)
        if nl_solve_J is not None:
            for dep in extract_variables(nl_solve_J):
                deps.setdefault(var_id(dep), dep)
        deps = sorted(deps.values(), key=var_id)
        if x in deps:
            deps.remove(x)
        deps.insert(0, x)
        nl_deps = tuple(nl_deps.values())

        if all(isinstance(bc._function_arg, ufl.classes.Zero) for bc in bcs):
            rhs_bc = expr_zero(F)
        else:
            rhs_bc = -action(J, x)
        deps, bc_nodes, bc_gs = unpack_bcs(bcs, deps=deps)
        hbcs = tuple(map(homogenized_bc, bcs))

        if cache_adjoint_jacobian is None:
            cache_adjoint_jacobian = cache_jacobian if cache_jacobian is not None else is_cached(J)  # noqa: E501
        if cache_tlm_jacobian is None:
            cache_tlm_jacobian = cache_jacobian
        if cache_jacobian is None:
            cache_jacobian = is_cached(J) and all(is_cached(bc._function_arg) for bc in bcs)  # noqa: E501

        (solver_parameters, linear_solver_parameters,
         ic, J_ic) = process_solver_parameters(solver_parameters, linear=linear)  # noqa: E501
        if adjoint_solver_parameters is None:
            adjoint_solver_parameters = process_adjoint_solver_parameters(linear_solver_parameters)  # noqa: E501
            adj_ic = J_ic
        else:
            (_, adjoint_solver_parameters,
             _, adj_ic) = process_solver_parameters(adjoint_solver_parameters, linear=True)  # noqa: E501
        if tlm_solver_parameters is None:
            tlm_solver_parameters = linear_solver_parameters
        else:
            (_, tlm_solver_parameters,
             _, _) = process_solver_parameters(tlm_solver_parameters, linear=True)  # noqa: E501

        form_compiler_parameters = \
            process_form_compiler_parameters(form_compiler_parameters)
        if match_quadrature:
            update_parameters(
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
        self._bc_gs = bc_gs
        self._bc_nodes = bc_nodes
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
        if isinstance(self._rhs, ufl.classes.BaseForm):
            self._rhs = ufl.replace(self._rhs, replace_map)
        self._rhs_bc = ufl.replace(self._rhs_bc, replace_map)
        self._J = ufl.replace(self._J, replace_map)
        if self._nl_solve_J is not None:
            self._nl_solve_J = ufl.replace(self._nl_solve_J, replace_map)
        self._lhs = self._J if self._linear else self._F

        # Used only to update bc values
        del self._bcs

        if self._forward_b_cache is not None:
            cached_form, mat_forms, non_cached_form = self._forward_b_cache
            cached_form = ufl.replace(cached_form, replace_map)
            mat_forms = {dep_id: ufl.replace(mat_form, replace_map)
                         for dep_id, mat_form in mat_forms.items()}
            non_cached_form = ufl.replace(non_cached_form, replace_map)
            self._forward_b_cache = (cached_form, mat_forms, non_cached_form)

        for dep_index, (dF_forms, dF_cofunctions) in self._adjoint_b_cache.items():  # noqa: E501
            self._adjoint_b_cache[dep_index] = \
                (ufl.replace(dF_forms, replace_map),
                 ufl.replace(dF_cofunctions, replace_map))

    @property
    def _pre_process_required(self):
        return super()._pre_process_required or len(self._hbcs) > 0

    def _pre_process(self):
        for bc in self._bcs:
            # Apply any boundary condition updates
            bc.function_arg
        super()._pre_process()

    def _assemble(self, form, bcs=None, *,
                  eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            form = ufl.replace(form, dict(zip(eq_deps, deps)))

        return assemble(
            form, bcs=bcs,
            form_compiler_parameters=self._form_compiler_parameters,
            mat_type=self._linear_solver_parameters.get("mat_type", None))

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

        if isinstance(self._rhs_bc, ufl.classes.ZeroBaseForm):
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
        if isinstance(non_cached_form, ufl.classes.ZeroBaseForm):
            b = var_new_conjugate_dual(self.x())
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
            var_axpy(b, 1.0, cached_b)

        # Boundary conditions
        for bc in self._hbcs:
            bc.apply(b.riesz_representation("l2"))

        return x_0, b

    def _reconstruct_bcs(self, deps=None, *, tlm_map=None):
        if deps is None:
            deps = self.dependencies()

        bcs = []
        assert len(self._hbcs) == len(self._bc_gs)
        for hbc, g in zip(self._hbcs, self._bc_gs):
            if isinstance(g, BCIndex):
                g = deps[g]
                if tlm_map is not None:
                    g = tlm_map[g]
            elif isinstance(g, ufl.classes.Zero) or tlm_map is not None:
                g = None
            bcs.append(hbc.reconstruct(g=g))

        return tuple(bcs)

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

        solver, _, b_bc = assemble_linear_solver(
            form, bcs=bcs,
            form_compiler_parameters=self._form_compiler_parameters,
            linear_solver_parameters=linear_solver_parameters)
        return solver, b_bc

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
        solver, _, b_bc = value
        return solver, b_bc

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()
        bcs = self._reconstruct_bcs(deps=deps)

        if self._solver_parameters.get("mat_type", "aij") != "matfree" \
                and self._linear:
            x_0, b = self._assemble_rhs(x, bcs, deps=deps)

            J_solver, _ = self._linear_solver(
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
            solve(self._replace(self._F, deps) == 0, x, bcs,
                  J=self._replace(J, deps),
                  form_compiler_parameters=self._form_compiler_parameters,
                  solver_parameters=self._solver_parameters)

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        eq_nl_deps = self.nonlinear_dependencies()

        if len(self._hbcs) == 0:
            adj_x_0 = adj_x
        else:
            adj_x_0 = var_copy(adj_x)
            for bc in self._hbcs:
                bc.apply(adj_x_0)
        dF_bc = None

        for dep_index, dep_B in dep_Bs.items():
            if dep_index not in self._adjoint_b_cache:
                dep = self.dependencies()[dep_index]
                dF_forms = ufl.classes.Form([])
                dF_cofunctions = expr_zero(self._F)
                for weight, comp in iter_expr(self._F):
                    if isinstance(comp, ufl.classes.Form):
                        dF_term = derivative(weight * comp, dep)
                        dF_term = eliminate_zeros(dF_term)
                        if not isinstance(dF_term, ufl.classes.ZeroBaseForm):
                            dF_forms = dF_forms + adjoint(dF_term)
                    elif isinstance(comp, ufl.classes.Cofunction):
                        # Note: Ignores weight dependencies
                        dF_term = ufl.conj(weight) * derivative(comp, dep)
                        dF_term = eliminate_zeros(dF_term)
                        if not isinstance(dF_term, ufl.classes.ZeroBaseForm):
                            dF_cofunctions = dF_cofunctions + dF_term
                    else:
                        raise TypeError(f"Unexpected type: {type(comp)}")
                self._adjoint_b_cache[dep_index] = (dF_forms, dF_cofunctions)
            dF_forms, dF_cofunctions = self._adjoint_b_cache[dep_index]

            # Forms
            if not dF_forms.empty():
                if self._cache_rhs_assembly and is_cached(dF_forms):
                    mat, _ = self._assemble_cached(
                        ("cached_adjoint_rhs_mat", dep_index), dF_forms,
                        eq_deps=eq_nl_deps, deps=nl_deps)
                    dep_B.sub(matrix_multiply(mat, adj_x_0))
                else:
                    dF_forms = action(dF_forms, adj_x_0)
                    dF_forms = self._assemble(dF_forms,
                                              eq_deps=eq_nl_deps, deps=nl_deps)
                    dep_B.sub(dF_forms)

            # Cofunctions
            for weight, dF_term in iter_expr(dF_cofunctions,
                                             evaluate_weights=True):
                assert isinstance(dF_term, ufl.classes.Coargument)
                dep_B.sub((weight, adj_x_0))

            # Boundary conditions
            if dep_index in self._bc_nodes:
                if dF_bc is None:
                    if self._cache_rhs_assembly and is_cached(self._J):
                        mat, _ = self._assemble_cached(
                            "cached_adjoint_rhs_mat_bc", adjoint(self._J),
                            eq_deps=eq_nl_deps, deps=nl_deps)
                        dF_bc = matrix_multiply(mat, adj_x_0)
                    else:
                        dF_bc = action(adjoint(self._J), adj_x_0)
                        dF_bc = self._assemble(dF_bc,
                                               eq_deps=eq_nl_deps, deps=nl_deps)  # noqa: E501

                dF_term = var_new(dF_bc).assign(
                    dF_bc - adj_x.riesz_representation("l2"),
                    subset=self._bc_nodes[dep_index])
                dep_B.sub(dF_term)

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
        if len(self._hbcs) == 0:
            b_0 = b
        else:
            b_0 = var_copy(b)
            for bc in self._hbcs:
                bc.apply(adj_x)
                bc.apply(b_0.riesz_representation("l2"))

        if self._adjoint_solver_parameters.get("mat_type", "aij") != "matfree":
            J_solver, _ = self._linear_solver(
                adjoint(self._J), bcs=self._hbcs,
                linear_solver_parameters=self._adjoint_solver_parameters,
                eq_deps=self.nonlinear_dependencies(), deps=nl_deps,
                cache_key="adjoint_J_solver" if self._cache_adjoint_jacobian else None)  # noqa: E501
            J_solver.solve(adj_x, b_0)
        else:
            solve(self._nonlinear_replace(adjoint(self._J), nl_deps) == b_0,
                  adj_x, self._hbcs,
                  solver_parameters=self._adjoint_solver_parameters)

        for bc in self._hbcs:
            bc.reconstruct(g=b.riesz_representation("l2")).apply(adj_x)
        return adj_x

    def _tangent_linear_rhs(self, tlm_map):
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
        return tlm_rhs

    def tangent_linear(self, tlm_map):
        x = self.x()
        tlm_rhs = self._tangent_linear_rhs(tlm_map)
        tlm_bcs = self._reconstruct_bcs(tlm_map=tlm_map)
        if isinstance(tlm_rhs, ufl.classes.ZeroBaseForm) \
                and all(isinstance(bc._function_arg, ufl.classes.Zero) for bc in tlm_bcs):  # noqa: E501
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


class LocalEquationSolver(EquationSolver):
    """Represents the solution of a linear finite element variational problem,
    for the case where the matrix is element-wise local block diagonal.

    Remaining arguments are passed to the :class:`.EquationSolver` constructor.
    """

    def __init__(self, eq, x, *,
                 form_compiler_parameters=None, cache_jacobian=None,
                 cache_adjoint_jacobian=None, cache_tlm_jacobian=None,
                 cache_rhs_assembly=None, match_quadrature=None):
        super().__init__(
            eq, x,
            form_compiler_parameters=form_compiler_parameters,
            solver_parameters={"ksp_type": "preonly",
                               "pc_type": "lu"},
            cache_jacobian=cache_jacobian,
            cache_adjoint_jacobian=cache_adjoint_jacobian,
            cache_tlm_jacobian=cache_tlm_jacobian,
            cache_rhs_assembly=cache_rhs_assembly,
            match_quadrature=match_quadrature)
        if not self._linear:
            raise ValueError("Must be a linear variational problem")

    def _assemble_linear_solver(self, form, bcs, *,
                                linear_solver_parameters,
                                eq_deps, deps):
        if len(bcs) > 0:
            raise ValueError("Unexpected boundary conditions")
        if deps is not None:
            assert len(eq_deps) == len(deps)
            form = ufl.replace(form, dict(zip(eq_deps, deps)))

        solver = LocalSolver(
            form, form_compiler_parameters=self._form_compiler_parameters)
        return solver, None

    def _assemble_linear_solver_cached(self, key, form, bcs, *,
                                       linear_solver_parameters,
                                       eq_deps, deps):
        if len(bcs) > 0:
            raise ValueError("Unexpected boundary conditions")
        if deps is not None:
            assert len(eq_deps) == len(deps)
            replace_map = dict(zip(eq_deps, deps))
        else:
            replace_map = None

        var_update_caches(*eq_deps, value=deps)
        value = self._solver_cache[key]()
        if value is None:
            self._solver_cache[key], value = \
                local_solver_cache().local_solver(
                    form,
                    form_compiler_parameters=self._form_compiler_parameters,
                    replace_map=replace_map)
        return value, None

    def tangent_linear(self, tlm_map):
        x = self.x()
        tlm_rhs = self._tangent_linear_rhs(tlm_map)
        if len(self._hbcs) != 0:
            raise RuntimeError("Unexpected boundary conditions")
        if tlm_rhs.empty():
            return ZeroAssignment(tlm_map[x])
        else:
            return LocalEquationSolver(
                self._J == tlm_rhs, tlm_map[x],
                form_compiler_parameters=self._form_compiler_parameters,
                cache_jacobian=self._cache_tlm_jacobian,
                cache_adjoint_jacobian=self._cache_adjoint_jacobian,
                cache_tlm_jacobian=self._cache_tlm_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly)
