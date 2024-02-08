"""Projection operations with FEniCS.
"""

from .backend import LocalSolver, TestFunction, TrialFunction
from ..interface import var_space, var_update_caches

from ..equation import ZeroAssignment

from .caches import local_solver_cache
from .solve import EquationSolver

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "LocalProjection",
        "Projection"
    ]


class Projection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`.

    :arg x: A DOLFIN `Function` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`, or a :class:`ufl.Form` defining the
        right-hand-side of the finite element variational problem. Should not
        depend on `x`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.fenics.solve.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *args, **kwargs):
        space = var_space(x)
        test, trial = TestFunction(space), TrialFunction(space)
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(rhs, test) * ufl.dx
        super().__init__(ufl.inner(trial, test) * ufl.dx == rhs, x,
                         *args, **kwargs)


class LocalProjection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`, for the case where the mass
    matrix is element-wise local block diagonal.

    :arg x: A DOLFIN `Function` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`, or a :class:`ufl.Form` defining the
        right-hand-side of the finite element variational problem. Should not
        depend on `x`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.fenics.solve.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *,
                 form_compiler_parameters=None, cache_jacobian=None,
                 cache_rhs_assembly=None, match_quadrature=None):
        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(trial, test) * ufl.dx
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(rhs, test) * ufl.dx

        super().__init__(
            lhs == rhs, x,
            form_compiler_parameters=form_compiler_parameters,
            solver_parameters={"linear_solver": "direct"},
            cache_jacobian=cache_jacobian,
            cache_rhs_assembly=cache_rhs_assembly,
            match_quadrature=match_quadrature)
        self._local_solver_type = LocalSolver.SolverType.Cholesky

    def _local_solver(self, *args, cache_key=None, **kwargs):
        if cache_key is None:
            return self._assemble_local_solver(*args, **kwargs)
        else:
            return self._assemble_local_solver_cached(
                cache_key, *args, **kwargs)

    def _assemble_local_solver(self, form, *,
                               eq_deps, deps):
        if deps is not None:
            assert len(eq_deps) == len(deps)
            form = ufl.replace(form, dict(zip(eq_deps, deps)))

        return LocalSolver(
            form, solver_type=self._local_solver_type)

    def _assemble_local_solver_cached(self, key, form, *,
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
                local_solver_cache().local_solver(
                    form, self._local_solver_type,
                    replace_map=replace_map)
        return value

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()

        assert self._linear
        assert len(self._hbcs) == 0
        x_0, b = self._assemble_rhs(x, (), deps=deps)
        assert x_0 is x

        J_solver = self._local_solver(
            self._J, eq_deps=eq_deps, deps=deps,
            cache_key="J_solver_local" if self._cache_jacobian else None)

        J_solver.solve_local(x.vector(), b, x.function_space().dofmap())

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        adj_x = self.new_adj_x()
        eq_nl_deps = self.nonlinear_dependencies()

        assert len(self._hbcs) == 0
        J_solver = self._local_solver(
            self._J, eq_deps=eq_nl_deps, deps=nl_deps,
            cache_key="J_solver_local" if self._cache_jacobian else None)

        J_solver.solve_local(adj_x.vector(), b.vector(),
                             adj_x.function_space().dofmap())
        return adj_x

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()
        tlm_rhs = self._tangent_linear_rhs(tlm_map)
        assert len(self._hbcs) == 0
        if tlm_rhs.empty():
            return ZeroAssignment(tlm_map[x])
        else:
            return LocalProjection(
                tlm_map[x], tlm_rhs,
                form_compiler_parameters=self._form_compiler_parameters,
                cache_jacobian=self._cache_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly)
