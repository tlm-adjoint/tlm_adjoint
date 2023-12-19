"""This module includes additional functionality for use with Firedrake.
"""

from .backend import (
    FunctionSpace, Interpolator, Tensor, TestFunction, TrialFunction,
    VertexOnlyMesh, backend_Cofunction, backend_Constant, backend_Function,
    backend_assemble, complex_mode)
from ..interface import (
    check_space_type, comm_dup_cached, is_var, space_new, var_assign, var_comm,
    var_id, var_inner, var_is_scalar, var_new, var_new_conjugate_dual,
    var_replacement, var_scalar_value, var_space, var_space_type,
    var_update_caches, var_zero, weakref_method)
from .backend_code_generator_interface import assemble, matrix_multiply
from .backend_interface import ReplacementCofunction, ReplacementFunction

from ..caches import Cache
from ..equation import Equation, ZeroAssignment

from .caches import form_dependencies, form_key, parameters_key
from .equations import (
    EquationSolver, ExprEquation, derivative, extract_dependencies)
from .functions import ReplacementConstant, eliminate_zeros, iter_expr

import itertools
import numpy as np
import pyop2
import ufl

__all__ = \
    [
        "LocalSolverCache",
        "local_solver_cache",
        "set_local_solver_cache",

        "ExprAssignment",
        "LocalProjection",
        "PointInterpolation"
    ]


def LocalSolver(form, *,
                form_compiler_parameters=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    local_solver = backend_assemble(
        Tensor(form).inv,
        form_compiler_parameters=form_compiler_parameters)

    def solve_local(self, x, b):
        matrix_multiply(self, b, tensor=x)
    local_solver._tlm_adjoint__solve_local = weakref_method(
        solve_local, local_solver)

    return local_solver


class LocalSolverCache(Cache):
    """A :class:`.Cache` for element-wise local block diagonal linear solver
    data.
    """

    def local_solver(self, form, *,
                     form_compiler_parameters=None, replace_map=None):
        """Compute data for an element-wise local block diagonal linear
        solver and cache the result, or return a previously cached result.

        :arg form: An arity two :class:`ufl.Form`, defining the element-wise
            local block diagonal matrix.
        :arg form_compiler_parameters: Form compiler parameters.
        :arg replace_map: A :class:`Mapping` defining a map from symbolic
            variables to values.
        :returns: A :class:`tuple` `(value_ref, value)`. `value` is a
            :class:`firedrake.matrix.Matrix` storing the assembled inverse
            matrix, and `value_ref` is a :class:`.CacheRef` storing a reference
            to `value`.
        """

        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        form = eliminate_zeros(form)
        if replace_map is None:
            assemble_form = form
        else:
            assemble_form = ufl.replace(form, replace_map)

        key = (form_key(form, assemble_form),
               parameters_key(form_compiler_parameters))

        def value():
            return LocalSolver(
                assemble_form,
                form_compiler_parameters=form_compiler_parameters)

        return self.add(key, value,
                        deps=form_dependencies(form, assemble_form))


_local_solver_cache = LocalSolverCache()


def local_solver_cache():
    """
    :returns: The default :class:`.LocalSolverCache`.
    """

    return _local_solver_cache


def set_local_solver_cache(local_solver_cache):
    """Set the default :class:`.LocalSolverCache`.

    :arg local_solver_cache: The new default :class:`.LocalSolverCache`.
    """

    global _local_solver_cache
    _local_solver_cache = local_solver_cache


class LocalProjection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`, for the case where the mass
    matrix is element-wise local block diagonal.

    :arg x: A :class:`firedrake.function.Function` defining the forward
        solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`, or a :class:`ufl.Form` defining the
        right-hand-side of the finite element variational problem. Should not
        depend on `x`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.firedrake.equations.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *,
                 form_compiler_parameters=None, cache_jacobian=None,
                 cache_rhs_assembly=None, match_quadrature=None):
        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(trial, test) * ufl.dx
        if not isinstance(rhs, ufl.classes.BaseForm):
            rhs = ufl.inner(rhs, test) * ufl.dx

        super().__init__(
            lhs == rhs, x,
            form_compiler_parameters=form_compiler_parameters,
            solver_parameters={},
            cache_jacobian=cache_jacobian,
            cache_rhs_assembly=cache_rhs_assembly,
            match_quadrature=match_quadrature)

    def forward_solve(self, x, deps=None):
        eq_deps = self.dependencies()

        if self._cache_rhs_assembly:
            b = self._cached_rhs(deps)
        elif deps is None:
            b = assemble(
                self._rhs,
                form_compiler_parameters=self._form_compiler_parameters)
        else:
            b = assemble(
                self._replace(self._rhs, deps),
                form_compiler_parameters=self._form_compiler_parameters)

        if self._cache_jacobian:
            var_update_caches(*eq_deps, value=deps)
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = \
                    local_solver_cache().local_solver(
                        self._lhs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        replace_map=self._replace_map(deps))
        else:
            local_solver = LocalSolver(
                self._lhs,
                form_compiler_parameters=self._form_compiler_parameters)

        local_solver._tlm_adjoint__solve_local(x, b)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        eq_nl_deps = self.nonlinear_dependencies()

        if self._cache_jacobian:
            var_update_caches(*eq_nl_deps, value=nl_deps)
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = \
                    local_solver_cache().local_solver(
                        self._lhs,
                        form_compiler_parameters=self._form_compiler_parameters,  # noqa: E501
                        replace_map=self._nonlinear_replace_map(nl_deps))
        else:
            local_solver = LocalSolver(
                self._lhs,
                form_compiler_parameters=self._form_compiler_parameters)

        adj_x = self.new_adj_x()
        local_solver._tlm_adjoint__solve_local(adj_x, b)
        return adj_x

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.ZeroBaseForm(self._rhs.arguments())
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    for weight, comp in iter_expr(self._rhs):
                        # Note: Ignores weight dependencies
                        tlm_rhs = (tlm_rhs
                                   + weight * derivative(comp, dep, argument=tau_dep))  # noqa: E501

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.ZeroBaseForm):
            return ZeroAssignment(tlm_map[x])
        else:
            return LocalProjection(
                tlm_map[x], tlm_rhs,
                form_compiler_parameters=self._form_compiler_parameters,
                cache_jacobian=self._cache_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly)


def vmesh_coords_map(vmesh, X_coords):
    comm = comm_dup_cached(vmesh.comm)
    N, _ = X_coords.shape

    vmesh_coords = vmesh.coordinates.dat.data_ro
    Nm, _ = vmesh_coords.shape

    vmesh_coords_indices = {tuple(vmesh_coords[i, :]): i for i in range(Nm)}
    vmesh_coords_map = np.full(Nm, -1, dtype=np.int_)
    for i in range(N):
        key = tuple(X_coords[i, :])
        if key in vmesh_coords_indices:
            vmesh_coords_map[vmesh_coords_indices[key]] = i
    if (vmesh_coords_map < 0).any():
        raise RuntimeError("Failed to find vertex map")

    vmesh_coords_map = comm.allgather(vmesh_coords_map)
    if len(tuple(itertools.chain(*vmesh_coords_map))) != N:
        raise RuntimeError("Failed to find vertex map")

    return vmesh_coords_map


class PointInterpolation(Equation):
    r"""Represents interpolation of a scalar-valued function at given points.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg X: A scalar variable, or a :class:`Sequence` of scalar variables,
        defining the forward solution.
    :arg y: A scalar-valued :class:`firedrake.function.Function` to
        interpolate.
    :arg X_coords: A :class:`numpy.ndarray` defining the coordinates at which
        to interpolate `y`. Shape is `(n, d)` where `n` is the number of
        interpolation points and `d` is the geometric dimension. Ignored if `P`
        is supplied.
    :arg tolerance: :class:`firedrake.mesh.VertexOnlyMesh` tolerance.
    """

    def __init__(self, X, y, X_coords=None, *, tolerance=None,
                 _interp=None):
        if is_var(X):
            X = (X,)

        for x in X:
            check_space_type(x, "primal")
            if not var_is_scalar(x):
                raise ValueError("Solution must be a scalar variable, or a "
                                 "Sequence of scalar variables")
        check_space_type(y, "primal")

        if X_coords is None:
            if _interp is None:
                raise TypeError("X_coords required")
        else:
            if len(X) != X_coords.shape[0]:
                raise ValueError("Invalid number of variables")
        if not isinstance(y, backend_Function):
            raise TypeError("y must be a Function")
        if len(y.ufl_shape) > 0:
            raise ValueError("y must be a scalar-valued Function")

        interp = _interp
        if interp is None:
            y_space = y.function_space()
            vmesh = VertexOnlyMesh(y_space.mesh(), X_coords,
                                   tolerance=tolerance)
            vspace = FunctionSpace(vmesh, "Discontinuous Lagrange", 0)
            interp = Interpolator(TestFunction(y_space), vspace)
            if not hasattr(interp, "_tlm_adjoint__vmesh_coords_map"):
                interp._tlm_adjoint__vmesh_coords_map = vmesh_coords_map(vmesh, X_coords)  # noqa: E501

        super().__init__(X, list(X) + [y], nl_deps=[], ic=False, adj_ic=False)
        self._interp = interp

    def forward_solve(self, X, deps=None):
        if is_var(X):
            X = (X,)
        y = (self.dependencies() if deps is None else deps)[-1]

        Xm = space_new(self._interp.V)
        self._interp.interpolate(y, output=Xm)

        X_values = var_comm(Xm).allgather(Xm.dat.data_ro)
        vmesh_coords_map = self._interp._tlm_adjoint__vmesh_coords_map
        for x_val, index in zip(itertools.chain(*X_values),
                                itertools.chain(*vmesh_coords_map)):
            X[index].assign(x_val)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_var(adj_X):
            adj_X = (adj_X,)

        if dep_index < len(adj_X):
            return adj_X[dep_index]
        elif dep_index == len(adj_X):
            adj_Xm = space_new(self._interp.V, space_type="conjugate_dual")

            vmesh_coords_map = self._interp._tlm_adjoint__vmesh_coords_map
            rank = var_comm(adj_Xm).rank
            # This line must be outside the loop to avoid deadlocks
            adj_Xm_data = adj_Xm.dat.data
            for i, j in enumerate(vmesh_coords_map[rank]):
                adj_Xm_data[i] = var_scalar_value(adj_X[j])

            F = var_new_conjugate_dual(self.dependencies()[-1])
            self._interp.interpolate(adj_Xm, transpose=True, output=F)
            return (-1.0, F)
        else:
            raise IndexError("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()
        y = self.dependencies()[-1]

        tlm_y = tlm_map[y]
        if tlm_y is None:
            return ZeroAssignment([tlm_map[x] for x in X])
        else:
            return PointInterpolation([tlm_map[x] for x in X], tlm_y,
                                      _interp=self._interp)


class ExprAssignment(ExprEquation):
    r"""Represents an evaluation of `rhs`, storing the result in `x`. Uses
    :meth:`firedrake.function.Function.assign` or
    :meth:`firedrake.cofunction.Cofunction.assign` to perform the evaluation.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: A :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to
        evaluate. Should not depend on `x`.
    :arg subset: A :class:`pyop2.types.set.Subset`. If provided then defines a
        subset of degrees of freedom at which to evaluate `rhs`. Other degrees
        of freedom are set to zero.
    """

    def __init__(self, x, rhs, *,
                 subset=None):
        deps, nl_deps = extract_dependencies(
            rhs, space_type=var_space_type(x))
        if var_id(x) in deps:
            raise ValueError("Invalid non-linear dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        if subset is not None:
            subset = pyop2.Subset(subset.superset, subset.indices)

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs
        self._subset = subset
        self._subset_kwargs = {} if subset is None else {"subset": subset}

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()

        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        rhs = self._replace(self._rhs, deps)
        if self._subset is not None:
            var_zero(x)
        x.assign(rhs, **self._subset_kwargs)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        eq_deps = self.dependencies()
        if dep_index < 0 or dep_index >= len(eq_deps):
            raise IndexError("dep_index out of bounds")
        elif dep_index == 0:
            return adj_x

        dep = eq_deps[dep_index]
        if len(dep.ufl_shape) > 0:
            if not isinstance(dep, (backend_Cofunction, ReplacementCofunction,
                                    backend_Function, ReplacementFunction)):
                raise NotImplementedError("Case not implemented")

            if complex_mode:
                # Used to work around a missing conjugate, see below
                adj_x_ = var_new_conjugate_dual(adj_x)
                adj_x_.dat.data[:] = adj_x.dat.data_ro.conjugate()
                adj_x = adj_x_
                del adj_x_
            else:
                adj_x = adj_x.riesz_representation("l2")

            test = TestFunction(dep)
            # dF = derivative(action(cotest, self._rhs), dep, argument=trial)
            dF = derivative(self._rhs, dep, argument=test)
            # dF = action(adjoint(dF), adj_x)
            # Missing a conjugate, see below
            dF = ufl.replace(dF, {test: adj_x})
            dF = ufl.algorithms.expand_derivatives(dF)
            dF = eliminate_zeros(dF)
            dF = self._nonlinear_replace(dF, nl_deps)

            # F = assemble(dF)
            F = var_new(dep)
            F.assign(dF, **self._subset_kwargs)

            if complex_mode:
                # The conjugate which would be introduced by adjoint(...).
                # Above we take the conjugate of the adj_x dofs, and this is
                # reversed here, so we have the required action of the adjoint
                # of the derivative on adj_x.
                F_ = var_new_conjugate_dual(F)
                F_.dat.data[:] = F.dat.data_ro.conjugate()
                F = F_
            else:
                F = F.riesz_representation("l2")
        else:
            dF = derivative(self._rhs, dep, argument=ufl.classes.IntValue(1))
            dF = ufl.algorithms.expand_derivatives(dF)
            dF = eliminate_zeros(dF)
            dF = self._nonlinear_replace(dF, nl_deps)

            if isinstance(dep, (backend_Constant, ReplacementConstant)):
                dF = var_new_conjugate_dual(adj_x).assign(
                    dF, **self._subset_kwargs)
                F = var_new_conjugate_dual(dep)
                var_assign(F, var_inner(adj_x, dF))
            elif isinstance(dep, (backend_Cofunction, ReplacementCofunction,
                                  backend_Function, ReplacementFunction)):
                e = var_space(dep).ufl_element()
                F = var_new_conjugate_dual(dep)
                if (e.family(), e.degree(), e.value_shape) == ("Real", 0, ()):
                    dF = var_new_conjugate_dual(adj_x).assign(
                        dF, **self._subset_kwargs)
                    F.dat.data[:] = var_inner(adj_x, dF)
                else:
                    F.assign(adj_x, **self._subset_kwargs)
                    F *= dF(()).conjugate()
            else:
                raise TypeError(f"Unexpected type: {type(F)}")
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.Zero(shape=x.ufl_shape)
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
            return ExprAssignment(tlm_map[x], tlm_rhs, subset=self._subset)
