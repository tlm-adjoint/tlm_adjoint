"""Interpolation operations with Firedrake.
"""

from .backend import (
    FunctionSpace, Interpolator, TestFunction, VertexOnlyMesh,
    backend_Cofunction, backend_Constant, backend_Function)
from ..interface import (
    check_space_type, comm_dup_cached, is_var, space_new, var_assign, var_comm,
    var_copy, var_id, var_inner, var_is_scalar, var_new_conjugate_dual,
    var_replacement, var_scalar_value, var_space)

from ..equation import Equation, ZeroAssignment
from ..manager import manager_disabled

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_dependencies,
    extract_variables)
from .variables import ReplacementConstant

import itertools
import numpy as np
import ufl

__all__ = \
    [
        "ExprInterpolation",
        "PointInterpolation"
    ]


@manager_disabled()
def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is None:
        check_space_type(x, "primal")
    else:
        check_space_type(x, "conjugate_dual")
        check_space_type(adj_x, "conjugate_dual")
    for dep in extract_variables(expr):
        check_space_type(dep, "primal")

    expr = eliminate_zeros(expr)

    if adj_x is None:
        if isinstance(x, backend_Constant):
            x.assign(expr)
        elif isinstance(x, backend_Function):
            x.interpolate(expr)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
    elif isinstance(x, backend_Constant):
        if len(x.ufl_shape) > 0:
            raise ValueError("Scalar Constant required")
        expr_val = var_new_conjugate_dual(adj_x)
        interpolate_expression(expr_val, expr)
        var_assign(x, var_inner(adj_x, expr_val))
    elif isinstance(x, backend_Cofunction):
        adj_x_space = var_space(adj_x).dual()
        interp = Interpolator(expr, adj_x_space)
        adj_x = var_copy(adj_x)
        adj_x.dat.data[:] = adj_x.dat.data_ro.conjugate()
        interp._interpolate(adj_x, transpose=True, output=x)
        x.dat.data[:] = x.dat.data_ro.conjugate()
    else:
        raise TypeError(f"Unexpected type: {type(x)}")


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
        dF = eliminate_zeros(dF)
        dF = self._nonlinear_replace(dF, nl_deps)

        F = var_new_conjugate_dual(dep)
        interpolate_expression(F, dF, adj_x=adj_x)
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(x)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = (tlm_rhs
                               + derivative(self._rhs, dep, argument=tau_dep))

        if isinstance(tlm_rhs, ufl.classes.Zero):
            return ZeroAssignment(tlm_map[x])
        else:
            return ExprInterpolation(tlm_map[x], tlm_rhs)


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
        self._interp._interpolate(y, output=Xm)

        X_values = var_comm(Xm).allgather(Xm.dat.data_ro)
        vmesh_coords_map = self._interp._tlm_adjoint__vmesh_coords_map
        for x_val, index in zip(itertools.chain(*X_values),
                                itertools.chain(*vmesh_coords_map)):
            X[index].assign(x_val)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_var(adj_X):
            adj_X = (adj_X,)
        if dep_index != len(self.X()):
            raise ValueError("Unexpected dep_index")

        adj_Xm = space_new(self._interp.V, space_type="conjugate_dual")

        vmesh_coords_map = self._interp._tlm_adjoint__vmesh_coords_map
        rank = var_comm(adj_Xm).rank
        # This line must be outside the loop to avoid deadlocks
        adj_Xm_data = adj_Xm.dat.data
        for i, j in enumerate(vmesh_coords_map[rank]):
            adj_Xm_data[i] = var_scalar_value(adj_X[j])

        F = var_new_conjugate_dual(self.dependencies()[-1])
        self._interp._interpolate(adj_Xm, transpose=True, output=F)
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, tlm_map):
        X = self.X()
        y = self.dependencies()[-1]

        tlm_y = tlm_map[y]
        if tlm_y is None:
            return ZeroAssignment([tlm_map[x] for x in X])
        else:
            return PointInterpolation([tlm_map[x] for x in X], tlm_y,
                                      _interp=self._interp)
