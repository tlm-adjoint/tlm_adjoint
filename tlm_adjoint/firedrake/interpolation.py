"""Interpolation operations with Firedrake.
"""

from .backend import (
    Interpolator, backend_Cofunction, backend_Constant, backend_Function)
from ..interface import (
    check_space_types, var_assign, var_assign_conjugate, var_axpy,
    var_axpy_conjugate, var_copy_conjugate, var_id, var_inner, var_new,
    var_new_conjugate, var_new_conjugate_dual, var_replacement, var_zero)

from ..equation import ZeroAssignment
from ..manager import manager_disabled

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_dependencies,
    iter_expr)
from .variables import ReplacementConstant

import ufl

__all__ = \
    [
        "ExprInterpolation"
    ]


@manager_disabled()
def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is not None:
        check_space_types(x, adj_x)

    expr = eliminate_zeros(expr)

    if adj_x is None:
        if isinstance(x, backend_Constant):
            x.assign(expr)
        elif isinstance(x, backend_Cofunction):
            var_zero(x)
            for weight, comp in iter_expr(expr):
                var_axpy(x, weight, var_new(x).interpolate(comp))
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
        Interpolator(expr, adj_x.function_space().dual())._interpolate(
            adj_x, adjoint=True, output=x)
    elif isinstance(x, backend_Function):
        (weight, expr), = iter_expr(expr)
        if weight != 1.0 or not isinstance(expr, ufl.classes.Coargument):
            raise NotImplementedError("Case not implemented")
        Interpolator(adj_x, x.function_space())._interpolate(
            output=x)
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
        deps, nl_deps = extract_dependencies(rhs)
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
