"""Assignment operations with Firedrake.
"""

from .backend import (
    TestFunction, backend_Cofunction, backend_Constant, backend_Function,
    complex_mode)
from ..interface import (
    var_assign, var_id, var_inner, var_new, var_new_conjugate_dual,
    var_replacement, var_space, var_space_type, var_zero)

from ..equation import ZeroAssignment

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_dependencies)
from .variables import (
    ReplacementCofunction, ReplacementConstant, ReplacementFunction)

import ufl

__all__ = \
    [
        "ExprAssignment"
    ]


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
            raise ValueError("Invalid dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

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
        if dep_index <= 0 or dep_index >= len(eq_deps):
            raise ValueError("Unexpected dep_index")

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
                del F_
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

        tlm_rhs = expr_zero(x)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = (tlm_rhs
                               + derivative(self._rhs, dep, argument=tau_dep))

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.Zero):
            return ZeroAssignment(tlm_map[x])
        else:
            return ExprAssignment(tlm_map[x], tlm_rhs, subset=self._subset)
