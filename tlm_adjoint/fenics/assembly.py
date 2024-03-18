"""Finite element assembly operations with FEniCS.
"""

from .backend import Parameters, adjoint, parameters
from ..interface import (
    check_space_type, register_functional_term_eq, var_assign, var_id,
    var_is_scalar, var_replacement, var_scalar_value)

from ..equation import ZeroAssignment

from .backend_interface import assemble
from .expr import (
    ExprEquation, action, derivative, eliminate_zeros, expr_zero,
    extract_dependencies)
from .parameters import (
    form_compiler_quadrature_parameters, process_form_compiler_parameters,
    update_parameters)

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "Assembly"
    ]


if "tlm_adjoint" not in parameters:
    parameters.add(Parameters("tlm_adjoint"))
_parameters = parameters["tlm_adjoint"]
if "Assembly" not in _parameters:
    _parameters.add(Parameters("Assembly"))
if "match_quadrature" not in _parameters["Assembly"]:
    _parameters["Assembly"].add("match_quadrature", False)
del _parameters


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
            update_parameters(
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
        dF = eliminate_zeros(dF)
        if dF.empty():
            return None

        dF = self._nonlinear_replace(dF, nl_deps)
        if self._arity == 0:
            # dF = adjoint(dF)
            dF = ufl.classes.Form(
                [integral.reconstruct(integrand=ufl.conj(integral.integrand()))
                 for integral in dF.integrals()])
            dF = assemble(
                dF, form_compiler_parameters=self._form_compiler_parameters)
            return (-var_scalar_value(adj_x), dF)
        elif self._arity == 1:
            dF = action(adjoint(dF), adj_x)
            dF = assemble(
                dF, form_compiler_parameters=self._form_compiler_parameters)
            return (-1.0, dF)
        else:
            raise ValueError("Must be an arity 0 or arity 1 form")

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(self._rhs)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = tlm_rhs + derivative(self._rhs, dep, argument=tau_dep)  # noqa: E501

        if tlm_rhs.empty():
            return ZeroAssignment(tlm_map[x])
        else:
            return Assembly(
                tlm_map[x], tlm_rhs,
                form_compiler_parameters=self._form_compiler_parameters)


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.Form,
    functional_term_eq_form)
