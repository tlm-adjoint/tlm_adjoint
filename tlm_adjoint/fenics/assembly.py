"""Finite element assembly operations with FEniCS.
"""

from .backend import (
    Parameters, TestFunction, adjoint, as_backend_type, backend_Constant,
    backend_DirichletBC, backend_Function, backend_ScalarType,
    backend_assemble, backend_assemble_system, parameters)
from ..interface import (
    check_space_type, check_space_types, register_functional_term_eq,
    space_new, var_assign, var_id, var_is_scalar, var_replacement,
    var_scalar_value, var_space_type)

from ..equation import ZeroAssignment

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_dependencies)
from .parameters import (
    form_compiler_quadrature_parameters, process_form_compiler_parameters,
    update_parameters)

import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "Assembly"
    ]


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    if len(bcs) > 0:
        test = TestFunction(form.arguments()[0].function_space())
        if len(test.ufl_shape) == 0:
            zero = backend_Constant(0.0)
        else:
            zero = backend_Constant(np.zeros(test.ufl_shape,
                                             dtype=backend_ScalarType))
        dummy_rhs = ufl.inner(zero, test) * ufl.dx
        A, b_bc = assemble_system(
            form, dummy_rhs, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        if b_bc.norm("linf") == 0.0:
            b_bc = None
    else:
        A = assemble(
            form, form_compiler_parameters=form_compiler_parameters)
        b_bc = None

    return A, b_bc


def assemble(form, tensor=None, bcs=None, *,
             form_compiler_parameters=None):
    if tensor is not None and hasattr(tensor, "_tlm_adjoint__function"):
        check_space_type(tensor._tlm_adjoint__function, "conjugate_dual")
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)

    form = eliminate_zeros(form)
    b = backend_assemble(form, tensor=tensor,
                         form_compiler_parameters=form_compiler_parameters)
    for bc in bcs:
        bc.apply(b)
    return b


def assemble_system(A_form, b_form, bcs=None, *,
                    form_compiler_parameters=None):
    A_form = eliminate_zeros(A_form)
    b_form = eliminate_zeros(b_form)
    return backend_assemble_system(
        A_form, b_form, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters)


def matrix_copy(A):
    return A.copy()


def matrix_multiply(A, x, *,
                    tensor=None, addto=False, action_type="conjugate_dual"):
    if isinstance(x, backend_Function):
        x = x.vector()
    if tensor is not None and isinstance(tensor, backend_Function):
        tensor = tensor.vector()
    if tensor is None:
        if hasattr(A, "_tlm_adjoint__form") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
            tensor = space_new(
                A._tlm_adjoint__form.arguments()[0].function_space(),
                space_type=var_space_type(x._tlm_adjoint__function,
                                          rel_space_type=action_type))
            tensor = tensor.vector()
        else:
            return A * x
    elif hasattr(tensor, "_tlm_adjoint__function") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
        check_space_types(tensor._tlm_adjoint__function,
                          x._tlm_adjoint__function,
                          rel_space_type=action_type)

    x_v = as_backend_type(x).vec()
    tensor_v = as_backend_type(tensor).vec()
    if addto:
        as_backend_type(A).mat().multAdd(x_v, tensor_v, tensor_v)
    else:
        as_backend_type(A).mat().mult(x_v, tensor_v)
    tensor.apply("insert")

    return tensor


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


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.Form,
    functional_term_eq_form)
