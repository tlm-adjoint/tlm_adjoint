"""Finite element assembly operations with Firedrake.
"""

from .backend import (
    LinearSolver, adjoint, backend_DirichletBC, backend_Function,
    backend_Matrix, backend_assemble, complex_mode, parameters)
from ..interface import (
    check_space_type, check_space_types, is_var, register_functional_term_eq,
    space_new, var_assign, var_id, var_is_scalar, var_new_conjugate_dual,
    var_replacement, var_scalar_value, var_space_type)

from ..equation import ZeroAssignment
from ..patch import patch_method

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_coefficients,
    extract_dependencies, iter_expr)
from .parameters import (
    form_compiler_quadrature_parameters, process_form_compiler_parameters,
    update_parameters)

import petsc4py.PETSc as PETSc
import ufl

__all__ = \
    [
        "Assembly"
    ]


def _assemble(form, tensor=None, bcs=None, *,
              form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    form = eliminate_zeros(form)
    if isinstance(form, ufl.classes.ZeroBaseForm):
        raise ValueError("Form cannot be a ZeroBaseForm")
    if len(form.arguments()) == 1:
        b = backend_assemble(
            form, tensor=tensor,
            form_compiler_parameters=form_compiler_parameters,
            mat_type=mat_type)
        for bc in bcs:
            bc.apply(b.riesz_representation("l2"))
    else:
        b = backend_assemble(
            form, tensor=tensor, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters,
            mat_type=mat_type)

    return b


def _assemble_system(A_form, b_form=None, bcs=None, *,
                     form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    A = _assemble(
        A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
        mat_type=mat_type)

    if len(bcs) > 0:
        F = backend_Function(A_form.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)

        if b_form is None:
            b = _assemble(
                -ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

            with b.dat.vec_ro as b_v:
                if b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) == 0.0:
                    b = None
        else:
            b = _assemble(
                b_form - ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)
    else:
        if b_form is None:
            b = None
        else:
            b = _assemble(
                b_form,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

    A._tlm_adjoint__lift_bcs = False

    return A, b


@patch_method(LinearSolver, "_lifted")
def LinearSolver_lifted(self, orig, orig_args, b):
    if getattr(self.A, "_tlm_adjoint__lift_bcs", True):
        return orig_args()
    else:
        return b


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    return _assemble_system(form, bcs=bcs,
                            form_compiler_parameters=form_compiler_parameters,
                            mat_type=mat_type)


def assemble(form, tensor=None, bcs=None, *,
             form_compiler_parameters=None, mat_type=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    b = _assemble(
        form, tensor=tensor, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters, mat_type=mat_type)

    return b


def matrix_copy(A):
    if not isinstance(A, backend_Matrix):
        raise TypeError("Unexpected matrix type")

    options_prefix = A.petscmat.getOptionsPrefix()
    A_copy = backend_Matrix(A.a, A.bcs, A.mat_type,
                            A.M.sparsity, A.M.dtype,
                            options_prefix=options_prefix)

    assert A.petscmat.assembled
    A_copy.petscmat.axpy(1.0, A.petscmat)
    assert A_copy.petscmat.assembled

    # MatAXPY does not propagate the options prefix
    A_copy.petscmat.setOptionsPrefix(options_prefix)

    if hasattr(A, "_tlm_adjoint__lift_bcs"):
        A_copy._tlm_adjoint__lift_bcs = A._tlm_adjoint__lift_bcs

    return A_copy


def matrix_multiply(A, x, *,
                    tensor=None, addto=False, action_type="conjugate_dual"):
    if tensor is None:
        tensor = space_new(
            A.a.arguments()[0].function_space(),
            space_type=var_space_type(x, rel_space_type=action_type))
    else:
        check_space_types(tensor, x, rel_space_type=action_type)

    if addto:
        with x.dat.vec_ro as x_v, tensor.dat.vec as tensor_v:
            A.petscmat.multAdd(x_v, tensor_v, tensor_v)
    else:
        with x.dat.vec_ro as x_v, tensor.dat.vec_wo as tensor_v:
            A.petscmat.mult(x_v, tensor_v)

    return tensor


_parameters = parameters.setdefault("tlm_adjoint", {})
_parameters.setdefault("Assembly", {})
_parameters["Assembly"].setdefault("match_quadrature", False)
del _parameters


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
                        dF = ufl.algorithms.expand_derivatives(dF)
                        dF = eliminate_zeros(dF)
                        for dF_term_weight, dF_term in iter_expr(weight * dF,
                                                                 evaluate_weights=True):  # noqa: E501
                            assert isinstance(dF_term, ufl.classes.Coargument)
                            dep_B.sub((-dF_term_weight.conjugate(), adj_x))
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


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.BaseForm,
    functional_term_eq_form)
