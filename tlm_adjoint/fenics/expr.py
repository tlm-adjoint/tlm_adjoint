"""Symbolic expression functionality.
"""

from .backend import (
    FunctionSpace, TensorFunctionSpace, TestFunction, TrialFunction,
    VectorFunctionSpace, backend_action, backend_Constant, backend_Function,
    backend_ScalarType, cpp_Constant)
from ..interface import (
    VariableStateChangeError, add_replacement_interface, check_space_type,
    is_var, manager_disabled, var_id, var_is_replacement, var_replacement,
    var_space)

from ..equation import Equation

from collections.abc import Sequence
import functools
import numbers
import numpy as np
try:
    import ufl_legacy as ufl
except ModuleNotFoundError:
    import ufl

__all__ = \
    [
        "Zero",
        "eliminate_zeros",

        "Replacement"
    ]


def form_cached(key):
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(expr):
            if isinstance(expr, ufl.classes.Form) and key in expr._cache:
                value = expr._cache[key]
            else:
                value = fn(expr)
                if isinstance(expr, ufl.classes.Form):
                    assert key not in expr._cache
                    expr._cache[key] = value
            return value
        return wrapped
    return wrapper


def as_ufl(expr):
    if isinstance(expr, (ufl.classes.Expr, numbers.Complex)):
        return ufl.as_ufl(expr)
    elif isinstance(expr, ufl.classes.Form):
        return expr
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")


@form_cached("_tlm_adjoint__extract_coefficients")
def extract_coefficients(expr):
    if isinstance(expr, Sequence):
        expr = ufl.as_vector(tuple(map(as_ufl, expr)))
    else:
        expr = as_ufl(expr)

    deps = ufl.algorithms.extract_coefficients(expr)

    if len(set(map(var_id, deps))) != len(deps):
        raise RuntimeError("Invalid dependencies")
    return deps


@form_cached("_tlm_adjoint__extract_variables")
def extract_variables(expr):
    deps = sorted((dep for dep in extract_coefficients(expr) if is_var(dep)),
                  key=var_id)
    return tuple(deps)


def extract_derivative_variables(expr, dep):
    dexpr = derivative(expr, dep, enable_automatic_argument=False)
    return extract_variables(dexpr)


def extract_dependencies(expr, *, space_type=None):
    deps = {var_id(dep): dep for dep in extract_variables(expr)}

    nl_deps = {}
    for dep in deps.values():
        for nl_dep in extract_derivative_variables(expr, dep):
            nl_deps.setdefault(var_id(nl_dep), nl_dep)
    nl_deps = {nl_dep_id: nl_deps[nl_dep_id]
               for nl_dep_id in sorted(nl_deps.keys())}

    assert len(set(nl_deps.keys()).difference(set(deps.keys()))) == 0
    if space_type is not None:
        for dep in deps.values():
            check_space_type(dep, space_type)

    return deps, nl_deps


@manager_disabled()
def is_valid_r0_space(space):
    e = space.ufl_element()
    if (e.family(), e.degree()) != ("Real", 0):
        return False
    elif len(e.value_shape()) == 0:
        r = backend_Function(space)
        r.assign(backend_Constant(-1.0))
        return (r.vector().max() == -1.0)
    else:
        r = backend_Function(space)
        r_arr = -np.arange(1, np.prod(r.ufl_shape) + 1,
                           dtype=backend_ScalarType)
        r_arr.shape = r.ufl_shape
        r.assign(backend_Constant(r_arr))
        for i, r_c in enumerate(r.split(deepcopy=True)):
            if r_c.vector().max() != -(i + 1):
                return False
        else:
            return True


def r0_space(x):
    domain = var_space(x)._tlm_adjoint__space_interface_attrs["domain"]
    domain = domain.ufl_cargo()
    if not hasattr(domain, "_tlm_adjoint__r0_space"):
        if len(x.ufl_shape) == 0:
            space = FunctionSpace(domain, "R", 0)
        elif len(x.ufl_shape) == 1:
            dim, = ufl.shape
            space = VectorFunctionSpace(domain, "R", 0, dim=dim)
        else:
            space = TensorFunctionSpace(domain, "R", degree=0,
                                        shape=x.ufl_shape)
        if not is_valid_r0_space(space):
            raise RuntimeError("Invalid space")
        domain._tlm_adjoint__r0_space = space
    return domain._tlm_adjoint__r0_space


def derivative_space(x):
    space = var_space(x)
    if space.ufl_domain() is not None:
        return space
    elif space.ufl_element().family() == "Real":
        return r0_space(x)
    else:
        raise RuntimeError("Unable to determine space")


def _derivative(expr, x, argument=None):
    expr = as_ufl(expr)
    if argument is None:
        dexpr = ufl.derivative(expr, x)
        dexpr = ufl.algorithms.expand_derivatives(dexpr)
    else:
        if isinstance(expr, ufl.classes.Expr):
            dexpr = ufl.derivative(expr, x, argument=argument)
            dexpr = ufl.algorithms.expand_derivatives(dexpr)
        elif isinstance(expr, ufl.classes.Form):
            if len(ufl.algorithms.extract_arguments(argument)) \
                    > len(ufl.algorithms.extract_arguments(x)):
                dexpr = ufl.derivative(expr, x, argument=argument)
                dexpr = ufl.algorithms.expand_derivatives(dexpr)
            else:
                dexpr = ufl.derivative(expr, x)
                dexpr = ufl.algorithms.expand_derivatives(dexpr)
                if not dexpr.empty():
                    dexpr = action(dexpr, argument)
        else:
            raise TypeError(f"Unexpected type: {type(expr)}")
    return dexpr


def derivative(expr, x, argument=None, *,
               enable_automatic_argument=True):
    expr_arguments = ufl.algorithms.extract_arguments(expr)
    arity = len(expr_arguments)

    if argument is None and enable_automatic_argument:
        Argument = {0: TestFunction, 1: TrialFunction}[arity]
        argument = Argument(derivative_space(x))

    for expr_argument in expr_arguments:
        if expr_argument.number() >= arity:
            raise ValueError("Unexpected argument")
    if argument is not None:
        for expr_argument in ufl.algorithms.extract_arguments(argument):
            if expr_argument.number() < arity:
                raise ValueError("Invalid argument")

    return _derivative(expr, x, argument=argument)


def action(form, coefficient):
    return backend_action(form, coefficient=coefficient)


class Zero:
    """Mixin for defining a zero-valued variable. Used for zero-valued
    variables for which UFL zero elimination should not be applied.
    """

    def _tlm_adjoint__var_interface_update_state(self):
        raise VariableStateChangeError("Cannot call _update_state interface "
                                       "of Zero")


def expr_zero(expr):
    if isinstance(expr, ufl.classes.Form):
        return ufl.classes.Form([])
    elif isinstance(expr, ufl.classes.Expr):
        return ufl.classes.Zero(shape=expr.ufl_shape,
                                free_indices=expr.ufl_free_indices,
                                index_dimensions=expr.ufl_index_dimensions)
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")


@form_cached("_tlm_adjoint__eliminate_zeros")
def eliminate_zeros(expr):
    """Apply zero elimination for :class:`.Zero` objects in the supplied
    :class:`ufl.core.expr.Expr` or :class:`ufl.Form`.

    :arg expr: A :class:`ufl.core.expr.Expr` or :class:`ufl.Form`.
    :returns: A :class:`ufl.core.expr.Expr` or :class:`ufl.Form` with zero
        elimination applied. May return `expr`.
    """

    replace_map = {c: expr_zero(c)
                   for c in extract_variables(expr)
                   if isinstance(c, Zero)}
    if len(replace_map) == 0:
        simplified_expr = expr
    else:
        simplified_expr = ufl.replace(expr, replace_map)

    return simplified_expr


class Replacement(ufl.classes.Coefficient):
    """Represents a symbolic variable but with no value.
    """

    def __init__(self, x, count):
        super().__init__(var_space(x), count=count)
        add_replacement_interface(self, x)


def new_count():
    return cpp_Constant(0.0).id()


def replaced_form(form):
    replace_map = {}
    for c in extract_variables(form):
        if not var_is_replacement(c):
            c_rep = var_replacement(c)
            if c_rep is not c:
                replace_map[c] = c_rep
    return ufl.replace(form, replace_map)


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
