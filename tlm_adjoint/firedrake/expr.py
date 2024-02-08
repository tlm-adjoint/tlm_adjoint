"""Symbolic expression functionality.
"""

from .backend import (
    TestFunction, TrialFunction, backend_Constant, complex_mode)
from ..interface import (
    VariableStateChangeError, check_space_type, is_var, var_id,
    var_is_replacement, var_replacement, var_space)

from ..equation import Equation

from collections.abc import Sequence
import functools
import itertools
import numbers
import ufl

__all__ = \
    [
        "Zero",
        "eliminate_zeros",

        "Replacement"
    ]


def iter_expr(expr, *, evaluate_weights=False):
    if isinstance(expr, ufl.classes.FormSum):
        for weight, comp in zip(expr.weights(), expr.components()):
            if evaluate_weights:
                weight = complex(weight)
                if weight.imag == 0:
                    weight = weight.real
            yield (weight, comp)
    elif isinstance(expr, (ufl.classes.Action, ufl.classes.Coargument,
                           ufl.classes.Cofunction, ufl.classes.Expr,
                           ufl.classes.Form)):
        yield (1, expr)
    elif isinstance(expr, ufl.classes.ZeroBaseForm):
        return
        yield
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")


def form_cached(key):
    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped(expr, *args, **kwargs):
            if isinstance(expr, ufl.classes.Form) and key in expr._cache:
                value = expr._cache[key]
            else:
                value = fn(expr, *args, **kwargs)
                if isinstance(expr, ufl.classes.Form):
                    assert key not in expr._cache
                    expr._cache[key] = value
            return value
        return wrapped
    return wrapper


@form_cached("_tlm_adjoint__extract_coefficients")
def extract_coefficients(expr):
    def as_ufl(expr):
        if isinstance(expr, (ufl.classes.BaseForm,
                             ufl.classes.Expr,
                             numbers.Complex)):
            return ufl.as_ufl(expr)
        elif isinstance(expr, Sequence):
            return ufl.as_vector(tuple(map(as_ufl, expr)))
        else:
            raise TypeError(f"Unexpected type: {type(expr)}")

    deps = []
    for c in (ufl.coefficient.BaseCoefficient, backend_Constant):
        c_deps = {}
        for dep in itertools.chain.from_iterable(map(
                lambda expr: ufl.algorithms.extract_type(as_ufl(expr), c),
                itertools.chain.from_iterable(iter_expr(as_ufl(expr))))):
            c_deps[dep.count()] = dep
        deps.extend(sorted(c_deps.values(), key=lambda dep: dep.count()))
    return deps


def extract_derivative_coefficients(expr, dep):
    dexpr = derivative(expr, dep, enable_automatic_argument=False)
    dexpr = ufl.algorithms.expand_derivatives(dexpr)
    return extract_coefficients(dexpr)


def extract_dependencies(expr, *, space_type=None):
    deps = {}
    nl_deps = {}
    for dep in extract_coefficients(expr):
        if is_var(dep):
            deps.setdefault(var_id(dep), dep)
            for nl_dep in extract_derivative_coefficients(expr, dep):
                if is_var(nl_dep):
                    nl_deps.setdefault(var_id(dep), dep)
                    nl_deps.setdefault(var_id(nl_dep), nl_dep)

    deps = {dep_id: deps[dep_id]
            for dep_id in sorted(deps.keys())}
    nl_deps = {nl_dep_id: nl_deps[nl_dep_id]
               for nl_dep_id in sorted(nl_deps.keys())}

    assert len(set(nl_deps.keys()).difference(set(deps.keys()))) == 0
    if space_type is not None:
        for dep in deps.values():
            check_space_type(dep, space_type)

    return deps, nl_deps


def with_coefficient(expr, x):
    if isinstance(x, ufl.classes.Coefficient):
        return expr, {}, {}
    else:
        x_coeff = ufl.classes.Coefficient(var_space(x))
        replace_map = {x: x_coeff}
        replace_map_inverse = {x_coeff: x}
        return ufl.replace(expr, replace_map), replace_map, replace_map_inverse


def derivative(expr, x, argument=None, *,
               enable_automatic_argument=True):
    expr_arguments = ufl.algorithms.extract_arguments(expr)
    arity = len(expr_arguments)

    if argument is None and enable_automatic_argument:
        Argument = {0: TestFunction, 1: TrialFunction}[arity]
        argument = Argument(var_space(x))

    for expr_argument in expr_arguments:
        if expr_argument.number() >= arity:
            raise ValueError("Unexpected argument")
    if argument is not None:
        for expr_argument in ufl.algorithms.extract_arguments(argument):
            if expr_argument.number() < arity - int(isinstance(x, ufl.classes.Cofunction)):  # noqa: E501
                raise ValueError("Invalid argument")

    expr, replace_map, replace_map_inverse = with_coefficient(expr, x)
    x = replace_map.get(x, x)
    if argument is not None:
        argument = ufl.replace(argument, replace_map)

    if any(isinstance(comp, ufl.classes.Action)
           for _, comp in iter_expr(expr)):
        dexpr = None
        for weight, comp in iter_expr(expr):
            if isinstance(comp, ufl.classes.Action):
                if complex_mode:
                    # See Firedrake issue #3346
                    raise NotImplementedError("Complex case not implemented")

                dcomp = ufl.algorithms.expand_derivatives(
                    ufl.derivative(ufl.as_ufl(weight), x, argument=argument))
                if not isinstance(dcomp, ufl.classes.Zero):
                    raise NotImplementedError("Weight derivatives not "
                                              "implemented")

                dcomp = ufl.algorithms.expand_derivatives(
                    ufl.derivative(comp.left(), x, argument=argument))
                dcomp = weight * ufl.classes.Action(dcomp, comp.right())
                dexpr = dcomp if dexpr is None else dexpr + dcomp

                dcomp = ufl.algorithms.expand_derivatives(
                    ufl.derivative(comp.right(), x, argument=argument))
                dcomp = weight * ufl.classes.Action(comp.left(), dcomp)
                dexpr = dcomp if dexpr is None else dexpr + dcomp
            else:
                dcomp = ufl.derivative(weight * comp, x, argument=argument)
                dexpr = dcomp if dexpr is None else dexpr + dcomp
        assert dexpr is not None
    else:
        dexpr = ufl.derivative(expr, x, argument=argument)

    dexpr = ufl.algorithms.expand_derivatives(dexpr)
    return ufl.replace(dexpr, replace_map_inverse)


def expr_zero(expr):
    if isinstance(expr, ufl.classes.BaseForm):
        return ufl.classes.ZeroBaseForm(expr.arguments())
    elif isinstance(expr, ufl.classes.Expr):
        return ufl.classes.Zero(shape=expr.ufl_shape,
                                free_indices=expr.ufl_free_indices,
                                index_dimensions=expr.ufl_index_dimensions)
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")


class Zero:
    """Mixin for defining a zero-valued variable. Used for zero-valued
    variables for which UFL zero elimination should not be applied.
    """

    def _tlm_adjoint__var_interface_update_state(self):
        raise VariableStateChangeError("Cannot call _update_state interface "
                                       "of Zero")


@form_cached("_tlm_adjoint__eliminate_zeros")
def eliminate_zeros(expr):
    """Apply zero elimination for :class:`.Zero` objects in the supplied
    :class:`ufl.core.expr.Expr` or :class:`ufl.form.BaseForm`.

    :arg expr: A :class:`ufl.core.expr.Expr` or :class:`ufl.form.BaseForm`.
    :returns: A :class:`ufl.core.expr.Expr` or :class:`ufl.form.BaseForm` with
        zero elimination applied. May return `expr`.
    """

    replace_map = {c: expr_zero(c)
                   for c in extract_coefficients(expr)
                   if isinstance(c, Zero)}
    if len(replace_map) == 0:
        simplified_expr = expr
    else:
        simplified_expr = ufl.replace(expr, replace_map)

    if isinstance(simplified_expr, ufl.classes.BaseForm):
        nonempty_expr = expr_zero(expr)
        for weight, comp in iter_expr(simplified_expr):
            if not isinstance(comp, ufl.classes.Form) or not comp.empty():
                nonempty_expr = nonempty_expr + weight * comp
        simplified_expr = nonempty_expr

    return simplified_expr


class Replacement:
    """Represents a symbolic variable but with no value.
    """


def new_count(counted_class):
    # __slots__ workaround
    class Counted(ufl.utils.counted.Counted):
        pass

    return Counted(counted_class=counted_class).count()


def replaced_form(form):
    replace_map = {}
    for c in extract_coefficients(form):
        if is_var(c) and not var_is_replacement(c):
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
                    if isinstance(eq_dep, (ufl.classes.Expr,
                                           ufl.classes.Cofunction))}

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
                if isinstance(eq_nl_dep, (ufl.classes.Expr,
                                          ufl.classes.Cofunction))}

    def _nonlinear_replace(self, expr, nl_deps):
        replace_map = self._nonlinear_replace_map(nl_deps)
        return ufl.replace(expr, replace_map)
