#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module is used by both the FEniCS and Firedrake backends, and includes
functionality for handling :class:`ufl.Coefficient` objects and boundary
conditions.
"""

from .backend import (
    TestFunction, TrialFunction, backend_Constant, backend_DirichletBC,
    backend_ScalarType)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, add_interface, comm_parent, is_var,
    space_comm, var_caches, var_comm, var_dtype, var_derivative_space, var_id,
    var_is_cached, var_is_checkpointed, var_is_replacement, var_is_static,
    var_linf_norm, var_name, var_replacement, var_scalar_value, var_space,
    var_space_type)
from ..interface import VariableInterface as _VariableInterface

from ..caches import Caches
from ..manager import manager_disabled
from ..overloaded_float import SymbolicFloat

import numpy as np
import ufl
import weakref

__all__ = \
    [
        "Constant",
        "extract_coefficients",

        "Zero",
        "ZeroConstant",
        "eliminate_zeros",

        "Replacement",
        "ReplacementConstant",
        "ReplacementFunction",

        "DirichletBC",
        "HomogeneousDirichletBC"
    ]


class ConstantSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return self._tlm_adjoint__space_interface_attrs["dtype"]

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        domain = self._tlm_adjoint__space_interface_attrs["domain"]
        return Constant(name=name, domain=domain, space=self,
                        space_type=space_type, static=static, cache=cache,
                        checkpoint=checkpoint)


class ConstantInterface(_VariableInterface):
    def _space(self):
        return self._tlm_adjoint__var_interface_attrs["space"]

    def _derivative_space(self):
        return self._tlm_adjoint__var_interface_attrs["derivative_space"](self)

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _dtype(self):
        return self._tlm_adjoint__var_interface_attrs["dtype"]

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self._tlm_adjoint__var_interface_attrs["name"](self)

    def _state(self):
        return self._tlm_adjoint__var_interface_attrs["state"][0]

    def _update_state(self):
        self._tlm_adjoint__var_interface_attrs["state"][0] += 1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__var_interface_attrs["checkpoint"]

    def _caches(self):
        if "caches" not in self._tlm_adjoint__var_interface_attrs:
            self._tlm_adjoint__var_interface_attrs["caches"] \
                = Caches(self)
        return self._tlm_adjoint__var_interface_attrs["caches"]

    @manager_disabled()
    def _zero(self):
        if len(self.ufl_shape) == 0:
            value = 0.0
        else:
            value = np.zeros(self.ufl_shape, dtype=var_dtype(self))
            value = backend_Constant(value)
        self.assign(value)

    @manager_disabled()
    def _assign(self, y):
        if isinstance(y, SymbolicFloat):
            y = y.value
        if isinstance(y, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)):
            if len(self.ufl_shape) == 0:
                value = y
            else:
                value = np.full(self.ufl_shape, y, dtype=var_dtype(self))
                value = backend_Constant(value)
        elif isinstance(y, backend_Constant):
            value = y
        else:
            raise TypeError(f"Unexpected type: {type(y)}")
        self.assign(value)

    @manager_disabled()
    def _axpy(self, alpha, x, /):
        if isinstance(x, SymbolicFloat):
            x = x.value
        if isinstance(x, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)):
            if len(self.ufl_shape) == 0:
                value = (var_scalar_value(self) + alpha * x)
            else:
                value = self.values() + alpha * x
                value.shape = self.ufl_shape
                value = backend_Constant(value)
        elif isinstance(x, backend_Constant):
            if len(self.ufl_shape) == 0:
                value = (var_scalar_value(self)
                         + alpha * var_scalar_value(x))
            else:
                value = self.values() + alpha * x.values()
                value.shape = self.ufl_shape
                value = backend_Constant(value)
        elif is_var(x):
            value = (var_scalar_value(self)
                     + alpha * var_scalar_value(x))
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
        self.assign(value)

    def _inner(self, y):
        if isinstance(y, backend_Constant):
            return y.values().conjugate().dot(self.values())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _sum(self):
        return self.values().sum()

    def _linf_norm(self):
        return abs(self.values()).max(initial=0.0)

    def _local_size(self):
        comm = var_comm(self)
        if comm.rank == 0:
            if len(self.ufl_shape) == 0:
                return 1
            else:
                return np.prod(self.ufl_shape)
        else:
            return 0

    def _global_size(self):
        if len(self.ufl_shape) == 0:
            return 1
        else:
            return np.prod(self.ufl_shape)

    def _local_indices(self):
        comm = var_comm(self)
        if comm.rank == 0:
            if len(self.ufl_shape) == 0:
                return slice(0, 1)
            else:
                return slice(0, np.prod(self.ufl_shape))
        else:
            return slice(0, 0)

    def _get_values(self):
        comm = var_comm(self)
        if comm.rank == 0:
            values = self.values().copy()
        else:
            values = np.array([], dtype=var_dtype(self))
        return values

    @manager_disabled()
    def _set_values(self, values):
        comm = var_comm(self)
        if comm.rank != 0:
            values = None
        values = comm.bcast(values, root=0)
        if len(self.ufl_shape) == 0:
            values.shape = (1,)
            self.assign(values[0])
        else:
            values.shape = self.ufl_shape
            self.assign(backend_Constant(values))

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementConstant(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return len(self.ufl_shape) == 0

    def _scalar_value(self):
        # assert var_is_scalar(self)
        return var_dtype(self)(self)

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__var_interface_attrs


def constant_value(value=None, shape=None):
    if value is None:
        if shape is None:
            shape = ()
    elif shape is not None:
        value_ = value
        if not isinstance(value_, np.ndarray):
            value_ = np.array(value_)
        if value_.shape != shape:
            raise ValueError("Invalid shape")
        del value_

    # Default value
    if value is None:
        if len(shape) == 0:
            value = 0.0
        else:
            value = np.zeros(shape, dtype=backend_ScalarType)

    return value


class Constant(backend_Constant):
    """Extends the backend `Constant` class.

    :arg value: The initial value. `None` indicates a value of zero.
    :arg name: A :class:`str` name.
    :arg domain: The domain on which the :class:`Constant` is defined.
    :arg space: The space on which the :class:`Constant` is defined.
    :arg space_type: The space type for the :class:`Constant`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg shape: A :class:`tuple` of :class:`int` objects defining the shape of
        the value.
    :arg comm: The communicator for the :class:`Constant`.
    :arg static: Defines the default value for `cache` and `checkpoint`.
    :arg cache: Defines whether results involving this :class:`Constant` may be
        cached. Default `static`.
    :arg checkpoint: Defines whether a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage` should store this
        :class:`Constant` by value (`checkpoint=True`) or reference
        (`checkpoint=False`). Default `not static`.

    Remaining arguments are passed to the backend `Constant` constructor.
    """

    def __init__(self, value=None, *args, name=None, domain=None, space=None,
                 space_type="primal", shape=None, comm=None, static=False,
                 cache=None, checkpoint=None, **kwargs):
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")

        if domain is None and space is not None:
            domains = space.ufl_domains()
            if len(domains) > 0:
                domain, = domains
            del domains

        # Shape initialization / checking
        if space is not None:
            if shape is None:
                shape = space.ufl_element().value_shape()
            elif shape != space.ufl_element().value_shape():
                raise ValueError("Invalid shape")

        value = constant_value(value, shape)

        # Default comm
        if comm is None:
            if space is None:
                comm = DEFAULT_COMM
            else:
                comm = comm_parent(space_comm(space))

        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static

        super().__init__(
            value, *args, name=name, domain=domain, space=space,
            comm=comm, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)
        self._tlm_adjoint__var_interface_attrs.d_setitem("checkpoint", checkpoint)  # noqa: E501

    def __new__(cls, value=None, *args, name=None, domain=None,
                space_type="primal", shape=None, static=False, cache=None,
                checkpoint=None, **kwargs):
        if issubclass(cls, ufl.classes.Coefficient) or domain is None:
            return object().__new__(cls)
        else:
            # For Firedrake
            value = constant_value(value, shape)
            if space_type not in {"primal", "conjugate",
                                  "dual", "conjugate_dual"}:
                raise ValueError("Invalid space type")
            if cache is None:
                cache = static
            if checkpoint is None:
                checkpoint = not static
            F = super().__new__(cls, value, domain=domain)
            F.rename(name=name)
            F._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
            F._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
            F._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)
            F._tlm_adjoint__var_interface_attrs.d_setitem("checkpoint", checkpoint)  # noqa: E501
            return F


class Zero:
    """Mixin for defining a zero-valued variable. Used for zero-valued
    variables for which UFL zero elimination should not be applied.
    """

    def _tlm_adjoint__var_interface_assign(self, y):
        raise RuntimeError("Cannot call _assign interface of Zero")

    def _tlm_adjoint__var_interface_axpy(self, alpha, x, /):
        raise RuntimeError("Cannot call _axpy interface of Zero")

    def _tlm_adjoint__var_interface_set_values(self, values):
        raise RuntimeError("Cannot call _set_values interface of Zero")

    def _tlm_adjoint__var_interface_update_state(self):
        raise RuntimeError("Cannot call _update_state interface of Zero")


class ZeroConstant(Constant, Zero):
    """A :class:`Constant` which is flagged as having a value of zero.

    Arguments are passed to the :class:`Constant` constructor, together with
    `static=True`, `cache=True`, and `checkpoint=False`.
    """

    def __init__(self, *, name=None, domain=None, space=None,
                 space_type="primal", shape=None, comm=None):
        Constant.__init__(
            self, name=name, domain=domain, space=space, space_type=space_type,
            shape=shape, comm=comm, static=True, cache=True, checkpoint=False)
        if var_linf_norm(self) != 0.0:
            raise RuntimeError("ZeroConstant is not zero-valued")

    def __new__(cls, *args, shape=None, **kwargs):
        return Constant.__new__(
            cls, constant_value(shape=shape), *args,
            shape=shape, static=True, cache=True, checkpoint=False, **kwargs)

    def assign(self, *args, **kwargs):
        raise RuntimeError("Cannot call assign method of ZeroConstant")


def as_coefficient(x):
    if isinstance(x, ufl.classes.Coefficient):
        return x

    # For Firedrake

    if not isinstance(x, backend_Constant):
        raise TypeError("Unexpected type")

    if not hasattr(x, "_tlm_adjoint__Coefficient"):
        if is_var(x):
            space = var_space(x)
        else:
            if len(x.ufl_shape) == 0:
                element = ufl.classes.FiniteElement("R", None, 0)
            elif len(x.ufl_shape) == 1:
                element = ufl.classes.VectorElement("R", None, 0,
                                                    dim=x.ufl_shape[0])
            else:
                element = ufl.classes.TensorElement("R", None, 0,
                                                    shape=x.ufl_shape)
            space = ufl.classes.FunctionSpace(None, element)

        x._tlm_adjoint__Coefficient = ufl.classes.Coefficient(space)

    return x._tlm_adjoint__Coefficient


def with_coefficient(expr, x):
    x_coeff = as_coefficient(x)
    if x_coeff is x:
        return expr, {}, {}
    else:
        # For Firedrake
        replace_map = {x: x_coeff}
        replace_map_inverse = {x_coeff: x}
        return ufl.replace(expr, replace_map), replace_map, replace_map_inverse


def with_coefficients(expr):
    if isinstance(expr, ufl.classes.Form) \
            and "_tlm_adjoint__form_with_coefficients" in expr._cache:
        return expr._cache["_tlm_adjoint__form_with_coefficients"]

    if issubclass(backend_Constant, ufl.classes.Coefficient):
        replace_map = {}
    else:
        # For Firedrake
        constants = tuple(sorted(ufl.algorithms.extract_type(expr, backend_Constant),  # noqa: E501
                          key=lambda c: c.count()))
        replace_map = dict(zip(constants, map(as_coefficient, constants)))
    replace_map_inverse = {c_coeff: c
                           for c, c_coeff in replace_map.items()}

    expr_with_coeffs = ufl.replace(expr, replace_map)
    if isinstance(expr, ufl.classes.Form):
        expr._cache["_tlm_adjoint__form_with_coefficients"] = \
            (expr_with_coeffs, replace_map, replace_map_inverse)
    return expr_with_coeffs, replace_map, replace_map_inverse


def extract_coefficients(expr):
    """
    :returns: Variables on which the supplied :class:`ufl.core.expr.Expr` or
        :class:`ufl.Form` depends.
    """

    if isinstance(expr, ufl.classes.Form) \
            and "_tlm_adjoint__form_coefficients" in expr._cache:
        return expr._cache["_tlm_adjoint__form_coefficients"]

    if issubclass(backend_Constant, ufl.classes.Coefficient):
        cls = (ufl.classes.Coefficient,)
    else:
        # For Firedrake
        cls = (ufl.classes.Coefficient, backend_Constant)
    deps = []
    for c in cls:
        deps.extend(sorted(ufl.algorithms.extract_type(expr, c),
                           key=lambda dep: dep.count()))

    if isinstance(expr, ufl.classes.Form):
        expr._cache["_tlm_adjoint__form_coefficients"] = deps
    return deps


def diff(expr, x):
    expr, replace_map, replace_map_inverse = with_coefficient(expr, x)
    dexpr = ufl.diff(expr, replace_map.get(x, x))
    dexpr = ufl.algorithms.expand_derivatives(dexpr)
    return ufl.replace(dexpr, replace_map_inverse)


def derivative(expr, x, argument=None, *,
               enable_automatic_argument=True):
    expr_arguments = ufl.algorithms.extract_arguments(expr)
    arity = len(expr_arguments)

    if argument is None and enable_automatic_argument:
        Argument = {0: TestFunction, 1: TrialFunction}[arity]
        argument = Argument(var_derivative_space(x))

    for expr_argument in expr_arguments:
        if expr_argument.number() >= arity:
            raise ValueError("Unexpected argument")
    if isinstance(argument, ufl.classes.Argument) and argument.number() < arity:  # noqa: E501
        raise ValueError("Invalid argument")

    if isinstance(expr, ufl.classes.Expr):
        expr, replace_map, replace_map_inverse = with_coefficient(expr, x)
    else:
        expr, replace_map, replace_map_inverse = with_coefficients(expr)

    if argument is not None:
        argument, _, argument_replace_map_inverse = with_coefficients(argument)
        replace_map_inverse.update(argument_replace_map_inverse)

    dexpr = ufl.derivative(expr, replace_map.get(x, x), argument=argument)
    return ufl.replace(dexpr, replace_map_inverse)


def eliminate_zeros(expr, *, force_non_empty_form=False):
    """Apply zero elimination for
    :class:`tlm_adjoint._code_generator.functions.Zero` objects in the supplied
    :class:`ufl.core.expr.Expr` or :class:`ufl.Form`.

    :arg expr: A :class:`ufl.core.expr.Expr` or :class:`ufl.Form`.
    :arg force_non_empty_form: If `True` and if `expr` is a :class:`ufl.Form`,
        then the returned form is guaranteed to be non-empty, and may be
        assembled.
    :returns: A :class:`ufl.core.expr.Expr` or :class:`ufl.Form` with zero
        elimination applied. May return `expr`.
    """

    if isinstance(expr, ufl.classes.Form) \
            and "_tlm_adjoint__simplified_form" in expr._cache:
        simplified_expr = expr._cache["_tlm_adjoint__simplified_form"]
    else:
        replace_map = {}
        for c in extract_coefficients(expr):
            if isinstance(c, Zero):
                replace_map[c] = ufl.classes.Zero(shape=c.ufl_shape)

        if len(replace_map) == 0:
            simplified_expr = expr
        else:
            simplified_expr = ufl.replace(expr, replace_map)

        if isinstance(expr, ufl.classes.Form):
            expr._cache["_tlm_adjoint__simplified_form"] = simplified_expr

    if force_non_empty_form \
            and isinstance(simplified_expr, ufl.classes.Form) \
            and simplified_expr.empty():
        if "_tlm_adjoint__simplified_form_non_empty" in expr._cache:
            simplified_expr = expr._cache["_tlm_adjoint__simplified_form_non_empty"]  # noqa: E501
        else:
            # Inefficient, but it is very difficult to generate a non-empty but
            # zero valued form
            arguments = expr.arguments()
            zero = ZeroConstant()
            if len(arguments) == 0:
                domain, = expr.ufl_domains()
                simplified_expr = zero * ufl.ds(domain)
            elif len(arguments) == 1:
                test, = arguments
                simplified_expr = ufl.inner(zero, test[tuple(0 for _ in test.ufl_shape)]) * ufl.ds  # noqa: E501
            else:
                test, trial = arguments
                simplified_expr = zero * ufl.inner(trial[tuple(0 for _ in trial.ufl_shape)],  # noqa: E501
                                                   test[tuple(0 for _ in test.ufl_shape)]) * ufl.ds  # noqa: E501

            if isinstance(expr, ufl.classes.Form):
                expr._cache["_tlm_adjoint__simplified_form_non_empty"] = simplified_expr  # noqa: E501

    return simplified_expr


class DirichletBC(backend_DirichletBC):
    """Extends the backend `DirichletBC`.

    :arg static: A flag that indicates that the value for this
        :class:`DirichletBC` will not change, and which determines whether
        calculations involving this :class:`DirichletBC` can be cached. If
        `None` then autodetected from the value.

    Remaining arguments are passed to the backend `DirichletBC` constructor.
    """

    # Based on FEniCS 2019.1.0 DirichletBC API
    def __init__(self, V, g, sub_domain, *args,
                 static=None, _homogeneous=False, **kwargs):
        super().__init__(V, g, sub_domain, *args, **kwargs)

        if static is None:
            for dep in extract_coefficients(
                    g if isinstance(g, ufl.classes.Expr)
                    else Constant(g, static=True)):
                # The 'static' flag for variables is only a hint. 'not
                # checkpointed' is a guarantee that the variable will never
                # appear as the solution to an Equation.
                if not is_var(dep) or var_is_checkpointed(dep):
                    static = False
                    break
            else:
                static = True

        self._tlm_adjoint__static = static
        self._tlm_adjoint__cache = static
        self._tlm_adjoint__homogeneous = _homogeneous

    def homogenize(self):
        """Homogenize the :class:`DirichletBC`, setting its value to zero.
        """

        if self._tlm_adjoint__static:
            raise RuntimeError("Cannot call homogenize method for static "
                               "DirichletBC")
        if not self._tlm_adjoint__homogeneous:
            super().homogenize()
            self._tlm_adjoint__homogeneous = True

    def set_value(self, *args, **kwargs):
        """Set the :class:`DirichletBC` value.

        Arguments are passed to the base class `set_value` method.
        """

        if self._tlm_adjoint__static:
            raise RuntimeError("Cannot call set_value method for static "
                               "DirichletBC")
        super().set_value(*args, **kwargs)


class HomogeneousDirichletBC(DirichletBC):
    """A :class:`DirichletBC` whose value is zero.

    Arguments are passed to the :class:`DirichletBC` constructor, together with
    `static=True`.
    """

    # Based on FEniCS 2019.1.0 DirichletBC API
    def __init__(self, V, sub_domain, *args, **kwargs):
        shape = V.ufl_element().value_shape()
        if len(shape) == 0:
            g = 0.0
        else:
            g = np.zeros(shape, dtype=backend_ScalarType)
        super().__init__(V, g, sub_domain, *args, static=True,
                         _homogeneous=True, **kwargs)


def bcs_is_static(bcs):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    for bc in bcs:
        if not getattr(bc, "_tlm_adjoint__static", False):
            return False
    return True


def bcs_is_cached(bcs):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    for bc in bcs:
        if not getattr(bc, "_tlm_adjoint__cache", False):
            return False
    return True


def bcs_is_homogeneous(bcs):
    if isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    for bc in bcs:
        if not getattr(bc, "_tlm_adjoint__homogeneous", False):
            return False
    return True


class ReplacementInterface(_VariableInterface):
    def _space(self):
        return self.ufl_function_space()

    def _derivative_space(self):
        return self._tlm_adjoint__var_interface_attrs.get(
            "derivative_space", lambda x: var_space(x))(self)

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self._tlm_adjoint__var_interface_attrs["name"]

    def _state(self):
        return -1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__var_interface_attrs["checkpoint"]

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _replacement(self):
        return self

    def _is_replacement(self):
        return True


class Replacement(ufl.classes.Coefficient):
    """A :class:`ufl.Coefficient` representing a symbolic variable but with no
    value.
    """

    def __init__(self, x):
        space = var_space(x)

        x_domains = x.ufl_domains()
        if len(x_domains) == 0:
            domain = None
        else:
            domain, = x_domains

        super().__init__(space, count=x.count())
        self._tlm_adjoint__domain = domain
        add_interface(self, ReplacementInterface,
                      {"id": var_id(x), "name": var_name(x),
                       "space": space,
                       "space_type": var_space_type(x),
                       "static": var_is_static(x),
                       "cache": var_is_cached(x),
                       "checkpoint": var_is_checkpointed(x),
                       "caches": var_caches(x)})

    def ufl_domain(self):
        return self._tlm_adjoint__domain

    def ufl_domains(self):
        if self._tlm_adjoint__domain is None:
            return ()
        else:
            return (self._tlm_adjoint__domain,)


class ReplacementConstant(Replacement):
    """Represents a symbolic constant, but has no value.
    """

    def __init__(self, x):
        super().__init__(x)
        self._tlm_adjoint__var_interface_attrs["derivative_space"] \
            = x._tlm_adjoint__var_interface_attrs["derivative_space"]


class ReplacementFunction(Replacement):
    """Represents a symbolic backend `Function`, but has no value.
    """

    def function_space(self):
        return var_space(self)


def replaced_form(form):
    replace_map = {}
    for c in extract_coefficients(form):
        if is_var(c) and not var_is_replacement(c):
            c_rep = var_replacement(c)
            if c_rep is not c:
                replace_map[c] = c_rep
    return ufl.replace(form, replace_map)


def define_var_alias(x, parent, *, key):
    if x is not parent:
        if "alias" in x._tlm_adjoint__var_interface_attrs:
            alias_parent, alias_key = x._tlm_adjoint__var_interface_attrs["alias"]  # noqa: E501
            alias_parent = alias_parent()
            if alias_parent is None or alias_parent is not parent \
                    or alias_key != key:
                raise ValueError("Invalid alias data")
        else:
            x._tlm_adjoint__var_interface_attrs["alias"] \
                = (weakref.ref(parent), key)
            x._tlm_adjoint__var_interface_attrs.d_setitem(
                "space_type", var_space_type(parent))
            x._tlm_adjoint__var_interface_attrs.d_setitem(
                "static", var_is_static(parent))
            x._tlm_adjoint__var_interface_attrs.d_setitem(
                "cache", var_is_cached(parent))
            x._tlm_adjoint__var_interface_attrs.d_setitem(
                "checkpoint", var_is_checkpointed(parent))
            x._tlm_adjoint__var_interface_attrs.d_setitem(
                "state", parent._tlm_adjoint__var_interface_attrs["state"])
