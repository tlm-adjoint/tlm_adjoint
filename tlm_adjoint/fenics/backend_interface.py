#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    as_backend_type, backend_Constant, backend_Function, backend_FunctionSpace,
    backend_ScalarType, backend_Vector, cpp_PETScVector)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, add_interface, check_space_type,
    check_space_types, comm_dup_cached, function_copy, function_dtype,
    function_new, function_scalar_value, function_space, function_space_type,
    new_function_id, new_space_id, register_finalize_adjoint_derivative_action,
    register_functional_term_eq, register_subtract_adjoint_derivative_action,
    space_id, subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base)
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import (
    assemble, is_valid_r0_space, r0_space)

from .equations import Assembly
from .functions import (
    Caches, Constant, ConstantInterface, ConstantSpaceInterface, Function,
    ReplacementFunction, Zero, define_function_alias)
from ..override import override_method

from ..manager import manager_disabled
from ..overloaded_float import SymbolicFloat

import functools
import numpy as np
import ufl
import warnings

__all__ = \
    [
        "new_scalar_function"
    ]


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, *args, domain=None, space=None,
                     comm=None, **kwargs):
    if domain is not None and hasattr(domain, "ufl_domain"):
        domain = domain.ufl_domain()
    if comm is None:
        comm = DEFAULT_COMM

    orig(self, *args, **kwargs)

    self.ufl_domain = lambda: domain
    if domain is None:
        self.ufl_domains = lambda: ()
    else:
        self.ufl_domains = lambda: (domain,)

    if space is None:
        space = self.ufl_function_space()
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_function_id(), "name": lambda x: x.name(),
                   "state": 0, "space": space,
                   "form_derivative_space": lambda x: r0_space(x),
                   "space_type": "primal", "dtype": self.values().dtype.type,
                   "static": False, "cache": False, "checkpoint": True})


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        return Function(self, name=name, space_type=space_type, static=static,
                        cache=cache, checkpoint=checkpoint)


_FunctionSpace_add_interface = True


def FunctionSpace_add_interface_disabled(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        global _FunctionSpace_add_interface
        add_interface = _FunctionSpace_add_interface
        _FunctionSpace_add_interface = False
        try:
            return fn(*args, **kwargs)
        finally:
            _FunctionSpace_add_interface = add_interface
    return wrapped_fn


@override_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    if _FunctionSpace_add_interface:
        add_interface(self, FunctionSpaceInterface,
                      {"comm": comm_dup_cached(self.mesh().mpi_comm()),
                       "id": new_space_id()})


def check_vector_size(fn):
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.vector().size() != self.function_space().dofmap().global_dimension():  # noqa: E501
            raise RuntimeError("Unexpected vector size")
        return fn(self, *args, **kwargs)
    return wrapped_fn


class FunctionInterface(_FunctionInterface):
    def _space(self):
        return self._tlm_adjoint__function_interface_attrs["space"]

    def _form_derivative_space(self):
        return function_space(self)

    def _space_type(self):
        return self._tlm_adjoint__function_interface_attrs["space_type"]

    def _id(self):
        return self._tlm_adjoint__function_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        return self._tlm_adjoint__function_interface_attrs["state"]

    def _update_state(self):
        state = self._tlm_adjoint__function_interface_attrs["state"]
        self._tlm_adjoint__function_interface_attrs.d_setitem("state", state + 1)  # noqa: E501

    def _is_static(self):
        return self._tlm_adjoint__function_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__function_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__function_interface_attrs["checkpoint"]

    def _caches(self):
        if "caches" not in self._tlm_adjoint__function_interface_attrs:
            self._tlm_adjoint__function_interface_attrs["caches"] \
                = Caches(self)
        return self._tlm_adjoint__function_interface_attrs["caches"]

    @check_vector_size
    def _zero(self):
        self.vector().zero()

    @manager_disabled()
    @check_vector_size
    def _assign(self, y):
        if isinstance(y, SymbolicFloat):
            y = y.value()
        if isinstance(y, backend_Function):
            if self.vector().local_size() != y.vector().local_size():
                raise ValueError("Invalid function space")
            self.vector().zero()
            self.vector().axpy(1.0, y.vector())
        elif isinstance(y, (int, np.integer, float, np.floating)):
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(y))
            else:
                y_arr = np.full(self.ufl_shape, y)
                self.assign(backend_Constant(y_arr))
        elif isinstance(y, Zero):
            self.vector().zero()
        elif isinstance(y, backend_Constant):
            self.assign(y)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    @manager_disabled()
    @check_vector_size
    def _axpy(self, alpha, x, /):
        if isinstance(x, SymbolicFloat):
            x = x.value()
        if isinstance(x, backend_Function):
            if self.vector().local_size() != x.vector().local_size():
                raise ValueError("Invalid function space")
            self.vector().axpy(alpha, x.vector())
        elif isinstance(x, (int, np.integer, float, np.floating)):
            x_ = function_new(self)
            if len(self.ufl_shape) == 0:
                x_.assign(backend_Constant(x))
            else:
                x_arr = np.full(self.ufl_shape, x)
                x_.assign(backend_Constant(x_arr))
            self.vector().axpy(alpha, x_.vector())
        elif isinstance(x, Zero):
            pass
        elif isinstance(x, backend_Constant):
            x_ = backend_Function(self.function_space())
            x_.assign(x)
            self.vector().axpy(alpha, x_.vector())
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    @manager_disabled()
    @check_vector_size
    def _inner(self, y):
        if isinstance(y, backend_Function):
            if self.vector().local_size() != y.vector().local_size():
                raise ValueError("Invalid function space")
            inner = y.vector().inner(self.vector())
        elif isinstance(y, Zero):
            inner = 0.0
        elif isinstance(y, backend_Constant):
            y_ = backend_Function(self.function_space())
            y_.assign(y)
            inner = y_.vector().inner(self.vector())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")
        return inner

    @check_vector_size
    def _sum(self):
        return self.vector().sum()

    @check_vector_size
    def _linf_norm(self):
        return self.vector().norm("linf")

    @check_vector_size
    def _local_size(self):
        return self.vector().local_size()

    @check_vector_size
    def _global_size(self):
        return self.function_space().dofmap().global_dimension()

    @check_vector_size
    def _local_indices(self):
        return slice(*self.function_space().dofmap().ownership_range())

    @check_vector_size
    def _get_values(self):
        return self.vector().get_local().copy()  # copy likely not required

    @check_vector_size
    def _set_values(self, values):
        if not np.can_cast(values, function_dtype(self)):
            raise ValueError("Invalid dtype")
        if values.shape != (self.vector().local_size(),):
            raise ValueError("Invalid shape")
        self.vector().set_local(values)
        self.vector().apply("insert")

    @check_vector_size
    def _new(self, *, name=None, static=False, cache=None, checkpoint=None,
             rel_space_type="primal"):
        y = function_copy(self, name=name, static=static, cache=cache,
                          checkpoint=checkpoint)
        y.vector().zero()
        space_type = function_space_type(self, rel_space_type=rel_space_type)
        y._tlm_adjoint__function_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        return y

    @manager_disabled()
    @check_vector_size
    def _copy(self, *, name=None, static=False, cache=None, checkpoint=None):
        y = self.copy(deepcopy=True)
        if name is not None:
            y.rename(name, "a Function")
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static
        y._tlm_adjoint__function_interface_attrs.d_setitem("space_type", function_space_type(self))  # noqa: E501
        y._tlm_adjoint__function_interface_attrs.d_setitem("static", static)
        y._tlm_adjoint__function_interface_attrs.d_setitem("cache", cache)
        y._tlm_adjoint__function_interface_attrs.d_setitem("checkpoint", checkpoint)  # noqa: E501
        return y

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementFunction(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return (is_valid_r0_space(self.function_space())
                and len(self.ufl_shape) == 0)

    @check_vector_size
    def _scalar_value(self):
        # assert function_is_scalar(self)
        return self.vector().max()

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__function_interface_attrs


@override_method(backend_Function, "__init__")
@FunctionSpace_add_interface_disabled
def Function__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    self.vector()

    if not isinstance(as_backend_type(self.vector()), cpp_PETScVector):
        raise RuntimeError("PETSc backend required")

    add_interface(self, FunctionInterface,
                  {"id": new_function_id(), "state": 0, "space_type": "primal",
                   "static": False, "cache": False, "checkpoint": True})

    space = self.function_space()
    if isinstance(args[0], backend_FunctionSpace) \
            and args[0].id() == space.id():
        id = space_id(args[0])
    else:
        id = new_space_id()
    add_interface(space, FunctionSpaceInterface,
                  {"comm": comm_dup_cached(space.mesh().mpi_comm()), "id": id})
    self._tlm_adjoint__function_interface_attrs["space"] = space


# Aim for compatibility with FEniCS 2019.1.0 API
@override_method(backend_Function, "function_space")
def Function_function_space(self, orig, orig_args):
    if hasattr(self, "_tlm_adjoint__function_interface_attrs") \
            and "space" in self._tlm_adjoint__function_interface_attrs:
        return self._tlm_adjoint__function_interface_attrs["space"]
    else:
        return orig_args()


# Aim for compatibility with FEniCS 2019.1.0 API
@override_method(backend_Function, "split")
def Function_split(self, orig, orig_args, deepcopy=False):
    Y = orig_args()
    if not deepcopy:
        for i, y in enumerate(Y):
            define_function_alias(y, self, key=("split", i))
    return Y


def subtract_adjoint_derivative_action_backend_constant_vector(x, alpha, y):
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_types(x, y._tlm_adjoint__function)

    if len(x.ufl_shape) == 0:
        x.assign(function_scalar_value(x) - alpha * y.max())
    else:
        value = x.values()
        y_fn = backend_Function(r0_space(x))
        y_fn.vector().axpy(1.0, y)
        for i, y_fn_c in enumerate(y_fn.split(deepcopy=True)):
            value[i] -= alpha * y_fn_c.vector().max()
        value.shape = x.ufl_shape
        x.assign(backend_Constant(value))


def subtract_adjoint_derivative_action_backend_function_vector(x, alpha, y):
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_types(x, y._tlm_adjoint__function)

    if x.vector().local_size() != y.local_size():
        raise ValueError("Invalid function space")
    x.vector().axpy(-alpha, y)


def subtract_adjoint_derivative_action_function_form(x, alpha, y):
    check_space_type(x, "conjugate_dual")
    if alpha != 1.0:
        y = backend_Constant(alpha) * y
    if hasattr(x, "_tlm_adjoint__fenics_adj_b"):
        x._tlm_adjoint__fenics_adj_b = x._tlm_adjoint__fenics_adj_b - y
    else:
        x._tlm_adjoint__fenics_adj_b = -y


register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Function), object,
    subtract_adjoint_derivative_action_base,
    replace=True)
register_subtract_adjoint_derivative_action(
    backend_Constant, backend_Vector,
    subtract_adjoint_derivative_action_backend_constant_vector)
register_subtract_adjoint_derivative_action(
    backend_Function, backend_Vector,
    subtract_adjoint_derivative_action_backend_function_vector)
register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Function), ufl.classes.Form,
    subtract_adjoint_derivative_action_function_form)


def finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__fenics_adj_b"):
        y = assemble(x._tlm_adjoint__fenics_adj_b)
        subtract_adjoint_derivative_action(x, (-1.0, y))
        delattr(x, "_tlm_adjoint__fenics_adj_b")


register_finalize_adjoint_derivative_action(finalize_adjoint_derivative_action)


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    (SymbolicFloat, backend_Constant, backend_Function), ufl.classes.Form,
    functional_term_eq_form)


def new_scalar_function(*, name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    warnings.warn("new_scalar_function is deprecated -- "
                  "use Float instead",
                  DeprecationWarning, stacklevel=2)
    return Constant(0.0, name=name, comm=comm, static=static, cache=cache,
                    checkpoint=checkpoint)
