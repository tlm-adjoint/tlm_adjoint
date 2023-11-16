#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    as_backend_type, backend_Constant, backend_Function, backend_FunctionSpace,
    backend_ScalarType, backend_Vector, cpp_PETScVector)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, VariableInterface, add_interface,
    check_space_types, comm_dup_cached, new_space_id, new_var_id,
    register_functional_term_eq, register_subtract_adjoint_derivative_action,
    space_id, subtract_adjoint_derivative_action_base, var_copy, var_linf_norm,
    var_lock_state, var_scalar_value, var_space, var_space_type)
from .backend_code_generator_interface import r0_space

from ..equations import Conversion
from ..manager import manager_disabled
from ..override import override_method

from .equations import Assembly
from .functions import (
    Caches, ConstantInterface, ConstantSpaceInterface, ReplacementFunction,
    Zero, define_var_alias)

import functools
import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "Function",

        "ZeroFunction",

        "to_fenics"
    ]


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, *args, domain=None, space=None,
                     comm=None, **kwargs):
    if domain is not None and hasattr(domain, "ufl_domain"):
        domain = domain.ufl_domain()
    if comm is None:
        comm = DEFAULT_COMM

    orig(self, *args, **kwargs)

    if space is None:
        space = self.ufl_function_space()
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_var_id(), "name": lambda x: x.name(),
                   "state": [0], "space": space,
                   "derivative_space": lambda x: r0_space(x),
                   "space_type": "primal", "dtype": self.values().dtype.type,
                   "static": False, "cache": False})


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        return Function(self, name=name, space_type=space_type, static=static,
                        cache=cache)


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


def check_vector(fn):
    @functools.wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        def space_sizes(self):
            dofmap = self.function_space().dofmap()
            n0, n1 = dofmap.ownership_range()
            return (n1 - n0, dofmap.global_dimension())

        def vector_sizes(self):
            return (self.vector().local_size(), self.vector().size())

        space = self.function_space()
        vector = self.vector()
        if vector_sizes(self) != space_sizes(self):
            raise RuntimeError("Unexpected vector size")

        return_value = fn(self, *args, **kwargs)

        if self.function_space() is not space:
            raise RuntimeError("Unexpected space")
        if self.vector() is not vector:
            raise RuntimeError("Unexpected vector")
        if vector_sizes(self) != space_sizes(self):
            raise RuntimeError("Unexpected vector size")

        return return_value
    return wrapped_fn


class FunctionInterface(VariableInterface):
    def _space(self):
        return self._tlm_adjoint__var_interface_attrs["space"]

    def _derivative_space(self):
        return var_space(self)

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        return self._tlm_adjoint__var_interface_attrs["state"][0]

    def _update_state(self):
        self._tlm_adjoint__var_interface_attrs["state"][0] += 1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _caches(self):
        if "caches" not in self._tlm_adjoint__var_interface_attrs:
            self._tlm_adjoint__var_interface_attrs["caches"] = Caches(self)
        return self._tlm_adjoint__var_interface_attrs["caches"]

    @check_vector
    def _zero(self):
        self.vector().zero()

    @manager_disabled()
    @check_vector
    def _assign(self, y):
        if isinstance(y, (int, np.integer,
                          float, np.floating)):
            if len(self.ufl_shape) != 0:
                raise ValueError("Invalid shape")
            self.assign(backend_Constant(y))
        elif isinstance(y, backend_Function):
            if space_id(y.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            self.vector().zero()
            self.vector().axpy(1.0, y.vector())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    @check_vector
    def _axpy(self, alpha, x, /):
        if isinstance(x, backend_Function):
            if space_id(x.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            self.vector().axpy(alpha, x.vector())
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    @check_vector
    def _inner(self, y):
        if isinstance(y, backend_Function):
            if space_id(y.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            return y.vector().inner(self.vector())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    @check_vector
    def _linf_norm(self):
        return self.vector().norm("linf")

    @check_vector
    def _local_size(self):
        return self.vector().local_size()

    @check_vector
    def _global_size(self):
        return self.function_space().dofmap().global_dimension()

    @check_vector
    def _local_indices(self):
        return slice(*self.function_space().dofmap().ownership_range())

    @check_vector
    def _get_values(self):
        return self.vector().get_local().copy()  # copy likely not required

    @check_vector
    def _set_values(self, values):
        self.vector().set_local(values)
        self.vector().apply("insert")

    @check_vector
    def _new(self, *, name=None, static=False, cache=None,
             rel_space_type="primal"):
        y = var_copy(self, name=name, static=static, cache=cache)
        y.vector().zero()
        space_type = var_space_type(self, rel_space_type=rel_space_type)
        y._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        return y

    @manager_disabled()
    @check_vector
    def _copy(self, *, name=None, static=False, cache=None):
        y = self.copy(deepcopy=True)
        if name is not None:
            y.rename(name, "a Function")
        if cache is None:
            cache = static
        y._tlm_adjoint__var_interface_attrs.d_setitem("space_type", var_space_type(self))  # noqa: E501
        y._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        y._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)
        return y

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementFunction(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return False

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__var_interface_attrs


class Function(backend_Function):
    """Extends the DOLFIN `Function` class.

    :arg space_type: The space type for the :class:`.Function`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines whether the :class:`.Function` is static, meaning that
        it is stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.Function` may be
        cached. Default `static`.

    Remaining arguments are passed to the DOLFIN `Function` constructor.
    """

    def __init__(self, *args, space_type="primal", static=False, cache=None,
                 **kwargs):
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if cache is None:
            cache = static

        super().__init__(*args, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)


class ZeroFunction(Function, Zero):
    """A :class:`.Function` which is flagged as having a value of zero.

    Arguments are passed to the :class:`.Function` constructor, together with
    `static=True` and `cache=True`.
    """

    def __init__(self, *args, **kwargs):
        Function.__init__(
            self, *args, **kwargs,
            static=True, cache=True)
        var_lock_state(self)
        if var_linf_norm(self) != 0.0:
            raise RuntimeError("ZeroFunction is not zero-valued")


# Aim for compatibility with FEniCS 2019.1.0 API


@override_method(backend_Function, "__init__")
@FunctionSpace_add_interface_disabled
def Function__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    # Creates a reference to the vector so that unexpected changes can be
    # detected
    self.vector()

    if not isinstance(as_backend_type(self.vector()), cpp_PETScVector):
        raise RuntimeError("PETSc backend required")

    add_interface(self, FunctionInterface,
                  {"id": new_var_id(), "state": [0],
                   "space_type": "primal", "static": False, "cache": False})

    space = self.function_space()
    if isinstance(args[0], backend_FunctionSpace) and args[0].id() == space.id():  # noqa: E501
        id = space_id(args[0])
    else:
        id = new_space_id()
    add_interface(space, FunctionSpaceInterface,
                  {"comm": comm_dup_cached(space.mesh().mpi_comm()), "id": id})
    self._tlm_adjoint__var_interface_attrs["space"] = space


@override_method(backend_Function, "function_space")
def Function_function_space(self, orig, orig_args):
    if hasattr(self, "_tlm_adjoint__var_interface_attrs") \
            and "space" in self._tlm_adjoint__var_interface_attrs:
        return self._tlm_adjoint__var_interface_attrs["space"]
    else:
        return orig_args()


@override_method(backend_Function, "split")
def Function_split(self, orig, orig_args, deepcopy=False):
    Y = orig_args()
    if not deepcopy:
        for i, y in enumerate(Y):
            define_var_alias(y, self, key=("split", i))
    return Y


def subtract_adjoint_derivative_action_backend_constant_vector(x, alpha, y):
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_types(x, y._tlm_adjoint__function)

    if len(x.ufl_shape) == 0:
        x.assign(var_scalar_value(x) - alpha * y.max())
    else:
        value = x.values()
        y_fn = backend_Function(r0_space(x))
        y_fn.vector().axpy(1.0, y)
        for i, y_fn_c in enumerate(y_fn.split(deepcopy=True)):
            value[i] -= alpha * y_fn_c.vector().max()
        value.shape = x.ufl_shape
        x.assign(backend_Constant(value))


def to_fenics(y, space, *, name=None):
    """Convert a variable to a DOLFIN `Function`.

    :arg y: A variable.
    :arg space: The space for the return value.
    :arg name: A :class:`str` name.
    :returns: The DOLFIN `Function`.
    """

    x = Function(space, space_type=var_space_type(y), name=name)
    Conversion(x, y).solve()
    return x


def subtract_adjoint_derivative_action_backend_function_vector(x, alpha, y):
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_types(x, y._tlm_adjoint__function)

    if x.vector().local_size() != y.local_size():
        raise ValueError("Invalid function space")
    x.vector().axpy(-alpha, y)


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


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.Form,
    functional_term_eq_form)
