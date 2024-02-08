"""This module includes functionality for interacting with Firedrake variables
and Dirichlet boundary conditions.
"""

from .backend import (
    FiniteElement, TensorElement, VectorElement, backend_Constant,
    backend_ScalarType)
from ..interface import (
    SpaceInterface, VariableInterface, add_replacement_interface, space_comm,
    var_comm, var_dtype, var_is_cached, var_is_static, var_linf_norm,
    var_lock_state, var_scalar_value, var_space, var_space_type)

from ..caches import Caches
from ..manager import paused_manager

from .expr import Replacement, Zero

import numbers
import numpy as np
import ufl
import weakref

__all__ = \
    [
        "Constant",

        "ZeroConstant",

        "ReplacementConstant",
        "ReplacementFunction",
        "ReplacementZeroConstant",
        "ReplacementZeroFunction"
    ]


class ConstantSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return self._tlm_adjoint__space_interface_attrs["dtype"]

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        domain = self._tlm_adjoint__space_interface_attrs["domain"]
        return Constant(name=name, domain=domain, space=self,
                        space_type=space_type, static=static, cache=cache)


class ConstantInterface(VariableInterface):
    def _space(self):
        return self._tlm_adjoint__var_interface_attrs["space"]

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

    def _caches(self):
        if "caches" not in self._tlm_adjoint__var_interface_attrs:
            self._tlm_adjoint__var_interface_attrs["caches"] \
                = Caches(self)
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _zero(self):
        if len(self.ufl_shape) == 0:
            value = 0.0
        else:
            value = np.zeros(self.ufl_shape, dtype=var_dtype(self))
            value = backend_Constant(value)
        self.assign(value)

    def _assign(self, y):
        if isinstance(y, numbers.Complex):
            if len(self.ufl_shape) != 0:
                raise ValueError("Invalid shape")
            self.assign(y)
        elif isinstance(y, backend_Constant):
            if y.ufl_shape != self.ufl_shape:
                raise ValueError("Invalid shape")
            self.assign(y)
        else:
            if len(self.ufl_shape) != 0:
                raise ValueError("Invalid shape")
            self.assign(var_scalar_value(y))

    def _axpy(self, alpha, x, /):
        if isinstance(x, backend_Constant):
            if x.ufl_shape != self.ufl_shape:
                raise ValueError("Invalid shape")
            if len(self.ufl_shape) == 0:
                self.assign(self + alpha * x)
            else:
                value = self.values() + alpha * x.values()
                value.shape = self.ufl_shape
                value = backend_Constant(value)
                self.assign(value)
        else:
            if len(self.ufl_shape) == 0:
                self.assign(
                    var_scalar_value(self) + alpha * var_scalar_value(x))
            else:
                raise ValueError("Invalid shape")

    def _inner(self, y):
        if isinstance(y, backend_Constant):
            if y.ufl_shape != self.ufl_shape:
                raise ValueError("Invalid shape")
            return y.values().conjugate().dot(self.values())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _linf_norm(self):
        values = self.values()
        if len(values) == 0:
            return var_dtype(self)(0.0).real.dtype.type(0.0)
        else:
            return abs(values).max()

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
        if "replacement" not in self._tlm_adjoint__var_interface_attrs:
            count = self._tlm_adjoint__var_interface_attrs["replacement_count"]
            if isinstance(self, Zero):
                self._tlm_adjoint__var_interface_attrs["replacement"] = \
                    ReplacementZeroConstant(self, count=count)
            else:
                self._tlm_adjoint__var_interface_attrs["replacement"] = \
                    ReplacementConstant(self, count=count)
        return self._tlm_adjoint__var_interface_attrs["replacement"]

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return len(self.ufl_shape) == 0

    def _scalar_value(self):
        # assert var_is_scalar(self)
        value, = self.values()
        return var_dtype(self)(value)

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
    """Extends the :class:`firedrake.constant.Constant` class.

    :arg value: The initial value. `None` indicates a value of zero.
    :arg name: A :class:`str` name.
    :arg domain: The domain on which the :class:`.Constant` is defined.
    :arg space: The space on which the :class:`.Constant` is defined.
    :arg space_type: The space type for the :class:`.Constant`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg shape: A :class:`tuple` of :class:`int` objects defining the shape of
        the value.
    :arg comm: The communicator for the :class:`.Constant`.
    :arg static: Defines whether the :class:`.Constant` is static, meaning that
        it is stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.Constant` may be
        cached. Default `static`.

    Remaining arguments are passed to the :class:`firedrake.constant.Constant`
    constructor.
    """

    def __init__(self, value=None, *args, name=None, domain=None, space=None,
                 space_type="primal", shape=None, comm=None, static=False,
                 cache=None, **kwargs):
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
                shape = space.ufl_element().value_shape
            elif shape != space.ufl_element().value_shape:
                raise ValueError("Invalid shape")

        value = constant_value(value, shape)

        # Default comm
        if comm is None and space is not None:
            comm = space_comm(space)

        if cache is None:
            cache = static

        with paused_manager():
            super().__init__(
                value, *args, name=name, domain=domain, space=space,
                comm=comm, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)

    def __new__(cls, value=None, *args, name=None, domain=None,
                space_type="primal", shape=None, static=False, cache=None,
                **kwargs):
        if domain is None:
            return object().__new__(cls)
        else:
            value = constant_value(value, shape)
            if space_type not in {"primal", "conjugate",
                                  "dual", "conjugate_dual"}:
                raise ValueError("Invalid space type")
            if cache is None:
                cache = static
            F = super().__new__(cls, value, domain=domain)
            F.rename(name=name)
            F._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
            F._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
            F._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)
            return F


class ZeroConstant(Constant, Zero):
    """A :class:`.Constant` which is flagged as having a value of zero.

    Arguments are passed to the :class:`.Constant` constructor, together with
    `static=True` and `cache=True`.
    """

    def __init__(self, *, name=None, domain=None, space=None,
                 space_type="primal", shape=None, comm=None):
        Constant.__init__(
            self, name=name, domain=domain, space=space, space_type=space_type,
            shape=shape, comm=comm, static=True, cache=True)
        var_lock_state(self)
        if var_linf_norm(self) != 0.0:
            raise RuntimeError("ZeroConstant is not zero-valued")

    def __new__(cls, *args, shape=None, **kwargs):
        return Constant.__new__(
            cls, constant_value(shape=shape), *args,
            shape=shape, static=True, cache=True, **kwargs)


def constant_space(shape, *, domain=None):
    if domain is None:
        cell = None
    else:
        cell = domain.ufl_cell()

    if len(shape) == 0:
        element = FiniteElement("R", cell, 0)
    elif len(shape) == 1:
        dim, = shape
        element = VectorElement("R", cell, 0, dim=dim)
    else:
        element = TensorElement("R", cell, 0, shape=shape)

    return ufl.classes.FunctionSpace(domain, element)


class ReplacementConstant(Replacement, ufl.classes.ConstantValue,
                          ufl.utils.counted.Counted):
    """Represents a symbolic :class:`firedrake.constant.Constant`, but has no
    value.
    """

    def __init__(self, x, count):
        Replacement.__init__(self)
        ufl.classes.ConstantValue.__init__(self)
        ufl.utils.counted.Counted.__init__(
            self, count=count, counted_class=x._counted_class)
        self._tlm_adjoint__ufl_shape = tuple(x.ufl_shape)
        add_replacement_interface(self, x)

    def __repr__(self):
        return f"<{type(self)} with count {self.count()}>"

    @property
    def ufl_shape(self):
        return self._tlm_adjoint__ufl_shape


class ReplacementFunction(Replacement, ufl.classes.Coefficient):
    """Represents a symbolic :class:`firedrake.function.Function`, but has no
    value.
    """

    def __init__(self, x, count):
        Replacement.__init__(self)
        ufl.classes.Coefficient.__init__(self, var_space(x), count=count)
        add_replacement_interface(self, x)

    def __new__(cls, x, *args, **kwargs):
        return ufl.classes.Coefficient.__new__(cls, var_space(x),
                                               *args, **kwargs)


class ReplacementZeroConstant(ReplacementConstant, Zero):
    """Represents a symbolic :class:`firedrake.constant.Constant` which is
    zero, but has no value.
    """

    def __init__(self, *args, **kwargs):
        ReplacementConstant.__init__(self, *args, **kwargs)
        Zero.__init__(self)


class ReplacementZeroFunction(ReplacementFunction, Zero):
    """Represents a symbolic :class:`firedrake.function.Function` which is
    zero, but has no value.
    """

    def __init__(self, *args, **kwargs):
        ReplacementFunction.__init__(self, *args, **kwargs)
        Zero.__init__(self)


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
                "state", parent._tlm_adjoint__var_interface_attrs["state"])
