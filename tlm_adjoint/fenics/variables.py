"""FEniCS variables.
"""

from .backend import (
    as_backend_type, backend_Constant, backend_Function, backend_ScalarType,
    backend_Vector)
from ..interface import (
    SpaceInterface, VariableInterface, check_space_types,
    register_subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base, space_comm, space_dtype, space_eq,
    space_id, var_axpy, var_comm, var_copy, var_dtype, var_is_cached,
    var_is_static, var_linf_norm, var_lock_state, var_new, var_scalar_value,
    var_space_type)

from ..caches import Caches
from ..equations import Conversion

from .expr import Replacement, Zero, r0_space

import functools
import numbers
import numpy as np
import weakref

__all__ = \
    [
        "Constant",
        "Function",

        "ZeroConstant",
        "ZeroFunction",

        "ReplacementConstant",
        "ReplacementFunction",
        "ReplacementZeroConstant",
        "ReplacementZeroFunction",

        "to_fenics"
    ]


class ConstantSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return self._tlm_adjoint__space_interface_attrs["dtype"]

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _eq(self, other):
        return (space_id(self) == space_id(other)
                or (isinstance(other, type(self))
                    and space_comm(self).py2f() == space_comm(other).py2f()
                    and space_dtype(self) == space_dtype(other)
                    and self == other))

    def _global_size(self):
        shape = self.ufl_element().value_shape()
        if len(shape) == 0:
            return 1
        else:
            return np.prod(shape)

    def _local_indices(self):
        comm = space_comm(self)
        if comm.rank == 0:
            shape = self.ufl_element().value_shape()
            if len(shape) == 0:
                return slice(0, 1)
            else:
                return slice(0, np.prod(shape))
        else:
            return slice(0, 0)

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
        if isinstance(y, numbers.Real):
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
    """Extends the DOLFIN `Constant` class.

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

    Remaining arguments are passed to the DOLFIN `Constant` constructor.
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
                shape = space.ufl_element().value_shape()
            elif shape != space.ufl_element().value_shape():
                raise ValueError("Invalid shape")

        value = constant_value(value, shape)

        # Default comm
        if comm is None and space is not None:
            comm = space_comm(space)

        if cache is None:
            cache = static

        super().__init__(
            value, *args, name=name, domain=domain, space=space,
            comm=comm, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _eq(self, other):
        return (space_id(self) == space_id(other)
                or (isinstance(other, type(self))
                    and self == other))

    def _global_size(self):
        return self.dofmap().global_dimension()

    def _local_indices(self):
        n0, n1 = self.dofmap().ownership_range()
        return slice(n0, n1)

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        return Function(self, name=name, space_type=space_type, static=static,
                        cache=cache)


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

    @check_vector
    def _assign(self, y):
        if isinstance(y, numbers.Real):
            if len(self.ufl_shape) != 0:
                raise ValueError("Invalid shape")
            self.assign(backend_Constant(y))
        elif isinstance(y, backend_Function):
            if not space_eq(y.function_space(), self.function_space()):
                raise ValueError("Invalid function space")
            self.vector().zero()
            self.vector().axpy(1.0, y.vector())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    @check_vector
    def _axpy(self, alpha, x, /):
        if isinstance(x, backend_Function):
            if not space_eq(x.function_space(), self.function_space()):
                raise ValueError("Invalid function space")
            self.vector().axpy(alpha, x.vector())
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    @check_vector
    def _inner(self, y):
        if isinstance(y, backend_Function):
            if not space_eq(y.function_space(), self.function_space()):
                raise ValueError("Invalid function space")
            return y.vector().inner(self.vector())
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    @check_vector
    def _linf_norm(self):
        return self.vector().norm("linf")

    @check_vector
    def _get_values(self):
        return self.vector().get_local().copy()  # copy likely not required

    @check_vector
    def _set_values(self, values):
        self.vector().set_local(values)
        self.vector().apply("insert")

    @check_vector
    def _to_petsc(self, vec):
        self_v = as_backend_type(self.vector()).vec()
        self_v.copy(result=vec)

    @check_vector
    def _from_petsc(self, vec):
        self_v = as_backend_type(self.vector()).vec()
        vec.copy(result=self_v)
        self.vector().apply("insert")

    @check_vector
    def _new(self, *, name=None, static=False, cache=None,
             rel_space_type="primal"):
        y = var_copy(self, name=name, static=static, cache=cache)
        y.vector().zero()
        space_type = var_space_type(self, rel_space_type=rel_space_type)
        y._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        return y

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
        if "replacement" not in self._tlm_adjoint__var_interface_attrs:
            count = self._tlm_adjoint__var_interface_attrs["replacement_count"]
            if isinstance(self, Zero):
                self._tlm_adjoint__var_interface_attrs["replacement"] = \
                    ReplacementZeroFunction(self, count=count)
            else:
                self._tlm_adjoint__var_interface_attrs["replacement"] = \
                    ReplacementFunction(self, count=count)
        return self._tlm_adjoint__var_interface_attrs["replacement"]

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


class ReplacementConstant(Replacement):
    """Represents a symbolic DOLFIN `Constant`, but has no value.
    """


class ReplacementFunction(Replacement):
    """Represents a symbolic DOLFIN `Function`, but has no value.
    """


class ReplacementZeroConstant(ReplacementConstant, Zero):
    """Represents a symbolic DOLFIN `Constant` which is zero, but has no value.
    """

    def __init__(self, *args, **kwargs):
        ReplacementConstant.__init__(self, *args, **kwargs)
        Zero.__init__(self)


class ReplacementZeroFunction(ReplacementFunction, Zero):
    """Represents a symbolic DOLFIN `Function` which is zero, but has no value.
    """

    def __init__(self, *args, **kwargs):
        ReplacementFunction.__init__(self, *args, **kwargs)
        Zero.__init__(self)


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


def subtract_adjoint_derivative_action_backend_constant_vector(x, alpha, y):
    if hasattr(y, "_tlm_adjoint__function"):
        check_space_types(x, y._tlm_adjoint__function)

    if len(x.ufl_shape) == 0:
        value = y.max()
    else:
        value = np.zeros_like(x.values()).flatten()
        y_fn = backend_Function(r0_space(x))
        y_fn.vector().axpy(1.0, y)
        for i, y_fn_c in enumerate(y_fn.split(deepcopy=True)):
            value[i] = y_fn_c.vector().max()
        value.shape = x.ufl_shape
        value = backend_Constant(value)

    y = var_new(x)
    y.assign(value)
    var_axpy(x, -alpha, y)


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
