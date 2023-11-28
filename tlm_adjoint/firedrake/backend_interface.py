#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    FiniteElement, TensorElement, VectorElement, backend_Cofunction,
    backend_CofunctionSpace, backend_Constant, backend_Function,
    backend_FunctionSpace, backend_ScalarType)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, VariableInterface, add_interface,
    check_space_type, comm_dup_cached, is_var, new_space_id, new_var_id,
    register_garbage_cleanup, register_functional_term_eq,
    register_subtract_adjoint_derivative_action, relative_space_type, space_id,
    subtract_adjoint_derivative_action_base, var_caches, var_id, var_is_alias,
    var_is_cached, var_is_static, var_linf_norm, var_lock_state, var_name,
    var_space, var_space_type)
from .backend_code_generator_interface import r0_space

from ..equations import Conversion
from ..override import override_method, override_property

from .equations import Assembly
from .functions import (
    Caches, ConstantInterface, ConstantSpaceInterface, Replacement,
    ReplacementConstant, ReplacementFunction, ReplacementInterface, Zero,
    define_var_alias)

import mpi4py.MPI as MPI
import numbers
import petsc4py.PETSc as PETSc
import pyop2
import ufl

__all__ = \
    [
        "Cofunction",
        "Function",

        "ZeroFunction",

        "ReplacementCofunction",

        "to_firedrake"
    ]


# Aim for compatibility with Firedrake API


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, value, domain=None, *,
                     name=None, space=None, comm=None,
                     **kwargs):
    orig(self, value, domain=domain, name=name, **kwargs)

    if name is None:
        name = self.name
    if comm is None:
        comm = DEFAULT_COMM

    if space is None:
        if domain is None:
            cell = None
        else:
            cell = domain.ufl_cell()
        if len(self.ufl_shape) == 0:
            element = FiniteElement("R", cell, 0)
        elif len(self.ufl_shape) == 1:
            element = VectorElement("R", cell, 0, dim=self.ufl_shape[0])
        else:
            element = TensorElement("R", cell, 0, shape=self.ufl_shape)
        space = ufl.classes.FunctionSpace(domain, element)
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_var_id(), "name": lambda x: name,
                   "state": [0], "space": space,
                   "derivative_space": lambda x: r0_space(x),
                   "space_type": "primal", "dtype": self.dat.dtype.type,
                   "static": False, "cache": False,
                   "replacement": ReplacementConstant(space)})


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        if space_type in {"primal", "conjugate"}:
            if "space" in self._tlm_adjoint__space_interface_attrs:
                space = self._tlm_adjoint__space_interface_attrs["space"]
            else:
                space = self._tlm_adjoint__space_interface_attrs["space_dual"].dual()  # noqa: E501
            return Function(space, name=name, space_type=space_type,
                            static=static, cache=cache)
        elif space_type in {"dual", "conjugate_dual"}:
            if "space_dual" in self._tlm_adjoint__space_interface_attrs:
                space_dual = self._tlm_adjoint__space_interface_attrs["space_dual"]  # noqa: E501
            else:
                space_dual = self._tlm_adjoint__space_interface_attrs["space"].dual()  # noqa: E501
            return Cofunction(space_dual, name=name, space_type=space_type,
                              static=static, cache=cache)
        else:
            raise ValueError("Invalid space type")


@override_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id()})


@override_method(backend_FunctionSpace, "dual")
def FunctionSpace_dual(self, orig, orig_args):
    if "space_dual" not in self._tlm_adjoint__space_interface_attrs:
        self._tlm_adjoint__space_interface_attrs["space_dual"] = orig_args()
    space_dual = self._tlm_adjoint__space_interface_attrs["space_dual"]
    if "space" not in space_dual._tlm_adjoint__space_interface_attrs:
        space_dual._tlm_adjoint__space_interface_attrs["space"] = self
    return space_dual


@override_method(backend_CofunctionSpace, "__init__")
def CofunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space_dual": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id()})


@override_method(backend_CofunctionSpace, "dual")
def CofunctionSpace_dual(self, orig, orig_args):
    if "space" not in self._tlm_adjoint__space_interface_attrs:
        self._tlm_adjoint__space_interface_attrs["space"] = orig_args()
    space = self._tlm_adjoint__space_interface_attrs["space"]
    if "space_dual" not in space._tlm_adjoint__space_interface_attrs:
        space._tlm_adjoint__space_interface_attrs["space_dual"] = self
    return space


class FunctionInterfaceBase(VariableInterface):
    def _comm(self):
        return self._tlm_adjoint__var_interface_attrs["comm"]

    def _space(self):
        return self.function_space()

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _dtype(self):
        return self.dat.dtype.type

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        dat, count = self._tlm_adjoint__var_interface_attrs["state"]
        if count is not None and count > dat.dat_version:
            raise RuntimeError("Invalid state")
        return dat.dat_version

    def _update_state(self):
        dat, count = self._tlm_adjoint__var_interface_attrs["state"]
        if count is None or count == dat.dat_version:
            # Make sure that the dat version has been incremented at least once
            dat.increment_dat_version()
        elif count > dat.dat_version:
            raise RuntimeError("Invalid state")
        self._tlm_adjoint__var_interface_attrs["state"][1] = dat.dat_version

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
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def _linf_norm(self):
        with self.dat.vec_ro as x_v:
            linf_norm = x_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)
        return linf_norm

    def _local_size(self):
        with self.dat.vec_ro as x_v:
            local_size = x_v.getLocalSize()
        return local_size

    def _global_size(self):
        with self.dat.vec_ro as x_v:
            size = x_v.getSize()
        return size

    def _local_indices(self):
        with self.dat.vec_ro as x_v:
            local_range = x_v.getOwnershipRange()
        return slice(*local_range)

    def _get_values(self):
        with self.dat.vec_ro as x_v:
            x_a = x_v.getArray(True)
        return x_a.copy()

    def _set_values(self, values):
        with self.dat.vec_wo as x_v:
            x_v.setArray(values)

    def _replacement(self):
        replacement = self._tlm_adjoint__var_interface_attrs["replacement"]
        if not is_var(replacement):
            add_interface(replacement, ReplacementInterface,
                          {"id": var_id(self), "name": var_name(self),
                           "space": var_space(self),
                           "space_type": var_space_type(self),
                           "static": var_is_static(self),
                           "cache": var_is_cached(self),
                           "caches": var_caches(self)})
        return replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return False

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__var_interface_attrs


class FunctionInterface(FunctionInterfaceBase):
    def _derivative_space(self):
        return self.function_space()

    def _assign(self, y):
        if isinstance(y, backend_Cofunction):
            y = y.riesz_representation("l2")
        if isinstance(y, numbers.Complex):
            if len(self.ufl_shape) != 0:
                raise ValueError("Invalid shape")
            self.assign(backend_Constant(y))
        elif isinstance(y, backend_Function):
            if space_id(y.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            with self.dat.vec_wo as x_v, y.dat.vec_ro as y_v:
                y_v.copy(result=x_v)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _axpy(self, alpha, x, /):
        if isinstance(x, backend_Cofunction):
            x = x.riesz_representation("l2")
        if isinstance(x, backend_Function):
            if space_id(x.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                y_v.axpy(alpha, x_v)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    def _inner(self, y):
        if isinstance(y, backend_Function):
            y = y.riesz_representation("l2")
        if isinstance(y, backend_Cofunction):
            if space_id(y.function_space()) != space_id(self.function_space().dual()):  # noqa: E501
                raise ValueError("Invalid function space")
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                inner = x_v.dot(y_v)
            return inner
        else:
            raise TypeError(f"Unexpected type: {type(y)}")


class Function(backend_Function):
    """Extends :class:`firedrake.function.Function`.

    :arg space_type: The space type for the :class:`.Function`. `'primal'` or
        `'conjugate'`.
    :arg static: Defines whether the :class:`.Function` is static, meaning that
        it is stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.Function` may
        be cached. Default `static`.

    Remaining arguments are passed to the :class:`firedrake.function.Function`
    constructor.
    """

    def __init__(self, *args, space_type="primal", static=False, cache=None,
                 **kwargs):
        if space_type not in {"primal", "conjugate"}:
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


@override_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, function_space, val=None,
                     *args, **kwargs):
    orig_args()
    comm = self.comm
    if pyop2.mpi.is_pyop2_comm(comm):
        # Work around Firedrake issue #3043
        comm = self.function_space().comm
    add_interface(self, FunctionInterface,
                  {"comm": comm_dup_cached(comm), "id": new_var_id(),
                   "state": [self.dat, getattr(self.dat, "dat_version", None)],
                   "space_type": "primal", "static": False, "cache": False,
                   "replacement": ReplacementFunction(self.function_space())})
    if isinstance(val, backend_Function):
        define_var_alias(self, val, key=("Function__init__",))


@override_method(backend_Function, "__getattr__")
def Function__getattr__(self, orig, orig_args, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return orig_args()


@override_method(backend_Function, "riesz_representation")
def Function_riesz_representation(self, orig, orig_args,
                                  riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "primal")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


@override_property(backend_Function, "subfunctions", cached=True)
def Function_subfunctions(self, orig):
    Y = orig()
    for i, y in enumerate(Y):
        define_var_alias(y, self, key=("subfunctions", i))
    return Y


@override_method(backend_Function, "sub")
def Function_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not var_is_alias(y):
        define_var_alias(y, self, key=("sub", i))
    return y


class CofunctionInterface(FunctionInterfaceBase):
    def _assign(self, y):
        if isinstance(y, backend_Function):
            y = y.riesz_representation("l2")
        if isinstance(y, backend_Cofunction):
            if space_id(y.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            with self.dat.vec_wo as x_v, y.dat.vec_ro as y_v:
                y_v.copy(result=x_v)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _axpy(self, alpha, x, /):
        if isinstance(x, backend_Function):
            x = x.riesz_representation("l2")
        if isinstance(x, backend_Cofunction):
            if space_id(x.function_space()) != space_id(self.function_space()):
                raise ValueError("Invalid function space")
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                y_v.axpy(alpha, x_v)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    def _inner(self, y):
        if isinstance(y, backend_Cofunction):
            y = y.riesz_representation("l2")
        if isinstance(y, backend_Function):
            if space_id(y.function_space()) != space_id(self.function_space().dual()):  # noqa: E501
                raise ValueError("Invalid function space")
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                inner = x_v.dot(y_v)
            return inner
        else:
            raise TypeError(f"Unexpected type: {type(y)}")


class Cofunction(backend_Cofunction):
    """Extends the :class:`firedrake.cofunction.Cofunction` class.

    :arg space_type: The space type for the :class:`.Cofunction`.
        `'conjugate'` or `'conjugate_dual'`.
    :arg static: Defines whether the :class:`.Cofunction` is static, meaning
        that it is stored by reference in checkpointing/replay, and an
        associated tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.Cofunction` may
        be cached. Default `static`.

    Remaining arguments are passed to the
    :class:`firedrake.cofunction.Cofunction` constructor.
    """

    def __init__(self, *args, space_type="conjugate_dual", static=False,
                 cache=None, **kwargs):
        if space_type not in {"dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if cache is None:
            cache = static

        super().__init__(*args, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)


@override_method(backend_Cofunction, "__init__")
def Cofunction__init__(self, orig, orig_args, function_space, val=None,
                       *args, **kwargs):
    orig_args()
    add_interface(self, CofunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_var_id(),
                   "state": [self.dat, getattr(self.dat, "dat_version", None)],
                   "space_type": "conjugate_dual", "static": False,
                   "cache": False,
                   "replacement": ReplacementCofunction(self.function_space())})  # noqa: E501
    if isinstance(val, backend_Cofunction):
        define_var_alias(self, val, key=("Cofunction__init__",))


@override_method(backend_Cofunction, "riesz_representation")
def Cofunction_riesz_representation(self, orig, orig_args,
                                    riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "conjugate_dual")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


class ReplacementCofunction(Replacement, ufl.classes.Cofunction):
    """Represents a symbolic :class:`firedrake.cofunction.Cofunction`, but has
    no value.
    """

    def __init__(self, space):
        Replacement.__init__(self)
        ufl.classes.Cofunction.__init__(self, space)


def to_firedrake(y, space, *, name=None):
    """Convert a variable to a :class:`firedrake.function.Function` or
    :class:`firedrake.cofunction.Cofunction`.

    :arg y: A variable.
    :arg space: The space for the return value.
    :arg name: A :class:`str` name.
    :returns: The :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`.
    """

    space_type = var_space_type(y)
    if space_type in {"primal", "conjugate"}:
        if isinstance(space, backend_FunctionSpace):
            x = Function(space, space_type=space_type, name=name)
        else:
            x = Function(space.dual(), space_type=space_type, name=name)
    elif space_type in {"dual", "conjugate_dual"}:
        if isinstance(space, backend_FunctionSpace):
            x = Cofunction(space.dual(), space_type=space_type, name=name)
        else:
            x = Cofunction(space, space_type=space_type, name=name)
    else:
        raise ValueError("Invalid space type")
    Conversion(x, y).solve()
    return x


def garbage_cleanup_internal_comm(comm):
    if not MPI.Is_finalized() and not PETSc.Sys.isFinalized() \
            and not pyop2.mpi.PYOP2_FINALIZED \
            and comm.py2f() != MPI.COMM_NULL.py2f():
        if pyop2.mpi.is_pyop2_comm(comm):
            raise RuntimeError("Should not call garbage_cleanup directly on a "
                               "PyOP2 communicator")
        internal_comm = comm.Get_attr(pyop2.mpi.innercomm_keyval)
        if internal_comm is not None and internal_comm.py2f() != MPI.COMM_NULL.py2f():  # noqa: E501
            PETSc.garbage_cleanup(internal_comm)


register_garbage_cleanup(garbage_cleanup_internal_comm)


register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Cofunction, backend_Function), object,
    subtract_adjoint_derivative_action_base,
    replace=True)


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.Form,
    functional_term_eq_form)
