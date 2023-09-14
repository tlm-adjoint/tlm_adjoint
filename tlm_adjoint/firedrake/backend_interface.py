#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    backend_Constant, backend_Function, backend_FunctionSpace,
    backend_ScalarType)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, add_interface, check_space_type,
    comm_dup_cached, function_comm, function_dtype, function_is_alias,
    function_space, new_function_id, new_space_id, register_garbage_cleanup,
    register_finalize_adjoint_derivative_action, register_functional_term_eq,
    register_subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base)
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import assemble, r0_space

from ..manager import manager_disabled
from ..override import override_method, override_property
from ..overloaded_float import SymbolicFloat

from .equations import Assembly
from .functions import (
    Caches, ConstantInterface, ConstantSpaceInterface, Function,
    ReplacementFunction, Zero, define_function_alias)

import mpi4py.MPI as MPI
import numpy as np
import petsc4py.PETSc as PETSc
import pyop2
import ufl

__all__ = \
    [
    ]


# Aim for compatibility with Firedrake API


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, value, domain=None, *,
                     name=None, space=None, comm=None,
                     **kwargs):
    const_name = name
    if const_name is not None:
        # Work around Firedrake issue #3079
        chars = []
        for char in const_name:
            if char.isascii() and (char.isalnum() or char == "_"):
                chars.append(char)
        const_name = "".join(chars)

    orig(self, value, domain=domain, name=const_name, **kwargs)

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
            element = ufl.classes.FiniteElement("R", cell, 0)
        elif len(self.ufl_shape) == 1:
            element = ufl.classes.VectorElement("R", cell, 0,
                                                dim=self.ufl_shape[0])
        else:
            element = ufl.classes.TensorElement("R", cell, 0,
                                                shape=self.ufl_shape)
        space = ufl.classes.FunctionSpace(domain, element)
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_function_id(), "name": lambda x: name,
                   "state": 0, "space": space,
                   "form_derivative_space": lambda x: r0_space(x),
                   "space_type": "primal", "dtype": self.dat.dtype.type,
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


@override_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_space_id()})


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return self._tlm_adjoint__function_interface_attrs["comm"]

    def _space(self):
        return self.function_space()

    def _form_derivative_space(self):
        return function_space(self)

    def _space_type(self):
        return self._tlm_adjoint__function_interface_attrs["space_type"]

    def _dtype(self):
        return self.dat.dtype.type

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

    def _zero(self):
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    @manager_disabled()
    def _assign(self, y):
        if isinstance(y, SymbolicFloat):
            y = y.value()
        if isinstance(y, backend_Function):
            with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise ValueError("Invalid function space")
                y_v.copy(result=x_v)
        elif isinstance(y, (int, np.integer,
                            float, np.floating,
                            complex, np.complexfloating)):
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(y))
            else:
                y_arr = np.full(self.ufl_shape, y)
                self.assign(backend_Constant(y_arr))
        elif isinstance(y, Zero):
            with self.dat.vec_wo as x_v:
                x_v.zeroEntries()
        elif isinstance(y, backend_Constant):
            self.assign(y)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            comm = function_comm(self)
            if comm.rank != 0:
                values = None
            values = comm.bcast(values, root=0)
            self.dat.data[:] = values

    @manager_disabled()
    def _axpy(self, alpha, x, /):
        if isinstance(x, SymbolicFloat):
            x = x.value()
        if isinstance(x, backend_Function):
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                if y_v.getLocalSize() != x_v.getLocalSize():
                    raise ValueError("Invalid function space")
                y_v.axpy(alpha, x_v)
        elif isinstance(x, (int, np.integer,
                            float, np.floating,
                            complex, np.complexfloating)):
            self.assign(self + alpha * x)
        elif isinstance(x, Zero):
            pass
        elif isinstance(x, backend_Constant):
            self.assign(self + alpha * x)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            comm = function_comm(self)
            if comm.rank != 0:
                values = None
            values = comm.bcast(values, root=0)
            self.dat.data[:] = values

    @manager_disabled()
    def _inner(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise ValueError("Invalid function space")
                inner = x_v.dot(y_v)
        elif isinstance(y, Zero):
            inner = 0.0
        elif isinstance(y, backend_Constant):
            y_ = backend_Function(self.function_space())
            y_.assign(y)
            with self.dat.vec_ro as x_v, y_.dat.vec_ro as y_v:
                inner = x_v.dot(y_v)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")
        return inner

    def _sum(self):
        with self.dat.vec_ro as x_v:
            sum = x_v.sum()
        return sum

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
            with x_v as x_v_a:
                values = x_v_a.copy()
        return values

    def _set_values(self, values):
        if not np.can_cast(values, function_dtype(self)):
            raise ValueError("Invalid dtype")
        with self.dat.vec as x_v:
            if values.shape != (x_v.getLocalSize(),):
                raise ValueError("Invalid shape")
            x_v.setArray(values)

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementFunction(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        False

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__function_interface_attrs


@override_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    comm = self.comm
    if pyop2.mpi.is_pyop2_comm(comm):
        # Work around Firedrake issue #3043
        comm = self.function_space().comm
    add_interface(self, FunctionInterface,
                  {"comm": comm_dup_cached(comm), "id": new_function_id(),
                   "state": 0, "space_type": "primal", "static": False,
                   "cache": False, "checkpoint": True})


@override_method(backend_Function, "__getattr__")
def Function__getattr__(self, orig, orig_args, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return orig_args()


@override_property(backend_Function, "subfunctions", cached=True)
def Function_subfunctions(self, orig):
    Y = orig()
    for i, y in enumerate(Y):
        define_function_alias(y, self, key=("subfunctions", i))
    return Y


@override_method(backend_Function, "sub")
def Function_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not function_is_alias(y):
        define_function_alias(y, self, key=("sub", i))
    return y


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


def subtract_adjoint_derivative_action_function_form(x, alpha, y):
    check_space_type(x, "conjugate_dual")
    if alpha != 1.0:
        y = backend_Constant(alpha) * y
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        x._tlm_adjoint__firedrake_adj_b = x._tlm_adjoint__firedrake_adj_b - y
    else:
        x._tlm_adjoint__firedrake_adj_b = -y


register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Function), object,
    subtract_adjoint_derivative_action_base,
    replace=True)
register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Function), ufl.classes.Form,
    subtract_adjoint_derivative_action_function_form)


def finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        y = assemble(x._tlm_adjoint__firedrake_adj_b)
        subtract_adjoint_derivative_action(x, (-1.0, y))
        delattr(x, "_tlm_adjoint__firedrake_adj_b")


register_finalize_adjoint_derivative_action(finalize_adjoint_derivative_action)


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    (SymbolicFloat, backend_Constant, backend_Function), ufl.classes.Form,
    functional_term_eq_form)
