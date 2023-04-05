#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from .backend import FunctionSpace, UnitIntervalMesh, backend, \
    backend_Constant, backend_Function, backend_FunctionSpace, \
    backend_ScalarType, info
from ..interface import DEFAULT_COMM, SpaceInterface, \
    add_finalize_adjoint_derivative_action, add_functional_term_eq, \
    add_interface, add_subtract_adjoint_derivative_action, check_space_types, \
    comm_dup_cached, function_comm, function_dtype, function_is_alias, \
    function_is_scalar, function_scalar_value, new_function_id, new_space_id, \
    space_id, space_new, subtract_adjoint_derivative_action
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import assemble, is_valid_r0_space

from ..overloaded_float import SymbolicFloat

from .caches import form_neg
from .equations import Assembly
from .functions import Caches, Constant, ConstantInterface, \
    ConstantSpaceInterface, Function, ReplacementFunction, Zero, \
    define_function_alias

from functools import cached_property
import numpy as np
import petsc4py.PETSc as PETSc
import ufl
import warnings

__all__ = \
    [
        "new_scalar_function",

        "RealFunctionSpace",
        "default_comm",
        "function_space_id",
        "function_space_new",
        "info",
        "warning"
    ]


# Aim for compatibility with Firedrake API, git master revision
# efb48f4f178ae4989c146640025641cf0cc00a0e, Apr 19 2021
def _Constant__init__(self, value, domain=None, *,
                      name=None, space=None, comm=None,
                      **kwargs):
    if comm is None:
        comm = DEFAULT_COMM
    backend_Constant._tlm_adjoint__orig___init__(self, value, domain=domain,
                                                 **kwargs)

    if name is None:
        # Following FEniCS 2019.1.0 behaviour
        name = f"f_{self.count():d}"

    if space is None:
        space = self.ufl_function_space()
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_function_id(), "name": name, "state": 0,
                   "space": space, "space_type": "primal",
                   "dtype": self.dat.dtype.type, "static": False,
                   "cache": False, "checkpoint": True})


assert not hasattr(backend_Constant, "_tlm_adjoint__orig___init__")
backend_Constant._tlm_adjoint__orig___init__ = backend_Constant.__init__
backend_Constant.__init__ = _Constant__init__


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


def _FunctionSpace__init__(self, *args, **kwargs):
    backend_FunctionSpace._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionSpaceInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_space_id()})


assert not hasattr(backend_FunctionSpace, "_tlm_adjoint__orig___init__")
backend_FunctionSpace._tlm_adjoint__orig___init__ = backend_FunctionSpace.__init__  # noqa: E501
backend_FunctionSpace.__init__ = _FunctionSpace__init__


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return self._tlm_adjoint__function_interface_attrs["comm"]

    def _space(self):
        return self.function_space()

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
            dtype = function_dtype(self)
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(dtype(y)),
                            annotate=False, tlm=False)
            else:
                y_arr = np.full(self.ufl_shape, dtype(y), dtype=dtype)
                self.assign(backend_Constant(y_arr),
                            annotate=False, tlm=False)
        elif isinstance(y, Zero):
            with self.dat.vec_wo as x_v:
                x_v.zeroEntries()
        elif isinstance(y, backend_Constant):
            self.assign(y, annotate=False, tlm=False)
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

    def _axpy(self, alpha, x, /):
        dtype = function_dtype(self)
        alpha = dtype(alpha)
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
            self.assign(self + alpha * dtype(x),
                        annotate=False, tlm=False)
        elif isinstance(x, Zero):
            pass
        elif isinstance(x, backend_Constant):
            self.assign(self + alpha * x, annotate=False, tlm=False)
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
            y_.assign(y, annotate=False, tlm=False)
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
        values.setflags(write=False)
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
        return (is_valid_r0_space(self.function_space())
                and len(self.ufl_shape) == 0)

    def _scalar_value(self):
        # assert function_is_scalar(self)
        with self.dat.vec_ro as x_v:
            value = x_v.sum() / x_v.getSize()
        return value

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__function_interface_attrs


def _Function__init__(self, *args, **kwargs):
    backend_Function._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_function_id(),
                   "state": 0, "space_type": "primal", "static": False,
                   "cache": False, "checkpoint": True})


assert not hasattr(backend_Function, "_tlm_adjoint__orig___init__")
backend_Function._tlm_adjoint__orig___init__ = backend_Function.__init__
backend_Function.__init__ = _Function__init__


def _Function__getattr__(self, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return backend_Function._tlm_adjoint__orig__getattr__(self, key)


assert not hasattr(backend_Function, "_tlm_adjoint__orig__getattr__")
backend_Function._tlm_adjoint__orig__getattr__ = backend_Function.__getattr__
backend_Function.__getattr__ = _Function__getattr__


# Aim for compatibility with Firedrake API, git master revision
# c0b45ce2123fdeadf358df1d5655ce42f3b3d74b, Feb 1 2023
@cached_property
def _Function_subfunctions(self):
    Y = backend_Function._tlm_adjoint__orig_subfunctions.__get__(self,
                                                                 type(self))
    for i, y in enumerate(Y):
        define_function_alias(y, self, key=("subfunctions", i))
    return Y


assert not hasattr(backend_Function, "_tlm_adjoint__orig_subfunctions")
backend_Function._tlm_adjoint__orig_subfunctions = backend_Function.subfunctions  # noqa: E501
backend_Function.subfunctions = _Function_subfunctions
backend_Function.subfunctions.__set_name__(
    backend_Function.subfunctions, "_tlm_adjoint___Function_subfunctions")


# Aim for compatibility with Firedrake API, git master revision
# f322d327db1efb56e8078f4883a2d62fa0f63c45, Oct 26 2022
def _Function_sub(self, i):
    self.subfunctions
    y = backend_Function._tlm_adjoint__orig_sub(self, i)
    if not function_is_alias(y):
        define_function_alias(y, self, key=("sub", i))
    return y


assert not hasattr(backend_Function, "_tlm_adjoint__orig_sub")
backend_Function._tlm_adjoint__orig_sub = backend_Function.sub
backend_Function.sub = _Function_sub


def new_scalar_function(*, name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    return Constant(0.0, name=name, comm=comm, static=static, cache=cache,
                    checkpoint=checkpoint)


def _subtract_adjoint_derivative_action(x, y):
    if isinstance(y, ufl.classes.Form) \
            and isinstance(x, (backend_Constant, backend_Function)):
        if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
            x._tlm_adjoint__firedrake_adj_b += form_neg(y)
        else:
            x._tlm_adjoint__firedrake_adj_b = form_neg(y)
    elif isinstance(x, backend_Constant):
        dtype = function_dtype(x)
        if isinstance(y, backend_Function) and function_is_scalar(y):
            alpha = 1.0
        elif isinstance(y, tuple) \
                and len(y) == 2 \
                and isinstance(y[0], (int, np.integer,
                                      float, np.floating,
                                      complex, np.complexfloating)) \
                and isinstance(y[1], backend_Function) \
                and function_is_scalar(y[1]):
            alpha, y = y
            alpha = dtype(alpha)
        else:
            return NotImplemented
        check_space_types(x, y)
        y_value = function_scalar_value(y)
        x.assign(dtype(x) - alpha * y_value, annotate=False, tlm=False)
    else:
        return NotImplemented


add_subtract_adjoint_derivative_action(backend,
                                       _subtract_adjoint_derivative_action)


def _finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        y = assemble(x._tlm_adjoint__firedrake_adj_b)
        subtract_adjoint_derivative_action(x, (-1.0, y))
        delattr(x, "_tlm_adjoint__firedrake_adj_b")


add_finalize_adjoint_derivative_action(backend,
                                       _finalize_adjoint_derivative_action)


def _functional_term_eq(x, term):
    if isinstance(term, ufl.classes.Form) \
            and len(term.arguments()) == 0 \
            and isinstance(x, (SymbolicFloat, backend_Constant, backend_Function)):  # noqa: E501
        return Assembly(x, term)
    else:
        return NotImplemented


add_functional_term_eq(backend, _functional_term_eq)


def default_comm():
    warnings.warn("default_comm is deprecated -- "
                  "use DEFAULT_COMM instead",
                  DeprecationWarning, stacklevel=2)
    return DEFAULT_COMM


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_scalar_function instead",
                  DeprecationWarning, stacklevel=2)
    if comm is None:
        comm = DEFAULT_COMM
    return FunctionSpace(UnitIntervalMesh(comm.size, comm=comm), "R", 0)


def function_space_id(*args, **kwargs):
    warnings.warn("function_space_id is deprecated -- use space_id instead",
                  DeprecationWarning, stacklevel=2)
    return space_id(*args, **kwargs)


def function_space_new(*args, **kwargs):
    warnings.warn("function_space_new is deprecated -- use space_new instead",
                  DeprecationWarning, stacklevel=2)
    return space_new(*args, **kwargs)


# def info(message):


def warning(message):
    warnings.warn("warning is deprecated -- use logging.warning instead",
                  DeprecationWarning, stacklevel=2)
    warnings.warn(message, RuntimeWarning)
