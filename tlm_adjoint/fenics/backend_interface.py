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
    backend_ScalarType, backend_Vector, info
from ..interface import InterfaceException, SpaceInterface, \
    add_finalize_adjoint_derivative_action, add_functional_term_eq, \
    add_interface, add_new_real_function, \
    add_subtract_adjoint_derivative_action, add_time_system_eq, \
    function_caches, function_copy, function_is_cached, \
    function_is_checkpointed, function_is_static, function_new, \
    new_function_id, new_space_id, space_id, space_new, \
    subtract_adjoint_derivative_action
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import assemble, is_valid_r0_space, \
    r0_space

from .caches import form_neg
from .equations import AssembleSolver, EquationSolver
from .functions import Caches, Constant, Function, Replacement, Zero

import mpi4py.MPI as MPI
import numpy as np
import types
import ufl
import warnings

__all__ = \
    [
        "RealFunctionSpace",
        "default_comm",
        "function_space_id",
        "function_space_new",
        "info",
        "warning"
    ]


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.mesh().mpi_comm()

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self, name=name, static=static, cache=cache,
                        checkpoint=checkpoint)


def _FunctionSpace__init__(self, *args, **kwargs):
    backend_FunctionSpace._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionSpaceInterface,
                  {"id": new_space_id()})


backend_FunctionSpace._tlm_adjoint__orig___init__ = backend_FunctionSpace.__init__  # noqa: E501
backend_FunctionSpace.__init__ = _FunctionSpace__init__


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return self.function_space().mesh().mpi_comm()

    def _space(self):
        return self._tlm_adjoint__function_interface_attrs["space"]

    def _id(self):
        return self._tlm_adjoint__function_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        return self._tlm_adjoint__function_interface_attrs["state"]

    def _update_state(self):
        self._tlm_adjoint__function_interface_attrs["state"] += 1

    def _is_static(self):
        return self._tlm_adjoint__function_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__function_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__function_interface_attrs["checkpoint"]

    def _caches(self):
        if not hasattr(self, "_tlm_adjoint__caches"):
            self._tlm_adjoint__caches = Caches(self)
        return self._tlm_adjoint__caches

    def _update_caches(self, value=None):
        if value is None:
            value = self
        function_caches(self).update(value)

    def _zero(self):
        self.vector().zero()

    def _assign(self, y):
        if isinstance(y, backend_Function):
            if self.vector().local_size() != y.vector().local_size():
                raise InterfaceException("Invalid function space")
            self.vector().zero()
            self.vector().axpy(1.0, y.vector())
        elif isinstance(y, (int, float)):
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(float(y)),
                            annotate=False, tlm=False)
            else:
                y_arr = np.full(self.ufl_shape, float(y), dtype=np.float64)
                self.assign(backend_Constant(y_arr),
                            annotate=False, tlm=False)
        elif isinstance(y, Zero):
            self.vector().zero()
        else:
            assert isinstance(y, backend_Constant)
            self.assign(y, annotate=False, tlm=False)

    def _axpy(self, *args):  # self, alpha, x
        alpha, x = args
        alpha = float(alpha)
        if isinstance(x, backend_Function):
            if self.vector().local_size() != x.vector().local_size():
                raise InterfaceException("Invalid function space")
            self.vector().axpy(alpha, x.vector())
        elif isinstance(x, (int, float)):
            x_ = function_new(self)
            if len(self.ufl_shape) == 0:
                x_.assign(backend_Constant(float(x)),
                          annotate=False, tlm=False)
            else:
                x_arr = np.full(self.ufl_shape, float(x), dtype=np.float64)
                x_.assign(backend_Constant(x_arr),
                          annotate=False, tlm=False)
            self.vector().axpy(alpha, x_.vector())
        elif isinstance(x, Zero):
            pass
        else:
            assert isinstance(x, backend_Constant)
            x_ = backend_Function(self.function_space())
            x_.assign(x, annotate=False, tlm=False)
            self.vector().axpy(alpha, x_.vector())

    def _inner(self, y):
        if isinstance(y, backend_Function):
            if self.vector().local_size() != y.vector().local_size():
                raise InterfaceException("Invalid function space")
            inner = self.vector().inner(y.vector())
        elif isinstance(y, Zero):
            inner = 0.0
        else:
            assert isinstance(y, backend_Constant)
            y_ = backend_Function(self.function_space())
            y_.assign(y, annotate=False, tlm=False)
            inner = self.vector().inner(y_.vector())
        return inner

    def _max_value(self):
        return self.vector().max()

    def _sum(self):
        return self.vector().sum()

    def _linf_norm(self):
        return self.vector().norm("linf")

    def _local_size(self):
        return self.vector().local_size()

    def _global_size(self):
        return self.function_space().dofmap().global_dimension()

    def _local_indices(self):
        return slice(*self.function_space().dofmap().ownership_range())

    def _get_values(self):
        values = self.vector().get_local().view()
        values.setflags(write=False)
        if not np.can_cast(values, np.float64):
            raise InterfaceException("Invalid dtype")
        return values

    def _set_values(self, values):
        if not np.can_cast(values, backend_ScalarType):
            raise InterfaceException("Invalid dtype")
        if values.shape != (self.vector().local_size(),):
            raise InterfaceException("Invalid shape")
        self.vector().set_local(values)
        self.vector().apply("insert")

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        y = function_copy(self, name=name, static=static, cache=cache,
                          checkpoint=checkpoint)
        y.vector().zero()
        return y

    def _copy(self, name=None, static=False, cache=None, checkpoint=None):
        y = self.copy(deepcopy=True)
        if name is not None:
            y.rename(name, "a Function")
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static
        self._tlm_adjoint__function_interface_attrs["static"] = static
        self._tlm_adjoint__function_interface_attrs["cache"] = cache
        self._tlm_adjoint__function_interface_attrs["checkpoint"] = checkpoint
        # Backwards compatibility
        self.is_static = types.MethodType(Function.is_static, self)
        self.is_cached = types.MethodType(Function.is_cached, self)
        self.is_checkpointed = types.MethodType(Function.is_checkpointed, self)
        return y

    def _tangent_linear(self, name=None):
        if function_is_static(self):
            return None
        else:
            return function_new(self, name=name, static=False,
                                cache=function_is_cached(self),
                                checkpoint=function_is_checkpointed(self))

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = Replacement(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_real(self):
        return (is_valid_r0_space(self.function_space())
                and len(self.ufl_shape) == 0)

    def _real_value(self):
        # assert is_real_function(self)
        return self.vector().max()


def _Function__init__(self, *args, **kwargs):
    backend_Function._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionInterface,
                  {"id": new_function_id(), "state": 0,
                   "static": False, "cache": False, "checkpoint": True})
    if isinstance(args[0], backend_FunctionSpace):
        space = args[0]
    else:
        space = backend_Function._tlm_adjoint__orig_function_space(self)
    self._tlm_adjoint__function_interface_attrs["space"] = space


backend_Function._tlm_adjoint__orig___init__ = backend_Function.__init__
backend_Function.__init__ = _Function__init__


def _Function_function_space(self):
    if hasattr(self, "_tlm_adjoint__function_interface_attrs") \
            and "space" in self._tlm_adjoint__function_interface_attrs:
        return self._tlm_adjoint__function_interface_attrs["space"]
    else:
        return backend_Function._tlm_adjoint__orig_function_space(self)


backend_Function._tlm_adjoint__orig_function_space = backend_Function.function_space  # noqa: E501
backend_Function.function_space = _Function_function_space


def _new_real_function(name=None, comm=None, static=False, cache=None,
                       checkpoint=None):
    return Constant(0.0, name=name, comm=comm, static=static, cache=cache,
                    checkpoint=checkpoint)


add_new_real_function(backend, _new_real_function)


def _subtract_adjoint_derivative_action(x, y):
    if isinstance(y, backend_Vector):
        y = (1.0, y)
    if isinstance(y, ufl.classes.Form) \
            and isinstance(x, (backend_Constant, backend_Function)):
        if hasattr(x, "_tlm_adjoint__fenics_adj_b"):
            x._tlm_adjoint__fenics_adj_b += form_neg(y)
        else:
            x._tlm_adjoint__fenics_adj_b = form_neg(y)
    elif isinstance(y, tuple) \
            and len(y) == 2 \
            and isinstance(y[0], (int, float)) \
            and isinstance(y[1], backend_Vector):
        alpha, y = y
        alpha = float(alpha)
        if isinstance(x, backend_Constant):
            if len(x.ufl_shape) == 0:
                # annotate=False, tlm=False
                x.assign(float(x) - alpha * y.max())
            else:
                y_fn = Function(r0_space(x))

                # Ordering check
                check_values = np.arange(np.prod(x.ufl_shape),
                                         dtype=np.float64)
                # annotate=False, tlm=False
                y_fn.assign(backend_Constant(check_values.reshape(x.ufl_shape)))  # noqa: E501
                for i, y_fn_c in enumerate(y_fn.split(deepcopy=True)):
                    assert y_fn_c.vector().max() == check_values[i]
                y_fn.vector().zero()

                value = x.values()
                y_fn.vector().axpy(1.0, y)
                for i, y_fn_c in enumerate(y_fn.split(deepcopy=True)):
                    value[i] -= alpha * y_fn_c.vector().max()
                value.shape = x.ufl_shape
                # annotate=False, tlm=False
                x.assign(backend_Constant(value))
        elif isinstance(x, backend_Function):
            if x.vector().local_size() != y.local_size():
                raise InterfaceException("Invalid function space")
            x.vector().axpy(-alpha, y)
        else:
            return NotImplemented
    else:
        return NotImplemented


add_subtract_adjoint_derivative_action(backend,
                                       _subtract_adjoint_derivative_action)


def _finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__fenics_adj_b"):
        if isinstance(x, backend_Constant):
            y = assemble(x._tlm_adjoint__fenics_adj_b)
            subtract_adjoint_derivative_action(x, (-1.0, y))
        else:
            assemble(x._tlm_adjoint__fenics_adj_b, tensor=x.vector(),
                     add_values=True)
        delattr(x, "_tlm_adjoint__fenics_adj_b")


add_finalize_adjoint_derivative_action(backend,
                                       _finalize_adjoint_derivative_action)


def _functional_term_eq(term, x):
    if isinstance(term, ufl.classes.Form) \
            and len(term.arguments()) == 0 \
            and isinstance(x, (backend_Constant, backend_Function)):
        return AssembleSolver(term, x)
    else:
        return NotImplemented


add_functional_term_eq(backend, _functional_term_eq)


def _time_system_eq(*args, **kwargs):
    if len(args) >= 1:
        eq = args[0]
    elif "eq" in kwargs:
        eq = kwargs["eq"]
    else:
        return NotImplemented

    if len(args) >= 2:
        x = args[1]
    elif "x" in kwargs:
        x = kwargs["x"]
    else:
        return NotImplemented

    if isinstance(eq, ufl.classes.Equation) \
            and isinstance(x, backend_Function):
        return EquationSolver(*args, **kwargs)
    else:
        return NotImplemented


add_time_system_eq(backend, _time_system_eq)


def default_comm():
    warnings.warn("default_comm is deprecated -- "
                  "use mpi4py.MPI.COMM_WORLD instead",
                  DeprecationWarning, stacklevel=2)
    return MPI.COMM_WORLD


def RealFunctionSpace(comm=MPI.COMM_WORLD):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_real_function instead",
                  DeprecationWarning, stacklevel=2)
    return FunctionSpace(UnitIntervalMesh(comm, comm.size), "R", 0)


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
