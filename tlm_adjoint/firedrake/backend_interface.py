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
from ..interface import InterfaceException, SpaceInterface, \
    add_finalize_adjoint_derivative_action, add_functional_term_eq, \
    add_interface, add_new_real_function, \
    add_subtract_adjoint_derivative_action, add_time_system_eq, \
    function_assign, function_caches, function_comm, function_is_cached, \
    function_is_checkpointed, function_is_static, function_new, \
    is_real_function, new_function_id, new_space_id, real_function_value, \
    space_id, space_new, subtract_adjoint_derivative_action
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import assemble, is_valid_r0_space

from .caches import form_neg
from .equations import AssembleSolver, EquationSolver
from .functions import Caches, Constant, Function, Replacement, Zero

import mpi4py.MPI as MPI
import numpy as np
import petsc4py.PETSc as PETSc
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
        return self.comm

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
        return self.comm

    def _space(self):
        return self.function_space()

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
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def _assign(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                y_v.copy(result=x_v)
        elif isinstance(y, (int, float)):
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(float(y)),
                            annotate=False, tlm=False)
            else:
                y_arr = np.full(self.ufl_shape, float(y), dtype=np.float64)
                self.assign(backend_Constant(y_arr),
                            annotate=False, tlm=False)
        elif isinstance(y, Zero):
            with self.dat.vec_wo as x_v:
                x_v.zeroEntries()
        else:
            assert isinstance(y, backend_Constant)
            self.assign(y, annotate=False, tlm=False)

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            values = function_comm(self).bcast(values, root=0)
            self.dat.data[:] = values

    def _axpy(self, *args):  # self, alpha, x
        alpha, x = args
        alpha = float(alpha)
        if isinstance(x, backend_Function):
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                if y_v.getLocalSize() != x_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                y_v.axpy(alpha, x_v)
        elif isinstance(x, (int, float)):
            self.assign(self + alpha * float(x), annotate=False, tlm=False)
        elif isinstance(x, Zero):
            pass
        else:
            assert isinstance(x, backend_Constant)
            self.assign(self + alpha * x, annotate=False, tlm=False)

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            values = function_comm(self).bcast(values, root=0)
            self.dat.data[:] = values

    def _inner(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                inner = x_v.dot(y_v)
        elif isinstance(y, Zero):
            inner = 0.0
        else:
            assert isinstance(y, backend_Constant)
            y_ = backend_Function(self.function_space())
            y_.assign(y, annotate=False, tlm=False)
            with self.dat.vec_ro as x_v, y_.dat.vec_ro as y_v:
                inner = x_v.dot(y_v)
        return inner

    def _max_value(self):
        with self.dat.vec_ro as x_v:
            max = x_v.max()[1]
        return max

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
        if not np.can_cast(values, np.float64):
            raise InterfaceException("Invalid dtype")
        return values

    def _set_values(self, values):
        if not np.can_cast(values, backend_ScalarType):
            raise InterfaceException("Invalid dtype")
        with self.dat.vec as x_v:
            if values.shape != (x_v.getLocalSize(),):
                raise InterfaceException("Invalid shape")
            x_v.setArray(values)

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self.function_space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint)

    def _copy(self, name=None, static=False, cache=None, checkpoint=None):
        y = function_new(self, name=name, static=static, cache=cache,
                         checkpoint=checkpoint)
        function_assign(y, self)
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
        with self.dat.vec_ro as x_v:
            max = x_v.max()[1]
        return max


def _Function__init__(self, *args, **kwargs):
    backend_Function._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionInterface,
                  {"id": new_function_id(), "state": 0,
                   "static": False, "cache": False, "checkpoint": True})


backend_Function._tlm_adjoint__orig___init__ = backend_Function.__init__
backend_Function.__init__ = _Function__init__


def _new_real_function(name=None, comm=None, static=False, cache=None,
                       checkpoint=None):
    return Constant(0.0, name=name, comm=comm, static=static, cache=cache,
                    checkpoint=checkpoint)


add_new_real_function(backend, _new_real_function)


def _subtract_adjoint_derivative_action(x, y):
    if isinstance(y, ufl.classes.Form) \
            and isinstance(x, (backend_Constant, backend_Function)):
        if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
            x._tlm_adjoint__firedrake_adj_b += form_neg(y)
        else:
            x._tlm_adjoint__firedrake_adj_b = form_neg(y)
    elif isinstance(x, backend_Constant):
        if isinstance(y, backend_Function) and is_real_function(y):
            alpha = 1.0
        elif isinstance(y, tuple) \
                and len(y) == 2 \
                and isinstance(y[0], (int, float)) \
                and isinstance(y[1], backend_Function) \
                and is_real_function(y[1]):
            alpha, y = y
            alpha = float(alpha)
        else:
            return NotImplemented
        y_value = real_function_value(y)
        # annotate=False, tlm=False
        x.assign(float(x) - alpha * y_value)
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


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_real_function instead",
                  DeprecationWarning, stacklevel=2)
    if comm is None:
        comm = MPI.COMM_WORLD
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
