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

from .backend import *
from .backend_code_generator_interface import assemble, copy_parameters_dict
from .interface import *
from .interface import FunctionInterface as _FunctionInterface

from .caches import clear_caches, form_neg
from .functions import Caches, Constant, Function, Replacement, Zero, \
    is_r0_function

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc
import ufl
import sys
import warnings

__all__ = \
    [
        "InterfaceException",

        "is_space",
        "space_id",
        "space_new",

        "is_function",
        "function_assign",
        "function_axpy",
        "function_caches",
        "function_comm",
        "function_copy",
        "function_get_values",
        "function_global_size",
        "function_id",
        "function_inner",
        "function_is_cached",
        "function_is_checkpointed",
        "function_is_static",
        "function_linf_norm",
        "function_local_indices",
        "function_local_size",
        "function_max_value",
        "function_name",
        "function_new",
        "function_replacement",
        "function_set_values",
        "function_space",
        "function_state",
        "function_sum",
        "function_tangent_linear",
        "function_update_state",
        "function_zero",

        "is_real_function",
        "new_real_function",
        "real_function_value",

        "clear_caches",
        "copy_parameters_dict",
        "default_comm",
        "finalize_adjoint_derivative_action",
        "info",
        "subtract_adjoint_derivative_action",

        "Constant",
        "Function",
        "Replacement",

        "RealFunctionSpace",
        "function_space_id",
        "function_space_new",
        "warning"
    ]


def default_comm():
    return MPI.COMM_WORLD


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self, name=name, static=static, cache=cache,
                        checkpoint=checkpoint)


def _WithGeometry__init__(self, *args, **kwargs):
    _WithGeometry__init__._tlm_adjoint__orig(self, *args, **kwargs)
    id = _space_id_counter[0]
    _space_id_counter[0] += 1
    add_interface(self, FunctionSpaceInterface,
                  {"id": id})


_space_id_counter = [0]
_WithGeometry__init__._tlm_adjoint__orig = backend_WithGeometry.__init__
backend_WithGeometry.__init__ = _WithGeometry__init__


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return self.comm

    def _space(self):
        return self.function_space()

    def _id(self):
        return self.count()

    def _name(self):
        return self.name()

    def _state(self):
        if not hasattr(self, "_tlm_adjoint__state"):
            self._tlm_adjoint__state = 0
        return self._tlm_adjoint__state

    def _update_state(self):
        if hasattr(self, "_tlm_adjoint__state"):
            self._tlm_adjoint__state += 1
        else:
            self._tlm_adjoint__state = 1

    def _is_static(self):
        if hasattr(self, "is_static"):
            return self.is_static()
        else:
            return False

    def _is_cached(self):
        if hasattr(self, "is_cached"):
            return self.is_cached()
        else:
            return False

    def _is_checkpointed(self):
        if hasattr(self, "is_checkpointed"):
            return self.is_checkpointed()
        else:
            return True

    def _caches(self):
        if not hasattr(self, "_tlm_adjoint__caches"):
            self._tlm_adjoint__caches = Caches(self)
        return self._tlm_adjoint__caches

    def _zero(self):
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def _assign(self, y):
        if isinstance(y, backend_Function):
            if is_r0_function(self):
                # Work around Firedrake bug (related to issue #1459?)
                self.dat.data[:] = y.dat.data_ro
            else:
                with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                    if x_v.getLocalSize() != y_v.getLocalSize():
                        raise InterfaceException("Invalid function space")
                    y_v.copy(result=x_v)
        elif isinstance(y, (int, float)):
            if is_r0_function(self):
                # Work around Firedrake issue #1459
                self.dat.data[:] = float(y)
            else:
                with self.dat.vec_wo as x_v:
                    x_v.set(float(y))
        elif isinstance(y, Zero):
            with self.dat.vec_wo as x_v:
                x_v.zeroEntries()
        else:
            assert isinstance(y, backend_Constant)
            if len(y.ufl_shape) == 0:
                with self.dat.vec_wo as x_v:
                    x_v.set(float(y))
            else:
                self.assign(y, annotate=False, tlm=False)

    def _axpy(self, alpha, y):
        if isinstance(y, backend_Function):
            if is_r0_function(self):
                # Work around Firedrake bug (related to issue #1459?)
                self.dat.data[:] += alpha * y.dat.data_ro
            else:
                with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                    if x_v.getLocalSize() != y_v.getLocalSize():
                        raise InterfaceException("Invalid function space")
                    x_v.axpy(alpha, y_v)
        elif isinstance(y, Zero):
            pass
        else:
            assert isinstance(y, backend_Constant)
            if is_r0_function(self):
                # Work around Firedrake bug (related to issue #1459?)
                if len(self.ufl_shape) == 0:
                    self.dat.data[:] += alpha * float(y)
                else:
                    x = function_new(self)
                    x.assign(y, annotate=False, tlm=False)
                    self.dat.data[:] = x.dat.data_ro + alpha * y.dat.data_ro
            else:
                self += alpha * y  # annotate=False, tlm=False

    def _inner(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                inner = x_v.dot(y_v)
            return inner
        elif isinstance(y, Zero):
            return 0.0
        else:
            assert isinstance(y, backend_Constant)
            if len(y.ufl_shape) == 0:
                with self.dat.vec_ro as x_v:
                    sum = x_v.sum()
                return sum * float(y)
            else:
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
            values = x_v.getArray(readonly=True)
        return values

    def _set_values(self, values):
        with self.dat.vec_wo as x_v:
            if (x_v.getLocalSize(),) != values.shape:
                raise InterfaceException("Invalid function space")
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
        if hasattr(self, "tangent_linear"):
            return self.tangent_linear(name=name)
        elif function_is_static(self):
            return None
        else:
            return function_new(self, name=name, static=False,
                                cache=function_is_cached(self),
                                checkpoint=function_is_checkpointed(self))

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = Replacement(self)
        return self._tlm_adjoint__replacement


def _Function__init__(self, *args, **kwargs):
    _Function__init__._tlm_adjoint__orig(self, *args, **kwargs)
    add_interface(self, FunctionInterface)


_Function__init__._tlm_adjoint__orig = backend_Function.__init__
backend_Function.__init__ = _Function__init__


def is_real_function(x):
    return is_r0_function(x) and len(x.ufl_shape) == 0


def new_real_function(name=None, domain=None, comm=None, static=False,
                      cache=None, checkpoint=None):
    return Constant(0.0, name=name, domain=domain, comm=comm, static=static,
                    cache=cache, checkpoint=checkpoint)


def real_function_value(x):
    return function_max_value(x)


# def clear_caches(*deps):


def info(message):
    sys.stdout.write(f"{message:s}\n")
    sys.stdout.flush()


# def copy_parameters_dict(parameters):


def subtract_adjoint_derivative_action(x, y):
    if y is None:
        pass
    elif isinstance(y, ufl.classes.Form):
        if hasattr(x, "_tlm_adjoint__adj_b"):
            x._tlm_adjoint__adj_b += form_neg(y)
        else:
            x._tlm_adjoint__adj_b = form_neg(y)
    else:
        if isinstance(y, tuple):
            alpha, y = y
        else:
            alpha = 1.0
        function_axpy(x, -alpha, y)


def finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__adj_b"):
        function_axpy(x, 1.0, assemble(x._tlm_adjoint__adj_b))
        delattr(x, "_tlm_adjoint__adj_b")


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_real_function instead",
                  DeprecationWarning, stacklevel=2)
    if comm is None:
        comm = default_comm()
    return FunctionSpace(UnitIntervalMesh(comm.size, comm=comm), "R", 0)


def function_space_id(*args, **kwargs):
    warnings.warn("function_space_id is deprecated -- use space_id instead",
                  DeprecationWarning, stacklevel=2)
    return space_id(*args, **kwargs)


def function_space_new(*args, **kwargs):
    warnings.warn("function_space_new is deprecated -- use space_new instead",
                  DeprecationWarning, stacklevel=2)
    return space_new(*args, **kwargs)


def warning(message):
    warnings.warn("warning is deprecated -- use warnings.warn instead",
                  DeprecationWarning, stacklevel=2)
    warnings.warn(message, RuntimeWarning)
