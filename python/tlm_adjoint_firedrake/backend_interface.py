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
from .backend_code_generator_interface import copy_parameters_dict, \
    is_real_function
from .interface import *
from .interface import FunctionInterface as _FunctionInterface

from .caches import FunctionCaches, clear_caches, form_neg
from .functions import Function, ReplacementFunction

import ufl
import sys

__all__ = \
    [
        "InterfaceException",

        "Function",
        "RealFunctionSpace",
        "ReplacementFunction",
        "clear_caches",
        "copy_parameters_dict",
        "default_comm",
        "function_alias",
        "function_assign",
        "function_axpy",
        "function_caches",
        "function_comm",
        "function_copy",
        "function_finalize_adjoint_derivative_action",
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
        "function_subtract_adjoint_derivative_action",
        "function_sum",
        "function_tangent_linear",
        "function_tlm_depth",
        "function_update_state",
        "function_zero",
        "info",
        "is_function",
        "space_id",
        "space_new",
        "warning"
    ]


class FunctionInterface(_FunctionInterface):
    def comm(self):
        return self._x.comm

    def space(self):
        return self._x.function_space()

    def id(self):
        return self._x.count()

    def name(self):
        return self._x.name()

    def state(self):
        if not hasattr(self._x, "_tlm_adjoint__state"):
            self._x._tlm_adjoint__state = 0
        return self._x._tlm_adjoint__state

    def update_state(self):
        if hasattr(self._x, "_tlm_adjoint__state"):
            self._x._tlm_adjoint__state += 1
        else:
            self._x._tlm_adjoint__state = 1

    def is_static(self):
        if hasattr(self._x, "is_static"):
            return self._x.is_static()
        else:
            return False

    def is_cached(self):
        if hasattr(self._x, "is_cached"):
            return self._x.is_cached()
        else:
            return False

    def is_checkpointed(self):
        if hasattr(self._x, "is_checkpointed"):
            return self._x.is_checkpointed()
        else:
            return True

    def tlm_depth(self):
        if hasattr(self._x, "tlm_depth"):
            return self._x.tlm_depth()
        else:
            return 0

    def caches(self):
        if not hasattr(self._x, "_tlm_adjoint__function_caches"):
            self._x._tlm_adjoint__function_caches = FunctionCaches(self._x)
        return self._x._tlm_adjoint__function_caches

    def zero(self):
        with self._x.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def assign(self, y):
        if isinstance(y, (int, float)):
            if is_real_function(self._x):
                # Work around Firedrake issue #1459
                self._x.dat.data[:] = y
            else:
                with self._x.dat.vec_wo as x_v:
                    x_v.set(float(y))
        else:
            if is_real_function(self._x):
                # Work around Firedrake bug (related to issue #1459?)
                self._x.dat.data[:] = y.dat.data_ro
            else:
                with self._x.dat.vec_wo as x_v, y.dat.vec_ro as y_v:
                    y_v.copy(result=x_v)

    def axpy(self, alpha, y):
        if is_real_function(self._x):
            # Work around Firedrake bug (related to issue #1459?)
            self._x.dat.data[:] += alpha * y.dat.data_ro
        else:
            with self._x.dat.vec as x_v, y.dat.vec_ro as y_v:
                x_v.axpy(alpha, y_v)

    def inner(self, y):
        with self._x.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
            inner = x_v.dot(y_v)
        return inner

    def max_value(self):
        with self._x.dat.vec_ro as x_v:
            max = x_v.max()[1]
        return max

    def sum(self):
        with self._x.dat.vec_ro as x_v:
            sum = x_v.sum()
        return sum

    def linf_norm(self):
        import petsc4py.PETSc as PETSc
        with self._x.dat.vec_ro as x_v:
            linf_norm = x_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)
        return linf_norm

    def local_size(self):
        with self._x.dat.vec_ro as x_v:
            local_size = x_v.getLocalSize()
        return local_size

    def global_size(self):
        with self._x.dat.vec_ro as x_v:
            size = x_v.getSize()
        return size

    def local_indices(self):
        with self._x.dat.vec_ro as x_v:
            local_range = x_v.getOwnershipRange()
        return slice(*local_range)

    def get_values(self):
        with self._x.dat.vec_ro as x_v:
            values = x_v.getArray(readonly=True)
        return values

    def set_values(self, values):
        with self._x.dat.vec_wo as x_v:
            x_v.setArray(values)

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._x.function_space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)

    def copy(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        y = function_new(self._x, name=name, static=static, cache=cache,
                         checkpoint=checkpoint, tlm_depth=tlm_depth)
        function_assign(y, self._x)
        return y

    def tangent_linear(self, name=None):
        if hasattr(self._x, "tangent_linear"):
            return self._x.tangent_linear(name=name)
        elif function_is_static(self._x):
            return None
        else:
            return function_new(self._x, name=name, static=False,
                                cache=function_is_cached(self._x),
                                checkpoint=function_is_checkpointed(self._x),
                                tlm_depth=function_tlm_depth(self._x) + 1)

    def replacement(self):
        if not hasattr(self._x, "_tlm_adjoint__replacement"):
            self._x._tlm_adjoint__replacement = ReplacementFunction(self._x)
        return self._x._tlm_adjoint__replacement

    def alias(self):
        return Function(self._x.function_space(), name=function_name(self._x),
                        static=function_is_static(self._x),
                        cache=function_is_cached(self._x),
                        checkpoint=function_is_checkpointed(self._x),
                        tlm_depth=function_tlm_depth(self._x), val=self._x.dat)


_orig_Function__init__ = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function__init__(self, *args, **kwargs)
    self._tlm_adjoint__function_interface = FunctionInterface(self)


backend_Function.__init__ = _Function__init__


class FunctionSpaceInterface(SpaceInterface):
    _id_counter = [0]

    def __init__(self, space):
        SpaceInterface.__init__(self, space)
        self._id = self._id_counter[0]
        self._id_counter[0] += 1

    def id(self):
        return self._id

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._space, name=name, static=static, cache=cache,
                        checkpoint=checkpoint, tlm_depth=tlm_depth)


_orig_WithGeometry__init__ = backend_WithGeometry.__init__


def _WithGeometry__init__(self, *args, **kwargs):
    _orig_WithGeometry__init__(self, *args, **kwargs)
    self._tlm_adjoint__space_interface = FunctionSpaceInterface(self)


backend_WithGeometry.__init__ = _WithGeometry__init__


# def clear_caches(*deps):


def info(message):
    sys.stdout.write(f"{message:s}\n")
    sys.stdout.flush()


def warning(message):
    sys.stderr.write(f"{message:s}\n")
    sys.stderr.flush()


# def copy_parameters_dict(parameters):


def RealFunctionSpace(comm=None):
    if comm is None:
        comm = default_comm()
    return FunctionSpace(UnitIntervalMesh(comm.size, comm=comm), "R", 0)


def default_comm():
    import mpi4py.MPI as MPI
    return MPI.COMM_WORLD


def function_subtract_adjoint_derivative_action(x, y):
    if y is None:
        pass
    elif isinstance(y, tuple):
        alpha, y = y
        function_axpy(x, -alpha, y)
    elif isinstance(y, ufl.classes.Form):
        if hasattr(x, "_tlm_adjoint__adj_b"):
            x._tlm_adjoint__adj_b += form_neg(y)
        else:
            x._tlm_adjoint__adj_b = form_neg(y)
    else:
        function_axpy(x, -1.0, y)


def function_finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__adj_b"):
        function_axpy(x, 1.0, backend_assemble(x._tlm_adjoint__adj_b))
        delattr(x, "_tlm_adjoint__adj_b")
