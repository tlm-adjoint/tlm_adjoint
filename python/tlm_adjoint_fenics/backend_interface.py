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
from .backend_code_generator_interface import copy_parameters_dict
from .interface import *
from .interface import FunctionInterface as _FunctionInterface

from .caches import clear_caches, form_neg
from .functions import Caches, Constant, Function, Replacement

import ufl
import sys

__all__ = \
    [
        "InterfaceException",

        "Function",
        "RealFunctionSpace",
        "Replacement",
        "clear_caches",
        "copy_parameters_dict",
        "default_comm",
        "finalize_adjoint_derivative_action",
        "function_alias",
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
        "function_tlm_depth",
        "function_update_state",
        "function_zero",
        "info",
        "is_function",
        "new_real_function",
        "space_id",
        "space_new",
        "subtract_adjoint_derivative_action",
        "warning"
    ]


class FunctionInterface(_FunctionInterface):
    def comm(self):
        return self._x.function_space().mesh().mpi_comm()

    def space(self):
        return self._x.function_space()

    def id(self):
        return self._x.id()

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
        if not hasattr(self._x, "_tlm_adjoint__caches"):
            self._x._tlm_adjoint__caches = Caches(self._x)
        return self._x._tlm_adjoint__caches

    def zero(self):
        self._x.vector().zero()

    def assign(self, y):
        if isinstance(y, (int, float)):
            self._x.vector()[:] = float(y)
        else:
            self._x.vector().zero()
            self._x.vector().axpy(1.0, y.vector())

    def axpy(self, alpha, y):
        self._x.vector().axpy(alpha, y.vector())

    def inner(self, y):
        return self._x.vector().inner(y.vector())

    def max_value(self):
        return self._x.vector().max()

    def sum(self):
        return self._x.vector().sum()

    def linf_norm(self):
        return self._x.vector().norm("linf")

    def local_size(self):
        return self._x.vector().local_size()

    def global_size(self):
        return self._x.function_space().dofmap().global_dimension()

    def local_indices(self):
        return slice(*self._x.function_space().dofmap().ownership_range())

    def get_values(self):
        values = self._x.vector().get_local().view()
        values.setflags(write=False)
        return values

    def set_values(self, values):
        self._x.vector().set_local(values)
        self._x.vector().apply("insert")

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        y = function_copy(self._x, name=name, static=static, cache=cache,
                          checkpoint=checkpoint, tlm_depth=tlm_depth)
        y.vector().zero()
        return y

    def copy(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        y = self._x.copy(deepcopy=True)
        if name is not None:
            y.rename(name, "a Function")
        y.is_static = lambda: static
        if cache is None:
            cache = static
        y.is_cached = lambda: cache
        if checkpoint is None:
            checkpoint = not static
        y.is_checkpointed = lambda: checkpoint
        y.tlm_depth = lambda: tlm_depth
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
            self._x._tlm_adjoint__replacement = Replacement(self._x)
        return self._x._tlm_adjoint__replacement

    def alias(self):
        y = self._x.copy(deepcopy=False)
        y.rename(function_name(self._x), "a Function")
        static = function_is_static(self._x)
        y.is_static = lambda: static
        cache = function_is_cached(self._x)
        y.is_cached = lambda: cache
        checkpoint = function_is_checkpointed(self._x)
        y.is_checkpointed = lambda: checkpoint
        tlm_depth = function_tlm_depth(self._x)
        y.tlm_depth = lambda: tlm_depth
        return y


_orig_Function__init__ = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function__init__(self, *args, **kwargs)
    self._tlm_adjoint__function_interface = FunctionInterface(self)


backend_Function.__init__ = _Function__init__


class FunctionSpaceInterface(SpaceInterface):
    def id(self):
        return self._space.id()

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._space, name=name, static=static, cache=cache,
                        checkpoint=checkpoint, tlm_depth=tlm_depth)


_orig_FunctionSpace__init__ = FunctionSpace.__init__


def _FunctionSpace__init__(self, *args, **kwargs):
    _orig_FunctionSpace__init__(self, *args, **kwargs)
    self._tlm_adjoint__space_interface = FunctionSpaceInterface(self)


FunctionSpace.__init__ = _FunctionSpace__init__


# def clear_caches(*deps):


# def info(message):


def warning(message):
    sys.stderr.write(f"{message:s}\n")
    sys.stderr.flush()


# def copy_parameters_dict(parameters):


def RealFunctionSpace(comm=None):
    if comm is None:
        comm = default_comm()
    return FunctionSpace(UnitIntervalMesh(comm, comm.size), "R", 0)


def new_real_function(name=None, static=False, cache=None, checkpoint=None,
                      tlm_depth=0, comm=None):
    return Constant(0.0, name=name, static=static, cache=cache,
                    checkpoint=checkpoint, tlm_depth=tlm_depth, comm=comm)


def default_comm():
    return mpi_comm_world()


def subtract_adjoint_derivative_action(x, y):
    if y is None:
        pass
    elif isinstance(y, tuple):
        alpha, y = y
        if isinstance(x, backend_Function):
            if isinstance(y, backend_Function):
                y = y.vector()
            x.vector().axpy(-alpha, y)
        else:
            function_axpy(x, -alpha, y)
    elif isinstance(y, ufl.classes.Form):
        if hasattr(x, "_tlm_adjoint__adj_b"):
            x._tlm_adjoint__adj_b += form_neg(y)
        else:
            x._tlm_adjoint__adj_b = form_neg(y)
    else:
        if isinstance(x, backend_Function):
            if isinstance(y, backend_Function):
                y = y.vector()
            x.vector().axpy(-1.0, y)
        else:
            function_axpy(x, -1.0, y)


def finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__adj_b"):
        backend_assemble(x._tlm_adjoint__adj_b, tensor=x.vector(),
                         add_values=True)
        delattr(x, "_tlm_adjoint__adj_b")
