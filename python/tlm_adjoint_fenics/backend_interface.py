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


def default_comm():
    comm = mpi_comm_world()
    # FEniCS backwards compatibility
    if hasattr(comm, "tompi4py"):
        comm = comm.tompi4py()
    return comm


class FunctionSpaceInterface(SpaceInterface):
    def _id(self):
        return self.id()

    def _new(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        return Function(self, name=name, static=static, cache=cache,
                        checkpoint=checkpoint, tlm_depth=tlm_depth)


_orig_FunctionSpace__init__ = FunctionSpace.__init__


def _FunctionSpace__init__(self, *args, **kwargs):
    _orig_FunctionSpace__init__(self, *args, **kwargs)
    add_interface(self, FunctionSpaceInterface)


FunctionSpace.__init__ = _FunctionSpace__init__


def RealFunctionSpace(comm=None):
    if comm is None:
        # FEniCS backwards compatibility
        comm = mpi_comm_world()
    return FunctionSpace(UnitIntervalMesh(comm, comm.size), "R", 0)


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        comm = self.function_space().mesh().mpi_comm()
        # FEniCS backwards compatibility
        if hasattr(comm, "tompi4py"):
            comm = comm.tompi4py()
        return comm

    def _space(self):
        return self.function_space()

    def _id(self):
        return self.id()

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

    def _tlm_depth(self):
        if hasattr(self, "tlm_depth"):
            return self.tlm_depth()
        else:
            return 0

    def _caches(self):
        if not hasattr(self, "_tlm_adjoint__caches"):
            self._tlm_adjoint__caches = Caches(self)
        return self._tlm_adjoint__caches

    def _zero(self):
        self.vector().zero()

    def _assign(self, y):
        if isinstance(y, (int, float)):
            self.vector()[:] = float(y)
        else:
            self.vector().zero()
            self.vector().axpy(1.0, y.vector())

    def _axpy(self, alpha, y):
        self.vector().axpy(alpha, y.vector())

    def _inner(self, y):
        return self.vector().inner(y.vector())

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
        return values

    def _set_values(self, values):
        self.vector().set_local(values)
        self.vector().apply("insert")

    def _new(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        y = function_copy(self, name=name, static=static, cache=cache,
                          checkpoint=checkpoint, tlm_depth=tlm_depth)
        y.vector().zero()
        return y

    def _copy(self, name=None, static=False, cache=None, checkpoint=None,
              tlm_depth=0):
        y = self.copy(deepcopy=True)
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

    def _tangent_linear(self, name=None):
        if hasattr(self, "tangent_linear"):
            return self.tangent_linear(name=name)
        elif function_is_static(self):
            return None
        else:
            return function_new(self, name=name, static=False,
                                cache=function_is_cached(self),
                                checkpoint=function_is_checkpointed(self),
                                tlm_depth=function_tlm_depth(self) + 1)

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = Replacement(self)
        return self._tlm_adjoint__replacement

    def _alias(self):
        y = self.copy(deepcopy=False)
        y.rename(self.name(), "a Function")
        static = function_is_static(self)
        y.is_static = lambda: static
        cache = function_is_cached(self)
        y.is_cached = lambda: cache
        checkpoint = function_is_checkpointed(self)
        y.is_checkpointed = lambda: checkpoint
        tlm_depth = function_tlm_depth(self)
        y.tlm_depth = lambda: tlm_depth
        return y


_orig_Function__init__ = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function__init__(self, *args, **kwargs)
    add_interface(self, FunctionInterface)


backend_Function.__init__ = _Function__init__


def new_real_function(name=None, comm=None, static=False, cache=None,
                      checkpoint=None, tlm_depth=0):
    return Constant(0.0, name=name, static=static, cache=cache,
                    checkpoint=checkpoint, tlm_depth=tlm_depth, comm=comm)


# def clear_caches(*deps):


# def info(message):


def warning(message):
    sys.stderr.write(f"{message:s}\n")
    sys.stderr.flush()


# def copy_parameters_dict(parameters):


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
