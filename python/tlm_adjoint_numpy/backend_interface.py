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

from .interface import *
from .interface import FunctionInterface as _FunctionInterface

import copy
import numpy as np
import sys

__all__ = \
    [
        "InterfaceException",

        "Function",
        "FunctionSpace",
        "RealFunctionSpace",
        "Replacement",
        "clear_caches",
        "copy_parameters_dict",
        "default_comm",
        "finalize_adjoint_derivative_action",
        "function_alias",
        "function_assign",
        "function_axpy",
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


def clear_caches(*deps):
    pass


def info(message):
    sys.stdout.write(f"{message:s}\n")
    sys.stdout.flush()


def warning(message):
    sys.stderr.write(f"{message:s}\n")
    sys.stderr.flush()


def copy_parameters_dict(parameters):
    return copy.deepcopy(parameters)


class FunctionSpaceInterface(SpaceInterface):
    def id(self):
        return self._space.dim()

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._space, name=name, static=static, cache=cache,
                        checkpoint=checkpoint, tlm_depth=tlm_depth)


class FunctionSpace:
    def __init__(self, dim):
        self._dim = dim
        self._tlm_adjoint__space_interface = FunctionSpaceInterface(self)

    def dim(self):
        return self._dim


def RealFunctionSpace(comm=None):
    return FunctionSpace(1)


class SerialComm:
    # Interface as in mpi4py 3.0.1
    def allgather(self, sendobj):
        v = sendobj.view()
        v.setflags(write=False)
        return (v,)

    def barrier(self):
        pass

    # Interface as in mpi4py 3.0.1
    def bcast(self, obj, root=0):
        return copy.deepcopy(obj)

    def py2f(self):
        return 0

    @property
    def rank(self):
        return 0

    @property
    def size(self):
        return 1


_comm = SerialComm()


class FunctionInterface(_FunctionInterface):
    def comm(self):
        return _comm

    def space(self):
        return self._x.space()

    def id(self):
        return self._x.id()

    def name(self):
        return self._x.name()

    def state(self):
        return self._x.state()

    def update_state(self):
        self._x.update_state()

    def is_static(self):
        return self._x.is_static()

    def is_cached(self):
        return self._x.is_cached()

    def is_checkpointed(self):
        return self._x.is_checkpointed()

    def tlm_depth(self):
        return self._x.tlm_depth()

    def zero(self):
        self._x.vector()[:] = 0.0

    def assign(self, y):
        if isinstance(y, (int, float)):
            self._x.vector()[:] = y
        else:
            self._x.vector()[:] = y.vector()

    def axpy(self, alpha, y):
        self._x.vector()[:] += alpha * y.vector()

    def inner(self, y):
        return self._x.vector().dot(y.vector())

    def max_value(self):
        return self._x.vector().max()

    def sum(self):
        return self._x.vector().sum()

    def linf_norm(self):
        return abs(self._x.vector()).max()

    def local_size(self):
        return self._x.vector().shape[0]

    def global_size(self):
        return self._x.vector().shape[0]

    def local_indices(self):
        return slice(0, self._x.vector().shape[0])

    def get_values(self):
        values = self._x.vector().view()
        values.setflags(write=False)
        return values

    def set_values(self, values):
        self._x.vector()[:] = values

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._x.space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)

    def copy(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        return Function(self._x.space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth, _data=self._x.vector().copy())

    def tangent_linear(self, name=None):
        return self._x.tangent_linear(name=name)

    def replacement(self):
        return self._x.replacement()

    def alias(self):
        return Function(self._x.space(), name=self._x.name(),
                        static=self._x.is_static(), cache=self._x.is_cached(),
                        checkpoint=self._x.is_checkpointed(),
                        tlm_depth=self._x.tlm_depth(), _data=self._x.vector())


class Function:
    _id_counter = [0]

    def __init__(self, space, name=None, static=False, cache=None,
                 checkpoint=None, tlm_depth=0, _data=None):
        id = self._id_counter[0]
        self._id_counter[0] += 1
        if name is None:
            # Following FEniCS 2019.1.0 behaviour
            name = f"f_{id:d}"
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static

        self._space = space
        self._name = name
        self._state = 0
        self._static = static
        self._cache = cache
        self._checkpoint = checkpoint
        self._tlm_depth = tlm_depth
        self._replacement = None
        self._id = id
        if _data is None:
            self._data = np.zeros(space.dim(), dtype=np.float64)
        else:
            self._data = _data
        self._tlm_adjoint__function_interface = FunctionInterface(self)

    def space(self):
        return self._space

    def id(self):
        return self._id

    def name(self):
        return self._name

    def state(self):
        return self._state

    def update_state(self):
        self._state += 1

    def is_static(self):
        return self._static

    def is_cached(self):
        return self._cache

    def is_checkpointed(self):
        return self._checkpoint

    def tlm_depth(self):
        return self._tlm_depth

    def tangent_linear(self, name=None):
        if self.is_static():
            return None
        else:
            return Function(self.space(), name=name, static=False,
                            cache=self.is_cached(),
                            checkpoint=self.is_checkpointed(),
                            tlm_depth=self.tlm_depth() + 1)

    def replacement(self):
        if self._replacement is None:
            self._replacement = Replacement(self)
        return self._replacement

    def vector(self):
        return self._data


class ReplacementInterface(_FunctionInterface):
    def space(self):
        return self._x.space()

    def id(self):
        return self._x.id()

    def name(self):
        return self._x.name()

    def state(self):
        return -1

    def is_static(self):
        return self._x.is_static()

    def is_cached(self):
        return self._x.is_cached()

    def is_checkpointed(self):
        return self._x.is_checkpointed()

    def tlm_depth(self):
        return self._x.tlm_depth()

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return Function(self._x.space(), name=name, static=static, cache=cache,
                        checkpoint=checkpoint, tlm_depth=tlm_depth)


class Replacement:
    def __init__(self, x):
        self._space = x.space()
        self._name = x.name()
        self._static = x.is_static()
        self._cache = x.is_cached()
        self._checkpoint = x.is_checkpointed()
        self._tlm_depth = x.tlm_depth()
        self._id = x.id()
        self._tlm_adjoint__function_interface = ReplacementInterface(self)

    def space(self):
        return self._space

    def id(self):
        return self._id

    def name(self):
        return self._name

    def is_static(self):
        return self._static

    def is_cached(self):
        return self._cache

    def is_checkpointed(self):
        return self._checkpoint

    def tlm_depth(self):
        return self._tlm_depth


def new_real_function(name=None, static=False, cache=None, checkpoint=None,
                      tlm_depth=0, comm=None):
    return Function(FunctionSpace(1), name=name, static=static, cache=cache,
                    checkpoint=checkpoint, tlm_depth=tlm_depth)


def default_comm():
    return _comm


def subtract_adjoint_derivative_action(x, y):
    if y is None:
        pass
    elif isinstance(y, tuple):
        alpha, y = y
        if isinstance(y, Function):
            y = y.vector()
        if alpha == 1.0:
            x.vector()[:] -= y
        else:
            x.vector()[:] -= alpha * y
    else:
        if isinstance(y, Function):
            y = y.vector()
        x.vector()[:] -= y


def finalize_adjoint_derivative_action(x):
    pass
