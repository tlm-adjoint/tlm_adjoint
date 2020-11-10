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
import warnings

__all__ = \
    [
        "InterfaceException",

        "is_space",
        "space_comm",
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
        "function_update_caches",
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

        "Function",
        "FunctionSpace",
        "Replacement",

        "RealFunctionSpace",
        "function_space_id",
        "function_space_new",
        "warning"
    ]


class SerialComm:
    def Dup(self):
        return SerialComm()

    def Free():
        pass

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


def default_comm():
    return _comm


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return _comm

    def _id(self):
        return self.dim()

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self, name=name, static=static, cache=cache,
                        checkpoint=checkpoint)


class FunctionSpace:
    def __init__(self, dim):
        self._dim = dim
        add_interface(self, FunctionSpaceInterface)

    def dim(self):
        return self._dim


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return _comm

    def _space(self):
        return self.space()

    def _id(self):
        return self.id()

    def _name(self):
        return self.name()

    def _state(self):
        return self.state()

    def _update_state(self):
        self.update_state()

    def _is_static(self):
        return self.is_static()

    def _is_cached(self):
        return self.is_cached()

    def _is_checkpointed(self):
        return self.is_checkpointed()

    def _update_caches(self, value=None):
        pass

    def _zero(self):
        self.vector()[:] = 0.0

    def _assign(self, y):
        if isinstance(y, (int, float)):
            self.vector()[:] = y
        else:
            self.vector()[:] = y.vector()

    def _axpy(self, *args):  # self, alpha, x
        alpha, x = args
        self.vector()[:] += alpha * x.vector()

    def _inner(self, y):
        return self.vector().dot(y.vector())

    def _max_value(self):
        return self.vector().max()

    def _sum(self):
        return self.vector().sum()

    def _linf_norm(self):
        return abs(self.vector()).max()

    def _local_size(self):
        return self.vector().shape[0]

    def _global_size(self):
        return self.vector().shape[0]

    def _local_indices(self):
        return slice(0, self.vector().shape[0])

    def _get_values(self):
        values = self.vector().view()
        values.setflags(write=False)
        if not np.can_cast(values, np.float64):
            raise InterfaceException("Invalid dtype")
        return values

    def _set_values(self, values):
        if not np.can_cast(values, np.float64):
            raise InterfaceException("Invalid dtype")
        if values.shape != self.vector().shape:
            raise InterfaceException("Invalid shape")
        self.vector()[:] = values

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self.space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint)

    def _copy(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self.space(), name=name, static=static, cache=cache,
                        checkpoint=checkpoint, _data=self.vector().copy())

    def _tangent_linear(self, name=None):
        return self.tangent_linear(name=name)

    def _replacement(self):
        return self.replacement()


class Function:
    _id_counter = [0]

    def __init__(self, space, name=None, static=False, cache=None,
                 checkpoint=None, _data=None):
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
        self._replacement = None
        self._id = id
        if _data is None:
            self._data = np.zeros(space.dim(), dtype=np.float64)
        else:
            if not np.can_cast(_data, np.float64):
                raise InterfaceException("Invalid dtype")
            self._data = _data
        add_interface(self, FunctionInterface)

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

    def tangent_linear(self, name=None):
        if self.is_static():
            return None
        else:
            return Function(self.space(), name=name, static=False,
                            cache=self.is_cached(),
                            checkpoint=self.is_checkpointed())

    def replacement(self):
        if self._replacement is None:
            self._replacement = Replacement(self)
        return self._replacement

    def vector(self):
        return self._data


class ReplacementInterface(_FunctionInterface):
    def _space(self):
        return self.space()

    def _id(self):
        return self.id()

    def _name(self):
        return self.name()

    def _state(self):
        return -1

    def _is_static(self):
        return self.is_static()

    def _is_cached(self):
        return self.is_cached()

    def _is_checkpointed(self):
        return self.is_checkpointed()

    def _update_caches(self, value=None):
        if value is None:
            raise InterfaceException("value required")

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        return Function(self.space(), name=name, static=static, cache=cache,
                        checkpoint=checkpoint)

    def _replacement(self):
        return self


class Replacement:
    def __init__(self, x):
        self._space = x.space()
        self._name = x.name()
        self._static = x.is_static()
        self._cache = x.is_cached()
        self._checkpoint = x.is_checkpointed()
        self._id = x.id()
        add_interface(self, ReplacementInterface)

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


def is_real_function(x):
    return is_function(x) and x.space().dim() == 1


def new_real_function(name=None, comm=None, static=False, cache=None,
                      checkpoint=None):
    return Function(FunctionSpace(1), name=name, static=static, cache=cache,
                    checkpoint=checkpoint)


def real_function_value(x):
    assert is_real_function(x)
    value, = x.vector()
    return value


def clear_caches(*deps):
    pass


def info(message):
    sys.stdout.write(f"{message:s}\n")
    sys.stdout.flush()


def copy_parameters_dict(parameters):
    return copy.deepcopy(parameters)


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


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_real_function instead",
                  DeprecationWarning, stacklevel=2)
    return FunctionSpace(1)


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
