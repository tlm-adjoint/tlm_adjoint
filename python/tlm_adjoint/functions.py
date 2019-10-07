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
from .interface import *
from .interface import FunctionInterface as _FunctionInterface

import copy
import numpy as np
import ufl
import weakref

__all__ = \
    [
        "Constant",
        "DirichletBC",
        "Function",
        "Replacement",
        "bcs_is_cached",
        "bcs_is_static",
        "new_count"
    ]


class Caches:
    def __init__(self, x):
        self._caches = weakref.WeakValueDictionary()
        self._id = function_id(x)
        self._state = (self._id, function_state(x))

    def __len__(self):
        return len(self._caches)

    def clear(self):
        for cache in tuple(self._caches.valuerefs()):
            cache = cache()
            if cache is not None:
                cache.clear(self._id)
                assert(not cache.id() in self._caches)

    def add(self, cache):
        cache_id = cache.id()
        if cache_id not in self._caches:
            self._caches[cache_id] = cache

    def remove(self, cache):
        del(self._caches[cache.id()])

    def update(self, x):
        state = (function_id(x), function_state(x))
        if state != self._state:
            self.clear()
            self._state = state


class Alias:
    def __init__(self, obj):
        type(obj).__setattr__(self, "_tlm_adjoint__alias", obj)

    def __new__(cls, obj):
        class Alias(cls, type(obj)):
            pass
        return object.__new__(Alias)

    def __getattr__(self, key):
        return self._tlm_adjoint__alias.__getattr__(self, key)

    def __setattr__(self, key, value):
        return self._tlm_adjoint__alias.__setattr__(self, key, value)

    def __delattr__(self, key):
        self._tlm_adjoint__alias.__delattr__(self, key)

    def __dir__(self):
        return self._tlm_adjoint__alias.__dir__(self)


class ConstantSpaceInterface(SpaceInterface):
    def __init__(self, space, comm):
        SpaceInterface.__init__(self, space)
        self._comm = comm
        self._id = new_count()

    def id(self):
        return self._id

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        shape = self._space.ufl_element().value_shape()
        if len(shape) == 0:
            value = 0.0
        else:
            value = np.zeros(shape, dtype=np.float64)
        return Constant(value, comm=self._comm, name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)


class ConstantInterface(_FunctionInterface):
    def comm(self):
        return self._x.comm()

    def space(self):
        return self._x.ufl_function_space()

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
        return self._x.is_static()

    def is_cached(self):
        return self._x.is_cached()

    def is_checkpointed(self):
        return self._x.is_checkpointed()

    def tlm_depth(self):
        return self._x.tlm_depth()

    def caches(self):
        if not hasattr(self._x, "_tlm_adjoint__caches"):
            self._x._tlm_adjoint__caches = Caches(self._x)
        return self._x._tlm_adjoint__caches

    def zero(self):
        if len(self._x.ufl_shape) == 0:
            value = 0.0
        else:
            value = np.zeros(self._x.ufl_shape, dtype=np.float64)
            value = backend_Constant(value)
        self._x.assign(value)

    def assign(self, y):
        self._x.assign(y)

    def axpy(self, alpha, y):
        if len(self._x.ufl_shape) == 0:
            value = float(self._x) + alpha * float(y)
        else:
            value = self._x.values() + alpha * y.values()
            value = backend_Constant(value)
        self._x.assign(value)

    def inner(self, y):
        return (self._x.values() * y.values()).sum()

    def max_value(self):
        return self._x.values().max()

    def sum(self):
        return self._x.values().sum()

    def linf_norm(self):
        return abs(self._x.values()).max()

    def local_size(self):
        comm = self._x.comm()
        if comm.rank == 0:
            if len(self._x.ufl_shape) == 0:
                return 1
            else:
                return np.prod(self._x.ufl_shape)
        else:
            return 0

    def global_size(self):
        if len(self._x.ufl_shape) == 0:
            return 1
        else:
            return np.prod(self._x.ufl_shape)

    def local_indices(self):
        comm = self._x.comm()
        if comm.rank == 0:
            if len(self._x.ufl_shape) == 0:
                return slice(0, 1)
            else:
                return slice(0, np.prod(self._x.ufl_shape))
        else:
            return slice(0, 0)

    def get_values(self):
        comm = self._x.comm()
        if comm.rank == 0:
            values = self._x.values().view()
        else:
            values = np.array([], dtype=np.float64)
        values.setflags(write=False)
        return values

    def set_values(self, values):
        comm = self._x.comm()
        if comm.rank != 0:
            if len(self._x.value_shape) == 0:
                values = np.array([0.0], dtype=np.float64)
            else:
                values = np.zeros(self._x.value_shape, dtype=np.float64)
        values = comm.bcast(values, root=0)
        if len(self._x.value_shape) == 0:
            self._x.assign(values[0])
        else:
            self._x.assign(backend_Constant(values))

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        if len(self._x.ufl_shape) == 0:
            value = 0.0
        else:
            value = np.zeros(self._x.ufl_shape, dtype=np.float64)
        return Constant(value, comm=self._x.comm(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)

    def copy(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        if len(self._x.ufl_shape) == 0:
            value = float(self._x)
        else:
            value = self._x.values()
        return Constant(value, comm=self._x.comm(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)

    def tangent_linear(self, name=None):
        return self._x.tangent_linear(name=name)

    def replacement(self):
        if not hasattr(self._x, "_tlm_adjoint__replacement"):
            # Firedrake requires Constant.function_space() to return None
            self._x._tlm_adjoint__replacement = \
                Replacement(self._x, space=None)
        return self._x._tlm_adjoint__replacement

    def alias(self):
        return Alias(self._x)


class Constant(backend_Constant):
    def __init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        import mpi4py.MPI as MPI
        comm = kwargs.pop("comm", MPI.COMM_WORLD)
        static = kwargs.pop("static", True)
        cache = kwargs.pop("cache", None)
        if cache is None:
            cache = static
        checkpoint = kwargs.pop("checkpoint", None)
        if checkpoint is None:
            checkpoint = not static
        tlm_depth = kwargs.pop("tlm_depth", 0)

        # "name" constructor argument not supported by Firedrake
        if not hasattr(backend_Constant, "name"):
            name = kwargs.pop("name", None)

        backend_Constant.__init__(self, *args, **kwargs)
        self.__comm = comm
        self.__static = static
        self.__cache = cache
        self.__checkpoint = checkpoint
        self.__tlm_depth = tlm_depth
        self._tlm_adjoint__function_interface = ConstantInterface(self)

        if not hasattr(backend_Constant, "name"):
            if name is None:
                # Following FEniCS 2019.1.0 behaviour
                name = f"f_{self.count():d}"
            self.name = lambda: name

        space = self.ufl_function_space()
        if not hasattr(space, "_tlm_adjoint__space_interface"):
            space._tlm_adjoint__space_interface = \
                ConstantSpaceInterface(space, comm)

    def comm(self):
        return self.__comm

    def is_static(self):
        return self.__static

    def is_cached(self):
        return self.__cache

    def is_checkpointed(self):
        return self.__checkpoint

    def tlm_depth(self):
        return self.__tlm_depth

    def tangent_linear(self, name=None, static=False, cache=None,
                       checkpoint=None):
        if self.is_static():
            return None
        else:
            if len(self.ufl_shape) == 0:
                value = 0.0
            else:
                value = np.zeros(self.ufl_shape, dtype=np.float64)
            return Constant(value, comm=self.comm(), name=name, static=False,
                            cache=cache, checkpoint=checkpoint,
                            tlm_depth=self.tlm_depth() + 1)


class Function(backend_Function):
    def __init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        static = kwargs.pop("static", False)
        cache = kwargs.pop("cache", None)
        if cache is None:
            cache = static
        checkpoint = kwargs.pop("checkpoint", None)
        if checkpoint is None:
            checkpoint = not static
        tlm_depth = kwargs.pop("tlm_depth", 0)

        self.__static = static
        self.__cache = cache
        self.__checkpoint = checkpoint
        self.__tlm_depth = tlm_depth
        backend_Function.__init__(self, *args, **kwargs)

    def is_static(self):
        return self.__static

    def is_cached(self):
        return self.__cache

    def is_checkpointed(self):
        return self.__checkpoint

    def tlm_depth(self):
        return self.__tlm_depth

    def tangent_linear(self, name=None):
        if self.is_static():
            return None
        else:
            return function_new(self, name=name, static=False,
                                cache=self.is_cached(),
                                checkpoint=self.is_checkpointed(),
                                tlm_depth=self.tlm_depth() + 1)


class DirichletBC(backend_DirichletBC):
    def __init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        static = kwargs.pop("static", False)
        cache = kwargs.pop("cache", None)
        if cache is None:
            cache = static
        homogeneous = kwargs.pop("homogeneous", False)

        backend_DirichletBC.__init__(self, *args, **kwargs)
        self.__static = static
        self.__cache = cache
        self.__homogeneous = homogeneous

    def is_static(self):
        return self.__static

    def is_cached(self):
        return self.__cache

    def is_homogeneous(self):
        return self.__homogeneous

    def homogenize(self):
        if not self.__homogeneous:
            backend_DirichletBC.homogenize(self)
            self.__homogeneous = True


def bcs_is_static(bcs):
    for bc in bcs:
        if not hasattr(bc, "is_static") or not bc.is_static():
            return False
    return True


def bcs_is_cached(bcs):
    for bc in bcs:
        if not hasattr(bc, "is_cached") or not bc.is_cached():
            return False
    return True


def new_count():
    return backend_Constant(0).count()


class ReplacementInterface(_FunctionInterface):
    def __init__(self, x, space):
        _FunctionInterface.__init__(self, x)
        self._space = space

    def space(self):
        return self._space

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

    def caches(self):
        return self._x.caches()

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        return space_new(self._space, name=name, static=static, cache=cache,
                         checkpoint=checkpoint, tlm_depth=tlm_depth)

    def replacement(self):
        return self._x


class Replacement(ufl.classes.Coefficient):
    def __init__(self, x, *args, **kwargs):
        if len(args) > 0 or len(kwargs) > 0:
            def extract_args(x, space):
                return x, space
            x, space = extract_args(x, *args, **kwargs)
            x_space = function_space(x)
        else:
            space = function_space(x)
            x_space = space

        ufl.classes.Coefficient.__init__(self, x_space, count=new_count())
        self.__space = space
        self.__id = function_id(x)
        self.__name = function_name(x)
        self.__static = function_is_static(x)
        self.__cache = function_is_cached(x)
        self.__checkpoint = function_is_checkpointed(x)
        self.__tlm_depth = function_tlm_depth(x)
        self.__caches = function_caches(x)
        self._tlm_adjoint__function_interface = \
            ReplacementInterface(self, x_space)

    def function_space(self):
        return self.__space

    def id(self):
        return self.__id

    def name(self):
        return self.__name

    def is_static(self):
        return self.__static

    def is_cached(self):
        return self.__cache

    def is_checkpointed(self):
        return self.__checkpoint

    def tlm_depth(self):
        return self.__tlm_depth

    def caches(self):
        return self.__caches
