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

import copy
import ufl

__all__ = \
    [
        "Constant",
        "DirichletBC",
        "Function",
        "ReplacementFunction",
        "bcs_is_cached",
        "bcs_is_static",
        "new_count"
    ]


class Constant(backend_Constant):
    def __init__(self, *args, **kwargs):
        kwargs = copy.copy(kwargs)
        static = kwargs.pop("static", False)
        cache = kwargs.pop("cache", None)
        if cache is None:
            cache = static

        backend_Constant.__init__(self, *args, **kwargs)
        self.__static = static
        self.__cache = cache

    def is_static(self):
        return self.__static

    def is_cached(self):
        return self.__cache


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
    return Constant(0).count()


class ReplacementFunctionInterface(FunctionInterface):
    def space(self):
        return self._x.function_space()

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
        return Function(self._x.function_space(), name=name, static=static,
                        cache=cache, checkpoint=checkpoint,
                        tlm_depth=tlm_depth)


class ReplacementFunction(ufl.classes.Coefficient):
    def __init__(self, x):
        space = function_space(x)
        ufl.classes.Coefficient.__init__(self, space, count=new_count())
        self.__space = space
        self.__id = function_id(x)
        self.__name = function_name(x)
        self.__static = function_is_static(x)
        self.__checkpoint = function_is_checkpointed(x)
        self.__tlm_depth = function_tlm_depth(x)
        self.__caches = function_caches(x)
        self._tlm_adjoint__function_interface = \
            ReplacementFunctionInterface(self)

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
