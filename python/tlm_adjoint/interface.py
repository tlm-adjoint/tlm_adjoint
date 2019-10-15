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

import copy
import types

__all__ = \
    [
        "InterfaceException",

        "add_interface",

        "SpaceInterface",
        "is_space",
        "space_id",
        "space_new",

        "FunctionInterface",
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
        "function_zero"
    ]


class InterfaceException(Exception):
    pass


def add_interface(obj, interface_cls, attrs={}):
    interface_name = f"{interface_cls.prefix:s}"
    assert not hasattr(obj, interface_name)
    setattr(obj, interface_name, interface_cls)

    for name in interface_cls.names:
        attr_name = f"{interface_cls.prefix:s}{name:s}"
        assert not hasattr(obj, attr_name)
        setattr(obj, attr_name,
                types.MethodType(getattr(interface_cls, name), obj))

    attrs_name = f"{interface_cls.prefix:s}_attrs"
    assert not hasattr(obj, attrs_name)
    setattr(obj, attrs_name, copy.copy(attrs))


class SpaceInterface:
    prefix = "_tlm_adjoint__space_interface"
    names = ("_id", "_new")

    def __init__(self):
        raise InterfaceException("Cannot instantiate SpaceInterface object")

    def _id(self):
        raise InterfaceException("Method not overridden")

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        raise InterfaceException("Method not overridden")


def is_space(x):
    return hasattr(x, "_tlm_adjoint__space_interface")


def space_id(space):
    return space._tlm_adjoint__space_interface_id()


def space_new(space, name=None, static=False, cache=None, checkpoint=None):
    return space._tlm_adjoint__space_interface_new(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


class FunctionInterface:
    prefix = "_tlm_adjoint__function_interface"
    names = ("_comm", "_space", "_id", "_name", "_state", "_update_state",
             "_is_static", "_is_cached", "_is_checkpointed", "_caches",
             "_zero", "_assign", "_axpy", "_inner", "_max_value", "_sum",
             "_linf_norm", "_local_size", "_global_size", "_local_indices",
             "_get_values", "_set_values", "_new", "_copy", "_tangent_linear",
             "_replacement")

    def __init__(self):
        raise InterfaceException("Cannot instantiate FunctionInterface object")

    def _comm(self):
        raise InterfaceException("Method not overridden")

    def _space(self):
        raise InterfaceException("Method not overridden")

    def _id(self):
        raise InterfaceException("Method not overridden")

    def _name(self):
        raise InterfaceException("Method not overridden")

    def _state(self):
        raise InterfaceException("Method not overridden")

    def _update_state(self):
        raise InterfaceException("Method not overridden")

    def _is_static(self):
        raise InterfaceException("Method not overridden")

    def _is_cached(self):
        raise InterfaceException("Method not overridden")

    def _is_checkpointed(self):
        raise InterfaceException("Method not overridden")

    def _caches(self):
        raise InterfaceException("Method not overridden")

    def _zero(self):
        raise InterfaceException("Method not overridden")

    def _assign(self, y):
        raise InterfaceException("Method not overridden")

    def _axpy(self, alpha, y):
        raise InterfaceException("Method not overridden")

    def _inner(self, y):
        raise InterfaceException("Method not overridden")

    def _max_value(self):
        raise InterfaceException("Method not overridden")

    def _sum(self):
        raise InterfaceException("Method not overridden")

    def _linf_norm(self):
        raise InterfaceException("Method not overridden")

    def _local_size(self):
        raise InterfaceException("Method not overridden")

    def _global_size(self):
        raise InterfaceException("Method not overridden")

    def _local_indices(self):
        raise InterfaceException("Method not overridden")

    def _get_values(self):
        raise InterfaceException("Method not overridden")

    def _set_values(self, values):
        raise InterfaceException("Method not overridden")

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        raise InterfaceException("Method not overridden")

    def _copy(self, name=None, static=False, cache=None, checkpoint=None):
        raise InterfaceException("Method not overridden")

    def _tangent_linear(self, name=None):
        raise InterfaceException("Method not overridden")

    def _replacement(self):
        raise InterfaceException("Method not overridden")


def is_function(x):
    return hasattr(x, "_tlm_adjoint__function_interface")


def function_comm(x):
    return x._tlm_adjoint__function_interface_comm()


def function_space(x):
    return x._tlm_adjoint__function_interface_space()


def function_id(x):
    return x._tlm_adjoint__function_interface_id()


def function_name(x):
    return x._tlm_adjoint__function_interface_name()


def function_state(x):
    return x._tlm_adjoint__function_interface_state()


def function_update_state(*X):
    for x in X:
        x._tlm_adjoint__function_interface_update_state()


def function_is_static(x):
    return x._tlm_adjoint__function_interface_is_static()


def function_is_cached(x):
    return x._tlm_adjoint__function_interface_is_cached()


def function_is_checkpointed(x):
    return x._tlm_adjoint__function_interface_is_checkpointed()


def function_caches(x):
    return x._tlm_adjoint__function_interface_caches()


def function_zero(x):
    x._tlm_adjoint__function_interface_zero()


def function_assign(x, y):
    x._tlm_adjoint__function_interface_assign(y)


def function_axpy(x, alpha, y):
    x._tlm_adjoint__function_interface_axpy(alpha, y)


def function_inner(x, y):
    return x._tlm_adjoint__function_interface_inner(y)


def function_max_value(x):
    return x._tlm_adjoint__function_interface_max_value()


def function_sum(x):
    return x._tlm_adjoint__function_interface_sum()


def function_linf_norm(x):
    return x._tlm_adjoint__function_interface_linf_norm()


def function_local_size(x):
    return x._tlm_adjoint__function_interface_local_size()


def function_global_size(x):
    return x._tlm_adjoint__function_interface_global_size()


def function_local_indices(x):
    return x._tlm_adjoint__function_interface_local_indices()


def function_get_values(x):
    return x._tlm_adjoint__function_interface_get_values()


def function_set_values(x, values):
    x._tlm_adjoint__function_interface_set_values(values)


def function_new(x, name=None, static=False, cache=None, checkpoint=None):
    return x._tlm_adjoint__function_interface_new(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


def function_copy(x, name=None, static=False, cache=None, checkpoint=None):
    return x._tlm_adjoint__function_interface_copy(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


def function_tangent_linear(x, name=None):
    return x._tlm_adjoint__function_interface_tangent_linear(name=name)


def function_replacement(x):
    return x._tlm_adjoint__function_interface_replacement()
