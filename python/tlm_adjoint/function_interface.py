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

__all__ = \
    [
        "InterfaceException",

        "FunctionInterface",
        "function_alias",
        "function_assign",
        "function_axpy",
        "function_comm",
        "function_copy",
        "function_function_space",
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
        "function_set_values",
        "function_state",
        "function_sum",
        "function_tangent_linear",
        "function_tlm_depth",
        "function_update_state",
        "function_zero",
        "is_function",
        "replaced_function"
    ]


class InterfaceException(Exception):
    pass


class FunctionInterface:
    def __init__(self, x):
        self._x = x

    def comm(self):
        raise InterfaceException("Method not overridden")

    def function_space(self):
        raise InterfaceException("Method not overridden")

    def id(self):
        raise InterfaceException("Method not overridden")

    def name(self):
        raise InterfaceException("Method not overridden")

    def state(self):
        raise InterfaceException("Method not overridden")

    def update_state(self):
        raise InterfaceException("Method not overridden")

    def is_static(self):
        raise InterfaceException("Method not overridden")

    def is_cached(self):
        raise InterfaceException("Method not overridden")

    def is_checkpointed(self):
        raise InterfaceException("Method not overridden")

    def tlm_depth(self):
        raise InterfaceException("Method not overridden")

    def zero(self):
        raise InterfaceException("Method not overridden")

    def assign(self, y):
        raise InterfaceException("Method not overridden")

    def axpy(self, alpha, y):
        raise InterfaceException("Method not overridden")

    def inner(self, y):
        raise InterfaceException("Method not overridden")

    def max_value(self):
        raise InterfaceException("Method not overridden")

    def sum(self):
        raise InterfaceException("Method not overridden")

    def linf_norm(self):
        raise InterfaceException("Method not overridden")

    def local_size(self):
        raise InterfaceException("Method not overridden")

    def global_size(self):
        raise InterfaceException("Method not overridden")

    def local_indices(self):
        raise InterfaceException("Method not overridden")

    def get_values(self):
        raise InterfaceException("Method not overridden")

    def set_values(self, values):
        raise InterfaceException("Method not overridden")

    def new(self, name=None, static=False, cache=None, checkpoint=None,
            tlm_depth=0):
        raise InterfaceException("Method not overridden")

    def copy(self, name=None, static=False, cache=None, checkpoint=None,
             tlm_depth=0):
        raise InterfaceException("Method not overridden")

    def tangent_linear(self, name=None):
        raise InterfaceException("Method not overridden")

    def replaced(self):
        raise InterfaceException("Method not overridden")

    def alias(self):
        raise InterfaceException("Method not overridden")


def is_function(x):
    return hasattr(x, "_tlm_adjoint__interface")


def function_comm(x):
    return x._tlm_adjoint__interface.comm()


def function_function_space(x):
    return x._tlm_adjoint__interface.function_space()


def function_id(x):
    return x._tlm_adjoint__interface.id()


def function_name(x):
    return x._tlm_adjoint__interface.name()


def function_state(x):
    return x._tlm_adjoint__interface.state()


def function_update_state(*X):
    for x in X:
        x._tlm_adjoint__interface.update_state()


def function_is_static(x):
    return x._tlm_adjoint__interface.is_static()


def function_is_cached(x):
    return x._tlm_adjoint__interface.is_cached()


def function_is_checkpointed(x):
    return x._tlm_adjoint__interface.is_checkpointed()


def function_tlm_depth(x):
    return x._tlm_adjoint__interface.tlm_depth()


def function_zero(x):
    x._tlm_adjoint__interface.zero()


def function_assign(x, y):
    x._tlm_adjoint__interface.assign(y)


def function_axpy(x, alpha, y):
    x._tlm_adjoint__interface.axpy(alpha, y)


def function_inner(x, y):
    return x._tlm_adjoint__interface.inner(y)


def function_max_value(x):
    return x._tlm_adjoint__interface.max_value()


def function_sum(x):
    return x._tlm_adjoint__interface.sum()


def function_linf_norm(x):
    return x._tlm_adjoint__interface.linf_norm()


def function_local_size(x):
    return x._tlm_adjoint__interface.local_size()


def function_global_size(x):
    return x._tlm_adjoint__interface.global_size()


def function_local_indices(x):
    return x._tlm_adjoint__interface.local_indices()


def function_get_values(x):
    return x._tlm_adjoint__interface.get_values()


def function_set_values(x, values):
    x._tlm_adjoint__interface.set_values(values)


def function_new(x, name=None, static=False, cache=None, checkpoint=None,
                 tlm_depth=0):
    return x._tlm_adjoint__interface.new(
        name=name, static=static, cache=cache, checkpoint=checkpoint,
        tlm_depth=tlm_depth)


def function_copy(x, name=None, static=False, cache=None, checkpoint=None,
                  tlm_depth=0):
    return x._tlm_adjoint__interface.copy(
        name=name, static=static, cache=cache, checkpoint=checkpoint,
        tlm_depth=tlm_depth)


def function_tangent_linear(x, name=None):
    return x._tlm_adjoint__interface.tangent_linear(name=name)


def replaced_function(x):
    return x._tlm_adjoint__interface.replaced()


def function_alias(x):
    return x._tlm_adjoint__interface.alias()
