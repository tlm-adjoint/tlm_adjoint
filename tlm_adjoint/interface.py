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

import numpy as np

import copy
import logging
import sys
import warnings
import weakref

__all__ = \
    [
        "InterfaceException",

        "add_interface",
        "weakref_method",

        "SpaceInterface",
        "is_space",
        "new_space_id",
        "space_comm",
        "space_dtype",
        "space_id",
        "space_new",

        "FunctionInterface",
        "is_function",
        "function_assign",
        "function_axpy",
        "function_caches",
        "function_comm",
        "function_copy",
        "function_dtype",
        "function_get_values",
        "function_global_size",
        "function_id",
        "function_inner",
        "function_is_cached",
        "function_is_checkpointed",
        "function_is_replacement",
        "function_is_static",
        "function_linf_norm",
        "function_local_indices",
        "function_local_size",
        "function_name",
        "function_new",
        "function_new_tangent_linear",
        "function_replacement",
        "function_set_values",
        "function_space",
        "function_state",
        "function_sum",
        "function_update_caches",
        "function_update_state",
        "function_zero",
        "new_function_id",

        "function_is_scalar",
        "function_scalar_value",
        "new_scalar_function",

        "subtract_adjoint_derivative_action",
        "finalize_adjoint_derivative_action",

        "functional_term_eq",
        "time_system_eq",

        "function_max_value",
        "function_tangent_linear",
        "is_real_function",
        "new_real_function",
        "real_function_value"
    ]


class InterfaceException(Exception):
    pass


def weakref_method(fn, obj):
    if not hasattr(obj, "_tlm_adjoint__weakref_method_self_ref"):
        obj._tlm_adjoint__weakref_method_self_ref = weakref.ref(obj)
    self_ref = obj._tlm_adjoint__weakref_method_self_ref

    def wrapped_fn(*args, **kwargs):
        self = self_ref()
        if self is None:
            raise InterfaceException("Referent must be alive")
        return fn(self, *args, **kwargs)
    return wrapped_fn


def add_interface(obj, interface_cls, attrs={}):
    interface_name = f"{interface_cls.prefix:s}"
    assert not hasattr(obj, interface_name)
    setattr(obj, interface_name, interface_cls)

    for name in interface_cls.names:
        attr_name = f"{interface_cls.prefix:s}{name:s}"
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name,
                    weakref_method(getattr(interface_cls, name), obj))

    attrs_name = f"{interface_cls.prefix:s}_attrs"
    assert not hasattr(obj, attrs_name)
    setattr(obj, attrs_name, copy.copy(attrs))


class SpaceInterface:
    prefix = "_tlm_adjoint__space_interface"
    names = ("_comm", "_dtype", "_id", "_new")

    def __init__(self):
        raise InterfaceException("Cannot instantiate SpaceInterface object")

    def _comm(self):
        raise InterfaceException("Method not overridden")

    def _dtype(self):
        raise InterfaceException("Method not overridden")

    def _id(self):
        raise InterfaceException("Method not overridden")

    def _new(self, name=None, static=False, cache=None, checkpoint=None):
        raise InterfaceException("Method not overridden")


def is_space(x):
    return hasattr(x, "_tlm_adjoint__space_interface")


def space_comm(space):
    return space._tlm_adjoint__space_interface_comm()


def space_dtype(space):
    return space._tlm_adjoint__space_interface_dtype()


_space_id_counter = [0]


def new_space_id():
    space_id = _space_id_counter[0]
    _space_id_counter[0] += 1
    return space_id


def space_id(space):
    return space._tlm_adjoint__space_interface_id()


def space_new(space, name=None, static=False, cache=None, checkpoint=None):
    return space._tlm_adjoint__space_interface_new(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


class FunctionInterface:
    prefix = "_tlm_adjoint__function_interface"
    names = ("_comm", "_space", "_dtype", "_id", "_name", "_state",
             "_update_state", "_is_static", "_is_cached", "_is_checkpointed",
             "_caches", "_update_caches", "_zero", "_assign", "_axpy",
             "_inner", "_max_value", "_sum", "_linf_norm", "_local_size",
             "_global_size", "_local_indices", "_get_values", "_set_values",
             "_new", "_copy", "_new_tangent_linear", "_replacement",
             "_is_replacement", "_is_scalar", "_scalar_value")

    def __init__(self):
        raise InterfaceException("Cannot instantiate FunctionInterface object")

    def _comm(self):
        return space_comm(function_space(self))

    def _space(self):
        raise InterfaceException("Method not overridden")

    def _dtype(self):
        return space_dtype(function_space(self))

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

    def _update_caches(self, value=None):
        if value is None:
            if function_is_replacement(self):
                raise InterfaceException("value required")
            value = self
        function_caches(self).update(value)

    def _zero(self):
        raise InterfaceException("Method not overridden")

    def _assign(self, y):
        raise InterfaceException("Method not overridden")

    def _axpy(self, *args):  # self, alpha, x
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
        return space_new(function_space(self), name=name, static=static,
                         cache=cache, checkpoint=checkpoint)

    def _copy(self, name=None, static=False, cache=None, checkpoint=None):
        raise InterfaceException("Method not overridden")

    def _new_tangent_linear(self, name=None):
        if function_is_static(self):
            return None
        else:
            return function_new(self, name=name, static=False,
                                cache=function_is_cached(self),
                                checkpoint=function_is_checkpointed(self))

    def _replacement(self):
        raise InterfaceException("Method not overridden")

    def _is_replacement(self):
        raise InterfaceException("Method not overridden")

    def _is_scalar(self):
        raise InterfaceException("Method not overridden")

    def _scalar_value(self):
        raise InterfaceException("Method not overridden")


def is_function(x):
    return hasattr(x, "_tlm_adjoint__function_interface")


def function_comm(x):
    return x._tlm_adjoint__function_interface_comm()


def function_space(x):
    return x._tlm_adjoint__function_interface_space()


def function_dtype(x):
    return x._tlm_adjoint__function_interface_dtype()


_function_id_counter = [0]


def new_function_id():
    function_id = _function_id_counter[0]
    _function_id_counter[0] += 1
    return function_id


def function_id(x):
    return x._tlm_adjoint__function_interface_id()


def function_name(x):
    return x._tlm_adjoint__function_interface_name()


def function_state(x):
    return x._tlm_adjoint__function_interface_state()


def function_update_state(*X):
    for x in X:
        x._tlm_adjoint__function_interface_update_state()
    function_update_caches(*X)


def function_is_static(x):
    return x._tlm_adjoint__function_interface_is_static()


def function_is_cached(x):
    return x._tlm_adjoint__function_interface_is_cached()


def function_is_checkpointed(x):
    return x._tlm_adjoint__function_interface_is_checkpointed()


def function_caches(x):
    return x._tlm_adjoint__function_interface_caches()


def function_update_caches(*X, value=None):
    if value is None:
        for x in X:
            x._tlm_adjoint__function_interface_update_caches()
    else:
        if is_function(value):
            value = (value,)
        for x, x_value in zip(X, value):
            x._tlm_adjoint__function_interface_update_caches(value=x_value)


def function_zero(x):
    x._tlm_adjoint__function_interface_zero()


def function_assign(x, y):
    x._tlm_adjoint__function_interface_assign(y)


def function_axpy(*args):  # y, alpha, x
    y, alpha, x = args
    y._tlm_adjoint__function_interface_axpy(alpha, x)


def function_inner(x, y):
    return x._tlm_adjoint__function_interface_inner(y)


def function_max_value(x):
    warnings.warn("function_max_value is deprecated",
                  DeprecationWarning, stacklevel=2)
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
    warnings.warn("function_tangent_linear is deprecated -- "
                  "use function_new_tangent_linear instead",
                  DeprecationWarning, stacklevel=2)
    return function_new_tangent_linear(x, name=name)


def function_new_tangent_linear(x, name=None):
    return x._tlm_adjoint__function_interface_new_tangent_linear(name=name)


def function_replacement(x):
    return x._tlm_adjoint__function_interface_replacement()


def function_is_replacement(x):
    return x._tlm_adjoint__function_interface_is_replacement()


def is_real_function(x):
    warnings.warn("is_real_function is deprecated -- "
                  "use function_is_scalar instead",
                  DeprecationWarning, stacklevel=2)
    return function_is_scalar(x)


def real_function_value(x):
    warnings.warn("real_function_value is deprecated -- "
                  "use function_scalar_value instead",
                  DeprecationWarning, stacklevel=2)
    return function_scalar_value(x)


def add_new_real_function(backend, fn):
    warnings.warn("add_new_real_function is deprecated -- "
                  "use add_new_scalar_function instead",
                  DeprecationWarning, stacklevel=2)
    add_new_scalar_function(backend, fn)


def new_real_function(name=None, comm=None, static=False, cache=None,
                      checkpoint=None):
    warnings.warn("new_real_function is deprecated -- "
                  "use new_scalar_function instead",
                  DeprecationWarning, stacklevel=2)
    return new_scalar_function(name=name, comm=comm, static=static,
                               cache=cache, checkpoint=checkpoint)


def function_is_scalar(x):
    return x._tlm_adjoint__function_interface_is_scalar()


def function_scalar_value(x):
    if not function_is_scalar(x):
        raise InterfaceException("Invalid function")
    return x._tlm_adjoint__function_interface_scalar_value()


_new_scalar_function = {}


def add_new_scalar_function(backend, fn):
    assert backend not in _new_scalar_function
    _new_scalar_function[backend] = fn


def new_scalar_function(name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    new_scalar_function = tuple(_new_scalar_function.values())[0]
    return new_scalar_function(name=name, comm=comm, static=static,
                               cache=cache, checkpoint=checkpoint)


_subtract_adjoint_derivative_action = {}


def add_subtract_adjoint_derivative_action(backend, fn):
    assert backend not in _subtract_adjoint_derivative_action
    _subtract_adjoint_derivative_action[backend] = fn


def subtract_adjoint_derivative_action(x, y):
    for fn in _subtract_adjoint_derivative_action.values():
        if fn(x, y) != NotImplemented:
            break
    else:
        if y is None:
            pass
        elif is_function(y):
            if isinstance(y._tlm_adjoint__function_interface,
                          type(x._tlm_adjoint__function_interface)):
                function_axpy(x, -1.0, y)
            else:
                function_set_values(x,
                                    function_get_values(x)
                                    - function_get_values(y))
        elif isinstance(y, tuple) \
                and len(y) == 2 \
                and isinstance(y[0], (int, np.integer,
                                      float, np.floating,
                                      complex, np.complexfloating)) \
                and is_function(y[1]):
            alpha, y = y
            alpha = function_dtype(x)(alpha)
            if isinstance(y._tlm_adjoint__function_interface,
                          type(x._tlm_adjoint__function_interface)):
                function_axpy(x, -alpha, y)
            else:
                function_set_values(x,
                                    function_get_values(x)
                                    - alpha * function_get_values(y))
        else:
            raise InterfaceException("Unexpected case encountered in "
                                     "subtract_adjoint_derivative_action")


_finalize_adjoint_derivative_action = {}


def add_finalize_adjoint_derivative_action(backend, fn):
    assert backend not in _finalize_adjoint_derivative_action
    _finalize_adjoint_derivative_action[backend] = fn


def finalize_adjoint_derivative_action(x):
    for fn in _finalize_adjoint_derivative_action.values():
        fn(x)


_functional_term_eq = {}


def add_functional_term_eq(backend, fn):
    assert backend not in _functional_term_eq
    _functional_term_eq[backend] = fn


def functional_term_eq(term, x):
    for fn in _functional_term_eq.values():
        eq = fn(term, x)
        if eq != NotImplemented:
            return eq
    raise InterfaceException("Unexpected case encountered in "
                             "functional_term_eq")


_time_system_eq = {}


def add_time_system_eq(backend, fn):
    assert backend not in _time_system_eq
    _time_system_eq[backend] = fn


def time_system_eq(*args, **kwargs):
    for fn in _time_system_eq.values():
        eq = fn(*args, **kwargs)
        if eq != NotImplemented:
            return eq
    raise InterfaceException("Unexpected case encountered in "
                             "time_system_eq")


_logger = logging.getLogger("tlm_adjoint")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
