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

from collections.abc import Mapping
import copy
import functools
import logging
import sys
import warnings
import weakref

__all__ = \
    [
        "InterfaceException",

        "DEFAULT_COMM",

        "add_interface",
        "weakref_method",

        "SpaceInterface",
        "is_space",
        "new_space_id",
        "space_comm",
        "space_dtype",
        "space_id",
        "space_new",

        "check_space_type",
        "check_space_types",
        "check_space_types_conjugate",
        "check_space_types_conjugate_dual",
        "check_space_types_dual",
        "conjugate_dual_space_type",
        "conjugate_space_type",
        "dual_space_type",
        "no_space_type_checking",
        "relative_space_type",
        "space_type_warning",

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
        "function_new_conjugate",
        "function_new_conjugate_dual",
        "function_new_dual",
        "function_new_tangent_linear",
        "function_replacement",
        "function_set_values",
        "function_space",
        "function_space_type",
        "function_state",
        "function_sum",
        "function_update_caches",
        "function_update_state",
        "function_zero",
        "new_function_id",

        "function_is_scalar",
        "function_scalar_value",

        "function_is_alias",

        "subtract_adjoint_derivative_action",
        "finalize_adjoint_derivative_action",

        "functional_term_eq",
        "time_system_eq",

        "function_max_value",
        "is_real_function",
        "real_function_value"
    ]


class InterfaceException(Exception):  # noqa: N818
    def __init__(self, *args, **kwargs):
        warnings.warn("InterfaceException is deprecated",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


try:
    from mpi4py.MPI import COMM_WORLD as DEFAULT_COMM
except ImportError:
    # As for mpi4py 3.0.3 API
    class SerialComm:
        _id_counter = [-1]

        def __init__(self):
            self._id = self._id_counter[0]
            self._id_counter[0] -= 1

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 1

        def Dup(self, info=None):
            return SerialComm()

        def Free(self):
            pass

        def allgather(self, sendobj):
            return [copy.deepcopy(sendobj)]

        def barrier(self):
            pass

        def bcast(self, obj, root=0):
            return copy.deepcopy(obj)

        def gather(self, sendobj, root=0):
            assert root == 0
            return [copy.deepcopy(sendobj)]

        def py2f(self):
            return self._id

        def scatter(self, sendobj, root=0):
            assert root == 0
            sendobj, = sendobj
            return copy.deepcopy(sendobj)

    DEFAULT_COMM = SerialComm()


def weakref_method(fn, obj):
    if not hasattr(obj, "_tlm_adjoint__weakref_method_self_ref"):
        obj._tlm_adjoint__weakref_method_self_ref = weakref.ref(obj)
    self_ref = obj._tlm_adjoint__weakref_method_self_ref

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        self = self_ref()
        if self is None:
            raise RuntimeError("Referent must be alive")
        return fn(self, *args, **kwargs)
    return wrapped_fn


class protecteddict(Mapping):  # noqa: N801
    def __init__(self, *args, **kwargs):
        """
        A mapping where previous key: value pairs are partially protected from
        modification. d_ prefixed methods can be used to modify key: value
        pairs.
        """

        self._d = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Key '{key}' already set")
        self._d[key] = value

    def __iter__(self):
        for key in self._d:
            yield key

    def __len__(self):
        return len(self._d)

    def d_delitem(self, key):
        del self._d[key]

    def d_setitem(self, key, value):
        self._d[key] = value


def add_interface(obj, interface_cls, attrs=None):
    if attrs is None:
        attrs = {}

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
    setattr(obj, attrs_name, protecteddict(attrs))


class SpaceInterface:
    prefix = "_tlm_adjoint__space_interface"
    names = ("_comm", "_dtype", "_id", "_new")

    def __init__(self):
        raise RuntimeError("Cannot instantiate SpaceInterface object")

    def _comm(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        raise NotImplementedError("Method not overridden")

    def _id(self):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        raise NotImplementedError("Method not overridden")


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


def space_new(space, *, name=None, space_type="primal", static=False,
              cache=None, checkpoint=None):
    if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid space type")
    return space._tlm_adjoint__space_interface_new(
        name=name, space_type=space_type, static=static, cache=cache,
        checkpoint=checkpoint)


def relative_space_type(space_type, rel_space_type):
    space_type_fn = {"primal": lambda space_type: space_type,
                     "conjugate": conjugate_space_type,
                     "dual": dual_space_type,
                     "conjugate_dual": conjugate_dual_space_type}[rel_space_type]  # noqa: E501
    return space_type_fn(space_type)


def conjugate_space_type(space_type):
    return {"primal": "conjugate", "conjugate": "primal",
            "dual": "conjugate_dual", "conjugate_dual": "dual"}[space_type]


def dual_space_type(space_type):
    return {"primal": "dual", "conjugate": "conjugate_dual",
            "dual": "primal", "conjugate_dual": "conjugate"}[space_type]


def conjugate_dual_space_type(space_type):
    return {"primal": "conjugate_dual", "conjugate": "dual",
            "dual": "conjugate", "conjugate_dual": "primal"}[space_type]


_check_space_types = [True]


def no_space_type_checking(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        check_space_types = _check_space_types[0]
        _check_space_types[0] = False
        try:
            return fn(*args, **kwargs)
        finally:
            _check_space_types[0] = check_space_types
    return wrapped_fn


def space_type_warning(msg, *, stacklevel=1):
    if _check_space_types[0]:
        warnings.warn(msg, stacklevel=stacklevel + 1)


def check_space_type(x, space_type):
    if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid space type")
    if function_space_type(x) != space_type:
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types(x, y, *, rel_space_type="primal"):
    if function_space_type(x) != \
            function_space_type(y, rel_space_type=rel_space_type):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_conjugate(x, y):
    if function_space_type(x) != \
            function_space_type(y, rel_space_type="conjugate"):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_dual(x, y):
    if function_space_type(x) != \
            function_space_type(y, rel_space_type="dual"):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_conjugate_dual(x, y):
    if function_space_type(x) != \
            function_space_type(y, rel_space_type="conjugate_dual"):
        space_type_warning("Unexpected space type", stacklevel=2)


class FunctionInterface:
    prefix = "_tlm_adjoint__function_interface"
    names = ("_comm", "_space", "_space_type", "_dtype", "_id", "_name",
             "_state", "_update_state", "_is_static", "_is_cached",
             "_is_checkpointed", "_caches", "_zero", "_assign", "_axpy",
             "_inner", "_max_value", "_sum", "_linf_norm", "_local_size",
             "_global_size", "_local_indices", "_get_values", "_set_values",
             "_new", "_copy", "_replacement", "_is_replacement", "_is_scalar",
             "_scalar_value", "_is_alias")

    def __init__(self):
        raise RuntimeError("Cannot instantiate FunctionInterface object")

    def _comm(self):
        return space_comm(function_space(self))

    def _space(self):
        raise NotImplementedError("Method not overridden")

    def _space_type(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        return space_dtype(function_space(self))

    def _id(self):
        raise NotImplementedError("Method not overridden")

    def _name(self):
        raise NotImplementedError("Method not overridden")

    def _state(self):
        raise NotImplementedError("Method not overridden")

    def _update_state(self):
        raise NotImplementedError("Method not overridden")

    def _is_static(self):
        raise NotImplementedError("Method not overridden")

    def _is_cached(self):
        raise NotImplementedError("Method not overridden")

    def _is_checkpointed(self):
        raise NotImplementedError("Method not overridden")

    def _caches(self):
        raise NotImplementedError("Method not overridden")

    def _zero(self):
        raise NotImplementedError("Method not overridden")

    def _assign(self, y):
        raise NotImplementedError("Method not overridden")

    def _axpy(self, alpha, x, /):
        raise NotImplementedError("Method not overridden")

    def _inner(self, y):
        raise NotImplementedError("Method not overridden")

    def _max_value(self):
        raise NotImplementedError("Method not overridden")

    def _sum(self):
        raise NotImplementedError("Method not overridden")

    def _linf_norm(self):
        raise NotImplementedError("Method not overridden")

    def _local_size(self):
        raise NotImplementedError("Method not overridden")

    def _global_size(self):
        raise NotImplementedError("Method not overridden")

    def _local_indices(self):
        raise NotImplementedError("Method not overridden")

    def _get_values(self):
        raise NotImplementedError("Method not overridden")

    def _set_values(self, values):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, static=False, cache=None, checkpoint=None,
             rel_space_type="primal"):
        space_type = function_space_type(self, rel_space_type=rel_space_type)
        return space_new(function_space(self), name=name,
                         space_type=space_type, static=static, cache=cache,
                         checkpoint=checkpoint)

    def _copy(self, *, name=None, static=False, cache=None, checkpoint=None):
        y = function_new(self, name=name, static=static, cache=cache,
                         checkpoint=checkpoint)
        function_assign(y, self)
        return y

    def _replacement(self):
        raise NotImplementedError("Method not overridden")

    def _is_replacement(self):
        raise NotImplementedError("Method not overridden")

    def _is_scalar(self):
        raise NotImplementedError("Method not overridden")

    def _scalar_value(self):
        raise NotImplementedError("Method not overridden")

    def _is_alias(self):
        return False


def is_function(x):
    return hasattr(x, "_tlm_adjoint__function_interface")


def function_comm(x):
    return x._tlm_adjoint__function_interface_comm()


def function_space(x):
    return x._tlm_adjoint__function_interface_space()


def function_space_type(x, *, rel_space_type="primal"):
    space_type = x._tlm_adjoint__function_interface_space_type()
    return relative_space_type(space_type, rel_space_type)


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
            if function_is_replacement(x):
                raise TypeError("value required")
            function_caches(x).update(x)
    else:
        if is_function(value):
            value = (value,)
        assert len(X) == len(value)
        for x, x_value in zip(X, value):
            function_caches(x).update(x_value)


def function_zero(x):
    x._tlm_adjoint__function_interface_zero()
    function_update_state(x)


def function_assign(x, y):
    if is_function(y):
        check_space_types(x, y)
    x._tlm_adjoint__function_interface_assign(y)
    function_update_state(x)


def function_axpy(y, alpha, x, /):
    if is_function(x):
        check_space_types(y, x)
    y._tlm_adjoint__function_interface_axpy(alpha, x)
    function_update_state(y)


def function_inner(x, y):
    if is_function(y):
        check_space_types_conjugate_dual(x, y)
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
    function_update_state(x)


def function_new(x, *, name=None, static=False, cache=None, checkpoint=None,
                 rel_space_type="primal"):
    if rel_space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid relative space type")
    return x._tlm_adjoint__function_interface_new(
        name=name, static=static, cache=cache, checkpoint=checkpoint,
        rel_space_type=rel_space_type)


def function_new_conjugate(x, *, name=None, static=False, cache=None,
                           checkpoint=None):
    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="conjugate")


def function_new_dual(x, *, name=None, static=False, cache=None,
                      checkpoint=None):
    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="dual")


def function_new_conjugate_dual(x, *, name=None, static=False, cache=None,
                                checkpoint=None):
    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="conjugate_dual")


def function_copy(x, *, name=None, static=False, cache=None, checkpoint=None):
    return x._tlm_adjoint__function_interface_copy(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


def function_new_tangent_linear(x, *, name=None):
    if function_is_checkpointed(x):
        return function_new(x, name=name, static=function_is_static(x),
                            cache=function_is_cached(x),
                            checkpoint=True)
    else:
        return None


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


def function_is_scalar(x):
    return x._tlm_adjoint__function_interface_is_scalar()


def function_scalar_value(x):
    if not function_is_scalar(x):
        raise ValueError("Invalid function")
    return x._tlm_adjoint__function_interface_scalar_value()


def function_is_alias(x):
    return x._tlm_adjoint__function_interface_is_alias()


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
            check_space_types(x, y)
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
            check_space_types(x, y)
            if isinstance(y._tlm_adjoint__function_interface,
                          type(x._tlm_adjoint__function_interface)):
                function_axpy(x, -alpha, y)
            else:
                function_set_values(x,
                                    function_get_values(x)
                                    - alpha * function_get_values(y))
        else:
            raise RuntimeError("Unexpected case encountered in "
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
    raise RuntimeError("Unexpected case encountered in functional_term_eq")


_time_system_eq = {}


def add_time_system_eq(backend, fn):
    assert backend not in _time_system_eq
    _time_system_eq[backend] = fn


def time_system_eq(*args, **kwargs):
    for fn in _time_system_eq.values():
        eq = fn(*args, **kwargs)
        if eq != NotImplemented:
            return eq
    raise RuntimeError("Unexpected case encountered in time_system_eq")


_logger = logging.getLogger("tlm_adjoint")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
