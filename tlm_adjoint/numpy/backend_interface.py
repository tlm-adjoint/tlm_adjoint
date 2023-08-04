#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..interface import (
    DEFAULT_COMM, SpaceInterface, add_interface, comm_dup_cached,
    new_function_id, new_space_id, register_subtract_adjoint_derivative_action,
    space_id, space_new, subtract_adjoint_derivative_action_base)
from ..interface import FunctionInterface as _FunctionInterface

from ..caches import Caches
from ..overloaded_float import SymbolicFloat

import copy
import numpy as np
import warnings

__all__ = \
    [
        "default_dtype",
        "set_default_dtype",

        "Function",
        "FunctionSpace",

        "RealFunctionSpace",
        "copy_parameters_dict",
        "default_comm",
        "function_space_id",
        "function_space_new",
        "new_scalar_function",
        "info",
        "warning"
    ]


_default_dtype = np.float64


def default_dtype():
    return _default_dtype


def set_default_dtype(dtype):
    global _default_dtype
    if not issubclass(dtype, (np.floating, np.complexfloating)):
        raise ValueError("Invalid dtype")
    _default_dtype = dtype


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm()

    def _dtype(self):
        return self.dtype()

    def _id(self):
        return self.id()

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        return Function(self, name=name, space_type=space_type, static=static,
                        cache=cache, checkpoint=checkpoint)


class FunctionSpace:
    def __init__(self, dim, *,
                 dtype=None):
        comm = comm_dup_cached(DEFAULT_COMM)
        if comm.size > 1:
            raise RuntimeError("Serial only")
        if dtype is None:
            dtype = default_dtype()

        self._comm = comm
        self._id = new_space_id()
        self._dim = dim
        self._dtype = dtype
        add_interface(self, FunctionSpaceInterface)

    def comm(self):
        return self._comm

    def id(self):
        return self._id

    def dim(self):
        return self._dim

    def dtype(self):
        return self._dtype


class FunctionInterface(_FunctionInterface):
    def _space(self):
        return self.space()

    def _space_type(self):
        return self.space_type()

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

    def _caches(self):
        return self.caches()

    def _zero(self):
        self.vector()[:] = 0.0

    def _assign(self, y):
        if isinstance(y, SymbolicFloat):
            y = y.value()
        if isinstance(y, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)) \
                and np.can_cast(y, self.dtype()):
            self.vector()[:] = y
        elif isinstance(y, Function):
            if np.can_cast(y.dtype(), self.dtype()):
                self.vector()[:] = y.vector()
            else:
                raise ValueError("Invalid dtype")
        else:
            raise TypeError("Invalid type")

    def _axpy(self, alpha, x, /):
        if isinstance(x, SymbolicFloat):
            x = x.value()
        if isinstance(x, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)) \
                and np.can_cast(x, self.dtype()):
            self.vector()[:] += alpha * x
        elif isinstance(x, Function):
            if np.can_cast(x.dtype(), self.dtype()):
                self.vector()[:] += alpha * x.vector()
            else:
                raise ValueError("Invalid dtype")
        else:
            raise TypeError("Invalid type")

    def _inner(self, y):
        assert isinstance(y, Function)
        return y.vector().conjugate().dot(self.vector())

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
        return self.vector().copy()

    def _set_values(self, values):
        if not np.can_cast(values, self.dtype()):
            raise ValueError("Invalid dtype")
        if values.shape != self.vector().shape:
            raise ValueError("Invalid shape")
        self.vector()[:] = values

    def _replacement(self):
        return self.replacement()

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return self.space().dim() == 1

    def _scalar_value(self):
        # assert function_is_scalar(self)
        return self.vector()[0]


class Function:
    def __init__(self, space, *,
                 name=None, space_type="primal", static=False, cache=None,
                 checkpoint=None, _data=None):
        if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
            raise ValueError("Invalid space type")
        id = new_function_id()
        if name is None:
            # Following FEniCS 2019.1.0 behaviour
            name = f"f_{id:d}"
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static

        self._space = space
        self._space_type = space_type
        self._id = id
        self._name = name
        self._state = 0
        self._static = static
        self._cache = cache
        self._checkpoint = checkpoint
        self._replacement = None
        if _data is None:
            self._data = np.zeros(space.dim(), dtype=space.dtype())
        else:
            if _data.dtype.type != space.dtype():
                raise ValueError("Invalid dtype")
            elif _data.shape != (space.dim(),):
                raise ValueError("Invalid shape")
            self._data = _data
        add_interface(self, FunctionInterface)
        self._caches = Caches(self)

    def space(self):
        return self._space

    def space_type(self):
        return self._space_type

    def dtype(self):
        return self._space.dtype()

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

    def caches(self):
        return self._caches

    def replacement(self):
        if self._replacement is None:
            self._replacement = Replacement(self)
        return self._replacement

    def vector(self):
        return self._data


class ReplacementInterface(_FunctionInterface):
    def _space(self):
        return self.space()

    def _space_type(self):
        return self.space_type()

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

    def _caches(self):
        return self.caches()

    def _replacement(self):
        return self

    def _is_replacement(self):
        return True


class Replacement:
    def __init__(self, x):
        self._space = x.space()
        self._space_type = x.space_type()
        self._id = x.id()
        self._name = x.name()
        self._static = x.is_static()
        self._cache = x.is_cached()
        self._checkpoint = x.is_checkpointed()
        self._caches = x.caches()
        add_interface(self, ReplacementInterface)

    def space(self):
        return self._space

    def space_type(self):
        return self._space_type

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

    def caches(self):
        return self._caches


register_subtract_adjoint_derivative_action(
    Function, object,
    subtract_adjoint_derivative_action_base,
    replace=True)


def default_comm():
    warnings.warn("default_comm is deprecated",
                  DeprecationWarning, stacklevel=2)
    return DEFAULT_COMM


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use Float instead",
                  DeprecationWarning, stacklevel=2)
    return FunctionSpace(1)


def new_scalar_function(*, name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    warnings.warn("new_scalar_function is deprecated -- "
                  "use Float instead",
                  DeprecationWarning, stacklevel=2)
    return Function(FunctionSpace(1), name=name, static=static, cache=cache,
                    checkpoint=checkpoint)


def function_space_id(*args, **kwargs):
    warnings.warn("function_space_id is deprecated -- use space_id instead",
                  DeprecationWarning, stacklevel=2)
    return space_id(*args, **kwargs)


def function_space_new(*args, **kwargs):
    warnings.warn("function_space_new is deprecated -- use space_new instead",
                  DeprecationWarning, stacklevel=2)
    return space_new(*args, **kwargs)


def info(message):
    warnings.warn("info is deprecated -- use print instead",
                  DeprecationWarning, stacklevel=2)
    print(message)


def warning(message):
    warnings.warn("warning is deprecated -- use logging.warning instead",
                  DeprecationWarning, stacklevel=2)
    warnings.warn(message, RuntimeWarning)


def copy_parameters_dict(parameters):
    warnings.warn("copy_parameters_dict is deprecated -- "
                  "use copy.deepcopy instead",
                  DeprecationWarning, stacklevel=2)
    return copy.deepcopy(parameters)
