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

from ..caches import Caches
from ..functional import Functional as _Functional
from ..hessian import GeneralGaussNewton as _GaussNewton
from ..hessian_optimization import CachedGaussNewton as _CachedGaussNewton
from ..interface import InterfaceException, SpaceInterface, add_interface, \
    function_space, new_function_id, new_space_id, space_id, space_new
from ..interface import FunctionInterface as _FunctionInterface
from ..tlm_adjoint import DEFAULT_COMM

import copy
import numpy as np
import warnings

__all__ = \
    [
        "default_dtype",
        "set_default_dtype",

        "CachedGaussNewton",
        "Functional",
        "GaussNewton",
        "new_scalar_function",

        "Function",
        "FunctionSpace",

        "RealFunctionSpace",
        "copy_parameters_dict",
        "default_comm",
        "function_space_id",
        "function_space_new",
        "info",
        "warning"
    ]


_default_dtype = [np.float64]


def default_dtype():
    return _default_dtype[0]


def set_default_dtype(dtype):
    _default_dtype[0] = dtype


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm()

    def _dtype(self):
        return self.dtype()

    def _id(self):
        return self.id()

    def _new(self, *, name=None, static=False, cache=None, checkpoint=None):
        return Function(self, name=name, static=static, cache=cache,
                        checkpoint=checkpoint)


class FunctionSpace:
    def __init__(self, dim, dtype=None):
        comm = DEFAULT_COMM
        if comm.size > 1:
            raise InterfaceException("Serial only")
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
        dtype = self.dtype()
        if isinstance(y, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)) \
                and np.can_cast(y, dtype):
            self.vector()[:] = dtype(y)
        elif isinstance(y, Function) and np.can_cast(y.dtype(), dtype):
            self.vector()[:] = y.vector()
        else:
            raise InterfaceException("Invalid type or dtype")

    def _axpy(self, *args):  # self, alpha, x
        alpha, x = args
        dtype = self.dtype()
        alpha = dtype(alpha)
        if isinstance(x, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)) \
                and np.can_cast(x, dtype):
            self.vector()[:] += alpha * dtype(x)
        elif isinstance(x, Function) and np.can_cast(x.dtype(), dtype):
            self.vector()[:] += alpha * x.vector()
        else:
            raise InterfaceException("Invalid type or dtype")

    def _inner(self, y):
        assert isinstance(y, Function)
        return y.vector().conjugate().dot(self.vector())

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
        return values

    def _set_values(self, values):
        dtype = self.dtype()
        if not np.can_cast(values, dtype):
            raise InterfaceException("Invalid dtype")
        if values.shape != self.vector().shape:
            raise InterfaceException("Invalid shape")
        self.vector()[:] = values

    def _copy(self, *, name=None, static=False, cache=None, checkpoint=None):
        return Function(self.space(), name=name, static=static, cache=cache,
                        checkpoint=checkpoint, _data=self.vector().copy())

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
    def __init__(self, space, name=None, static=False, cache=None,
                 checkpoint=None, _data=None):
        id = new_function_id()
        if name is None:
            # Following FEniCS 2019.1.0 behaviour
            name = f"f_{id:d}"
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static

        self._space = space
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
                raise InterfaceException("Invalid dtype")
            elif _data.shape != (space.dim(),):
                raise InterfaceException("Invalid shape")
            self._data = _data
        add_interface(self, FunctionInterface)
        self._caches = Caches(self)

    def space(self):
        return self._space

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
        self._id = x.id()
        self._name = x.name()
        self._static = x.is_static()
        self._cache = x.is_cached()
        self._checkpoint = x.is_checkpointed()
        self._caches = x.caches()
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

    def caches(self):
        return self._caches


def new_scalar_function(*, name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    return Function(FunctionSpace(1), name=name, static=static, cache=cache,
                    checkpoint=checkpoint)


class Functional(_Functional):
    def __init__(self, *, space=None, name=None, _fn=None):
        if space is None and _fn is None:
            space = function_space(new_scalar_function())

        super().__init__(space=space, name=name, _fn=_fn)


class GaussNewton(_GaussNewton):
    def __init__(self, forward, R_inv_action, B_inv_action=None,
                 *, J_space=None, manager=None):
        if J_space is None:
            J_space = function_space(new_scalar_function())

        super().__init__(
            forward, J_space, R_inv_action, B_inv_action=B_inv_action,
            manager=manager)


class CachedGaussNewton(_CachedGaussNewton):
    def __init__(self, X, R_inv_action, B_inv_action=None,
                 *, J_space=None, manager=None):
        if J_space is None:
            J_space = function_space(new_scalar_function())

        super().__init__(
            X, J_space, R_inv_action, B_inv_action=B_inv_action,
            manager=manager)


def default_comm():
    warnings.warn("default_comm is deprecated",
                  DeprecationWarning, stacklevel=2)
    return DEFAULT_COMM


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_scalar_function instead",
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
