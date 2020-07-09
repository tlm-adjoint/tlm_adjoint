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

from .backend_interface import *

from .manager import manager as _manager, set_manager

import numpy as np
import zlib

__all__ = \
    [
        "OptimizationException",

        "minimize_scipy"
    ]


class OptimizationException(Exception):
    pass


def minimize_scipy(forward, M0, J0=None, manager=None, **kwargs):
    """
    Gradient-based minimization using scipy.optimize.minimize.

    Arguments:

    forward  A callable which takes as input the control and returns the
             Functional to be minimized.
    M0       A function, or a list or tuple of functions. Control parameters
             initial guess.
    J0       (Optional) Initial functional. If supplied assumes that the
             forward has already been run, and processed by the equation
             manager, using the control parameters given by M0.
    manager  (Optional) The equation manager.

    Any remaining keyword arguments are passed directly to
    scipy.optimize.minimize.

    Returns a tuple
        (M, return_value)
    return M is the value of the control parameters obtained, and return_value
    is the return value of scipy.optimize.minimize.
    """

    if not isinstance(M0, (list, tuple)):
        (M,), return_value = minimize_scipy(forward, [M0], J0=J0,
                                            manager=manager, **kwargs)
        return M, return_value

    M0 = [m0 if is_function(m0) else m0.m() for m0 in M0]
    if manager is None:
        manager = _manager()
    comm = manager.comm()

    N = [0]
    for m in M0:
        N.append(N[-1] + function_local_size(m))
    size_global = comm.allgather(np.array(N[-1], dtype=np.int64))
    N_global = [0]
    for size in size_global:
        N_global.append(N_global[-1] + size)

    def get(F):
        x = np.full(N[-1], np.NAN, dtype=np.float64)
        for i, f in enumerate(F):
            x[N[i]:N[i + 1]] = function_get_values(f)

        x_global = comm.allgather(x)
        X = np.full(N_global[-1], np.NAN, dtype=np.float64)
        for i, x_p in enumerate(x_global):
            X[N_global[i]:N_global[i + 1]] = x_p
        return X

    def set(F, x):
        # Basic cross-process synchonization check
        check1 = np.array(zlib.adler32(x.data), dtype=np.uint32)
        check_global = comm.allgather(check1)
        for check2 in check_global:
            if check1 != check2:
                raise OptimizationException("Parallel desynchronization "
                                            "detected")

        x = x[N_global[comm.rank]:N_global[comm.rank + 1]]
        for i, f in enumerate(F):
            function_set_values(f, x[N[i]:N[i + 1]])

    M = [function_new(m0, static=function_is_static(m0),
                      cache=function_is_cached(m0),
                      checkpoint=function_is_checkpointed(m0))
         for m0 in M0]
    J = [J0]
    J_M = [M0]

    def fun(x):
        if not J[0] is None:
            return J[0].value()

        set(M, x)
        clear_caches(*M)

        old_manager = _manager()
        set_manager(manager)
        manager.reset()
        manager.stop()
        clear_caches()

        manager.start()
        J[0] = forward(*M)
        manager.stop()

        set_manager(old_manager)

        J_M[0] = M
        return J[0].value()

    def jac(x):
        fun(x)
        dJ = manager.compute_gradient(J[0], J_M[0])
        J[0] = None
        return get(dJ)

    from scipy.optimize import minimize
    return_value = minimize(fun, get(M0), jac=jac, **kwargs)

    set(M, return_value.x)
    clear_caches(*M)

    return M, return_value
