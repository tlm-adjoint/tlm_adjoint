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

from .interface import function_axpy, function_copy, function_get_values, \
    function_is_cached, function_is_checkpointed, function_is_static, \
    function_linf_norm, function_local_size, function_new, \
    function_set_values, is_function

from .base_caches import clear_caches
from .manager import manager as _manager, set_manager

import numpy as np

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
    comm = manager.comm().Dup()

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

        if comm.rank == 0:
            x_global = comm.gather(x, root=0)
            X = np.full(N_global[-1], np.NAN, dtype=np.float64)
            for i, x_p in enumerate(x_global):
                X[N_global[i]:N_global[i + 1]] = x_p
            return X
        else:
            comm.gather(x, root=0)
            return None

    def set(F, x):
        if comm.rank == 0:
            x = comm.scatter([x[N_global[rank]:N_global[rank + 1]]
                              for rank in range(comm.size)], root=0)
        else:
            assert x is None
            x = comm.scatter(None, root=0)
        for i, f in enumerate(F):
            function_set_values(f, x[N[i]:N[i + 1]])

    M = [function_new(m0, static=function_is_static(m0),
                      cache=function_is_cached(m0),
                      checkpoint=function_is_checkpointed(m0))
         for m0 in M0]
    J = [J0]
    J_M = [tuple(function_copy(m0) for m0 in M0), M0]

    def fun(x, force=False):
        set(M, x)
        clear_caches(*M)

        if not force and J[0] is not None:
            change_norm = 0.0
            for m, m0 in zip(M, J_M[0]):
                change = function_copy(m)
                function_axpy(change, -1.0, m0)
                change_norm = max(change_norm, function_linf_norm(change))
            if change_norm == 0.0:
                return J[0].value()

        J_M[0] = tuple(function_copy(m) for m in M)

        old_manager = _manager()
        set_manager(manager)
        manager.reset()
        manager.stop()
        clear_caches()

        manager.start()
        J[0] = forward(*M)
        manager.stop()

        set_manager(old_manager)

        J_M[1] = M

        return J[0].value()

    def fun_bcast(x):
        if comm.rank == 0:
            comm.bcast(("fun", None), root=0)
        return fun(x)

    def jac(x):
        if J_M[1] is None:
            fun(x, force=True)
        dJ = manager.compute_gradient(J[0], J_M[1])
        if manager._cp_method not in ["memory", "periodic_disk"]:
            J_M[1] = None
        return get(dJ)

    def jac_bcast(x):
        if comm.rank == 0:
            comm.bcast(("jac", None), root=0)
        return jac(x)

    from scipy.optimize import minimize
    if comm.rank == 0:
        x0 = get(M0)
        return_value = minimize(fun_bcast, x0, jac=jac_bcast, **kwargs)
        comm.bcast(("return", return_value), root=0)
        set(M, return_value.x)
    else:
        get(M0)
        while True:
            action, data = comm.bcast(None, root=0)
            if action == "fun":
                assert data is None
                fun(None)
            elif action == "jac":
                assert data is None
                jac(None)
            elif action == "return":
                assert data is not None
                return_value = data
                break
            else:
                raise OptimizationException(f"Unexpected action '{action:s}'")
        set(M, None)

    clear_caches(*M)
    comm.Free()

    return M, return_value
