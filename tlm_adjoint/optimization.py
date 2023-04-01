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

from .interface import comm_dup, function_axpy, function_copy, \
    function_get_values, function_is_cached, function_is_checkpointed, \
    function_is_static, function_linf_norm, function_local_size, \
    function_new, function_set_values, garbage_cleanup, is_function, space_comm

from .caches import clear_caches, local_caches
from .functional import Functional
from .manager import manager as _manager
from .manager import compute_gradient, reset_manager, restore_manager, \
    set_manager, start_manager, stop_manager

from collections.abc import Sequence
import numpy as np

__all__ = \
    [
        "minimize_scipy"
    ]


@local_caches
@restore_manager
def minimize_scipy(forward, M0, *, manager=None, **kwargs):
    """Provides an interface with :func:`scipy.optimize.minimize` for
    gradient-based optimization.

    Note that the control variable is gathered onto the root process so that
    the serial :func:`scipy.optimize.minimize` function may be used.

    All keyword arguments except for `manager` are passed to
    :func:`scipy.optimize.minimize`.

    **Important note:** No exception is raised if `return_value.success` is
    `False`. Calling code should check this attribute.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or :class:`Functional` defining
        the forward functional.
    :arg M0: A function or :class:`Sequence` of functions defining the control
        variable, and the initial guess for the optimization.
    :arg manager: An :class:`EquationManager` which should be used internally.
        `manager().new()` is used if not supplied.
    :returns: A :class:`tuple` `(M, return_value)`. `M` is function or a
        :class:`Sequence` of functions depending on the type of `M0`, and
        stores the result. `return_value` is the return value of
        :func:`scipy.optimize.minimize`.
    """

    if not isinstance(M0, Sequence):
        (M,), return_value = minimize_scipy(forward, [M0],
                                            manager=manager, **kwargs)
        return M, return_value

    if manager is None:
        manager = _manager().new()
    set_manager(manager)
    comm = comm_dup(manager.comm())

    N = [0]
    for m0 in M0:
        N.append(N[-1] + function_local_size(m0))
    if comm.rank == 0:
        size_global = comm.gather(np.array(N[-1], dtype=np.int64), root=0)
        N_global = [0]
        for size in size_global:
            N_global.append(N_global[-1] + size)
    else:
        comm.gather(np.array(N[-1], dtype=np.int64), root=0)

    def get(F):
        x = np.full(N[-1], np.NAN, dtype=np.float64)
        for i, f in enumerate(F):
            f_vals = function_get_values(f)
            if not np.can_cast(f_vals, x.dtype):
                raise ValueError("Invalid dtype")
            x[N[i]:N[i + 1]] = f_vals

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
    J = [None]
    J_M = [None, None]

    def fun(x, *, force=False):
        set(M, x)

        if not force and J[0] is not None:
            change_norm = 0.0
            assert len(M) == len(J_M[0])
            for m, m0 in zip(M, J_M[0]):
                change = function_copy(m)
                function_axpy(change, -1.0, m0)
                change_norm = max(change_norm, function_linf_norm(change))
            if change_norm == 0.0:
                J_val = J[0].value()
                if not isinstance(J_val, (float, np.floating)):
                    raise TypeError("Unexpected type")
                return J_val

        J_M[0] = tuple(function_copy(m) for m in M)

        reset_manager()
        stop_manager()
        clear_caches()

        start_manager()
        J[0] = forward(*M)
        if is_function(J[0]):
            J[0] = Functional(_fn=J[0])
        garbage_cleanup(space_comm(J[0].space()))
        stop_manager()

        J_M[1] = M

        J_val = J[0].value()
        if not isinstance(J_val, (float, np.floating)):
            raise TypeError("Unexpected type")
        return J_val

    def fun_bcast(x):
        if comm.rank == 0:
            comm.bcast(("fun", None), root=0)
        return fun(x)

    def jac(x):
        fun(x, force=J_M[1] is None)
        dJ = compute_gradient(J[0], J_M[1])
        if manager._cp_schedule.is_exhausted():
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
                raise ValueError(f"Unexpected action '{action:s}'")
        set(M, None)

    return M, return_value
