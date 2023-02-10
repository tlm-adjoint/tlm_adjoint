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

from .interface import function_assign, function_axpy, function_copy, \
    function_dtype, function_inner, function_is_cached, \
    function_is_checkpointed, function_is_static, function_linf_norm, \
    function_local_size, function_name, function_new, function_set_values, \
    garbage_cleanup, is_function, space_comm

from .caches import clear_caches, local_caches
from .functional import Functional
from .manager import manager as _manager, restore_manager, set_manager

from collections.abc import Sequence
import functools
import logging
import numpy as np

__all__ = \
    [
        "taylor_test",
        "taylor_test_tlm",
        "taylor_test_tlm_adjoint"
    ]


def wrapped_forward(forward):
    @functools.wraps(forward)
    def wrapped_forward(*M):
        J = forward(*M)
        if is_function(J):
            J = Functional(_fn=J)
        garbage_cleanup(space_comm(J.space()))
        return J

    return wrapped_forward


@local_caches
def taylor_test(forward, M, J_val, dJ=None, ddJ=None, seed=1.0e-2, dM=None,
                M0=None, size=5, manager=None):
    # Aims for similar behaviour to the dolfin-adjoint taylor_test function in
    # dolfin-adjoint 2017.1.0. Arguments based on dolfin-adjoint taylor_test
    # arguments
    #   forward (renamed from J)
    #   M (renamed from m)
    #   J_val (renamed from Jm)
    #   dJ (renamed from dJdm)
    #   ddJ (renamed from HJm)
    #   seed
    #   dM (renamed from perturbation_direction)
    #   M0 (renamed from value)
    #   size
    """
    Perform a Taylor remainder verification test.

    Arguments:

    forward  A callable which takes as input one or more functions defining the
             value of the control, and returns the functional.
    M        A function, or a sequence of functions. The control parameters.
    J_val    The reference functional value.
    dJ       (Optional if ddJ is supplied) A function, or a sequence of
             functions, storing the complex conjugate of the derivative of J
             with respect to M.
    ddJ      (Optional) A Hessian used to compute Hessian actions associated
             with the second derivative of J with respect to M.
    seed     (Optional) The maximum scaling for the perturbation is seed
             multiplied by the inf norm of the reference value (degrees of
             freedom inf norm) of the control (or 1 if this is less than 1).
    dM       A perturbation direction. Values generated using
             numpy.random.random are used if not supplied.
    M0       (Optional) The reference value of the control.
    size     (Optional) The number of perturbed forward runs used in the test.
    manager  (Optional) The equation manager.
    """

    if not isinstance(M, Sequence):
        if dJ is not None:
            dJ = (dJ,)
        if dM is not None:
            dM = (dM,)
        if M0 is not None:
            M0 = (M0,)
        return taylor_test(forward, (M,), J_val, dJ=dJ, ddJ=ddJ, seed=seed,
                           dM=dM, M0=M0, size=size, manager=manager)

    logger = logging.getLogger("tlm_adjoint.verification")
    forward = wrapped_forward(forward)
    if manager is None:
        manager = _manager()

    if M0 is None:
        M0 = tuple(map(function_copy, M))
    M1 = tuple(function_new(m, static=function_is_static(m),
                            cache=function_is_cached(m),
                            checkpoint=function_is_checkpointed(m))
               for m in M)

    def functions_inner(X, Y):
        inner = 0.0
        assert len(X) == len(Y)
        for x, y in zip(X, Y):
            inner += function_inner(x, y)
        return inner

    def functions_linf_norm(X):
        norm = 0.0
        for x in X:
            norm = max(norm, function_linf_norm(x))
        return norm

    # This combination seems to reproduce dolfin-adjoint behaviour
    eps = np.array([2 ** -p for p in range(size)], dtype=np.float64)
    eps = seed * eps * max(1.0, functions_linf_norm(M0))
    if dM is None:
        dM = tuple(function_new(m1, static=True) for m1 in M1)
        for dm in dM:
            dm_arr = np.random.random(function_local_size(dm))
            if issubclass(function_dtype(dm),
                          (complex, np.complexfloating)):
                dm_arr = dm_arr \
                    + 1.0j * np.random.random(function_local_size(dm))
            function_set_values(dm, dm_arr)
            del dm_arr

    J_vals = np.full(eps.shape, np.NAN, dtype=np.complex128)
    for i in range(eps.shape[0]):
        assert len(M0) == len(M1)
        assert len(M0) == len(dM)
        for m0, m1, dm in zip(M0, M1, dM):
            function_assign(m1, m0)
            function_axpy(m1, eps[i], dm)
        clear_caches()
        annotation_enabled, tlm_enabled = manager.stop()
        try:
            J_vals[i] = forward(*M1).value()
        finally:
            manager.start(annotate=annotation_enabled, tlm=tlm_enabled)
    if abs(J_vals.imag).max() == 0.0:
        J_vals = J_vals.real

    error_norms_0 = abs(J_vals - J_val)
    orders_0 = np.log(error_norms_0[1:] / error_norms_0[:-1]) / np.log(0.5)
    logger.info(f"Error norms, no adjoint   = {error_norms_0}")
    logger.info(f"Orders,      no adjoint   = {orders_0}")

    if ddJ is None:
        error_norms_1 = abs(J_vals - J_val
                            - eps * functions_inner(dM, dJ))
        orders_1 = np.log(error_norms_1[1:] / error_norms_1[:-1]) / np.log(0.5)
        logger.info(f"Error norms, with adjoint = {error_norms_1}")
        logger.info(f"Orders,      with adjoint = {orders_1}")
        return orders_1.min()
    else:
        if dJ is None:
            _, dJ, ddJ = ddJ.action(M, dM, M0=M0)
        else:
            dJ = functions_inner(dM, dJ)
            _, _, ddJ = ddJ.action(M, dM, M0=M0)
        error_norms_2 = abs(J_vals - J_val
                            - eps * dJ
                            - 0.5 * eps * eps * functions_inner(dM, ddJ))
        orders_2 = np.log(error_norms_2[1:] / error_norms_2[:-1]) / np.log(0.5)
        logger.info(f"Error norms, with adjoint = {error_norms_2}")
        logger.info(f"Orders,      with adjoint = {orders_2}")
        return orders_2.min()


@local_caches
def taylor_test_tlm(forward, M, tlm_order, seed=1.0e-2, dMs=None, size=5,
                    manager=None):
    if not isinstance(M, Sequence):
        if dMs is not None:
            dMs = tuple((dM,) for dM in dMs)
        return taylor_test_tlm(forward, (M,), tlm_order, seed=seed, dMs=dMs,
                               size=size, manager=manager)

    logger = logging.getLogger("tlm_adjoint.verification")
    forward = wrapped_forward(forward)
    if manager is None:
        manager = _manager()
    tlm_manager = manager.new("memory", {})
    tlm_manager.stop()

    M = tuple(function_copy(m, name=function_name(m),
                            static=function_is_static(m),
                            cache=function_is_cached(m),
                            checkpoint=function_is_checkpointed(m)) for m in M)
    M1 = tuple(function_new(m, static=function_is_static(m),
                            cache=function_is_cached(m),
                            checkpoint=function_is_checkpointed(m))
               for m in M)

    def functions_linf_norm(X):
        norm = 0.0
        for x in X:
            norm = max(norm, function_linf_norm(x))
        return norm

    eps = np.array([2 ** -p for p in range(size)], dtype=np.float64)
    eps = seed * eps * max(1.0, functions_linf_norm(M))
    if dMs is None:
        dMs = tuple(tuple(function_new(m, static=True) for m in M)
                    for i in range(tlm_order))
        for dM in dMs:
            for dm in dM:
                dm_arr = np.random.random(function_local_size(dm))
                if issubclass(function_dtype(dm),
                              (complex, np.complexfloating)):
                    dm_arr = dm_arr \
                        + 1.0j * np.random.random(function_local_size(dm))
                function_set_values(dm, dm_arr)
                del dm_arr

    @restore_manager
    def forward_tlm(dMs, *M):
        set_manager(tlm_manager)
        tlm_manager.reset()
        tlm_manager.stop()
        clear_caches()

        tlm_manager.configure_tlm(*[(M, dM) for dM in dMs])
        tlm_manager.start(annotate=False, tlm=True)
        J = forward(*M)
        for dM in dMs:
            J = J.tlm_functional((M, dM), manager=tlm_manager)

        return J

    J_val = forward_tlm(dMs[:-1], *M).value()
    dJ = forward_tlm(dMs, *M).value()

    J_vals = np.full(eps.shape, np.NAN, dtype=np.complex128)
    for i in range(eps.shape[0]):
        assert len(M) == len(M1)
        assert len(M) == len(dMs[-1])
        for m0, m1, dm in zip(M, M1, dMs[-1]):
            function_assign(m1, m0)
            function_axpy(m1, eps[i], dm)
        J_vals[i] = forward_tlm(dMs[:-1], *M1).value()
    if abs(J_vals.imag).max() == 0.0:
        J_vals = J_vals.real

    error_norms_0 = abs(J_vals - J_val)
    orders_0 = np.log(error_norms_0[1:] / error_norms_0[:-1]) / np.log(0.5)
    logger.info(f"Error norms, no tangent-linear   = {error_norms_0}")
    logger.info(f"Orders,      no tangent-linear   = {orders_0}")

    error_norms_1 = abs(J_vals - J_val - eps * dJ)
    orders_1 = np.log(error_norms_1[1:] / error_norms_1[:-1]) / np.log(0.5)
    logger.info(f"Error norms, with tangent-linear = {error_norms_1}")
    logger.info(f"Orders,      with tangent-linear = {orders_1}")
    return orders_1.min()


@local_caches
def taylor_test_tlm_adjoint(forward, M, adjoint_order, seed=1.0e-2, dMs=None,
                            size=5, manager=None):
    if not isinstance(M, Sequence):
        if dMs is not None:
            dMs = tuple((dM,) for dM in dMs)
        return taylor_test_tlm_adjoint(
            forward, (M,), adjoint_order, seed=seed, dMs=dMs, size=size,
            manager=manager)

    forward = wrapped_forward(forward)
    if manager is None:
        manager = _manager()
    tlm_manager = manager.new()
    tlm_manager.stop()

    M = tuple(function_copy(m, name=function_name(m),
                            static=function_is_static(m),
                            cache=function_is_cached(m),
                            checkpoint=function_is_checkpointed(m)) for m in M)

    if dMs is None:
        dM_test = None
        dMs = tuple(tuple(function_new(m, static=True) for m in M)
                    for i in range(adjoint_order - 1))
        for dM in dMs:
            for dm in dM:
                dm_arr = np.random.random(function_local_size(dm))
                if issubclass(function_dtype(dm),
                              (complex, np.complexfloating)):
                    dm_arr = dm_arr \
                        + 1.0j * np.random.random(function_local_size(dm))
                function_set_values(dm, dm_arr)
                del dm_arr
    else:
        dM_test = dMs[-1]
        dMs = dMs[:-1]

    @restore_manager
    def forward_tlm(*M, annotate=False):
        set_manager(tlm_manager)
        tlm_manager.reset()
        tlm_manager.stop()
        clear_caches()

        tlm_manager.configure_tlm(*[(M, dM) for dM in dMs],
                                  annotate=annotate)
        tlm_manager.start(annotate=annotate, tlm=True)
        J = forward(*M)
        for dM in dMs:
            J = J.tlm_functional((M, dM), manager=tlm_manager)

        return J

    J = forward_tlm(*M, annotate=True)
    J_val = J.value()
    dJ = tlm_manager.compute_gradient(J, M)

    return taylor_test(forward_tlm, M, J_val, dJ=dJ, seed=seed, dM=dM_test,
                       size=size, manager=tlm_manager)
