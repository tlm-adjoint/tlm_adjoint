#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""This module implements Taylor remainder convergence testing using the
approach described in

  - P. E. Farrell, D. A. Ham, S. W. Funke, and M. E. Rognes, 'Automated
    derivation of the adjoint of high-level transient finite element programs',
    SIAM Journal on Scientific Computing 35(4), pp. C369--C393, 2013, doi:
    10.1137/120873558

Specifically for a sufficiently regular functional :math:`J`, via Taylor's
theorem we have, for some direction :math:`\zeta` and with perturbation
magnitude controlled by :math:`\varepsilon`,

.. math::

    \left| J \left( m + \varepsilon \right) - J \left( m \right) \right|
        = O \left( \varepsilon \right),

.. math::

    \left| J \left( m + \varepsilon \right) - J \left( m \right)
        - \varepsilon dJ \left( m; \zeta \right) \right|
        = O \left( \varepsilon^2 \right),

where here :math:`dJ \left( m; \zeta \right)` denotes the directional
derivative of :math:`J` with respect to :math:`m` with direction :math:`\zeta`.
Here we refer to the quantity appearing on the left-hand-side in the first case
as the 'uncorrected Taylor remainder magnitude', and the quantity appearing on
the left-hand-side in the second case as the 'corrected Taylor remainder
magnitude'

A Taylor remainder convergence test considers some direction, and a number of
different values for :math:`\varepsilon`, and investigates the convergence
rates of the uncorrected and corrected Taylor remainder magnitudes, with the
directional derivative computed using a tangent-linear or adjoint. In a
successful verification the uncorrected Taylor remainder magnitude is observed
to converge to zero at first order, while the corrected Taylor remainder
magnitude is observed to converge to zero at second order.

There are a number of ways that a Taylor remainder convergence test can fail,
including:

  - The computed derivative is incorrect. This is the case that the test is
    designed to find, and indicates an error in the tangent-linear or adjoint
    calculation.
  - The considered values of :math:`\varepsilon` are too large, and the
    asymptotic convergence orders are not observable.
  - The considered values of :math:`\varepsilon` are too small, and iterative
    solver tolerances or floating point roundoff prevent the converge orders
    being observable.
  - The convergence order is higher than expected. For example if the
    directional derivative is zero then the uncorrected Taylor remainder
    magnitude can converge at higher than first order.

In principle higher order derivative calculations can be tested by considering
more terms in the Taylor expansion of the functional. In practice the
corresponding higher order convergence rate can mean that iterative solver
tolerances or floating point roundoff effects are more problematic. Instead,
one can verify the derivative of a derivative, by redefining :math:`J` to be a
directional derivative of some other functional :math:`K`, with the directional
derivative computed using a tangent-linear. A successful verification then once
again corresponds to second order convergence of the corrected Taylor remainder
magnitude.

The functions defined in this module log the uncorrected and corrected Taylor
remainder magnitudes, and also log the observed orders computed using a power
law fit between between consecutive pairs of values of :math:`\varepsilon`.
Logging is performed on a logging module logger, with name
`'tlm_adjoint.verification`' and with severity `logging.INFO`. The minimum
order computed for the corrected Taylor remainder magnitude is returned.

A typical test considers tangent-linears and adjoints up to the relevant order,
e.g. to verify Hessian calculations

.. code-block:: python

    min_order = taylor_test_tlm(forward, M, 1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, M, 1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, M, 2)
    assert min_order > 1.99
"""

from .interface import function_assign, function_axpy, function_copy, \
    function_dtype, function_inner, function_is_cached, \
    function_is_checkpointed, function_is_static, function_linf_norm, \
    function_local_size, function_name, function_new, function_set_values, \
    garbage_cleanup, is_function, space_comm

from .caches import clear_caches, local_caches
from .functional import Functional
from .manager import manager as _manager
from .manager import configure_tlm, paused_manager, reset_manager, \
    restore_manager, set_manager, start_manager, stop_manager
from .tlm_adjoint import EquationManager

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
@restore_manager
def taylor_test(forward, M, J_val, *, dJ=None, ddJ=None, seed=1.0e-2, dM=None,
                M0=None, size=5):
    r"""Perform a Taylor remainder convergence test. Aims for similar behaviour
    to the :func:`taylor_test` function in dolfin-adjoint 2017.1.0.

    Uncorrected and corrected Taylor remainder magnitudes are computed by
    repeatedly re-running the forward and evaluating the functional. The
    perturbation direction :math:`\zeta` is defined by the `dM` argument.
    :math:`\varepsilon` is set equal to

    .. math::

        \varepsilon = 2^{-p} \eta \max \left( 1,
            \left\| m \right\|_{l_\infty} \right)
            \quad \text{ for } p \in \left\{ 0, \ldots, P - 1 \right\},

    where the norm appearing here is defined to be the :math:`l_\infty` norm of
    the control value degree of freedom vector. The argument `seed` sets the
    value of :math:`\eta`, and the argument `size` sets the value of :math:`P`.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or
        :class:`tlm_adjoint.functional.Functional` defining the forward
        functional :math:`J`. Corresponds to the `J` argument in the
        dolfin-adjoint :func:`taylor_test` function.
    :arg M: A function or a :class:`Sequence` of functions defining the control
        variable :math:`m`. Corresponds to the `m` argument in the
        dolfin-adjoint :func:`taylor_test` function.
    :arg J_val: A scalar defining the value of the functional :math:`J` for
        control value defined by `M0`. Corresponds to the `Jm` argument in the
        dolfin-adjoint :func:`taylor_test` function.
    :arg dJ: A function or a :class:`Sequence` of functions defining a value
        for the derivative of the functional with respect to the control.
        Required if `ddJ` is not supplied. Corresponds to the `dJdm` argument
        in the dolfin-adjoint :func:`taylor_test` function.
    :arg ddJ: A :class:`tlm_adjoint.hessian.Hessian` used to compute the
        Hessian action on the considered perturbation direction. If supplied
        then a higher order corrected Taylor remainder magnitude is computed.
        If `dJ` is not supplied, also computes the first order directional
        derivative. Corresponds to the `HJm` argument in the dolfin-adjoint
        :func:`taylor_test` function.
    :arg seed: Defines the value of :math:`\eta`. Controls the magnitude of the
        perturbation. Corresponds to the `seed` argument in the dolfin-adjoint
        :func:`taylor_test` function.
    :arg dM: Defines the perturbation direction :math:`\zeta`. A direction with
        degrees of freedom vector real and (in the complex case) complex parts
        set by :func:`numpy.random.random` is used if not supplied. Corresponds
        to the `perturbation_direction` argument in the dolfin-adjoint
        :func:`taylor_test` function.
    :arg M0: Defines the value of the control at which the functional and
        derivatives are evaluated. `M` is used if not supplied. Corresponds to
        the `value` argument in the dolfin-adjoint :func:`taylor_test`
        function.
    :arg size: The number of values of :math:`\varepsilon` to consider.
        Corresponds to the `size` argument in the dolfin-adjoint
        :func:`taylor_test` function.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2 if `ddJ` is not supplied, and
        close to 3 if `ddJ` is supplied.
    """

    if not isinstance(M, Sequence):
        if dJ is not None:
            dJ = (dJ,)
        if dM is not None:
            dM = (dM,)
        if M0 is not None:
            M0 = (M0,)
        return taylor_test(forward, (M,), J_val, dJ=dJ, ddJ=ddJ, seed=seed,
                           dM=dM, M0=M0, size=size)

    logger = logging.getLogger("tlm_adjoint.verification")
    forward = wrapped_forward(forward)
    set_manager(EquationManager(cp_method="memory", cp_parameters={}))
    stop_manager()

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
        with paused_manager():
            J_vals[i] = forward(*M1).value()
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
def taylor_test_tlm(forward, M, tlm_order, *, seed=1.0e-2, dMs=None, size=5,
                    manager=None):
    r"""Perform a Taylor remainder convergence test for a functional :math:`J`
    defined to the `(tlm_order - 1)` th derivative of some functional
    :math:`K`. The `tlm_order` th derivative of :math:`K`, appearing in the
    corrected Taylor remainder magnitude, is computed using a `tlm_order` th
    order tangent-linear.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or
        :class:`tlm_adjoint.functional.Functional` defining the forward
        functional :math:`K`.
    :arg M: A function or a :class:`Sequence` of functions defining the control
        variable :math:`m` and its value.
    :arg tlm_order: An :class:`int` defining the tangent-linear order to
       test.
    :arg seed: Controls the perturbation magnitude. See :func:`taylor_test`.
    :arg dMs: A :class:`Sequence` of length `tlm_order` whose elements are each
        a function or a :class:`Sequence` of functions. The functional
        :math:`J` appearing in the definition of the Taylor remainder
        magnitudes is defined to be a `(tlm_adjoint - 1)` th derivative,
        defined by successively taking the derivative of :math:`K` with respect
        to the control and with directions defined by the `dM[:-1]` (with the
        directions considered in order). The perturbation direction
        :math:`\zeta` is defined by `dM[-1]` -- see :func:`taylor_test`.
        Directions with degrees of freedom vector real and (in the complex
        case) complex parts set by :func:`numpy.random.random` are used if not
        supplied.
    :arg size: The number of values of :math:`\varepsilon` to consider. See
        :func:`taylor_test`.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` used to
        create an internal manager via
        :meth:`tlm_adjoint.tlm_adjoint.EquationManager.new`. `manager()` is
        used if not supplied.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2.
    """

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
        reset_manager()
        stop_manager()
        clear_caches()

        configure_tlm(*[(M, dM) for dM in dMs])
        start_manager(annotate=False, tlm=True)
        J = forward(*M)
        for dM in dMs:
            J = J.tlm_functional((M, dM))

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
def taylor_test_tlm_adjoint(forward, M, adjoint_order, *, seed=1.0e-2,
                            dMs=None, size=5, manager=None):
    r"""Perform a Taylor remainder convergence test for a functional :math:`J`
    defined to the `(adjoint_order - 1)` th derivative of some functional
    :math:`K`. The `adjoint_order` th derivative of :math:`K`, appearing in the
    corrected Taylor remainder magnitude, is computed using an adjoint
    associated with an `(adjoint_order - 1)` th order tangent-linear.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or
        :class:`tlm_adjoint.functional.Functional` defining the forward
        functional :math:`K`.
    :arg M: A function or a :class:`Sequence` of functions defining the control
        variable :math:`m` and its value.
    :arg adjoint_order: An :class:`int` defining the adjoint order to test.
    :arg seed: Controls the perturbation magnitude. See :func:`taylor_test`.
    :arg dMs: A :class:`Sequence` of length `adjoint_order` whose elements are
        each a function or a :class:`Sequence` of functions. The functional
        :math:`J` appearing in the definition of the Taylor remainder
        magnitudes is defined to be a `(adjoint_order - 1)` th derivative,
        defined by successively taking the derivative of :math:`K` with respect
        to the control and with directions defined by the `dM[:-1]` (with the
        directions considered in order). The perturbation direction
        :math:`\zeta` is defined by `dM[-1]` -- see :func:`taylor_test`.
        Directions with degrees of freedom vector real and (in the complex
        case) complex parts set by :func:`numpy.random.random` are used if not
        supplied.
    :arg size: The number of values of :math:`\varepsilon` to consider. See
        :func:`taylor_test`.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` used to
        create an internal manager via
        :meth:`tlm_adjoint.tlm_adjoint.EquationManager.new`. `manager()` is
        used if not supplied.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2.
    """

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
        reset_manager()
        stop_manager()
        clear_caches()

        configure_tlm(*[(M, dM) for dM in dMs],
                      annotate=annotate)
        start_manager(annotate=annotate, tlm=True)
        J = forward(*M)
        for dM in dMs:
            J = J.tlm_functional((M, dM))

        return J

    J = forward_tlm(*M, annotate=True)
    J_val = J.value()
    dJ = tlm_manager.compute_gradient(J, M)

    return taylor_test(forward_tlm, M, J_val, dJ=dJ, seed=seed, dM=dM_test,
                       size=size)
