r"""This module implements Taylor remainder convergence testing using the
approach described in

    - P. E. Farrell, D. A. Ham, S. W. Funke, and M. E. Rognes, 'Automated
      derivation of the adjoint of high-level transient finite element
      programs', SIAM Journal on Scientific Computing 35(4), pp. C369--C393,
      2013, doi: 10.1137/120873558

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
      solver tolerances or floating point roundoff prevent the convergence
      orders being observable.
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

from .interface import (
    garbage_cleanup, packed, space_comm, var_copy, var_dtype, var_is_cached,
    var_is_static, var_local_size, var_name, var_new, var_set_values,
    vars_assign, vars_axpy, vars_copy, vars_inner, vars_linf_norm,
    var_locked, var_scalar_value)

from .caches import clear_caches, local_caches
from .manager import manager as _manager
from .manager import (
    compute_gradient, configure_tlm, manager_disabled, reset_manager,
    restore_manager, set_manager, start_manager, stop_manager, var_tlm)

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
        with var_locked(*M):
            J = forward(*M)
        garbage_cleanup(space_comm(J.space))
        return J

    return wrapped_forward


@local_caches
@manager_disabled()
def taylor_test(forward, M, J_val, *, dJ=None, ddJ=None, seed=1.0e-2, dM=None,
                M0=None, size=5):
    r"""Perform a Taylor remainder convergence test.

    Originally aimed for similar behaviour to the `taylor_test` function in
    dolfin-adjoint 2017.1.0.

    Uncorrected and corrected Taylor remainder magnitudes are computed by
    repeatedly re-running the forward and evaluating the functional. The
    perturbation direction :math:`\zeta` is defined by the `dM` argument.
    :math:`\varepsilon` is set equal to

    .. math::

        \varepsilon = 2^{-p} \eta
            \quad \text{ for } p \in \left\{ 0, \ldots, P - 1 \right\},

    where the norm appearing here is defined to be the :math:`l_\infty` norm of
    the control value degree of freedom vector. The argument `seed` sets the
    value of :math:`\eta`, and the argument `size` sets the value of :math:`P`.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional :math:`J`.
    :arg M: A variable or a :class:`Sequence` of variables defining the control
        :math:`m`.
    :arg J_val: A scalar defining the value of the functional :math:`J` for
        control value defined by `M0`.
    :arg dJ: A variable or a :class:`Sequence` of variables defining a value
        for the derivative of the functional with respect to the control.
        Required if `ddJ` is not supplied.
    :arg ddJ: A :class:`.Hessian` used to compute the Hessian action on the
        considered perturbation direction. If supplied then a higher order
        corrected Taylor remainder magnitude is computed. If `dJ` is not
        supplied, also computes the first order directional derivative.
    :arg seed: Defines the value of :math:`\eta`. Controls the magnitude of the
        perturbation.
    :arg dM: Defines the perturbation direction :math:`\zeta`. If not provided
        then the direction used has degrees of freedom real and (in the complex
        case) complex components set using :func:`numpy.random.random`, scaled
        by the :math:`l_\infty` norm of the degree of freedom vector for `M` if
        this is non-zero.
    :arg M0: Defines the value of the control at which the functional and
        derivatives are evaluated. `M` is used if not supplied.
    :arg size: The number of values of :math:`\varepsilon` to consider.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2 if `ddJ` is not supplied, and
        close to 3 if `ddJ` is supplied.
    """

    M = packed(M)
    if dJ is not None:
        dJ = packed(dJ)
    if dM is not None:
        dM = packed(dM)
    if M0 is not None:
        M0 = packed(M0)

    logger = logging.getLogger("tlm_adjoint.verification")
    forward = wrapped_forward(forward)

    if M0 is None:
        M0_val = vars_copy(M)
    else:
        M0_val = M0
    M1 = tuple(var_new(m, static=var_is_static(m),
                       cache=var_is_cached(m))
               for m in M)

    eps = seed * np.array([2 ** -p for p in range(size)], dtype=float)
    if dM is None:
        M_0_norm = vars_linf_norm(M)
        dM = tuple(var_new(m1, static=var_is_static(m1),
                           cache=var_is_cached(m1))
                   for m1 in M1)
        for dm in dM:
            dm_arr = np.random.random(var_local_size(dm))
            if issubclass(var_dtype(dm), np.complexfloating):
                dm_arr = dm_arr \
                    + 1.0j * np.random.random(var_local_size(dm))
            var_set_values(dm, dm_arr * (1.0 if M_0_norm == 0.0 else M_0_norm))
            del dm_arr

    J_vals = np.full(eps.shape, np.nan, dtype=complex)
    for i in range(eps.shape[0]):
        vars_assign(M1, M0_val)
        vars_axpy(M1, eps[i], dM)
        clear_caches()
        J_vals[i] = var_scalar_value(forward(*M1))

    error_norms_0 = abs(J_vals - J_val)
    orders_0 = np.log(error_norms_0[1:] / error_norms_0[:-1]) / np.log(0.5)
    logger.info(f"Error norms, no adjoint   = {error_norms_0}")
    logger.info(f"Orders,      no adjoint   = {orders_0}")

    if ddJ is None:
        error_norms_1 = abs(J_vals - J_val
                            - eps * vars_inner(dM, dJ))
        orders_1 = np.log(error_norms_1[1:] / error_norms_1[:-1]) / np.log(0.5)
        logger.info(f"Error norms, with adjoint = {error_norms_1}")
        logger.info(f"Orders,      with adjoint = {orders_1}")
        return orders_1.min()
    else:
        if dJ is None:
            _, dJ, ddJ = ddJ.action(M, dM, M0=M0)
        else:
            dJ = vars_inner(dM, dJ)
            _, _, ddJ = ddJ.action(M, dM, M0=M0)
        error_norms_2 = abs(J_vals - J_val
                            - eps * dJ
                            - 0.5 * eps * eps * vars_inner(dM, ddJ))
        orders_2 = np.log(error_norms_2[1:] / error_norms_2[:-1]) / np.log(0.5)
        logger.info(f"Error norms, with adjoint = {error_norms_2}")
        logger.info(f"Orders,      with adjoint = {orders_2}")
        return orders_2.min()


@local_caches
@restore_manager
def taylor_test_tlm(forward, M, tlm_order, *, seed=1.0e-2, dMs=None, size=5,
                    manager=None):
    r"""Perform a Taylor remainder convergence test for a functional :math:`J`
    defined to the `(tlm_order - 1)` th derivative of some functional
    :math:`K`. The `tlm_order` th derivative of :math:`K`, appearing in the
    corrected Taylor remainder magnitude, is computed using a `tlm_order` th
    order tangent-linear.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional :math:`K`.
    :arg M: A variable or a :class:`Sequence` of variables defining the control
        :math:`m` and its value.
    :arg tlm_order: An :class:`int` defining the tangent-linear order to
        test.
    :arg seed: Controls the perturbation magnitude. See :func:`.taylor_test`.
    :arg dMs: A :class:`Sequence` of length `tlm_order` whose elements are each
        a variable or a :class:`Sequence` of variables. The functional
        :math:`J` appearing in the definition of the Taylor remainder
        magnitudes is defined to be a `(tlm_adjoint - 1)` th derivative,
        defined by successively taking the derivative of :math:`K` with respect
        to the control and with directions defined by the `dM[:-1]` (with the
        directions considered in order). The perturbation direction
        :math:`\zeta` is defined by `dM[-1]` -- see :func:`.taylor_test`. By
        default `dMs[:-1]` have real and (in the complex case) complex
        components set using :func:`numpy.random.random`, and the default for
        `dMs[-1]` is set as described in :func:`.taylor_test` (see the `dM`
        argument).
    :arg size: The number of values of :math:`\varepsilon` to consider. See
        :func:`.taylor_test`.
    :arg manager: An :class:`.EquationManager` used to create an internal
        manager via :meth:`.EquationManager.new`. `manager()` is used if not
        supplied.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2.
    """

    M = packed(M)
    if dMs is not None:
        dMs = tuple(map(packed, dMs))

    logger = logging.getLogger("tlm_adjoint.verification")
    forward = wrapped_forward(forward)
    if manager is None:
        manager = _manager()
    manager = manager.new("memory", {})
    set_manager(manager)

    M = tuple(var_copy(m, name=var_name(m),
                       static=var_is_static(m),
                       cache=var_is_cached(m)) for m in M)
    M1 = tuple(var_new(m, static=var_is_static(m),
                       cache=var_is_cached(m))
               for m in M)

    eps = seed * np.array([2 ** -p for p in range(size)], dtype=float)
    if dMs is None:
        M_0_norm = vars_linf_norm(M)
        dMs = tuple(tuple(var_new(m, static=var_is_static(m),
                                  cache=var_is_cached(m))
                          for m in M)
                    for _ in range(tlm_order))
        for dM in dMs:
            for dm in dM:
                dm_arr = np.random.random(var_local_size(dm))
                if issubclass(var_dtype(dm), np.complexfloating):
                    dm_arr = dm_arr \
                        + 1.0j * np.random.random(var_local_size(dm))
                var_set_values(dm, dm_arr * (1.0 if M_0_norm == 0.0 else M_0_norm))  # noqa: E501
                del dm_arr

    def forward_tlm(dMs, *M):
        reset_manager()
        clear_caches()

        configure_tlm(*((M, dM) for dM in dMs))
        start_manager(annotate=False, tlm=True)
        J = forward(*M)
        stop_manager()
        for dM in dMs:
            J = var_tlm(J, (M, dM))

        return J

    J_val = var_scalar_value(forward_tlm(dMs[:-1], *M))
    dJ = var_scalar_value(forward_tlm(dMs, *M))

    J_vals = np.full(eps.shape, np.nan, dtype=complex)
    for i in range(eps.shape[0]):
        vars_assign(M1, M)
        vars_axpy(M1, eps[i], dMs[-1])
        J_vals[i] = var_scalar_value(forward_tlm(dMs[:-1], *M1))

    error_norms_0 = abs(J_vals - J_val)
    orders_0 = np.log(error_norms_0[1:] / error_norms_0[:-1]) / np.log(0.5)
    logger.info(f"Error norms, no tangent-linear   = {error_norms_0}")
    logger.info(f"Orders,      no tangent-linear   = {orders_0}")

    error_norms_1 = abs(J_vals - J_val - eps * dJ)
    orders_1 = np.log(error_norms_1[1:] / error_norms_1[:-1]) / np.log(0.5)
    logger.info(f"Error norms, with tangent-linear = {error_norms_1}")
    logger.info(f"Orders,      with tangent-linear = {orders_1}")

    reset_manager()
    return orders_1.min()


@local_caches
@restore_manager
def taylor_test_tlm_adjoint(forward, M, adjoint_order, *, seed=1.0e-2,
                            dMs=None, size=5, manager=None):
    r"""Perform a Taylor remainder convergence test for a functional :math:`J`
    defined to the `(adjoint_order - 1)` th derivative of some functional
    :math:`K`. The `adjoint_order` th derivative of :math:`K`, appearing in the
    corrected Taylor remainder magnitude, is computed using an adjoint
    associated with an `(adjoint_order - 1)` th order tangent-linear.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional :math:`K`.
    :arg M: A variable or a :class:`Sequence` of variables defining the control
        :math:`m` and its value.
    :arg adjoint_order: An :class:`int` defining the adjoint order to test.
    :arg seed: Controls the perturbation magnitude. See :func:`.taylor_test`.
    :arg dMs: A :class:`Sequence` of length `adjoint_order` whose elements are
        each a variable or a :class:`Sequence` of variables. The functional
        :math:`J` appearing in the definition of the Taylor remainder
        magnitudes is defined to be a `(adjoint_order - 1)` th derivative,
        defined by successively taking the derivative of :math:`K` with respect
        to the control and with directions defined by the `dM[:-1]` (with the
        directions considered in order). The perturbation direction
        :math:`\zeta` is defined by `dM[-1]` -- see :func:`.taylor_test`. By
        default `dMs[:-1]` have real and (in the complex case) complex
        components set using :func:`numpy.random.random`, and the default for
        `dMs[-1]` is set as described in :func:`.taylor_test` (see the `dM`
        argument).
    :arg size: The number of values of :math:`\varepsilon` to consider. See
        :func:`.taylor_test`.
    :arg manager: An :class:`.EquationManager` used to create an internal
        manager via :meth:`.EquationManager.new`. `manager()` is used if not
        supplied.
    :returns: The minimum order observed, via a power law fit between
        consecutive pairs of values of :math:`\varepsilon`, in the calculations
        for the corrected Taylor remainder magnitude. In a successful
        verification this should be close to 2.
    """

    M = packed(M)
    if dMs is not None:
        dMs = tuple(map(packed, dMs))

    forward = wrapped_forward(forward)
    if manager is None:
        manager = _manager()
    manager = manager.new()
    set_manager(manager)

    M = tuple(var_copy(m, name=var_name(m),
                       static=var_is_static(m),
                       cache=var_is_cached(m)) for m in M)

    if dMs is None:
        dM_test = None
        dMs = tuple(tuple(var_new(m, static=var_is_static(m),
                                  cache=var_is_cached(m))
                          for m in M)
                    for _ in range(adjoint_order - 1))
        for dM in dMs:
            for dm in dM:
                dm_arr = np.random.random(var_local_size(dm))
                if issubclass(var_dtype(dm), np.complexfloating):
                    dm_arr = dm_arr \
                        + 1.0j * np.random.random(var_local_size(dm))
                var_set_values(dm, dm_arr)
                del dm_arr
    else:
        dM_test = dMs[-1]
        dMs = dMs[:-1]

    @restore_manager
    def forward_tlm(*M, annotate=False):
        if not annotate:
            set_manager(manager.new("memory", {}))
        reset_manager()
        clear_caches()

        configure_tlm(*((M, dM) for dM in dMs),
                      annotate=annotate)
        start_manager(annotate=annotate, tlm=True)
        J = forward(*M)
        stop_manager()
        for dM in dMs:
            J = var_tlm(J, (M, dM))

        return J

    J = forward_tlm(*M, annotate=True)
    J_val = var_scalar_value(J)
    dJ = compute_gradient(J, M)

    reset_manager()
    return taylor_test(forward_tlm, M, J_val, dJ=dJ, seed=seed, dM=dM_test,
                       size=size)
