#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import check_space_types_conjugate_dual, function_axpy, \
    function_copy, function_get_values, function_is_cached, \
    function_is_checkpointed, function_is_static, function_name, \
    function_new, function_new_conjugate, function_set_values, is_function

from .caches import local_caches
from .equations import InnerProduct
from .functional import Functional
from .manager import manager as _manager
from .manager import compute_gradient, configure_tlm, function_tlm, \
    reset_manager, restore_manager, set_manager, start_manager, stop_manager
from .overloaded_float import Float, FloatSpace

from abc import ABC, abstractmethod
from collections.abc import Sequence
import functools

__all__ = \
    [
        "GaussNewton",
        "GeneralGaussNewton",
        "GeneralHessian",
        "Hessian"
    ]


def conjugate(X):
    if is_function(X):
        X = (X,)
    X_conj = tuple(function_new_conjugate(x) for x in X)
    assert len(X) == len(X_conj)
    for x, x_conj in zip(X, X_conj):
        function_set_values(x_conj, function_get_values(x).conjugate())
    return X_conj[0] if len(X_conj) == 1 else X_conj


class Hessian(ABC):
    r"""Represents a Hessian associated with a given forward model. Abstract
    base class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_gradient(self, M, M0=None):
        r"""Compute the (conjugate of the) derivative of a functional with
        respect to a control using an adjoint model.

        :arg M: A function or a :class:`Sequence` of functions defining the
            control variable.
        :arg M0: A function or a :class:`Sequence` of functions defining the
            control value. `M` is used if not supplied.
        :returns: The derivative. A function or :class:`Sequence` of functions,
            depending on the type of `M`.
        """

        raise NotImplementedError

    @abstractmethod
    def action(self, M, dM, M0=None):
        r"""Compute (the conjugate of) a Hessian action on some :math:`\zeta`
        using an adjoint of a tangent-linear model. i.e. considering
        derivatives to be column vectors, compute

        .. math::

            \left( \frac{d}{dm} \left[
                \frac{d \mathcal{J}}{d m}^T \zeta \right] \right)^{*,T}.

        :arg M: A function or a :class:`Sequence` of functions defining the
            control variable.
        :arg dM: A function or a :class:`Sequence` of functions defining
            :math:`\zeta`. The (conjugate of the) Hessian action on
            :math:`\zeta` is computed.
        :arg M0: A function or a :class:`Sequence` of functions defining the
            control value. `M` is used if not supplied.
        :returns: A tuple `(J, dJ, ddJ)`. `J` is the value of the functional.
            `dJ` is the value of :math:`\left( d \mathcal{J} / d m \right)^T
            \zeta`. `ddJ` stores the (conjugate of the) result of the Hessian
            action on :math:`\zeta`, and is a function or a :class:`Sequence`
            of functions depending on the type of `M`.
        """

        raise NotImplementedError

    def action_fn(self, m, m0=None):
        """Return a :class:`Callable` which can be used to compute Hessian
        actions.

        :arg m: A function defining the control variable.
        :arg m0: A function defining the control value. `m` is used if not
            supplied.
        :returns: A :class:`Callable` which accepts a single function argument,
            and returns the result of the Hessian action on that argument as a
            function. Note that the result is *not* the conjugate of the
            Hessian action on the input argument.
        """

        def action(dm):
            _, _, ddJ = self.action(m, dm, M0=m0)
            return conjugate(ddJ)

        return action


class GeneralHessian(Hessian):
    """Represents a Hessian associated with a given forward model. Calls to
    :meth:`compute_gradient` or :meth:`action` re-run the forward.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or
        :class:`tlm_adjoint.functional.Functional` defining the forward
        functional.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.
    """

    def __init__(self, forward, *, manager=None):
        if manager is None:
            manager = _manager().new()

        @functools.wraps(forward)
        def wrapped_forward(*M):
            J = forward(*M)
            if is_function(J):
                J = Functional(_fn=J)
            return J

        super().__init__()
        self._forward = wrapped_forward
        self._manager = manager

    @local_caches
    @restore_manager
    def compute_gradient(self, M, M0=None):
        if not isinstance(M, Sequence):
            J, (dJ,) = self.compute_gradient(
                (M,),
                M0=None if M0 is None else (M0,))
            return J, dJ

        set_manager(self._manager)
        reset_manager()
        stop_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(function_copy(m0, name=function_name(m),
                                static=function_is_static(m),
                                cache=function_is_cached(m),
                                checkpoint=function_is_checkpointed(m))
                  for m0, m in zip(M0, M))
        del M0

        start_manager()
        J = self._forward(*M)
        stop_manager()

        J_val = J.value()
        dJ = compute_gradient(J, M)

        return J_val, dJ

    @local_caches
    @restore_manager
    def action(self, M, dM, M0=None):
        if not isinstance(M, Sequence):
            J_val, dJ_val, (ddJ,) = self.action(
                (M,), (dM,),
                M0=None if M0 is None else (M0,))
            return J_val, dJ_val, ddJ

        set_manager(self._manager)
        reset_manager()
        stop_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(function_copy(m0, name=function_name(m),
                                static=function_is_static(m),
                                cache=function_is_cached(m),
                                checkpoint=function_is_checkpointed(m))
                  for m0, m in zip(M0, M))
        del M0

        dM = tuple(function_copy(dm, name=function_name(dm),
                                 static=function_is_static(dm),
                                 cache=function_is_cached(dm),
                                 checkpoint=function_is_checkpointed(dm))
                   for dm in dM)

        configure_tlm((M, dM))
        start_manager()
        J = self._forward(*M)
        dJ = J.tlm_functional((M, dM))
        stop_manager()

        J_val = J.value()
        dJ_val = dJ.value()
        ddJ = compute_gradient(dJ, M)

        return J_val, dJ_val, ddJ


class GaussNewton(ABC):
    r"""Represents a Gauss-Newton approximation for a Hessian. Abstract base
    class.

    In terms of matrices this defines a Hessian approximation

    .. math::

        H = J^* R_\text{obs}^{-1} J + B^{-1},

    where :math:`J` is the forward Jacobian. In a variational assimilation
    approach :math:`R_\text{obs}^{-1}` corresponds to the observational inverse
    covariance and :math:`B^{-1}` corresponds to the background inverse
    covariance.

    :arg R_inv_action: A :class:`Callable` which accepts one or more functions,
        and returns the action of the operator corresponding to
        :math:`R_\text{obs}^{-1}` on those functions, returning the result as a
        function or a :class:`Sequence` of functions.
    :arg B_inv_action: A :class:`Callable` which accepts one or more functions,
        and returns the action of the operator corresponding to :math:`B^{-1}`
        on those functions, returning the result as a function or a
        :class:`Sequence` of functions.
    :arg J_space: The space for the functional. `FloatSpace(Float)` is used if
        not supplied.
    """

    def __init__(self, R_inv_action, B_inv_action=None, *,
                 J_space=None):
        if J_space is None:
            J_space = FloatSpace(Float)

        self._J_space = J_space
        self._R_inv_action = R_inv_action
        self._B_inv_action = B_inv_action

    @abstractmethod
    def _setup_manager(self, M, dM, M0=None):
        raise NotImplementedError

    @restore_manager
    def action(self, M, dM, M0=None):
        r"""Compute (the conjugate of) a Hessian action on some :math:`\zeta`,
        using the Gauss-Newton approximation for the Hessian. i.e. compute

        .. math::

            \left( H \zeta \right)^{*,T}.

        :arg M: A function or a :class:`Sequence` of functions defining the
            control variable.
        :arg dM: A function or a :class:`Sequence` of functions defining
            :math:`\zeta`. The (conjugate of the) approximated Hessian action
            on :math:`\zeta` is computed.
        :arg M0: A function or a :class:`Sequence` of functions defining the
            control value. `M` is used if not supplied.
        :returns: The (conjugate of the) result of the approximated Hessian
            action on :math:`\zeta`. A function or a :class:`Sequence` of
            functions depending on the type of `M`.
        """

        if not isinstance(M, Sequence):
            ddJ, = self.action(
                (M,), (dM,),
                M0=None if M0 is None else (M0,))
            return ddJ

        manager, M, dM, X = self._setup_manager(M, dM, M0=M0)
        set_manager(manager)

        # J dM
        tau_X = tuple(function_tlm(x, (M, dM)) for x in X)
        # R^{-1} conj(J dM)
        R_inv_tau_X = self._R_inv_action(
            *tuple(function_copy(tau_x) for tau_x in tau_X))
        if not isinstance(R_inv_tau_X, Sequence):
            R_inv_tau_X = (R_inv_tau_X,)
        assert len(tau_X) == len(R_inv_tau_X)
        for tau_x, R_inv_tau_x in zip(tau_X, R_inv_tau_X):
            check_space_types_conjugate_dual(tau_x, R_inv_tau_x)

        # This defines the adjoint right-hand-side appropriately to compute a
        # J^* action
        start_manager()
        J = Functional(space=self._J_space)
        assert len(X) == len(R_inv_tau_X)
        for x, R_inv_tau_x in zip(X, R_inv_tau_X):
            J_term = function_new(J.function())
            InnerProduct(J_term, x, function_copy(R_inv_tau_x)).solve(
                tlm=False)
            J.addto(J_term, tlm=False)
        stop_manager()

        # Likelihood term: J^* R^{-1} conj(J dM)
        ddJ = compute_gradient(J, M)

        # Prior term: B^{-1} conj(dM)
        if self._B_inv_action is not None:
            B_inv_dM = self._B_inv_action(
                *tuple(function_copy(dm) for dm in dM))
            if not isinstance(B_inv_dM, Sequence):
                B_inv_dM = (B_inv_dM,)
            assert len(dM) == len(B_inv_dM)
            for dm, B_inv_dm in zip(dM, B_inv_dM):
                check_space_types_conjugate_dual(dm, B_inv_dm)
            assert len(ddJ) == len(B_inv_dM)
            for i, B_inv_dm in enumerate(B_inv_dM):
                function_axpy(ddJ[i], 1.0, B_inv_dm)

        return ddJ

    def action_fn(self, m, m0=None):
        """Return a :class:`Callable` which can be used to compute Hessian
        actions using the Gauss-Newton approximation.

        :arg m: A function defining the control variable.
        :arg m0: A function defining the control value. `m` is used if not
            supplied.
        :returns: A :class:`Callable` which accepts a single function argument,
            and returns the result of the approximated Hessian action on that
            argument as a function. Note that the result is *not* the conjugate
            of the approximated Hessian action on the input argument.
        """

        def action(dm):
            return conjugate(self.action(m, dm, M0=m0))

        return action


class GeneralGaussNewton(GaussNewton):
    """Represents a Gauss-Newton approximation to a Hessian associated with a
    given forward model. Calls to :meth:`GaussNewton.action` re-run the
    forward.

    :arg forward: A :class:`Callable` which accepts one or more function
        arguments, and which returns a function or :class:`Sequence` of
        functions defining the state.
    :arg R_inv_action: See :class:`GaussNewton`.
    :arg B_inv_action: See :class:`GaussNewton`.
    :arg J_space: The space for the functional. `FloatSpace(Float)` is used if
        not supplied.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.
    """

    def __init__(self, forward, R_inv_action, B_inv_action=None, *,
                 J_space=None, manager=None):
        if manager is None:
            manager = _manager().new()

        super().__init__(R_inv_action, B_inv_action=B_inv_action,
                         J_space=J_space)
        self._forward = forward
        self._manager = manager

    @local_caches
    @restore_manager
    def _setup_manager(self, M, dM, M0=None):
        set_manager(self._manager)
        reset_manager()
        stop_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(function_copy(m0, name=function_name(m),
                                static=function_is_static(m),
                                cache=function_is_cached(m),
                                checkpoint=function_is_checkpointed(m))
                  for m0, m in zip(M0, M))
        del M0

        dM = tuple(function_copy(dm, name=function_name(dm),
                                 static=function_is_static(dm),
                                 cache=function_is_cached(dm),
                                 checkpoint=function_is_checkpointed(dm))
                   for dm in dM)

        configure_tlm((M, dM), annotate=False)
        start_manager()
        X = self._forward(*M)
        if not isinstance(X, Sequence):
            X = (X,)
        stop_manager()

        return self._manager, M, dM, X
