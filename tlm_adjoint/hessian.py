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


class Hessian:
    def __init__(self):
        pass

    def compute_gradient(self, M, M0=None):
        raise NotImplementedError("Method not overridden")

    def action(self, M, dM, M0=None):
        raise NotImplementedError("Method not overridden")

    def action_fn(self, m, m0=None):
        """
        Return a callable which accepts a function defining dm, and returns the
        Hessian action.

        Arguments:

        m   A function defining the control
        m0  (Optional) A function defining the control value
        """

        def action(dm):
            _, _, ddJ = self.action(m, dm, M0=m0)
            return conjugate(ddJ)

        return action


class GeneralHessian(Hessian):
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
        """
        Evaluate a derivative. Re-evaluates the forward. Returns a tuple
            (J, dJ)
        where
        - J is the functional value
        - dJ is the complex conjugate of the derivative of J with respect to
          the parameters defined by M

        Arguments:

        M   A function, or a sequence of functions, defining the control
            parameters.
        M0  (Optional) A function, or a sequence of functions, defining the
            values of the control parameters.
        """

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
        """
        Evaluate a Hessian action. Re-evaluates the forward. Returns a tuple
            (J, dJ, ddJ)
        where
        - J is the functional value
        - dJ is the derivative of J with respect to the parameters defined by M
          evaluated with direction dM
        - ddJ is the complex conjugate of the action of the second derivative
          of the functional with respect to M on dM

        Arguments:

        M   A function, or a sequence of functions, defining the control
            parameters.
        dM  A function, or a sequence or functions, on which the Hessian acts.
        M0  (Optional) A function, or a sequence of functions, defining the
            values of the control parameters.
        """

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


class GaussNewton:
    def __init__(self, J_space, R_inv_action, B_inv_action=None):
        self._J_space = J_space
        self._R_inv_action = R_inv_action
        self._B_inv_action = B_inv_action

    def _setup_manager(self, M, dM, M0=None):
        raise NotImplementedError("Method not overridden")

    @restore_manager
    def action(self, M, dM, M0=None):
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
        def action(dm):
            return conjugate(self.action(m, dm, M0=m0))

        return action


class GeneralGaussNewton(GaussNewton):
    def __init__(self, forward, J_space, R_inv_action, B_inv_action=None,
                 *, manager=None):
        if manager is None:
            manager = _manager().new()

        super().__init__(J_space, R_inv_action, B_inv_action=B_inv_action)
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
