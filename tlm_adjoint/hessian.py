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
    function_name, function_new

from .caches import clear_caches
from .equations import InnerProductSolver
from .functional import Functional
from .manager import manager as _manager, restore_manager, set_manager

from collections.abc import Sequence
import warnings

__all__ = \
    [
        "GaussNewton",
        "GeneralGaussNewton",
        "GeneralHessian",
        "Hessian",
        "HessianException"
    ]


class HessianException(Exception):
    pass


class Hessian:
    def __init__(self):
        pass

    def compute_gradient(self, M, M0=None):
        raise HessianException("Abstract method not overridden")

    def action(self, M, dM, M0=None):
        raise HessianException("Abstract method not overridden")

    def action_fn(self, m, m0=None):
        """
        Return a callable which accepts a function defining dm, and returns the
        Hessian action as a NumPy array.

        Arguments:

        m   A function defining the control
        m0  (Optional) A function defining the control value
        """

        def action(dm):
            _, _, ddJ = self.action(m, dm, M0=m0)
            return function_get_values(ddJ)

        return action


class GeneralHessian(Hessian):
    def __init__(self, forward, manager=None):
        if manager is None:
            manager = _manager().new()

        super().__init__()
        self._forward = forward
        self._manager = manager

    @restore_manager
    def compute_gradient(self, M, M0=None):
        """
        Evaluate a derivative. Re-evaluates the forward. Returns a tuple
            (J, dJ)
        where
        - J is the functional value
        - dJ is the derivative of J with respect to the parameters defined by M

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
        self._manager.reset()
        self._manager.stop()
        clear_caches()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(function_copy(m0, name=function_name(m),
                                static=function_is_static(m),
                                cache=function_is_cached(m),
                                checkpoint=function_is_checkpointed(m))
                  for m0, m in zip(M0, M))
        del M0

        self._manager.start()
        J = self._forward(*M)
        self._manager.stop()

        J_val = J.value()
        dJ = self._manager.compute_gradient(J, M)

        return J_val, dJ

    @restore_manager
    def action(self, M, dM, M0=None):
        """
        Evaluate a Hessian action. Re-evaluates the forward. Returns a tuple
            (J, dJ, ddJ)
        where
        - J is the functional value
        - dJ is the derivative of J with respect to the parameters defined by M
          in in the direction dM
        - ddJ is the action, in the direction dM, of the second derivative of
          the functional with respect to M

        Arguments:

        M   A function, or a sequence of functions, defining the control
            parameters.
        dM  A function, or a sequence or functions, defining the Hessian action
            direction.
        M0  (Optional) A function, or a sequence of functions, defining the
            values of the control parameters.
        """

        if not isinstance(M, Sequence):
            J_val, dJ_val, (ddJ,) = self.action(
                (M,), (dM,),
                M0=None if M0 is None else (M0,))
            return J_val, dJ_val, ddJ

        set_manager(self._manager)
        self._manager.reset()
        self._manager.stop()
        clear_caches()

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

        self._manager.add_tlm(M, dM)
        self._manager.start()
        J = self._forward(*M)
        dJ = J.tlm(M, dM, manager=self._manager)
        self._manager.stop()

        J_val = J.value()
        dJ_val = dJ.value()
        ddJ = self._manager.compute_gradient(dJ, M)

        return J_val, dJ_val, ddJ


class GaussNewton:
    def __init__(self, adjoint_R_inv_action=None, adjoint_B_inv_action=None,
                 *, R_inv_action=None, B_inv_action=None):
        if adjoint_R_inv_action is None:
            if R_inv_action is None:
                raise HessianException("adjoint_R_inv_action argument "
                                       "required")
            else:
                warnings.warn("'R_inv_action argument' is deprecated -- "
                              "use 'adjoint_R_inv_action' instead",
                              DeprecationWarning, stacklevel=2)
                adjoint_R_inv_action = R_inv_action
        elif R_inv_action is not None:
            raise HessianException("Cannot supply both adjoint_R_inv_action "
                                   "and R_inv_action arguments")
        del R_inv_action

        if adjoint_B_inv_action is None:
            if B_inv_action is not None:
                warnings.warn("'B_inv_action argument' is deprecated -- "
                              "use 'adjoint_B_inv_action' instead",
                              DeprecationWarning, stacklevel=2)
                adjoint_B_inv_action = B_inv_action
        elif B_inv_action is not None:
            raise HessianException("Cannot supply both adjoint_B_inv_action "
                                   "and B_inv_action arguments")
        del B_inv_action

        self._adjoint_R_inv_action = adjoint_R_inv_action
        self._adjoint_B_inv_action = adjoint_B_inv_action

    def _setup_manager(self, M, dM, M0=None):
        raise HessianException("Abstract method not overridden")

    def action(self, M, dM, M0=None):
        if not isinstance(M, Sequence):
            ddJ, = self.action(
                (M,), (dM,),
                M0=None if M0 is None else (M0,))
            return ddJ

        manager, M, dM, X = self._setup_manager(M, dM, M0=M0)

        # J dM
        tau_X = tuple(manager.tlm(M, dM, x) for x in X)
        # R^{-1} J dM
        R_inv_tau_X = self._adjoint_R_inv_action(
            *tuple(function_copy(tau_x) for tau_x in tau_X))
        if not isinstance(R_inv_tau_X, Sequence):
            R_inv_tau_X = (R_inv_tau_X,)

        # This defines the adjoint right-hand-side appropriately to compute a
        # J^* action
        manager.start()
        J = Functional(name="J")
        assert len(X) == len(R_inv_tau_X)
        for x, R_inv_tau_x in zip(X, R_inv_tau_X):
            J_term = function_new(J.fn())
            InnerProductSolver(x, function_copy(R_inv_tau_x), J_term).solve(
                manager=manager, tlm=False)
            J.addto(J_term, manager=manager, tlm=False)
        manager.stop()

        # Likelihood term: J^* R^{-1} J dM
        ddJ = manager.compute_gradient(J, M)

        # Prior term
        if self._adjoint_B_inv_action is not None:
            B_inv_dM = self._adjoint_B_inv_action(
                *tuple(function_copy(dm) for dm in dM))
            if not isinstance(B_inv_dM, Sequence):
                B_inv_dM = (B_inv_dM,)
            assert len(ddJ) == len(B_inv_dM)
            for i, B_inv_dm in enumerate(B_inv_dM):
                function_axpy(ddJ[i], 1.0, B_inv_dm)

        return ddJ

    def action_fn(self, m, m0=None):
        def action(dm):
            ddJ = self.action(m, dm, M0=m0)
            return function_get_values(ddJ)

        return action


class GeneralGaussNewton(GaussNewton):
    def __init__(self, forward,
                 adjoint_R_inv_action, adjoint_B_inv_action=None,
                 *, R_inv_action=None, B_inv_action=None, manager=None):
        if manager is None:
            manager = _manager().new()

        super().__init__(
            adjoint_R_inv_action, adjoint_B_inv_action=adjoint_B_inv_action,
            R_inv_action=R_inv_action, B_inv_action=B_inv_action)
        self._forward = forward
        self._manager = manager

    @restore_manager
    def _setup_manager(self, M, dM, M0=None):
        set_manager(self._manager)
        self._manager.reset()
        self._manager.stop()
        clear_caches()

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

        self._manager.add_tlm(M, dM)
        # Possible optimization: We annotate all the TLM equations, but are
        # later only going to differentiate back through the forward
        self._manager.start()
        X = self._forward(*M)
        if not isinstance(X, Sequence):
            X = (X,)
        self._manager.stop()

        return self._manager, M, dM, X
