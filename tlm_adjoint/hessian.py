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

from .interface import function_copy, function_get_values, \
    function_is_cached, function_is_checkpointed, function_is_static, \
    function_name

from .caches import clear_caches
from .manager import manager as _manager, restore_manager, set_manager

from collections.abc import Sequence

__all__ = \
    [
        "Hessian",
        "HessianException"
    ]


class HessianException(Exception):
    pass


class Hessian:
    def __init__(self, forward, manager=None):
        """
        Manager for evaluation of Hessian actions.

        Arguments:

        forward  A callable which takes as input the control parameters and
                 returns the Functional whose Hessian action is to be computed.
        manager  (Optional) The equation manager used when computing Hessian
                 actions. If not specified a new manager is created on
                 instantiation using manager().new().
        """

        if manager is None:
            manager = _manager().new()

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
