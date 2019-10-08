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

    def compute_gradient(self, M):
        """
        Evaluate a derivative. Re-evaluates the forward. Returns a tuple
            (J, dJ)
        where
        - J is the functional value
        - dJ is the derivative of J with respect to the parameters defined by M

        Arguments:

        M   A Control or function, or a list or tuple of these, defining the
            derivative.
        """

        if not isinstance(M, (list, tuple)):
            J, (dJ,) = self.compute_gradient((M,))
            return J, dJ

        old_manager = _manager()
        set_manager(self._manager)
        self._manager.reset()
        self._manager.stop()
        clear_caches()

        self._manager.start()
        J = self._forward(*M)
        self._manager.stop()

        J_val = J.value()
        dJ = self._manager.compute_gradient(J, M)

        set_manager(old_manager)
        return J_val, dJ

    def action(self, M, dM):
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

        M   A Control or function, or a list or tuple of these, defining the
            Hessian.
        dM  A function, or list or tuple or functions, defining the Hessian
            action direction.
        """

        if not isinstance(M, (list, tuple)):
            J_val, dJ_val, (ddJ,) = self.action((M,), (dM,))
            return J_val, dJ_val, ddJ

        old_manager = _manager()
        set_manager(self._manager)
        self._manager.reset()
        self._manager.stop()
        clear_caches()

        self._manager.add_tlm(M, dM)
        self._manager.start()
        J = self._forward(*M)
        dJ = J.tlm(M, dM, manager=self._manager)
        self._manager.stop()

        J_val = J.value()
        dJ_val = dJ.value()
        ddJ = self._manager.compute_gradient(dJ, M)

        set_manager(old_manager)
        return J_val, dJ_val, ddJ

    def action_fn(self, m):
        """
        Return a callable which accepts a function defining dm, and returns the
        Hessian action as a NumPy array.

        Arguments:

        m   A Control or function
        """

        def action(dm):
            _, _, ddJ = self.action(m, dm)
            return function_get_values(ddJ)

        return action
