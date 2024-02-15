from .backend import Vector
from ..interface import (
    is_function, paused_space_type_checking, paused_var_state_lock_checking,
    var_assign, var_is_scalar, var_new, var_new_conjugate_dual,
    var_scalar_value)

from ..alias import Alias
from ..caches import clear_caches
from ..equations import Assignment, InnerProduct
from ..functional import Functional
from ..manager import (
    configure_tlm, compute_gradient, restore_manager, reset_manager,
    set_manager, start_manager, stop_manager)
from ..manager import manager as _manager

from .functions import Constant

import contextlib
import pyadjoint

__all__ = \
    [
        "Control",
        "ReducedFunctional"
    ]


class Control(Alias):
    @property
    def control(self):
        return self


class ReducedFunctional(pyadjoint.ReducedFunctional):
    def __init__(self, forward, M, *,
                 manager=None, clear_caches=True):
        if is_function(M):
            M = (M,)
        if manager is None:
            manager = _manager().new()

        self.__forward_fn = forward
        self.__M = tuple(map(
            lambda m: Control(m) if not isinstance(m, Control) else m,
            M))
        self.__manager = manager
        self.__J = None
        self.__J0 = None
        self.__adj_x = None
        self.__clear_caches = clear_caches

    @property
    def functional(self):
        if self.__J is None:
            self(self.__M)
        if var_is_scalar(self.__J):
            return var_scalar_value(self.__J)
        else:
            return self.__J

    @property
    def controls(self):
        return self.__M

    @restore_manager
    def __forward(self):
        if self.__clear_caches:
            clear_caches()
        set_manager(self.__manager)
        start_manager()

        J = self.__forward_fn(*self.__M)
        if not is_function(J):
            J = J.function()
        if self.__J is None:
            self.__J = var_new(J)
        Assignment(self.__J, J).solve()

        self.__J0 = Constant()
        # We need 'static=True' here as we are going to set the value of
        # the adjoint variable to initialize the adjoint
        self.__adj_x = var_new_conjugate_dual(self.__J, static=True)
        # This is the same trick as used in GaussNewton.action
        InnerProduct(self.__J0, self.__J, self.__adj_x).solve(tlm=False)

        stop_manager()
        if self.__clear_caches:
            clear_caches()

    @restore_manager
    def __call__(self, values):
        if is_function(values):
            values = (values,)
        if len(values) != len(self.__M):
            raise TypeError("Invalid values length")

        set_manager(self.__manager)
        reset_manager()
        stop_manager()

        for m, value in zip(self.__M, values):
            if m is not value:
                var_assign(m, value)

        self.__forward()
        J = var_new(self.__J)
        Assignment(J, self.__J).solve()
        return J

    @restore_manager
    def derivative(self, adj_input=1.0, options=None):
        if isinstance(adj_input, Vector):
            adj_input = adj_input.function
        if options is None:
            options = {}

        if self.__clear_caches:
            clear_caches()
        set_manager(self.__manager)

        if self.__J is None:
            self(self.__M)

        # This initializes the adjoint
        with paused_space_type_checking(), paused_var_state_lock_checking(self.__adj_x):  # noqa: E501
            var_assign(self.__adj_x, adj_input)

        dJ0 = compute_gradient(self.__J0, self.__M)

        if self.__clear_caches:
            clear_caches()
        return dJ0

    @restore_manager
    def hessian(self, m_dot, options=None):
        if is_function(m_dot):
            m_dot = (m_dot,)
        if len(m_dot) != len(self.__M):
            raise TypeError("Invalid m_dot length")
        if options is None:
            options = {}

        if self.__clear_caches:
            clear_caches()
        set_manager(self.__manager)
        reset_manager()
        stop_manager()

        dM = tuple(map(var_new, self.__M))
        for dm, m_dot_ in zip(dM, m_dot):
            var_assign(dm, m_dot_)
        configure_tlm((self.__M, dM))

        self.__forward()
        if not var_is_scalar(self.__J):
            raise RuntimeError("Unable to compute a Hessian action for a "
                               "non-scalar functional")
        J = Functional(_fn=self.__J)

        tau_J = J.tlm_functional((self.__M, dM), manager=self.__manager)
        ddJ0 = compute_gradient(tau_J, self.__M)

        if self.__clear_caches:
            clear_caches()
        return ddJ0

    def optimize_tape(self):
        pass

    def marked_controls(self):
        @contextlib.contextmanager
        def context_manager():
            yield
        return context_manager
