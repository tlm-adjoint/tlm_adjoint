from .interface import (
    Packed, check_space_types_conjugate_dual, packed, var_axpy, var_copy,
    var_copy_conjugate, var_is_cached, var_id, var_is_static, var_locked,
    var_name, var_scalar_value)

from .caches import local_caches
from .functional import Functional
from .markers import AdjointActionMarker
from .manager import manager as _manager
from .manager import (
    compute_gradient, configure_tlm, var_tlm, reset_manager, restore_manager,
    set_manager, start_manager, stop_manager)

from abc import ABC, abstractmethod
import warnings

__all__ = \
    [
        "GaussNewton",
        "GeneralGaussNewton",
        "GeneralHessian",
        "Hessian"
    ]


class Hessian(ABC):
    r"""Represents a Hessian associated with a given forward. Abstract base
    class.
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_gradient(self, M, M0=None):
        r"""Compute the (conjugate of the) derivative of a functional with
        respect to a control using an adjoint.

        :arg M: A variable or a :class:`Sequence` of variables defining the
            control.
        :arg M0: A variable or a :class:`Sequence` of variables defining the
            control value. `M` is used if not supplied.
        :returns: The (conjugate of the) derivative. A variable or
            :class:`Sequence` of variables, depending on the type of `M`.
        """

        raise NotImplementedError

    @abstractmethod
    def action(self, M, dM, M0=None):
        r"""Compute (the conjugate of) a Hessian action on some :math:`\zeta`
        using an adjoint of a tangent-linear. i.e. considering derivatives to
        be row vectors, compute

        .. math::

            \left( \frac{d}{dm} \left[
                \frac{d \mathcal{J}}{d m} \zeta \right] \right)^{*,T}.

        :arg M: A variable or a :class:`Sequence` of variables defining the
            control.
        :arg dM: A variable or a :class:`Sequence` of variables defining
            :math:`\zeta`. The (conjugate of the) Hessian action on
            :math:`\zeta` is computed.
        :arg M0: A variable or a :class:`Sequence` of variables defining the
            control value. `M` is used if not supplied.
        :returns: A tuple `(J, dJ, ddJ)`. `J` is the value of the functional.
            `dJ` is the value of :math:`\left( d \mathcal{J} / d m \right)
            \zeta`. `ddJ` stores the (conjugate of the) result of the Hessian
            action on :math:`\zeta`, and is a variable or a :class:`Sequence`
            of variables depending on the type of `M`.
        """

        raise NotImplementedError

    def action_fn(self, m, m0=None):
        warnings.warn("Hessian.action_fn is deprecated",
                      DeprecationWarning, stacklevel=2)

        def action(dm):
            _, _, ddJ = self.action(m, dm, M0=m0)
            return var_copy_conjugate(ddJ)

        return action


class GeneralHessian(Hessian):
    """Represents a Hessian associated with a given forward. Calls to
    :meth:`.GeneralHessian.compute_gradient` or :meth:`.GeneralHessian.action`
    re-run the forward.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional.
    :arg manager: An :class:`.EquationManager` used to create an internal
        manager via :meth:`.EquationManager.new`. `manager()` is used if not
        supplied.
    """

    def __init__(self, forward, *, manager=None):
        if manager is None:
            manager = _manager()
        manager = manager.new()

        forward_arg = forward

        def forward(*M):
            with var_locked(*M):
                return forward_arg(*M)

        super().__init__()
        self._forward = forward
        self._manager = manager

    @local_caches
    @restore_manager
    def compute_gradient(self, M, M0=None):
        M_packed = Packed(M)
        M = tuple(M_packed)
        if M0 is not None:
            M0 = packed(M0)

        set_manager(self._manager)
        reset_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(var_copy(m0, name=var_name(m),
                           static=var_is_static(m),
                           cache=var_is_cached(m))
                  for m0, m in zip(M0, M))
        del M0

        start_manager()
        J = self._forward(*M)
        stop_manager()

        J_val = var_scalar_value(J)
        dJ = compute_gradient(J, M)

        reset_manager()
        return J_val, M_packed.unpack(dJ)

    @local_caches
    @restore_manager
    def action(self, M, dM, M0=None):
        M_packed = Packed(M)
        M = tuple(M_packed)
        dM = packed(dM)
        if len(set(map(var_id, M)).intersection(map(var_id, dM))) > 0:
            raise ValueError("Direction and controls must be distinct")
        if M0 is not None:
            M0 = packed(M0)

        set_manager(self._manager)
        reset_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(var_copy(m0, name=var_name(m),
                           static=var_is_static(m),
                           cache=var_is_cached(m))
                  for m0, m in zip(M0, M))
        del M0

        dM = tuple(var_copy(dm, name=var_name(dm),
                            static=var_is_static(dm),
                            cache=var_is_cached(dm))
                   for dm in dM)

        configure_tlm((M, dM))
        start_manager()
        J = self._forward(*M)
        dJ = var_tlm(J, (M, dM))
        stop_manager()

        J_val = var_scalar_value(J)
        dJ_val = var_scalar_value(dJ)
        ddJ = compute_gradient(dJ, M)

        reset_manager()
        return J_val, dJ_val, M_packed.unpack(ddJ)


class GaussNewton(ABC):
    r"""Represents a Gauss-Newton approximation for a Hessian. Abstract base
    class.

    In terms of matrices this defines a Hessian approximation

    .. math::

        H = J^T R_\text{obs}^{-1} J + B^{-1},

    where :math:`J` is the forward Jacobian. In a variational assimilation
    approach :math:`R_\text{obs}^{-1}` corresponds to the observational inverse
    covariance and :math:`B^{-1}` corresponds to the background inverse
    covariance.

    :arg R_inv_action: A callable which accepts one or more variables, and
        returns the conjugate of the action of the operator corresponding to
        :math:`R_\text{obs}^{-1}` on those variables, returning the result as a
        variable or a :class:`Sequence` of variables.
    :arg B_inv_action: A callable which accepts one or more variables, and
        returns the conjugate of the action of the operator corresponding to
        :math:`B^{-1}` on those variables, returning the result as a variable
        or a :class:`Sequence` of variables.
    """

    def __init__(self, R_inv_action, B_inv_action=None):
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

        :arg M: A variable or a :class:`Sequence` of variables defining the
            control.
        :arg dM: A variable or a :class:`Sequence` of variables defining
            :math:`\zeta`. The (conjugate of the) approximated Hessian action
            on :math:`\zeta` is computed.
        :arg M0: A variable or a :class:`Sequence` of variables defining the
            control value. `M` is used if not supplied.
        :returns: The (conjugate of the) result of the approximated Hessian
            action on :math:`\zeta`. A variable or a :class:`Sequence` of
            variables depending on the type of `M`.
        """

        M_packed = Packed(M)
        M = tuple(M_packed)
        dM = packed(dM)
        if len(set(map(var_id, M)).intersection(map(var_id, dM))) > 0:
            raise ValueError("Direction and controls must be distinct")
        if M0 is not None:
            M0 = packed(M0)

        manager, M, dM, X = self._setup_manager(M, dM, M0=M0)
        set_manager(manager)

        # J dM
        tau_X = tuple(var_tlm(x, (M, dM)) for x in X)
        # conj[ R^{-1} J dM ]
        with var_locked(*tau_X):
            R_inv_tau_X = self._R_inv_action(*tau_X)
        R_inv_tau_X = packed(R_inv_tau_X)
        assert len(tau_X) == len(R_inv_tau_X)
        for tau_x, R_inv_tau_x in zip(tau_X, R_inv_tau_X):
            check_space_types_conjugate_dual(tau_x, R_inv_tau_x)

        # This defines the adjoint right-hand-side appropriately to compute a
        # J^T action
        start_manager()
        J = Functional()
        AdjointActionMarker(J, X, tuple(map(var_copy, R_inv_tau_X))).solve()
        stop_manager()

        # Likelihood term: conj[ J^T R^{-1} J dM ]
        ddJ = compute_gradient(J, M)

        # Prior term: conj[ B^{-1} dM ]
        if self._B_inv_action is not None:
            with var_locked(*dM):
                B_inv_dM = self._B_inv_action(*dM)
            B_inv_dM = packed(B_inv_dM)
            assert len(dM) == len(B_inv_dM)
            for dm, B_inv_dm in zip(dM, B_inv_dM):
                check_space_types_conjugate_dual(dm, B_inv_dm)
            assert len(ddJ) == len(B_inv_dM)
            for i, B_inv_dm in enumerate(B_inv_dM):
                var_axpy(ddJ[i], 1.0, B_inv_dm)

        reset_manager()
        return M_packed.unpack(ddJ)

    def action_fn(self, m, m0=None):
        warnings.warn("GaussNewton.action_fn is deprecated",
                      DeprecationWarning, stacklevel=2)

        def action(dm):
            return var_copy_conjugate(self.action(m, dm, M0=m0))

        return action


class GeneralGaussNewton(GaussNewton):
    """Represents a Gauss-Newton approximation to a Hessian associated with a
    given forward. Calls to :meth:`.GaussNewton.action` re-run the forward.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable or :class:`Sequence` of variables defining the
        state.
    :arg R_inv_action: See :class:`.GaussNewton`.
    :arg B_inv_action: See :class:`.GaussNewton`.
    :arg manager: An :class:`.EquationManager` used to create an internal
        manager via :meth:`.EquationManager.new`. `manager()` is used if not
        supplied.
    """

    def __init__(self, forward, R_inv_action, B_inv_action=None, *,
                 manager=None):
        if manager is None:
            manager = _manager()
        manager = manager.new()

        forward_arg = forward

        def forward(*M):
            with var_locked(*M):
                return forward_arg(*M)

        super().__init__(R_inv_action, B_inv_action=B_inv_action)
        self._forward = forward
        self._manager = manager

    @local_caches
    @restore_manager
    def _setup_manager(self, M, dM, M0=None):
        set_manager(self._manager)
        reset_manager()

        if M0 is None:
            M0 = M
        assert len(M0) == len(M)
        M = tuple(var_copy(m0, name=var_name(m),
                           static=var_is_static(m),
                           cache=var_is_cached(m))
                  for m0, m in zip(M0, M))
        del M0

        dM = tuple(var_copy(dm, name=var_name(dm),
                            static=var_is_static(dm),
                            cache=var_is_cached(dm))
                   for dm in dM)

        configure_tlm((M, dM), annotate=False)
        start_manager()
        X = self._forward(*M)
        X = packed(X)
        stop_manager()

        return self._manager, M, dM, X
