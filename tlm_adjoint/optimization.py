#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    comm_dup_cached, garbage_cleanup, is_var, paused_space_type_checking,
    var_axpy, var_comm, var_copy, var_dtype, var_get_values, var_global_size,
    var_is_cached, var_is_static, var_linf_norm, var_local_size, var_new,
    var_new_conjugate_dual, var_scalar_value, var_set_values, vars_assign,
    vars_axpy, vars_copy, vars_inner, vars_new, vars_new_conjugate_dual)

from .caches import clear_caches, local_caches
from .hessian import GeneralHessian as Hessian
from .manager import manager as _manager
from .manager import (
    compute_gradient, reset_manager, restore_manager, set_manager,
    start_manager, stop_manager)

from collections import deque
from collections.abc import Sequence
import functools
import logging
import numpy as np

__all__ = \
    [
        "minimize_scipy",

        "LBFGSHessianApproximation",
        "l_bfgs",
        "minimize_l_bfgs",
        "minimize_tao"
    ]


class ReducedFunctional:
    def __init__(self, forward, *,
                 manager=None):
        if manager is None:
            manager = _manager().new()

        self._manager = manager
        self._forward = forward
        self._M = None
        self._J = None

    @restore_manager
    def objective(self, M, *,
                  force=False):
        if not isinstance(M, Sequence):
            M = (M,)
        if self._M is not None and len(M) != len(self._M):
            raise ValueError("Invalid control")
        for m in M:
            if not issubclass(var_dtype(m), (float, np.floating)):
                raise ValueError("Invalid dtype")

        set_manager(self._manager)

        if force or self._M is None or self._J is None:
            self._M = None
            self._J = None
        else:
            assert len(M) == len(self._M)
            for m, m_val in zip(M, self._M):
                m_error = var_copy(m)
                var_axpy(m_error, -1.0, m_val)
                if var_linf_norm(m_error) != 0.0:
                    self._M = None
                    self._J = None
                    break

        if self._J is None:
            M = tuple(var_copy(m, static=var_is_static(m),
                               cache=var_is_cached(m))
                      for m in M)

            reset_manager()
            clear_caches()

            start_manager()
            J = self._forward(*M)
            stop_manager()

            self._M = M
            self._J = J

        assert self._M is not None
        assert self._J is not None

        J_val = var_scalar_value(self._J)
        if not isinstance(J_val, (float, np.floating)):
            raise ValueError("Invalid dtype")
        return J_val

    @restore_manager
    def gradient(self, M):
        if not isinstance(M, Sequence):
            dJ, = self.gradient((M,))
            return dJ

        set_manager(self._manager)

        _ = self.objective(M, force=self._manager._cp_schedule.is_exhausted)
        dJ = compute_gradient(self._J, self._M)

        for dJ_i in dJ:
            if not issubclass(var_dtype(dJ_i), (float, np.floating)):
                raise ValueError("Invalid dtype")
        return dJ

    def hessian_action(self, M, dM):
        if not isinstance(M, Sequence):
            ddJ, = self.hessian_action((M,), (dM,))
            return ddJ

        for m in M:
            if not issubclass(var_dtype(m), (float, np.floating)):
                raise ValueError("Invalid dtype")
        for dm in dM:
            if not issubclass(var_dtype(dm), (float, np.floating)):
                raise ValueError("Invalid dtype")

        ddJ = Hessian(self._forward, manager=self._manager.new())
        _, _, ddJ = ddJ.action(M, dM)

        for ddJ_i in ddJ:
            if not issubclass(var_dtype(ddJ_i), (float, np.floating)):
                raise ValueError("Invalid dtype")
        return ddJ


@local_caches
def minimize_scipy(forward, M0, *,
                   manager=None, **kwargs):
    """Provides an interface with :func:`scipy.optimize.minimize` for
    gradient-based optimization.

    Note that the control is gathered onto the root process so that the serial
    :func:`scipy.optimize.minimize` function may be used.

    All keyword arguments except for `manager` are passed to
    :func:`scipy.optimize.minimize`.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional.
    :arg M0: A variable or :class:`Sequence` of variables defining the control,
        and the initial guess for the optimization.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.
    :returns: A :class:`tuple` `(M, return_value)`. `M` is variable or a
        :class:`Sequence` of variables depending on the type of `M0`, and
        stores the result. `return_value` is the return value of
        :func:`scipy.optimize.minimize`.
    """

    if not isinstance(M0, Sequence):
        (M,), return_value = minimize_scipy(forward, (M0,),
                                            manager=manager, **kwargs)
        return M, return_value

    if manager is None:
        manager = _manager().new()
    comm = manager.comm()

    N = [0]
    for m0 in M0:
        N.append(N[-1] + var_local_size(m0))
    if comm.rank == 0:
        size_global = comm.gather(np.array(N[-1], dtype=np.int_), root=0)
        N_global = [0]
        for size in size_global:
            N_global.append(N_global[-1] + size)
    else:
        comm.gather(np.array(N[-1], dtype=np.int_), root=0)

    def get(F):
        x = np.full(N[-1], np.NAN, dtype=np.double)
        for i, f in enumerate(F):
            f_vals = var_get_values(f)
            if not np.can_cast(f_vals, x.dtype):
                raise ValueError("Invalid dtype")
            x[N[i]:N[i + 1]] = f_vals

        if comm.rank == 0:
            x_global = comm.gather(x, root=0)
            X = np.full(N_global[-1], np.NAN, dtype=np.double)
            for i, x_p in enumerate(x_global):
                X[N_global[i]:N_global[i + 1]] = x_p
            return X
        else:
            comm.gather(x, root=0)
            return None

    def set(F, x):
        if comm.rank == 0:
            x = comm.scatter([x[N_global[rank]:N_global[rank + 1]]
                              for rank in range(comm.size)], root=0)
        else:
            assert x is None
            x = comm.scatter(None, root=0)
        for i, f in enumerate(F):
            var_set_values(f, x[N[i]:N[i + 1]])

    M = tuple(var_new(m0, static=var_is_static(m0),
                      cache=var_is_cached(m0))
              for m0 in M0)
    J_hat = ReducedFunctional(forward, manager=manager)

    def fun(x):
        set(M, x)
        return J_hat.objective(M)

    def fun_bcast(x):
        if comm.rank == 0:
            comm.bcast(("fun", None), root=0)
        return fun(x)

    def jac(x):
        set(M, x)
        dJ = J_hat.gradient(M)
        return get(dJ)

    def jac_bcast(x):
        if comm.rank == 0:
            comm.bcast(("jac", None), root=0)
        return jac(x)

    def hessp(x, p):
        set(M, x)
        P = tuple(map(var_new, M))
        set(P, p)
        ddJ = J_hat.hessian_action(M, P)
        return get(ddJ)

    def hessp_bcast(x, p):
        if comm.rank == 0:
            comm.bcast(("hessp", None), root=0)
        return hessp(x, p)

    from scipy.optimize import minimize
    if comm.rank == 0:
        x0 = get(M0)
        return_value = minimize(fun_bcast, x0,
                                jac=jac_bcast, hessp=hessp_bcast, **kwargs)
        comm.bcast(("return", return_value), root=0)
        set(M, return_value.x)
    else:
        get(M0)
        while True:
            action, data = comm.bcast(None, root=0)
            if action == "fun":
                assert data is None
                fun(None)
            elif action == "jac":
                assert data is None
                jac(None)
            elif action == "hessp":
                assert data is None
                hessp(None, None)
            elif action == "return":
                assert data is not None
                return_value = data
                break
            else:
                raise ValueError(f"Unexpected action '{action:s}'")
        set(M, None)

    if not return_value.success:
        raise RuntimeError("Convergence failure")

    return M, return_value


def conjugate_dual_identity_action(*X):
    M_X = vars_new_conjugate_dual(X)
    with paused_space_type_checking():
        vars_assign(M_X, X)
    return M_X


def wrapped_action(M, *, copy=True):
    M_arg = M

    @functools.wraps(M_arg)
    def M(*X):
        M_X = M_arg(*vars_copy(X))
        if is_var(M_X):
            M_X = (M_X,)
        if len(M_X) != len(X):
            raise ValueError("Incompatible shape")
        return vars_copy(M_X) if copy else M_X

    return M


class LBFGSHessianApproximation:
    """L-BFGS Hessian matrix approximation.

    :arg m: Maximum number of vector pairs to retain in the L-BFGS Hessian
        matrix approximation.
    """

    def __init__(self, m):
        self._iterates = deque(maxlen=m)

    def append(self, S, Y, S_inner_Y):
        """Add a step + gradient change pair.

        :arg S: A variable or a :class:`Sequence` of variables defining the
            step.
        :arg Y: A variable or a :class:`Sequence` of variables defining the
            gradient change.
        :arg S_inner_Y: The projection of the gradient change onto the step.
            A separate argument so that a value consistent with
            that used in the line search can be supplied.
        """

        if is_var(S):
            S = (S,)
        if is_var(Y):
            Y = (Y,)
        if len(S) != len(Y):
            raise ValueError("Incompatible shape")
        for s in S:
            if not issubclass(var_dtype(s), (float, np.floating)):
                raise ValueError("Invalid dtype")
        for y in Y:
            if not issubclass(var_dtype(y), (float, np.floating)):
                raise ValueError("Invalid dtype")
        if S_inner_Y <= 0.0:
            raise ValueError("Invalid S_inner_Y")

        rho = 1.0 / S_inner_Y
        self._iterates.append((rho, vars_copy(S), vars_copy(Y)))

    def inverse_action(self, X, *,
                       H_0_action=None, theta=1.0):
        """Compute the action of the approximate Hessian matrix inverse on some
        given direction.

        Implements the L-BFGS Hessian matrix inverse action approximation as in
        Algorithm 7.4 of

            - Jorge Nocedal and Stephen J. Wright, 'Numerical optimization',
              Springer, New York, NY, 2006, Second edition,
              doi: 10.1007/978-0-387-40065-5

        Uses 'theta scaling' as in equation (3.7) of

            - Richard H. Byrd, Peihuang Lu, and Jorge Nocedal, and Ciyou Zhu,
              'A limited memory algorithm for bound constrained optimization',
              SIAM Journal on Scientific Computing, 16(5), 1190--1208, 1995,
              doi: 10.1137/0916069

        :arg X: A variable or a :class:`Sequence` of variables defining the
            direction on which to compute the approximate Hessian matrix
            inverse action.
        :arg H_0_action: A callable defining the action of the non-updated
            Hessian matrix inverse approximation on some direction. Accepts one
            or more variables as arguments, defining the direction, and returns
            a variable or a :class:`Sequence` of variables defining the action
            on this direction. Should correspond to a positive definite
            operator. An identity is used if not supplied.
        :returns: A variable or a :class:`Sequence` of variables storing the
            result.
        """

        if is_var(X):
            X = (X,)
        X = vars_copy(X)

        if H_0_action is None:
            H_0_action = wrapped_action(conjugate_dual_identity_action, copy=False)  # noqa: E501
        else:
            H_0_action = wrapped_action(H_0_action, copy=True)

        alphas = []
        for rho, S, Y in reversed(self._iterates):
            alpha = rho * vars_inner(S, X)
            vars_axpy(X, -alpha, Y)
            alphas.append(alpha)
        alphas.reverse()

        R = H_0_action(*X)
        if theta != 1.0:
            for r in R:
                var_set_values(r, var_get_values(r) / theta)

        assert len(self._iterates) == len(alphas)
        for (rho, S, Y), alpha in zip(self._iterates, alphas):
            beta = rho * vars_inner(R, Y)
            vars_axpy(R, alpha - beta, S)

        return R[0] if len(R) == 1 else R


def line_search_rank0_scipy_line_search(
        F, Fp, *,
        c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):

    def f(x):
        x, = x
        return F(x)

    def myfprime(x):
        x, = x
        return np.array([Fp(x)], dtype=np.double)

    if old_Fp_val is not None:
        old_Fp_val = np.array([old_Fp_val], dtype=np.double)

    from scipy.optimize import line_search
    alpha, _, _, new_fval, _, new_slope = line_search(
        f, myfprime,
        xk=np.array([0.0], dtype=np.double), pk=np.array([1.0], dtype=np.double),  # noqa: E501
        gfk=old_Fp_val, old_fval=old_F_val,
        c1=c1, c2=c2, **kwargs)
    if new_slope is None:
        alpha = None
    if alpha is None:
        new_fval = None
    return alpha, new_fval


def line_search_rank0_scipy_scalar_search_wolfe1(
        F, Fp, *,
        c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):
    from scipy.optimize.linesearch import scalar_search_wolfe1 as line_search
    alpha, phi, _ = line_search(
        F, Fp,
        phi0=old_F_val, derphi0=old_Fp_val,
        c1=c1, c2=c2, **kwargs)
    if alpha is None:
        phi = None
    return alpha, phi


_default_line_search_rank0 = line_search_rank0_scipy_scalar_search_wolfe1


def line_search(F, Fp, X, minus_P, *,
                c1=1.0e-4, c2=0.9,
                old_F_val=None, old_Fp_val=None,
                line_search_rank0=_default_line_search_rank0,
                line_search_rank0_kwargs=None,
                comm=None):
    if line_search_rank0_kwargs is None:
        line_search_rank0_kwargs = {}

    Fp = wrapped_action(Fp, copy=False)

    if is_var(X):
        X_rank1 = (X,)
    else:
        X_rank1 = X
    del X

    if is_var(minus_P):
        minus_P = (minus_P,)
    if len(minus_P) != len(X_rank1):
        raise ValueError("Incompatible shape")

    if comm is None:
        comm = var_comm(X_rank1[0])
    comm = comm_dup_cached(comm)

    F_last_X_rank0 = None
    F_last_F = None

    def F_rank0(x):
        nonlocal F_last_X_rank0, F_last_F

        X_rank0 = F_last_X_rank0 = float(x)
        del x

        X = vars_copy(X_rank1)
        vars_axpy(X, -X_rank0, minus_P)

        F_last_F = F(*X)
        return F_last_F

    Fp_last_X_rank0 = None
    Fp_last_Fp_rank1 = None
    Fp_last_Fp_rank0 = None

    def Fp_rank0(x):
        nonlocal Fp_last_X_rank0, Fp_last_Fp_rank1, Fp_last_Fp_rank0

        X_rank0 = Fp_last_X_rank0 = float(x)
        del x

        X = vars_copy(X_rank1)
        vars_axpy(X, -X_rank0, minus_P)

        Fp_last_Fp_rank1 = vars_copy(Fp(*X))
        Fp_last_Fp_rank0 = -vars_inner(minus_P, Fp_last_Fp_rank1)
        return Fp_last_Fp_rank0

    if old_F_val is None:
        old_F_val = F_rank0(0.0)

    if old_Fp_val is None:
        old_Fp_val_rank0 = Fp_rank0(0.0)
    else:
        if is_var(old_Fp_val):
            old_Fp_val = (old_Fp_val,)
        if len(old_Fp_val) != len(X_rank1):
            raise ValueError("Incompatible shape")
        old_Fp_val_rank0 = -vars_inner(minus_P, old_Fp_val)
    del old_Fp_val

    if comm.rank == 0:
        def F_rank0_bcast(x):
            comm.bcast(("F_rank0", (x,)), root=0)
            return F_rank0(x)

        def Fp_rank0_bcast(x):
            comm.bcast(("Fp_rank0", (x,)), root=0)
            return Fp_rank0(x)

        alpha, new_F_val = line_search_rank0(
            F_rank0_bcast, Fp_rank0_bcast, c1=c1, c2=c2,
            old_F_val=old_F_val, old_Fp_val=old_Fp_val_rank0,
            **line_search_rank0_kwargs)
        comm.bcast(("return", (alpha, new_F_val)), root=0)
    else:
        while True:
            action, data = comm.bcast(None, root=0)
            if action == "F_rank0":
                X_rank0, = data
                F_rank0(X_rank0)
            elif action == "Fp_rank0":
                X_rank0, = data
                Fp_rank0(X_rank0)
            elif action == "return":
                alpha, new_F_val = data
                break
            else:
                raise ValueError(f"Unexpected action '{action:s}'")

    if alpha is None:
        return None, old_Fp_val_rank0, None, None, None
    else:
        if new_F_val is None:
            if F_last_X_rank0 is None or F_last_X_rank0 != alpha:
                F_rank0(alpha)
            new_F_val = F_last_F

        if Fp_last_X_rank0 is None or Fp_last_X_rank0 != alpha:
            Fp_rank0(alpha)
        new_Fp_val_rank1 = Fp_last_Fp_rank1
        new_Fp_val_rank0 = Fp_last_Fp_rank0

        return (alpha, old_Fp_val_rank0, new_F_val,
                new_Fp_val_rank1[0] if len(new_Fp_val_rank1) == 1 else new_Fp_val_rank1,  # noqa: E501
                new_Fp_val_rank0)


def l_bfgs(F, Fp, X0, *,
           m=30, s_atol, g_atol, converged=None, max_its=1000,
           H_0_action=None, theta_scale=True, delta=1.0,
           M_action=None, M_inv_action=None,
           c1=1.0e-4, c2=0.9,
           comm=None):
    r"""Functional minimization using the L-BFGS algorithm.

    Implements Algorithm 7.5 of

        - Jorge Nocedal and Stephen J. Wright, 'Numerical optimization',
          Springer, New York, NY, 2006, Second edition,
          doi: 10.1007/978-0-387-40065-5

    in a more general inner product space.

    By default uses 'theta scaling' to define the initial Hessian matrix
    inverse approximation, based on the approach in equation (3.7) and point 7
    on p. 1204 of

        - Richard H. Byrd, Peihuang Lu, and Jorge Nocedal, and Ciyou Zhu, 'A
          limited memory algorithm for bound constrained optimization', SIAM
          Journal on Scientific Computing, 16(5), 1190--1208, 1995,
          doi: 10.1137/0916069

    and with an initial value for the scaling parameter based on the discussion
    in 'Implementation' in section 6.1 of

        - Jorge Nocedal and Stephen J. Wright, 'Numerical optimization',
          Springer, New York, NY, 2006, Second edition,
          doi: 10.1007/978-0-387-40065-5

    Precisely the Hessian matrix inverse approximation, before being updated,
    is scaled by :math:`1 / \theta` with, on the first iteration,

    .. math::

        \theta = \frac{\sqrt{\left| g_k^* M^{-1} g_k \right|}}{\delta}

    and on later iterations

    .. math::

        \theta = \frac{y_k^* H_0 y_k}{y_k^* s_k},

    where :math:`g_k`, :math:`y_k`, and :math:`s_k` are respectively the
    gradient, gradient change, and step, and where :math:`M^{-1}` and
    :math:`H_0` are defined by `M_inv_action` and `H_0_action` respectively.

    :arg F: A callable defining the functional. Accepts one or more variables
        as arguments, and returns the value of the functional. Input arguments
        should not be modified.
    :arg Fp: A callable defining the functional gradient. Accepts one or more
        variables as inputs, and returns a variable or :class:`Sequence` of
        variables storing the value of the gradient. Input arguments should not
        be modified.
    :arg X0: A variable or a :class:`Sequence` of variables defining the
        initial guess for the parameters.
    :arg m: The maximum number of step + gradient change pairs to use in the
        Hessian matrix inverse approximation.
    :arg s_atol: Absolute tolerance for the step change norm convergence
        criterion.
    :arg g_atol: Absolute tolerance for the gradient norm convergence
        criterion.
    :arg converged: A callable defining a callback, and which can be used to
        define custom convergence criteria. Takes the form

        .. code-block:: python

            def converged(it, F_old, F_new, X_new, G_new, S, Y):

        with

            - `it`: The iteration number.
            - `F_old`: The previous value of the functional.
            - `F_new`: The new value of the functional.
            - `X_new`: A variable or a :class:`Sequence` of variables defining
              the new value of the parameters.
            - `G_new`: A variable or a :class:`Sequence` of variables defining
              the new value of the gradient.
            - `S`: A variable or a :class:`Sequence` of variables defining the
              step.
            - `Y`: A variable or a sequence of variables defining the gradient
              change.

        Input variables should not be modified. Returns a :class:`bool`
        indicating whether the optimization has converged.
    :arg max_its: The maximum number of iterations.
    :arg H_0_action: A callable defining the action of the non-updated Hessian
        matrix inverse approximation on some direction. Accepts one or more
        variables as arguments, defining the direction, and returns a variable
        or a :class:`Sequence` of variables defining the action on this
        direction. Should correspond to a positive definite operator. An
        identity is used if not supplied.
    :arg theta_scale: Whether to apply 'theta scaling', discussed above.
    :arg delta: Controls the initial value of :math:`\theta` in 'theta
        scaling'. If `None` then on the first iteration :math:`\theta` is set
        equal to one.
    :arg M_action: A callable defining a primal space inner product,

        .. math::

            \left< x, y \right>_M = y^* M x,

        where :math:`x` and :math:`y` are degree of freedom vectors for primal
        space elements and :math:`M` is a Hermitian and positive definite
        matrix. Accepts one or more variables as arguments, defining the
        direction, and returns a variable or a :class:`Sequence` of variables
        defining the action of :math:`M` on this direction. An identity is used
        if not supplied. Required if `H_0_action` or `M_inv_action` are
        supplied.
    :arg M_inv_action: A callable defining a (conjugate) dual space inner
        product,

        .. math::

            \left< x, y \right>_{M^{-1}} = y^* M^{-1} x,

        where :math:`x` and :math:`y` are degree of freedom vectors for
        (conjugate) dual space elements and :math:`M` is as for `M_action`.
        Accepts one or more variables as arguments, defining the direction, and
        returns a variable or a :class:`Sequence` of variables defining the
        action of :math:`M^{-1}` on this direction. `H_0_action` is used if not
        supplied.
    :arg c1: Armijo condition parameter. :math:`c_1` in equation (3.6a) of

            - Jorge Nocedal and Stephen J. Wright, 'Numerical optimization',
              Springer, New York, NY, 2006, Second edition,
              doi: 10.1007/978-0-387-40065-5

    :arg c2: Curvature condition parameter. :math:`c_2` in equation (3.6b) of

            - Jorge Nocedal and Stephen J. Wright, 'Numerical optimization',
              Springer, New York, NY, 2006, Second edition,
              doi: 10.1007/978-0-387-40065-5

    :arg comm: A communicator.
    :returns: A :class:`tuple` `(X, (it, F_calls, Fp_calls, hessian_approx))`
        with

        - `X`: The solution. A variable or a :class:`tuple` of variables.
        - `it`: The number of iterations.
        - `F_calls`: The number of functional evaluations.
        - `Fp_calls`: The number of gradient evaluations.
        - `hessian_approx`: The :class:`LBFGSHessianApproximation`.
    """

    logger = logging.getLogger("tlm_adjoint.l_bfgs")

    F_arg = F
    F_calls = 0

    def F(*X):
        nonlocal F_calls

        F_calls += 1
        F_val = F_arg(*X)
        if not isinstance(F_val, (float, np.floating)):
            raise TypeError("Invalid type")
        return F_val

    Fp_arg = Fp
    Fp_calls = 0

    def Fp(*X):
        nonlocal Fp_calls

        Fp_calls += 1
        Fp_val = Fp_arg(*X)
        if is_var(Fp_val):
            Fp_val = (Fp_val,)
        if len(Fp_val) != len(X):
            raise ValueError("Incompatible shape")
        return Fp_val

    if is_var(X0):
        X0 = (X0,)

    if converged is None:
        def converged(it, F_old, F_new, X_new, G_new, S, Y):
            return False
    else:
        converged_arg = converged

        @functools.wraps(converged_arg)
        def converged(it, F_old, F_new, X_new, G_new, S, Y):
            return converged_arg(it, F_old, F_new,
                                 X_new[0] if len(X_new) == 1 else X_new,
                                 G_new[0] if len(G_new) == 1 else G_new,
                                 S[0] if len(S) == 1 else S,
                                 Y[0] if len(Y) == 1 else Y)

    if (H_0_action is None and M_inv_action is None) and M_action is not None:
        raise TypeError("If M_action is supplied, then H_0_action or "
                        "M_inv_action must be supplied")
    if (H_0_action is not None or M_inv_action is not None) and M_action is None:  # noqa: E501
        raise TypeError("If H_0_action or M_inv_action are supplied, then "
                        "M_action must be supplied")

    if H_0_action is None:
        def H_0_norm_sq(X):
            with paused_space_type_checking():
                return abs(vars_inner(X, X))
    else:
        H_0_norm_sq_H_0_action = wrapped_action(H_0_action, copy=True)

        def H_0_norm_sq(X):
            return abs(vars_inner(H_0_norm_sq_H_0_action(*X), X))

    if M_action is None:
        def M_norm_sq(X):
            with paused_space_type_checking():
                return abs(vars_inner(X, X))
    else:
        M_norm_sq_M_action = wrapped_action(M_action, copy=True)

        def M_norm_sq(X):
            return abs(vars_inner(X, M_norm_sq_M_action(*X)))
    del M_action

    if M_inv_action is None:
        M_inv_norm_sq = H_0_norm_sq
    else:
        M_inv_norm_sq_M_inv_action = wrapped_action(M_inv_action, copy=True)

        def M_inv_norm_sq(X):
            return abs(vars_inner(M_inv_norm_sq_M_inv_action(*X), X))
    del M_inv_action

    if comm is None:
        comm = var_comm(X0[0])

    X = vars_copy(X0)
    del X0
    old_F_val = F(*X)
    old_Fp_val = vars_copy(Fp(*X))
    old_Fp_norm_sq = M_inv_norm_sq(old_Fp_val)

    hessian_approx = LBFGSHessianApproximation(m)
    if theta_scale and delta is not None:
        theta = np.sqrt(old_Fp_norm_sq) / delta
    else:
        theta = 1.0

    it = 0
    logger.debug(f"L-BFGS: Iteration {it:d}, "
                 f"F calls {F_calls:d}, "
                 f"Fp calls {Fp_calls:d}, "
                 f"functional value {old_F_val:.6e}")
    while True:
        logger.debug(f"  Gradient norm = {np.sqrt(old_Fp_norm_sq):.6e}")
        if g_atol is not None and old_Fp_norm_sq <= g_atol * g_atol:
            break

        if it >= max_its:
            raise RuntimeError("L-BFGS: Maximum number of iterations exceeded")

        minus_P = hessian_approx.inverse_action(
            old_Fp_val,
            H_0_action=H_0_action, theta=theta)
        if is_var(minus_P):
            minus_P = (minus_P,)
        alpha, old_Fp_val_rank0, new_F_val, new_Fp_val, new_Fp_val_rank0 = line_search(  # noqa: E501
            F, Fp, X, minus_P, c1=c1, c2=c2,
            old_F_val=old_F_val, old_Fp_val=old_Fp_val,
            line_search_rank0=_default_line_search_rank0,
            comm=comm)
        if is_var(new_Fp_val):
            new_Fp_val = (new_Fp_val,)

        if alpha is None or alpha * old_Fp_val_rank0 >= 0.0:
            raise RuntimeError("L-BFGS: Line search failure")
        if new_F_val > old_F_val + c1 * alpha * old_Fp_val_rank0:
            raise RuntimeError("L-BFGS: Armijo condition not satisfied")
        if new_Fp_val_rank0 < c2 * old_Fp_val_rank0:
            raise RuntimeError("L-BFGS: Curvature condition not satisfied")
        if abs(new_Fp_val_rank0) > c2 * abs(old_Fp_val_rank0):
            logger.warning("L-BFGS: Strong curvature condition not satisfied")

        S = vars_new(minus_P)
        vars_axpy(S, -alpha, minus_P)
        vars_axpy(X, 1.0, S)

        Y = vars_copy(new_Fp_val)
        vars_axpy(Y, -1.0, old_Fp_val)

        # >=0 by curvature condition
        S_inner_Y = alpha * (new_Fp_val_rank0 - c2 * old_Fp_val_rank0)
        # >0 by definition of c2, and as steps are downhill
        S_inner_Y += alpha * (c2 - 1.0) * old_Fp_val_rank0

        hessian_approx.append(S, Y, S_inner_Y)
        if theta_scale:
            theta = H_0_norm_sq(Y) / S_inner_Y
        else:
            theta = 1.0

        garbage_cleanup(comm)
        it += 1
        logger.debug(f"L-BFGS: Iteration {it:d}, "
                     f"F calls {F_calls:d}, "
                     f"Fp calls {Fp_calls:d}, "
                     f"functional value {new_F_val:.6e}")
        if converged(it, old_F_val, new_F_val, X, new_Fp_val, S, Y):
            break
        if s_atol is not None:
            s_norm_sq = M_norm_sq(S)
            logger.debug(f"  Change norm = {np.sqrt(s_norm_sq):.6e}")
            if s_norm_sq <= s_atol * s_atol:
                break

        old_F_val = new_F_val
        old_Fp_val = new_Fp_val
        del new_F_val, new_Fp_val, new_Fp_val_rank0
        old_Fp_norm_sq = M_inv_norm_sq(old_Fp_val)

    return X[0] if len(X) == 1 else X, (it, F_calls, Fp_calls, hessian_approx)


@local_caches
def minimize_l_bfgs(forward, M0, *,
                    m=30, manager=None, **kwargs):
    """Functional minimization using the L-BFGS algorithm.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional.
    :arg M0: A variable or :class:`Sequence` of variables defining the control,
        and the initial guess for the optimization.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.

    Remaining arguments and the return value are described in the
    :func:`l_bfgs` documentation.
    """

    if not isinstance(M0, Sequence):
        (x,), optimization_data = minimize_l_bfgs(
            forward, (M0,),
            m=m, manager=manager, **kwargs)
        return x, optimization_data

    for m0 in M0:
        if not issubclass(var_dtype(m0), (float, np.floating)):
            raise ValueError("Invalid dtype")

    if manager is None:
        manager = _manager().new()
    comm = manager.comm()

    J_hat = ReducedFunctional(forward, manager=manager)

    X, optimization_data = l_bfgs(
        lambda *M: J_hat.objective(M), lambda *M: J_hat.gradient(M), M0,
        m=m, comm=comm, **kwargs)

    if is_var(X):
        X = (X,)
    return X, optimization_data


@local_caches
def minimize_tao(forward, M0, *,
                 method=None, gatol, grtol, gttol=0.0,
                 M_inv_action=None,
                 pre_callback=None, post_callback=None, manager=None):
    r"""Functional minimization using TAO.

    :arg forward: A callable which accepts one or more variable arguments, and
        which returns a variable defining the forward functional.
    :arg M0: A variable or :class:`Sequence` of variables defining the control,
        and the initial guess for the optimization.
    :arg method: TAO type. Defaults to `PETSc.TAO.Type.LMVM`.
    :arg gatol: TAO gradient absolute tolerance.
    :arg grtol: TAO gradient relative tolerance.
    :arg gttol: TAO gradient norm change relative tolerance.
    :arg M_inv_action: A callable defining a (conjugate) dual space inner
        product,

        .. math::

            \left< x, y \right>_{M^{-1}} = y^* M^{-1} x,

        where :math:`x` and :math:`y` are degree of freedom vectors for
        (conjugate) dual space elements and :math:`M` is a Hermitian and
        positive definite matrix. Accepts one or more variables as arguments,
        defining the direction, and returns a variable or a :class:`Sequence`
        of variables defining the action of :math:`M^{-1}` on this direction.
    :arg pre_callback: A callable accepting a single
        :class:`petsc4py.PETSc.TAO` argument. Used for detailed manual
        configuration. Called after all other configuration options are set.
    :arg post_callback: A callable accepting a single
        :class:`petsc4py.PETSc.TAO` argument. Called after the
        :meth:`petsc4py.PETSc.TAO.solve` method has been called.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.
    :returns: A variable or a :class:`Sequence` of variables storing the
        result.
    """

    if not isinstance(M0, Sequence):
        m, = minimize_tao(
            forward, (M0,),
            method=method, gatol=gatol, grtol=grtol, gttol=gttol,
            M_inv_action=M_inv_action,
            pre_callback=pre_callback, post_callback=post_callback,
            manager=manager)
        return m

    import petsc4py.PETSc as PETSc

    if method is None:
        method = PETSc.TAO.Type.LMVM
    for m0 in M0:
        if not issubclass(var_dtype(m0), (float, np.floating)):
            raise ValueError("Invalid dtype")
    if M_inv_action is not None:
        M_inv_action = wrapped_action(M_inv_action, copy=False)

    def from_petsc(y, X):
        with y as y_a:
            if not issubclass(y_a.dtype.type, (float, np.floating)):
                raise ValueError("Invalid dtype")
            if len(y_a.shape) != 1:
                raise ValueError("Invalid shape")

            i0 = 0
            if len(X) != len(indices):
                raise ValueError("Invalid length")
            for j, x in enumerate(X):
                i1 = i0 + var_local_size(x)
                if i1 > y_a.shape[0]:
                    raise ValueError("Invalid shape")
                if (i0, i1) != indices[j]:
                    raise ValueError("Invalid shape")
                var_set_values(x, y_a[i0:i1])
                i0 = i1
            if i0 != y_a.shape[0]:
                raise ValueError("Invalid shape")

    def to_petsc(x, Y):
        x_a = np.zeros(n, dtype=PETSc.ScalarType)

        i0 = 0
        if len(Y) != len(indices):
            raise ValueError("Invalid length")
        for j, y in enumerate(Y):
            y_a = var_get_values(y)

            if not issubclass(y_a.dtype.type, (float, np.floating)):
                raise ValueError("Invalid dtype")
            if not np.can_cast(y_a, x_a.dtype):
                raise ValueError("Invalid dtype")
            if len(y_a.shape) != 1:
                raise ValueError("Invalid shape")

            i1 = i0 + y_a.shape[0]
            if i1 > x_a.shape[0]:
                raise ValueError("Invalid shape")
                if (i0, i1) != indices[j]:
                    raise ValueError("Invalid shape")
            x_a[i0:i1] = y_a
            i0 = i1
        if i0 != x_a.shape[0]:
            raise ValueError("Invalid shape")

        x.setArray(x_a)

    if manager is None:
        manager = _manager().new()
    comm = manager.comm()
    comm = comm_dup_cached(comm, key="minimize_tao")

    tao = PETSc.TAO().create(comm=comm)
    tao.setType(method)
    tao.setTolerances(gatol=gatol, grtol=grtol, gttol=gttol)

    M = tuple(var_new(m0, static=var_is_static(m0),
                      cache=var_is_cached(m0))
              for m0 in M0)
    J_hat = ReducedFunctional(forward, manager=manager)

    def objective(tao, x):
        from_petsc(x, M)
        J_val = J_hat.objective(M)
        return J_val

    def gradient(tao, x, g):
        from_petsc(x, M)
        dJ = J_hat.gradient(M)
        to_petsc(g, dJ)

    def objective_gradient(tao, x, g):
        from_petsc(x, M)
        J_val = J_hat.objective(M)
        dJ = J_hat.gradient(M)
        to_petsc(g, dJ)
        return J_val

    def hessian(tao, x, H, P):
        H.getPythonContext().set_M(x)

    class Hessian:
        def __init__(self):
            self._shift = 0.0

        @functools.cached_property
        def _M(self):
            return tuple(var_new(m0, static=var_is_static(m0),
                                 cache=var_is_cached(m0))
                         for m0 in M0)

        def set_M(self, x):
            from_petsc(x, self._M)

        def shift(self, A, alpha):
            self._shift += alpha

        def mult(self, A, x, y):
            dM = tuple(map(var_new, self._M))
            from_petsc(x, dM)
            ddJ = J_hat.hessian_action(self._M, dM)
            to_petsc(y, ddJ)
            if self._shift != 0.0:
                y.axpy(self._shift, x)

    indices = []
    n = 0
    N = 0
    for m0 in M0:
        indices.append((n, n + var_local_size(m0)))
        n += var_local_size(m0)
        N += var_global_size(m0)
    H_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                        Hessian(), comm=comm)
    H_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    H_matrix.setUp()

    tao.setObjective(objective)
    tao.setGradient(gradient, None)
    tao.setObjectiveGradient(objective_gradient, None)
    tao.setHessian(hessian, H_matrix)

    if M_inv_action is not None:
        class GradientNorm:
            @functools.cached_property
            def _dJ(self):
                return tuple(map(var_new_conjugate_dual, M0))

            def mult(self, A, x, y):
                from_petsc(x, self._dJ)
                M_inv_X = M_inv_action(*self._dJ)
                to_petsc(y, M_inv_X)

        M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                GradientNorm(), comm=comm)
        M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        M_inv_matrix.setUp()
        tao.setGradientNorm(M_inv_matrix)

    x = H_matrix.getVecRight()
    to_petsc(x, M0)

    if pre_callback is not None:
        pre_callback(tao)
    tao.solve(x)
    if post_callback is not None:
        post_callback(tao)

    if tao.getConvergedReason() <= 0:
        raise RuntimeError("Convergence failure")

    from_petsc(x, M)

    tao.destroy()
    H_matrix.destroy()
    if M_inv_action is not None:
        M_inv_matrix.destroy()
    x.destroy()

    return M
