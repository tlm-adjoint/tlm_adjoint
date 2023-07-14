#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    comm_dup_cached, function_assign, function_axpy, function_comm,
    function_copy, function_dtype, function_get_values, function_inner,
    function_is_cached, function_is_checkpointed, function_is_static,
    function_linf_norm, function_local_size, function_new,
    function_new_conjugate_dual, function_set_values, garbage_cleanup,
    is_function, paused_space_type_checking, space_comm)

from .caches import clear_caches, local_caches
from .functional import Functional
from .manager import manager as _manager
from .manager import compute_gradient, reset_manager, restore_manager, \
    set_manager, start_manager, stop_manager

from collections import deque
from collections.abc import Sequence
import logging
import numpy as np

__all__ = \
    [
        "minimize_scipy",

        "LBFGSHessianApproximation",
        "l_bfgs",
        "minimize_l_bfgs"
    ]


@local_caches
@restore_manager
def minimize_scipy(forward, M0, *,
                   manager=None, **kwargs):
    """Provides an interface with :func:`scipy.optimize.minimize` for
    gradient-based optimization.

    Note that the control variable is gathered onto the root process so that
    the serial :func:`scipy.optimize.minimize` function may be used.

    All keyword arguments except for `manager` are passed to
    :func:`scipy.optimize.minimize`.

    **Important note:** No exception is raised if `return_value.success` is
    `False`. Calling code should check this attribute.

    :arg forward: A callable which accepts one or more function arguments, and
        which returns a function or :class:`tlm_adjoint.functional.Functional`
        defining the forward functional.
    :arg M0: A function or :class:`Sequence` of functions defining the control
        variable, and the initial guess for the optimization.
    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` which
        should be used internally. `manager().new()` is used if not supplied.
    :returns: A :class:`tuple` `(M, return_value)`. `M` is function or a
        :class:`Sequence` of functions depending on the type of `M0`, and
        stores the result. `return_value` is the return value of
        :func:`scipy.optimize.minimize`.
    """

    if not isinstance(M0, Sequence):
        (M,), return_value = minimize_scipy(forward, (M0,),
                                            manager=manager, **kwargs)
        return M, return_value

    if manager is None:
        manager = _manager().new()
    set_manager(manager)
    comm = manager.comm()

    N = [0]
    for m0 in M0:
        N.append(N[-1] + function_local_size(m0))
    if comm.rank == 0:
        size_global = comm.gather(np.array(N[-1], dtype=np.int64), root=0)
        N_global = [0]
        for size in size_global:
            N_global.append(N_global[-1] + size)
    else:
        comm.gather(np.array(N[-1], dtype=np.int64), root=0)

    def get(F):
        x = np.full(N[-1], np.NAN, dtype=np.float64)
        for i, f in enumerate(F):
            f_vals = function_get_values(f)
            if not np.can_cast(f_vals, x.dtype):
                raise ValueError("Invalid dtype")
            x[N[i]:N[i + 1]] = f_vals

        if comm.rank == 0:
            x_global = comm.gather(x, root=0)
            X = np.full(N_global[-1], np.NAN, dtype=np.float64)
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
            function_set_values(f, x[N[i]:N[i + 1]])

    M = [function_new(m0, static=function_is_static(m0),
                      cache=function_is_cached(m0),
                      checkpoint=function_is_checkpointed(m0))
         for m0 in M0]
    J = [None]
    J_M = [None, None]

    def fun(x, *, force=False):
        set(M, x)

        if not force and J[0] is not None:
            change_norm = 0.0
            assert len(M) == len(J_M[0])
            for m, m0 in zip(M, J_M[0]):
                change = function_copy(m)
                function_axpy(change, -1.0, m0)
                change_norm = max(change_norm, function_linf_norm(change))
            if change_norm == 0.0:
                J_val = J[0].value()
                if not isinstance(J_val, (float, np.floating)):
                    raise TypeError("Unexpected type")
                return J_val

        J_M[0] = tuple(function_copy(m) for m in M)

        reset_manager()
        stop_manager()
        clear_caches()

        start_manager()
        J[0] = forward(*M)
        if is_function(J[0]):
            J[0] = Functional(_fn=J[0])
        garbage_cleanup(space_comm(J[0].space()))
        stop_manager()

        J_M[1] = M

        J_val = J[0].value()
        if not isinstance(J_val, (float, np.floating)):
            raise TypeError("Unexpected type")
        return J_val

    def fun_bcast(x):
        if comm.rank == 0:
            comm.bcast(("fun", None), root=0)
        return fun(x)

    def jac(x):
        fun(x, force=J_M[1] is None)
        dJ = compute_gradient(J[0], J_M[1])
        if manager._cp_schedule.is_exhausted():
            J_M[1] = None
        return get(dJ)

    def jac_bcast(x):
        if comm.rank == 0:
            comm.bcast(("jac", None), root=0)
        return jac(x)

    from scipy.optimize import minimize
    if comm.rank == 0:
        x0 = get(M0)
        return_value = minimize(fun_bcast, x0, jac=jac_bcast, **kwargs)
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
            elif action == "return":
                assert data is not None
                return_value = data
                break
            else:
                raise ValueError(f"Unexpected action '{action:s}'")
        set(M, None)

    return M, return_value


def functions_assign(X, Y):
    assert len(X) == len(Y)
    for x, y in zip(X, Y):
        function_assign(x, y)


def functions_axpy(Y, alpha, X, /):
    assert len(Y) == len(X)
    for y, x in zip(Y, X):
        function_axpy(y, alpha, x)


def functions_copy(X):
    return tuple(map(function_copy, X))


def functions_inner(X, Y):
    assert len(X) == len(Y)
    inner = 0.0
    for x, y in zip(X, Y):
        inner += function_inner(x, y)
    return inner


def functions_new(X):
    return tuple(map(function_new, X))


def functions_new_conjugate_dual(X):
    return tuple(map(function_new_conjugate_dual, X))


def conjugate_dual_identity_action(*X):
    M_X = functions_new_conjugate_dual(X)
    with paused_space_type_checking():
        functions_assign(M_X, X)
    return M_X


def wrapped_action(M, *,
                   copy=True):
    M_arg = M

    def M(*X):
        M_X = M_arg(*X)
        if is_function(M_X):
            M_X = (M_X,)
        if len(M_X) != len(X):
            raise ValueError("Incompatible shape")
        return functions_copy(M_X) if copy else M_X

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

        :arg S: A function or a :class:`Sequence` of functions defining the
            step.
        :arg Y: A function or a :class:`Sequence` of functions defining the
            gradient change.
        :arg S_inner_Y: The projection of the gradient change onto the step.
            A separate argument so that a value consistent with
            that used in the line search can be supplied.
        """

        if is_function(S):
            S = (S,)
        if is_function(Y):
            Y = (Y,)
        if len(S) != len(Y):
            raise ValueError("Incompatible shape")
        for s in S:
            if not issubclass(function_dtype(s), (float, np.floating)):
                raise ValueError("Invalid dtype")
        for y in Y:
            if not issubclass(function_dtype(y), (float, np.floating)):
                raise ValueError("Invalid dtype")
        if S_inner_Y <= 0.0:
            raise ValueError("Invalid S_inner_Y")

        rho = 1.0 / S_inner_Y
        self._iterates.append((rho, functions_copy(S), functions_copy(Y)))

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

        :arg X: A function or a :class:`Sequence` of functions defining the
            direction on which to compute the approximate Hessian matrix
            inverse action.
        :arg H_0_action: A callable defining the action of the non-updated
            Hessian matrix inverse approximation on some direction. Accepts one
            or more functions as arguments, defining the direction, and returns
            a function or a :class:`Sequence` of functions defining the action
            on this direction. Should correspond to a positive definite
            operator. An identity is used if not supplied.
        :returns: A function or a :class:`Sequence` of functions storing the
            result.
        """

        if is_function(X):
            X = (X,)
        X = functions_copy(X)

        if H_0_action is None:
            H_0_action = wrapped_action(conjugate_dual_identity_action, copy=False)  # noqa: E501
        else:
            H_0_action = wrapped_action(H_0_action, copy=True)

        alphas = []
        for rho, S, Y in reversed(self._iterates):
            alpha = rho * functions_inner(S, X)
            functions_axpy(X, -alpha, Y)
            alphas.append(alpha)
        alphas.reverse()

        R = H_0_action(*X)
        if theta != 1.0:
            for r in R:
                function_set_values(r, function_get_values(r) / theta)

        assert len(self._iterates) == len(alphas)
        for (rho, S, Y), alpha in zip(self._iterates, alphas):
            beta = rho * functions_inner(R, Y)
            functions_axpy(R, alpha - beta, S)

        return R[0] if len(R) == 1 else R


def line_search_rank0_scipy_line_search(
        F, Fp, *,
        c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):

    def f(x):
        return F(x[0])

    def myfprime(x):
        return np.array([Fp(x[0])])

    if old_Fp_val is not None:
        old_Fp_val = np.array([old_Fp_val])

    from scipy.optimize import line_search
    alpha, fc, gc, new_fval, old_fval, new_slope = line_search(
        f, myfprime, xk=np.array([0.0]), pk=np.array([1.0]),
        gfk=old_Fp_val, old_fval=old_F_val, c1=c1, c2=c2,
        **kwargs)
    if new_slope is None:
        alpha = None
    if alpha is None:
        new_fval = None
    return alpha, new_fval


def line_search_rank0_scipy_scalar_search_wolfe1(
        F, Fp, *,
        c1, c2, old_F_val=None, old_Fp_val=None, **kwargs):
    from scipy.optimize.linesearch import scalar_search_wolfe1 as line_search
    alpha, phi, phi0 = line_search(
        F, Fp,
        phi0=old_F_val, derphi0=old_Fp_val, c1=c1, c2=c2,
        **kwargs)
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

    if is_function(X):
        X_rank1 = (X,)
    else:
        X_rank1 = X
    del X

    if is_function(minus_P):
        minus_P = (minus_P,)
    if len(minus_P) != len(X_rank1):
        raise ValueError("Incompatible shape")

    if comm is None:
        comm = function_comm(X_rank1[0])
    comm = comm_dup_cached(comm)

    last_F = [None, None]

    def F_rank0(x):
        X_rank0 = x
        if not isinstance(X_rank0, (float, np.floating)):
            raise TypeError("Invalid type")
        del x
        X = functions_copy(X_rank1)
        functions_axpy(X, -X_rank0, minus_P)
        last_F[0] = float(X_rank0)
        last_F[1] = F(*X)
        return last_F[1]

    last_Fp = [None, None, None]

    def Fp_rank0(x):
        X_rank0 = x
        if not isinstance(X_rank0, (float, np.floating)):
            raise TypeError("Invalid type")
        del x
        X = functions_copy(X_rank1)
        functions_axpy(X, -X_rank0, minus_P)
        last_Fp[0] = float(X_rank0)
        last_Fp[1] = functions_copy(Fp(*X))
        last_Fp[2] = -functions_inner(minus_P, last_Fp[1])
        return last_Fp[2]

    if old_F_val is None:
        old_F_val = F_rank0(0.0)

    if old_Fp_val is None:
        old_Fp_val_rank0 = Fp_rank0(0.0)
    else:
        if is_function(old_Fp_val):
            old_Fp_val = (old_Fp_val,)
        if len(old_Fp_val) != len(X_rank1):
            raise ValueError("Incompatible shape")
        old_Fp_val_rank0 = -functions_inner(minus_P, old_Fp_val)
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
            if last_F[0] is not None and last_F[0] == alpha:
                new_F_val = last_F[1]
            else:
                new_F_val = F_rank0(alpha)

        if last_Fp[0] is not None and last_Fp[0] == alpha:
            new_Fp_val_rank1 = last_Fp[1]
            new_Fp_val_rank0 = last_Fp[2]
        else:
            new_Fp_val_rank0 = Fp_rank0(alpha)
            assert last_Fp[0] == alpha
            new_Fp_val_rank1 = last_Fp[1]
            assert last_Fp[2] == new_Fp_val_rank0

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

    :arg F: A callable defining the functional. Accepts one or more functions
        as arguments, and returns the value of the functional. Input arguments
        should not be modified.
    :arg Fp: A callable defining the functional gradient. Accepts one or more
        functions as inputs, and returns a function or :class:`Sequence` of
        functions storing the value of the gradient. Input arguments should not
        be modified.
    :arg X0: A function or a :class:`Sequence` of functions defining the
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
            - `X_new`: A function or a :class:`Sequence` of functions defining
              the new value of the parameters.
            - `G_new`: A function or a :class:`Sequence` of functions defining
              the new value of the gradient.
            - `S`: A function or a :class:`Sequence` of functions defining the
              step.
            - `Y`: A function or a sequence of functions defining the gradient
              change.

        Returns a :class:`bool` indicating whether the optimization has
        converged.
    :arg max_its: The maximum number of iterations.
    :arg H_0_action: A callable defining the action of the non-updated Hessian
        matrix inverse approximation on some direction. Accepts one or more
        functions as arguments, defining the direction, and returns a function
        or a :class:`Sequence` of functions defining the action on this
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
        matrix. Accepts one or more functions as arguments, defining the
        direction, and returns a function or a :class:`Sequence` of functions
        defining the action of :math:`M` on this direction. An identity is used
        if not supplied. Required if `H_0_action` or `M_inv_action` are
        supplied.
    :arg M_inv_action: A callable defining a (conjugate) dual space inner
        product,

        .. math::

            \left< x, y \right>_{M^{-1}} = y^* M^{-1} x,

        where :math:`x` and :math:`y` are degree of freedom vectors for
        (conjugate) dual space elements and :math:`M` is as for `M_action`.
        Accepts one or more functions as arguments, defining the direction, and
        returns a function or a :class:`Sequence` of functions defining the
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

    :arg comm: An :class:`mpi4py.MPI.Comm`.
    :returns: A :class:`tuple` `(X, (it, F_calls, Fp_calls, hessian_approx))`
        with

        - `X`: The solution. A function or a :class:`tuple` of functions.
        - `it`: The number of iterations.
        - `F_calls`: The number of functional evaluations.
        - `Fp_calls`: The number of gradient evaluations.
        - `hessian_approx`: The :class:`LBFGSHessianApproximation`.
    """

    logger = logging.getLogger("tlm_adjoint.l_bfgs")

    F_arg = F
    F_calls = [0]

    def F(*X):
        F_calls[0] += 1
        F_val = F_arg(*X)
        if not isinstance(F_val, (float, np.floating)):
            raise TypeError("Invalid type")
        return F_val

    Fp_arg = Fp
    Fp_calls = [0]

    def Fp(*X):
        Fp_calls[0] += 1
        Fp_val = Fp_arg(*X)
        if is_function(Fp_val):
            Fp_val = (Fp_val,)
        if len(Fp_val) != len(X):
            raise ValueError("Incompatible shape")
        return Fp_val

    if is_function(X0):
        X0 = (X0,)

    if converged is None:
        def converged(it, F_old, F_new, X_new, G_new, S, Y):
            return False
    else:
        converged_arg = converged

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
                return abs(functions_inner(X, X))
    else:
        H_0_norm_sq_H_0_action = wrapped_action(H_0_action, copy=True)

        def H_0_norm_sq(X):
            return abs(functions_inner(H_0_norm_sq_H_0_action(*X), X))

    if M_action is None:
        def M_norm_sq(X):
            with paused_space_type_checking():
                return abs(functions_inner(X, X))
    else:
        M_norm_sq_M_action = wrapped_action(M_action, copy=True)

        def M_norm_sq(X):
            return abs(functions_inner(X, M_norm_sq_M_action(*X)))
    del M_action

    if M_inv_action is None:
        M_inv_norm_sq = H_0_norm_sq
    else:
        M_inv_norm_sq_M_inv_action = wrapped_action(M_inv_action, copy=True)

        def M_inv_norm_sq(X):
            return abs(functions_inner(M_inv_norm_sq_M_inv_action(*X), X))
    del M_inv_action

    if comm is None:
        comm = function_comm(X0[0])

    X = functions_copy(X0)
    del X0
    old_F_val = F(*X)
    old_Fp_val = functions_copy(Fp(*X))
    old_Fp_norm_sq = M_inv_norm_sq(old_Fp_val)

    hessian_approx = LBFGSHessianApproximation(m)
    if theta_scale and delta is not None:
        theta = np.sqrt(old_Fp_norm_sq) / delta
    else:
        theta = 1.0

    it = 0
    logger.debug(f"L-BFGS: Iteration {it:d}, "
                 f"F calls {F_calls[0]:d}, "
                 f"Fp calls {Fp_calls[0]:d}, "
                 f"functional value {old_F_val:.6e}")
    while True:
        logger.debug(f"  Gradient norm = {np.sqrt(old_Fp_norm_sq):.6e}")
        if g_atol is not None and old_Fp_norm_sq <= g_atol * g_atol:
            break

        minus_P = hessian_approx.inverse_action(
            old_Fp_val,
            H_0_action=H_0_action, theta=theta)
        if is_function(minus_P):
            minus_P = (minus_P,)
        alpha, old_Fp_val_rank0, new_F_val, new_Fp_val, new_Fp_val_rank0 = line_search(  # noqa: E501
            F, Fp, X, minus_P, c1=c1, c2=c2,
            old_F_val=old_F_val, old_Fp_val=old_Fp_val,
            line_search_rank0=_default_line_search_rank0,
            comm=comm)
        if is_function(new_Fp_val):
            new_Fp_val = (new_Fp_val,)

        if alpha is None or alpha * old_Fp_val_rank0 >= 0.0:
            raise RuntimeError("L-BFGS: Line search failure")
        if new_F_val > old_F_val + c1 * alpha * old_Fp_val_rank0:
            raise RuntimeError("L-BFGS: Armijo condition not satisfied")
        if new_Fp_val_rank0 < c2 * old_Fp_val_rank0:
            raise RuntimeError("L-BFGS: Curvature condition not satisfied")
        if abs(new_Fp_val_rank0) > c2 * abs(old_Fp_val_rank0):
            logger.warning("L-BFGS: Strong curvature condition not satisfied")

        S = functions_new(minus_P)
        functions_axpy(S, -alpha, minus_P)
        functions_axpy(X, 1.0, S)

        Y = functions_copy(new_Fp_val)
        functions_axpy(Y, -1.0, old_Fp_val)

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
                     f"F calls {F_calls[0]:d}, "
                     f"Fp calls {Fp_calls[0]:d}, "
                     f"functional value {new_F_val:.6e}")
        if converged(it, old_F_val, new_F_val, X, new_Fp_val, S, Y):
            break
        if s_atol is not None:
            s_norm_sq = M_norm_sq(S)
            logger.debug(f"  Change norm = {np.sqrt(s_norm_sq):.6e}")
            if s_norm_sq <= s_atol * s_atol:
                break

        if it >= max_its:
            raise RuntimeError("L-BFGS: Maximum number of iterations exceeded")

        old_F_val = new_F_val
        old_Fp_val = new_Fp_val
        del new_F_val, new_Fp_val, new_Fp_val_rank0
        old_Fp_norm_sq = M_inv_norm_sq(old_Fp_val)

    return X[0] if len(X) == 1 else X, (it, F_calls[0], Fp_calls[0], hessian_approx)  # noqa: E501


@local_caches
@restore_manager
def minimize_l_bfgs(forward, M0, *,
                    m=30, manager=None, **kwargs):
    """Functional minimization using the L-BFGS algorithm.

    :arg forward: A callable which accepts one or more function arguments, and
        which returns a function or :class:`tlm_adjoint.functional.Functional`
        defining the forward functional.
    :arg M0: A function or :class:`Sequence` of functions defining the control
        variable, and the initial guess for the optimization.
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
        if not issubclass(function_dtype(m0), (float, np.floating)):
            raise ValueError("Invalid dtype")

    if manager is None:
        manager = _manager().new()
    set_manager(manager)
    comm = manager.comm()

    M = [function_new(m0, static=function_is_static(m0),
                      cache=function_is_cached(m0),
                      checkpoint=function_is_checkpointed(m0))
         for m0 in M0]

    last_F = [None, None, None]

    def F(*X, force=False):
        if not force and last_F[0] is not None:
            change_norm = 0.0
            assert len(X) == len(last_F[0])
            for m, last_m in zip(X, last_F[0]):
                change = function_copy(m)
                function_axpy(change, -1.0, last_m)
                change_norm = max(change_norm, function_linf_norm(change))
            if change_norm == 0.0:
                return last_F[2].value()

        last_F[0] = functions_copy(X)
        functions_assign(M, X)

        reset_manager()
        stop_manager()

        last_F[1] = M
        start_manager()
        last_F[2] = forward(*last_F[1])
        if is_function(last_F[2]):
            last_F[2] = Functional(_fn=last_F[2])
        garbage_cleanup(comm)
        stop_manager()

        return last_F[2].value()

    def Fp(*X):
        F(*X, force=last_F[1] is None)
        dJ = compute_gradient(last_F[2], last_F[1])
        if manager._cp_schedule.is_exhausted():
            last_F[1] = None
        return dJ

    X, optimization_data = l_bfgs(
        F, Fp, M0,
        m=m, comm=comm, **kwargs)

    if is_function(X):
        X = (X,)
    return X, optimization_data
