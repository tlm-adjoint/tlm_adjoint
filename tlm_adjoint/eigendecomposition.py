#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 'eigendecompose' was originally developed by loosely following the slepc4py
# 3.6.0 demo demo/ex3.py. slepc4py 3.6.0 license information can be found in
# the 'eigendecompose' docstring.

from .interface import (
    check_space_type, comm_dup_cached, function_get_values,
    function_global_size, function_local_size, function_set_values,
    relative_space_type, space_comm, space_new)

import functools
import numpy as np

__all__ = \
    [
        "eigendecompose"
    ]


class PythonMatrix:
    def __init__(self, action):
        self._action = action

    def mult(self, A, x, y):
        import petsc4py.PETSc as PETSc

        with x as x_a:
            y_a = self._action(x_a)

        if not np.can_cast(y_a, PETSc.ScalarType):
            raise ValueError("Invalid dtype")
        if y_a.shape != (y.getLocalSize(),):
            raise ValueError("Invalid shape")

        y.setArray(y_a)


def wrapped_action(space, arg_space_type, action_space_type, action):
    action_arg = action

    @functools.wraps(action_arg)
    def action(x):
        x_a = x
        x = space_new(space, space_type=arg_space_type)
        function_set_values(x, x_a)

        y = action_arg(x)
        check_space_type(y, action_space_type)
        return function_get_values(y)

    return action


def eigendecompose(space, A_action, *, B_action=None, arg_space_type="primal",
                   action_space_type=None, N_eigenvalues=None,
                   solver_type=None, problem_type=None, which=None,
                   tolerance=1.0e-12, pre_callback=None, post_callback=None):
    # First written 2018-03-01
    r"""Interface with SLEPc via slepc4py, for the matrix free solution of
    eigenproblems

    .. math::

        A v = \lambda v,

    or generalized eigenproblems

    .. math::

        A v = \lambda B v.

    Originally developed by loosely following the slepc4py 3.6.0 demo
    demo/ex3.py. slepc4py 3.6.0 license information follows:

    .. code-block:: text

        =========================
        LICENSE: SLEPc for Python
        =========================

        :Author:  Lisandro Dalcin
        :Contact: dalcinl@gmail.com


        Copyright (c) 2015, Lisandro Dalcin.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions
        are met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    :arg space: The space for each eigenvector.
    :arg A_action: A callable. Accepts a single function argument, and returns
        a function containing the result after left multiplication of the input
        by :math:`A`.
    :arg B_action: A callable. Accepts a single function argument, and returns
        a function containing the result after left multiplication of the input
        by :math:`B`.
    :arg arg_space_type: The space type of eigenvectors. `'primal'`, `'dual'`,
        `'conjugate'`, or `'conjugate_dual'`.
    :arg action_space_type: The space type of the result of multiplication by
        :math:`A` or :math:`B`. `'primal'`, `'dual'`, `'conjugate'`, or
        `'conjugate_dual'`. Defaults to the space type conjugate dual to
        `arg_space_type`.
    :arg N_eigenvalues: An :class:`int`, the number of eigenvalues to attempt
        to compute. Defaults to the dimension of `space`.
    :arg problem_type: The eigenproblem type -- see
        :class:`slepc4py.SLEPc.EPS.ProblemType`. Defaults to
        `slepc4py.SLEPc.EPS.ProblemType.GNHEP` if `B_action` is supplied, or
        `slepc4py.SLEPc.EPS.ProblemType.NHEP` otherwise.
    :arg which: Which eigenvalues to attempt to compute -- see
        :class:`slepc4py.SLEPc.EPS.Which`. Defaults to
        `slepc4py.SLEPc.EPS.Which.LARGEST_MAGNITUDE`.
    :arg tolerance: Convergence tolerance. By default the convergence criterion
        is defined using `slepc4py.SLEPc.EPS.Conv.REL`.
    :arg pre_callback: A callable accepting a single
        :class:`slepc4py.SLEPc.EPS` argument. Used for detailed manual
        configuration. Called after all other configuration options are set,
        but before the :meth:`EPS.setUp` method is called.
    :arg post_callback: A callable accepting a single
        :class:`slepc4py.SLEPc.EPS` argument. Called after the
        :meth:`EPS.solve` method has been called.
    :returns: A :class:`tuple` `(lam, V)`. `lam` is a :class:`numpy.ndarray`
        containing eigenvalues. For non-Hermitian algorithms and a real build
        of PETSc, `V` is a :class:`tuple` `(V_r, V_i)`, where `V_r` and `V_i`
        are each a :class:`tuple` of functions containing respectively the real
        and complex parts of corresponding eigenvectors. Otherwise `V` is a
        :class:`tuple` of functions containing corresponding eigenvectors.
    """

    import petsc4py.PETSc as PETSc
    import slepc4py.SLEPc as SLEPc

    if arg_space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
        raise ValueError("Invalid space type")
    if action_space_type is None:
        action_space_type = relative_space_type(arg_space_type, "conjugate_dual")  # noqa: E501
    elif action_space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:  # noqa: E501
        raise ValueError("Invalid space type")

    A_action = wrapped_action(space, arg_space_type, action_space_type, A_action)  # noqa: E501
    if B_action is not None:
        B_action = wrapped_action(space, arg_space_type, action_space_type, B_action)  # noqa: E501

    if problem_type is None:
        if B_action is None:
            problem_type = SLEPc.EPS.ProblemType.NHEP
        else:
            problem_type = SLEPc.EPS.ProblemType.GNHEP
    if which is None:
        which = SLEPc.EPS.Which.LARGEST_MAGNITUDE

    X = space_new(space, space_type=arg_space_type)
    n, N = function_local_size(X), function_global_size(X)
    del X
    N_ev = N if N_eigenvalues is None else N_eigenvalues

    comm = space_comm(space)
    comm = comm_dup_cached(comm, key="eigendecompose")

    A_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                        PythonMatrix(A_action),
                                        comm=comm)
    A_matrix.setUp()

    if B_action is None:
        B_matrix = None
    else:
        B_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                            PythonMatrix(B_action),
                                            comm=comm)
        B_matrix.setUp()

    esolver = SLEPc.EPS().create(comm=comm)
    if solver_type is not None:
        esolver.setType(solver_type)
    esolver.setProblemType(problem_type)
    if B_matrix is None:
        esolver.setOperators(A_matrix)
    else:
        esolver.setOperators(A_matrix, B_matrix)
    esolver.setWhichEigenpairs(which)
    esolver.setDimensions(nev=N_ev,
                          ncv=SLEPc.DECIDE, mpd=SLEPc.DECIDE)
    esolver.setConvergenceTest(SLEPc.EPS.Conv.REL)
    esolver.setTolerances(tol=tolerance, max_it=SLEPc.DECIDE)

    if pre_callback is not None:
        pre_callback(esolver)
    esolver.setUp()
    esolver.solve()
    if post_callback is not None:
        post_callback(esolver)

    if esolver.getConverged() < N_ev:
        raise RuntimeError("Not all requested eigenpairs converged")
    if esolver.getConvergedReason() <= 0:
        raise RuntimeError("Convergence failure")

    lam = np.full(N_ev, np.NAN,
                  dtype=PETSc.RealType if esolver.isHermitian()
                  else PETSc.ComplexType)
    v_r = A_matrix.getVecRight()
    V_r = tuple(space_new(space, space_type=arg_space_type)
                for _ in range(N_ev))
    if issubclass(PETSc.ScalarType, (complex, np.complexfloating)):
        v_i = None
        V_i = None
    else:
        v_i = A_matrix.getVecRight()
        if esolver.isHermitian():
            V_i = None
        else:
            V_i = tuple(space_new(space, space_type=arg_space_type)
                        for _ in range(N_ev))
    for i in range(lam.shape[0]):
        lam_i = esolver.getEigenpair(i, v_r, v_i)
        if esolver.isHermitian():
            assert lam_i.imag == 0.0
            lam[i] = lam_i.real
            if v_i is not None:
                with v_i as v_i_a:
                    assert len(v_i_a) == 0 or abs(v_i_a).max() == 0.0
            # else:
            #     # Complex note: If v_i is None then v_r may be non-real
            #     pass
            with v_r as v_r_a:
                function_set_values(V_r[i], v_r_a)
        else:
            lam[i] = lam_i
            with v_r as v_r_a:
                function_set_values(V_r[i], v_r_a)
            if v_i is not None:
                with v_i as v_i_a:
                    function_set_values(V_i[i], v_i_a)

    esolver.destroy()
    A_matrix.destroy()
    if B_matrix is not None:
        B_matrix.destroy()
    v_r.destroy()
    if v_i is not None:
        v_i.destroy()

    if V_i is None:
        return lam, V_r
    else:
        return lam, (V_r, V_i)
