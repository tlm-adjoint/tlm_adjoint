#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 'eigendecompose' was originally developed by loosely following the slepc4py
# 3.6.0 demo demo/ex3.py. slepc4py 3.6.0 license information can be found in
# the 'eigendecompose' docstring.

from .interface import check_space_types, comm_dup, function_get_values, \
    function_global_size, function_local_size, function_set_values, \
    is_function, space_comm, space_new, space_type_warning

import functools
import numpy as np
import warnings
import weakref

__all__ = \
    [
        "eigendecompose"
    ]


_error_flag = False


def flag_errors(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        global _error_flag
        try:
            return fn(*args, **kwargs)
        except Exception:
            _error_flag = True
            raise
    return wrapped_fn


class PythonMatrix:
    def __init__(self, action):
        self._action = action

    @flag_errors
    def mult(self, A, x, y):
        import petsc4py.PETSc as PETSc

        with x as x_a:
            y_a = self._action(x_a)

        if not np.can_cast(y_a, PETSc.ScalarType):
            raise ValueError("Invalid dtype")
        if y_a.shape != (y.getLocalSize(),):
            raise ValueError("Invalid shape")

        y.setArray(y_a)


def wrapped_action(space, space_type, action_type, action):
    action_arg = action

    def action(x):
        x_a = x
        x = space_new(space, space_type=space_type)
        function_set_values(x, x_a)

        y = action_arg(x)
        if is_function(y):
            check_space_types(x, y, rel_space_type=action_type)
            y_a = function_get_values(y)
        else:
            warnings.warn("Action callable should return a function",
                          DeprecationWarning, stacklevel=2)
            y_a = y

        return y_a

    return action


def eigendecompose(space, A_action, *, B_action=None, space_type="primal",
                   action_type="dual", N_eigenvalues=None, solver_type=None,
                   problem_type=None, which=None, tolerance=1.0e-12,
                   configure=None):
    # First written 2018-03-01
    r"""
    Matrix-free interface with SLEPc via slepc4py, for the matrix free solution
    of eigenproblems

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
    :arg A_action: A :class:`Callable`. Accepts a single function argument, and
        returns a function containing the result after left multiplication of
        the input by :math:`A`.
    :arg B_action: A :class:`Callable`. Accepts a single function argument, and
        returns a function containing the result after left multiplication of
        the input by :math:`B`.
    :arg space_type: The space type of eigenvectors. `'primal'`, `'dual'`,
        `'conjugate'`, or `'conjugate_dual'`.
    :arg action_type: The space type relative to `space_type` of the result of
        multiplication by :math:`A` or :math:`B`. `'primal'`, `'dual'`, or
        `'conjugate_dual'`.
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
    :arg configure: A callable accepting a single :class:`slepc4py.SLEPc.EPS`
        argument. Used for detailed manual configuration. Called after all
        other configuration options are set, but before the :meth:`EPS.setUp`
        method is called.
    :returns: A :class:`tuple` `(lam, V)`. `lam` is a :class:`numpy.ndarray`
        containing eigenvalues. For non-Hermitian algorithms and a real build
        of PETSc, `V` is a :class:`tuple` `(V_r, V_i)`, where `V_r` and `V_i`
        are each a :class:`tuple` of functions containing respectively the real
        and complex parts of corresponding eigenvectors. Otherwise `V` is a
        :class:`tuple` of functions containing corresponding eigenvectors.
    """

    import petsc4py.PETSc as PETSc
    import slepc4py.SLEPc as SLEPc

    if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid space type")
    if action_type not in ["primal", "dual", "conjugate_dual"]:
        raise ValueError("Invalid action type")

    A_action = wrapped_action(space, space_type, action_type, A_action)
    if B_action is None:
        if action_type in ["dual", "conjugate_dual"]:
            space_type_warning("B_action argument expected with action type "
                               "'dual' or 'conjugate_dual'")
        else:
            assert action_type == "primal"
    else:
        B_action = wrapped_action(space, space_type, action_type, B_action)

    if problem_type is None:
        if B_action is None:
            problem_type = SLEPc.EPS.ProblemType.NHEP
        else:
            problem_type = SLEPc.EPS.ProblemType.GNHEP
    if which is None:
        which = SLEPc.EPS.Which.LARGEST_MAGNITUDE

    X = space_new(space, space_type=space_type)
    n, N = function_local_size(X), function_global_size(X)
    del X
    N_ev = N if N_eigenvalues is None else N_eigenvalues

    comm = space_comm(space)
    comm = comm_dup(comm)

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
    weakref.finalize(esolver, lambda comm: None,
                     comm)  # Hold a reference to the communicator
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
    if configure is not None:
        configure(esolver)
    esolver.setUp()

    assert not _error_flag
    esolver.solve()
    if _error_flag:
        raise RuntimeError("Error encountered in SLEPc.EPS.solve")
    if esolver.getConverged() < N_ev:
        raise RuntimeError("Not all requested eigenpairs converged")

    lam = np.full(N_ev, np.NAN,
                  dtype=PETSc.RealType if esolver.isHermitian()
                  else PETSc.ComplexType)
    v_r = A_matrix.getVecRight()
    V_r = tuple(space_new(space, space_type=space_type)
                for n in range(N_ev))
    if issubclass(PETSc.ScalarType, (complex, np.complexfloating)):
        v_i = None
        V_i = None
    else:
        v_i = A_matrix.getVecRight()
        if esolver.isHermitian():
            V_i = None
        else:
            V_i = tuple(space_new(space, space_type=space_type)
                        for n in range(N_ev))
    for i in range(lam.shape[0]):
        lam_i = esolver.getEigenpair(i, v_r, v_i)
        if esolver.isHermitian():
            assert lam_i.imag == 0.0
            lam[i] = lam_i.real
            if v_i is not None:
                with v_i as v_i_a:
                    assert abs(v_i_a).max() == 0.0
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

    if V_i is None:
        return lam, V_r
    else:
        return lam, (V_r, V_i)
