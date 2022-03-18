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

# 'eigendecompose' loosely follows the slepc4py 3.6.0 demo demo/ex3.py.
# slepc4py 3.6.0 license information follows.
#
# =========================
# LICENSE: SLEPc for Python
# =========================
#
# :Author:  Lisandro Dalcin
# :Contact: dalcinl@gmail.com
#
#
# Copyright (c) 2015, Lisandro Dalcin.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .interface import check_space_types, function_get_values, \
    function_global_size, function_local_size, function_set_values, \
    is_function, space_comm, space_new, space_type_warning

import numpy as np
import warnings

__all__ = \
    [
        "EigendecompositionException",
        "eigendecompose"
    ]


class EigendecompositionException(Exception):
    pass


_flagged_error = [False]


def flag_errors(fn):
    def wrapped_fn(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            _flagged_error[0] = True
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
            raise EigendecompositionException("Invalid dtype")
        if y_a.shape != (y.getLocalSize(),):
            raise EigendecompositionException("Invalid shape")

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
    """
    Matrix-free interface with SLEPc via slepc4py, loosely following
    the slepc4py 3.6.0 demo demo/ex3.py, for use in the calculation of a
    Hessian eigendecomposition with a real control space.

    Arguments:

    space          Eigenvector space.
    A_action       Callable accepting a function and returning a function,
                   defining the action of the left-hand-side matrix, e.g. as
                   returned by Hessian.action_fn.
    B_action       (Optional) Callable accepting a function and returning a
                   function, defining the action of the right-hand-side matrix.
    space_type     (Optional) "primal", "conjugate_primal", "dual", or
                   "conjugate_dual", defining the eigenvector space type.
    action_type    (Optional) "primal", "dual", or "conjugate_dual", whether
                   the action is in the same space as the eigenvectors, or the
                   associated dual or conjugate dual space.
    N_eigenvalues  (Optional) Number of eigenvalues to attempt to find.
                   Defaults to a full spectrum.
    solver_type    (Optional) The solver type.
    problem_type   (Optional) The problem type. If not supplied
                   slepc4py.SLEPc.EPS.ProblemType.NHEP or
                   slepc4py.SLEPc.EPS.ProblemType.GNHEP are used.
    which          (Optional) Which eigenvalues to find. Defaults to
                   slepc4py.SLEPc.EPS.Which.LARGEST_MAGNITUDE.
    tolerance      (Optional) Tolerance, using slepc4py.SLEPc.EPS.Conv.REL
                   convergence criterion.
    configure      (Optional) Function handle accepting the EPS. Can be used
                   for manual configuration.

    Returns:

    A tuple (lam, V), where lam is an array of eigenvalues. For Hermitian
    problems or with complex PETSc V is a tuple of functions containing
    corresponding eigenvectors. Otherwise V is a tuple (V_r, V_i) where V_r and
    V_i are each tuples of functions containing the real and imaginary parts of
    corresponding eigenvectors.
    """

    import petsc4py.PETSc as PETSc
    import slepc4py.SLEPc as SLEPc

    if space_type not in ["primal", "conjugate_primal", "dual", "conjugate_dual"]:  # noqa: E501
        raise EigendecompositionException("Invalid space type")
    if action_type not in ["primal", "dual", "conjugate_dual"]:
        raise EigendecompositionException("Invalid action type")

    A_action = wrapped_action(space, space_type, action_type, A_action)
    if B_action is None:
        if action_type in ["dual", "conjugate_dual"]:
            space_type_warning("'B_action' argument expected with action type "
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

    comm = space_comm(space)  # .Dup()

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
    if configure is not None:
        configure(esolver)
    esolver.setUp()

    assert not _flagged_error[0]
    esolver.solve()
    if _flagged_error[0]:
        raise EigendecompositionException("Error encountered in "
                                          "SLEPc.EPS.solve")
    if esolver.getConverged() < N_ev:
        raise EigendecompositionException("Not all requested eigenpairs "
                                          "converged")

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

    # comm.Free()

    if V_i is None:
        return lam, V_r
    else:
        return lam, (V_r, V_i)
