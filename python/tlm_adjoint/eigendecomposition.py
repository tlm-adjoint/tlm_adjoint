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

from .backend_interface import *

import numpy as np
import petsc4py.PETSc as PETSc

__all__ = \
    [
        "EigendecompositionException",
        "eigendecompose"
    ]


class EigendecompositionException(Exception):
    pass


def eigendecompose(space, A_action, B_matrix=None, N_eigenvalues=None,
                   solver_type=None, problem_type=None, which=None,
                   tolerance=1.0e-12, configure=None):
    # First written 2018-03-01
    """
    Matrix-free eigendecomposition using SLEPc via slepc4py, loosely following
    the slepc4py 3.6.0 demo demo/ex3.py.

    Arguments:

    space          Eigenspace.
    A_action       Function handle accepting a function and returning an array,
                   defining the action of the left-hand-side matrix.
    B_matrix       (Optional) Right-hand-side matrix in a generalized
                   eigendecomposition.
    N_eigenvalues  (Optional) Number of eigenvalues to attempt to find.
                   Defaults to a full eigendecomposition.
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

    A tuple (lam, V_r) for Hermitian problems, or (lam, (V_r, V_i)) otherwise,
    where lam is an array of eigenvalues, and V_r / V_i are tuples of functions
    containing the real and imaginary parts of the corresponding eigenvectors.
    """

    import slepc4py.SLEPc as SLEPc

    if problem_type is None:
        if B_matrix is None:
            problem_type = SLEPc.EPS.ProblemType.NHEP
        else:
            problem_type = SLEPc.EPS.ProblemType.GNHEP
    if which is None:
        which = SLEPc.EPS.Which.LARGEST_MAGNITUDE

    eps_error = [False]

    def flag_errors(fn):
        def wrapped_fn(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except:  # noqa: E722
                eps_error[0] = True
                raise
        return wrapped_fn

    class PythonMatrix:
        def __init__(self, action, X):
            self._action = action
            self._X = X

        @flag_errors
        def mult(self, A, x, y):
            function_set_values(self._X, x.getArray(readonly=True))
            y.setArray(self._action(self._X))

    X = space_new(space)
    n, N = function_local_size(X), function_global_size(X)
    N_ev = N if N_eigenvalues is None else N_eigenvalues

    comm = function_comm(X)  # .Dup()

    A_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                        PythonMatrix(A_action, X),
                                        comm=comm)
    A_matrix.setUp()

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

    esolver.solve()
    if eps_error[0]:
        raise EigendecompositionException("Error encountered in "
                                          "SLEPc.EPS.solve")
    if esolver.getConverged() < N_ev:
        raise EigendecompositionException("Not all requested eigenpairs "
                                          "converged")

    lam = np.empty(N_ev,
                   dtype=np.float64 if esolver.isHermitian() else np.complex64)
    V_r = tuple(function_new(X) for n in range(N_ev))
    if not esolver.isHermitian():
        V_i = tuple(function_new(X) for n in range(N_ev))
    v_r, v_i = A_matrix.getVecRight(), A_matrix.getVecRight()
    for i in range(lam.shape[0]):
        lam_i = esolver.getEigenpair(i, v_r, v_i)
        if esolver.isHermitian():
            lam[i] = lam_i.real
            assert lam_i.imag == 0.0
            function_set_values(V_r[i], v_r.getArray())
            assert abs(v_i.getArray()).max() == 0.0
        else:
            lam[i] = lam_i
            function_set_values(V_r[i], v_r.getArray())
            function_set_values(V_i[i], v_i.getArray())

    # comm.Free()

    return lam, (V_r if esolver.isHermitian() else (V_r, V_i))
