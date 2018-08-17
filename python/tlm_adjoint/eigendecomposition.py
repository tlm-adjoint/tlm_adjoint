#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 The University of Edinburgh
#
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

from .base_interface import *

import numpy

__all__ = \
  [
    "eigendecompose"
  ]

# First written 2018-03-01
def eigendecompose(space, A_action, B_matrix = None, N_eigenvalues = None,
  solver_type = None, problem_type = None, which = None, tolerance = 1.0e-12,
  configure = None):
  """
  Matrix-free eigendecomposition using SLEPc via slepc4py, loosely following the
  slepc4py 3.6.0 demo demo/ex3.py.
  
  Arguments:
  
  space          Function space.
  A_action       Function handle accepting a Function and returning an array,
                 defining the action of the left-hand-side matrix.
  B_matrix       (Optional) Right-hand-side matrix in a generalised
                 eigendecomposition.
  N_eigenvalues  (Optional) Number of eigenvalues to attempt to find. Defaults
                 to a full eigendecomposition.
  solver_type    (Optional) The solver type.
  problem_type   (Optional) The problem type. If not supplied
                 slepc4py.SLEPc.EPS.ProblemType.NHEP or
                 slepc4py.SLEPc.EPS.ProblemType.GNHEP are used.
  which          (Optional) Which eigenvalues to find. Defaults to
                 slepc4py.SLEPc.EPS.Which.LARGEST_MAGNITUDE.
  tolerance      (Optional) Tolerance, using slepc4py.SLEPc.EPS.Conv.REL
                 convergence criterion.
  configure      (Optional) Function handle accepting the EPS. Can be used for
                 manual configuration.
  
  Returns:
  
  A tuple (lam, V_r) for Hermitian problems, or (lam, (V_r, V_i)) otherwise,
  where lam is an array of eigenvalues, and V_r / V_i are lists of Function
  objects containing the real and imaginary parts of the corresponding
  eigenvectors.
  """

  import petsc4py.PETSc
  import slepc4py.SLEPc
  if which is None:
    which = slepc4py.SLEPc.EPS.Which.LARGEST_MAGNITUDE
  
  X = Function(space)
  class PythonMatrix:
    def __init__(self, action):
      self._action = action
  
    def mult(self, A, x, y):
      function_set_values(X, x.getArray(readonly = True))
      y.setArray(self._action(X))
  
  n, N = function_local_size(X), function_global_size(X)
  A_matrix = petsc4py.PETSc.Mat().createPython(((n, N), (n, N)), PythonMatrix(A_action), comm = function_comm(X))
  A_matrix.setUp()
  
  esolver = slepc4py.SLEPc.EPS().create(comm = function_comm(X))
  if not solver_type is None:
    esolver.setType(solver_type)
  if B_matrix is None:
    esolver.setProblemType(slepc4py.SLEPc.EPS.ProblemType.NHEP if problem_type is None else problem_type)
    esolver.setOperators(A_matrix)
  else:
    esolver.setProblemType(slepc4py.SLEPc.EPS.ProblemType.GNHEP if problem_type is None else problem_type)
    esolver.setOperators(A_matrix, B_matrix)
  esolver.setWhichEigenpairs(which)
  esolver.setDimensions(nev = N if N_eigenvalues is None else N_eigenvalues,
                        ncv = slepc4py.SLEPc.DECIDE, mpd = slepc4py.SLEPc.DECIDE)  
  esolver.setConvergenceTest(slepc4py.SLEPc.EPS.Conv.REL)
  esolver.setTolerances(tol = tolerance, max_it = slepc4py.SLEPc.DECIDE)
  if not configure is None:
    configure(esolver)
  esolver.setUp()
  
  esolver.solve()
  
  lam = numpy.empty(esolver.getConverged(), dtype = numpy.float64 if esolver.isHermitian() else numpy.complex64)
  V_r = [function_new(X) for n in range(N)]
  if not esolver.isHermitian():
    V_i = [function_new(X) for n in range(N)]
  v_r, v_i = A_matrix.getVecRight(), A_matrix.getVecRight()
  for i in range(lam.shape[0]):
    lam_i = esolver.getEigenpair(i, v_r, v_i)
    if esolver.isHermitian():
      lam[i] = lam_i.real
      function_set_values(V_r[i], v_r.getArray())
    else:
      lam[i] = lam_i
      function_set_values(V_r[i], v_r.getArray())
      function_set_values(V_i[i], v_i.getArray())
      
  return lam, (V_r if esolver.isHermitian() else (V_r, V_i))
