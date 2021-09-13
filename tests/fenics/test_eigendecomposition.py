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

from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics.backend_code_generator_interface import \
    function_vector, matrix_multiply

from test_base import *

import numpy as np
import pytest


@pytest.mark.fenics
@seed_test
def test_HEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    M = assemble(inner(test, trial) * dx)

    def M_action(x):
        y = function_new(x)
        matrix_multiply(M, function_vector(x), tensor=function_vector(y))
        return function_get_values(y)

    import slepc4py.SLEPc as SLEPc
    lam, V_r = eigendecompose(space, M_action,
                              problem_type=SLEPc.EPS.ProblemType.HEP)
    diff = Function(space)
    assert len(lam) == len(V_r)
    for lam_val, v_r in zip(lam, V_r):
        function_set_values(diff, M_action(v_r))
        function_axpy(diff, -lam_val, v_r)
        assert function_linf_norm(diff) < 1.0e-16


@pytest.mark.fenics
@seed_test
def test_NHEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    N = assemble(inner(test, trial.dx(0)) * dx)

    def N_action(x):
        y = function_new(x)
        matrix_multiply(N, function_vector(x), tensor=function_vector(y))
        return function_get_values(y)

    lam, (V_r, V_i) = eigendecompose(space, N_action)
    diff = Function(space)
    assert len(lam) == len(V_r)
    assert len(lam) == len(V_i)
    for lam_val, v_r, v_i in zip(lam, V_r, V_i):
        function_set_values(diff, N_action(v_r))
        function_axpy(diff, -lam_val.real, v_r)
        function_axpy(diff, +lam_val.imag, v_i)
        assert function_linf_norm(diff) < 1.0e-15
        function_set_values(diff, N_action(v_i))
        function_axpy(diff, -lam_val.real, v_i)
        function_axpy(diff, -lam_val.imag, v_r)
        assert function_linf_norm(diff) < 1.0e-15


@pytest.mark.fenics
@seed_test
def test_CachedHessian(setup_test):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitIntervalMesh(5)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(F):
        y = Function(space, name="y")
        EquationSolver(inner(test, trial) * dx == inner(test, F) * dx,
                       y, solver_parameters=ls_parameters_cg).solve()

        J = Functional(name="J")
        J.addto(inner(dot(y, y), dot(y, y)) * dx)
        return J

    F = Function(space, name="F", static=True)
    function_assign(F, 1.0)

    start_manager()
    J = forward(F)
    stop_manager()

    H = Hessian(forward)
    H_opt = CachedHessian(J)

    # Test consistency of matrix action for static direction

    zeta = Function(space, name="zeta", static=True)
    for i in range(5):
        function_set_values(zeta, np.random.random(function_local_size(zeta)))
        _, _, ddJ_opt = H_opt.action(F, zeta)
        _, _, ddJ = H.action(F, zeta)

        error = Function(space, name="error")
        function_assign(error, ddJ)
        function_axpy(error, -1.0, ddJ_opt)
        assert function_linf_norm(error) < 1.0e-14

    # Test consistency of eigenvalues

    lam, _ = eigendecompose(space, H.action_fn(F))
    assert max(abs(lam.imag)) == 0.0

    lam_opt, _ = eigendecompose(space, H_opt.action_fn(F))
    assert max(abs(lam_opt.imag)) == 0.0

    error = (np.array(sorted(lam.real), dtype=np.float64)
             - np.array(sorted(lam_opt.real), dtype=np.float64))
    assert abs(error).max() < 1.0e-14
