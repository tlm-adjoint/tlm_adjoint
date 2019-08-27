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
from tlm_adjoint_fenics import *

from test_base import leak_check

import pytest


@pytest.mark.fenics
@leak_check
def test_HEP():
    reset_manager("memory", {"replace": True})
    clear_caches()
    stop_manager()

    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    M = assemble(inner(test, trial) * dx)

    def M_action(F):
        return (M * F.vector()).get_local()

    import slepc4py.SLEPc as SLEPc
    lam, V_r = eigendecompose(space, M_action,
                              problem_type=SLEPc.EPS.ProblemType.HEP)
    diff = Function(space)
    for lam_val, v_r in zip(lam, V_r):
        function_set_values(diff, M_action(v_r))
        function_axpy(diff, -lam_val, v_r)
        assert(function_linf_norm(diff) < 1.0e-16)


@pytest.mark.fenics
@leak_check
def test_NHEP():
    reset_manager("memory", {"replace": True})
    clear_caches()
    stop_manager()

    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    N = assemble(inner(test, trial.dx(0)) * dx)

    def N_action(F):
        return (N * F.vector()).get_local()

    lam, (V_r, V_i) = eigendecompose(space, N_action)
    diff = Function(space)
    for lam_val, v_r, v_i in zip(lam, V_r, V_i):
        function_set_values(diff, N_action(v_r))
        function_axpy(diff, -float(lam_val.real), v_r)
        function_axpy(diff, +float(lam_val.imag), v_i)
        assert(function_linf_norm(diff) < 1.0e-8)
        function_set_values(diff, N_action(v_i))
        function_axpy(diff, -float(lam_val.real), v_i)
        function_axpy(diff, -float(lam_val.imag), v_r)
        assert(function_linf_norm(diff) < 1.0e-8)
