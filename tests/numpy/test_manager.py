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

from tlm_adjoint_numpy import *
from tlm_adjoint_numpy import manager as _manager

from test_base import *

import numpy as np
import pytest


@pytest.mark.numpy
def test_EmptySolver(setup_test, test_leaks):
    class EmptySolver(Equation):
        def __init__(self):
            Equation.__init__(self, [], [], nl_deps=[], ic_deps=[])

        def forward_solve(self, X, deps=None):
            pass

    space = FunctionSpace(100)
    space_0 = RealFunctionSpace()

    def forward(F):
        EmptySolver().solve()

        F_norm_sq = Function(space_0, name="F_norm_sq")
        NormSqSolver(F, F_norm_sq).solve()

        J = Functional(name="J", space=space_0)
        NormSqSolver(F_norm_sq, J.fn()).solve()
        return J

    F = Function(space, name="F")
    F.vector()[:] = np.arange(len(F.vector()), dtype=np.float64)

    start_manager()
    J = forward(F)
    stop_manager()

    manager = _manager()
    manager.finalize()
    manager.info()
    assert(len(manager._blocks) == 1)
    assert(len(manager._blocks[0]) == 3)
    assert(len(manager._blocks[0][0].X()) == 0)

    J_val = J.value()
    assert(abs(J_val - (F.vector() ** 2).sum() ** 2) == 0.0)

    dJ = compute_gradient(J, F)

    min_order = taylor_test(forward, F, J_val=J_val, dJ=dJ)
    assert(min_order > 2.00)

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, F, J_val=J_val, ddJ=ddJ)
    assert(min_order > 3.00)

    min_order = taylor_test_tlm(forward, F, tlm_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, F, adjoint_order=2)
    assert(min_order > 2.00)


@pytest.mark.numpy
def test_empty(setup_test, test_leaks):
    space = RealFunctionSpace()

    def forward(m):
        return Functional(name="J", space=space)

    m = Function(space, name="m", static=True)

    start_manager()
    J = forward(m)
    stop_manager()

    dJ = compute_gradient(J, m)
    assert(function_max_value(dJ) == 0.0)
