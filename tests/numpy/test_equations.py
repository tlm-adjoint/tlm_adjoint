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

from test_base import *

import numpy as np
import pytest


@pytest.mark.numpy
def test_SumSolver(setup_test, test_leaks):
    space = FunctionSpace(10)

    def forward(F):
        G = Function(space, name="G")
        AssignmentSolver(F, G).solve()

        J = Functional(name="J")
        SumSolver(G, J.fn()).solve()
        return J

    F = Function(space, name="F", static=True)
    function_set_values(F, np.random.random(function_local_size(F)))

    start_manager()
    J = forward(F)
    stop_manager()

    assert(J.value() == function_sum(F))

    dJ = compute_gradient(J, F)
    assert(abs(function_get_values(dJ) - 1.0).max() == 0.0)


@pytest.mark.numpy
def test_InnerProductSolver(setup_test, test_leaks):
    space = FunctionSpace(10)

    def forward(F):
        G = Function(space, name="G")
        AssignmentSolver(F, G).solve()

        J = Functional(name="J")
        InnerProductSolver(F, G, J.fn()).solve()
        return J

    F = Function(space, name="F", static=True)
    function_set_values(F, np.random.random(function_local_size(F)))

    start_manager()
    J = forward(F)
    stop_manager()

    dJ = compute_gradient(J, F)
    min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
    assert(min_order > 1.99)


@pytest.mark.numpy
def test_ContractionSolver(setup_test, test_leaks):
    space_0 = FunctionSpace(1)
    space = FunctionSpace(3)
    A = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
                 dtype=np.float64)

    def forward(m):
        x = Function(space, name="x")
        ContractionSolver(A, (1,), (m,), x).solve()

        norm_sq = Function(space_0, name="norm_sq")
        NormSqSolver(x, norm_sq).solve()

        J = Functional(name="J")
        NormSqSolver(norm_sq, J.fn()).solve()
        return x, J

    m = Function(space, name="m", static=True)
    function_set_values(m, np.array([7.0, 8.0, 9.0], dtype=np.float64))

    start_manager()
    x, J = forward(m)
    stop_manager()

    assert(abs(A.dot(m.vector()) - x.vector()).max() == 0.0)

    J_val = J.value()

    dJ = compute_gradient(J, m)

    def forward_J(m):
        return forward(m)[1]

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ)
    assert(min_order > 2.00)

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ)
    assert(min_order > 3.00)

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2)
    assert(min_order > 2.00)
