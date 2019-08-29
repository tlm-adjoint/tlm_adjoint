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
