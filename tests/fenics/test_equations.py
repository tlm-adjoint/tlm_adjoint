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
def test_AssignmentSolver():
    reset_manager("memory", {"replace": True})
    clear_caches()
    stop_manager()

    space = RealFunctionSpace()
    x = Function(space, name="x", static=True)
    function_assign(x, 16.0)

    def forward(x):
        y = [Function(space, name="y_%i" % i) for i in range(9)]
        z = Function(space, name="z")

        AssignmentSolver(x, y[0]).solve()
        for i in range(len(y) - 1):
            AssignmentSolver(y[i], y[i + 1]).solve()
        NormSqSolver(y[-1], z).solve()

        x_norm_sq = Function(space, name="x_norm_sq")
        NormSqSolver(x, x_norm_sq).solve()

        z_norm_sq = Function(space, name="z_norm_sq")
        NormSqSolver(z, z_norm_sq).solve()

        J = Functional(name="J", space=space)
        AxpySolver(z_norm_sq, 2.0, x_norm_sq, J.fn()).solve()

        K = Functional(name="K", space=space)
        AssignmentSolver(z_norm_sq, K.fn()).solve()

        return J, K

    start_manager()
    J, K = forward(x)
    stop_manager()

    assert(abs(J.value() - 66048.0) == 0.0)
    assert(abs(K.value() - 65536.0) == 0.0)

    dJs = compute_gradient([J, K], x)
    dm = Function(space, name="dm", static=True)
    function_assign(dm, 1.0)

    for forward_J, J_val, dJ in [(lambda x: forward(x)[0], J.value(), dJs[0]),
                                 (lambda x: forward(x)[1], K.value(), dJs[1])]:
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ, dM=dm)
        assert(min_order > 2.00)

        ddJ = Hessian(forward_J)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ, dM=dm)
        assert(min_order > 3.00)

        min_order = taylor_test_tlm(forward_J, x, tlm_order=1, dMs=(dm,))
        assert(min_order > 2.00)

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1,
                                            dMs=(dm,))
        assert(min_order > 2.00)

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2,
                                            dMs=(dm, dm))
        assert(min_order > 2.00)
