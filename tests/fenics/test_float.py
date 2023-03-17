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

from .test_base import *

import mpi4py.MPI as MPI
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
@seed_test
def test_L2_norm(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(u):
        norm_sq = Float(name="norm_sq")
        Assembly(norm_sq, u * u * dx).solve()
        return np.sqrt(norm_sq)

    u = Function(space, name="u")
    interpolate_expression(u, sin(pi * X[0]))

    start_manager()
    norm = forward(u)
    stop_manager()

    norm_ref = np.sqrt(assemble(u * u * dx))
    assert abs(function_scalar_value(norm) - norm_ref) == 0.0

    J_val = function_scalar_value(norm)
    dJ = compute_gradient(norm, u)

    min_order = taylor_test(forward, u, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, u, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, u, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, u, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, u, adjoint_order=2)
    assert min_order > 1.99
