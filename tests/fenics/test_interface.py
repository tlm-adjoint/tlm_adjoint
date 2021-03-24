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

from test_base import *

import pytest


@pytest.mark.fenics
def test_space_id(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = Function(space, name="F")

    assert space_id(space) == space_id(function_space(F))
    assert space_id(space) == space_id(F.function_space())

    F_copy = function_copy(F)
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_copy = F.copy()
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())
