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

from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

import mpi4py.MPI as MPI
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_FunctionSpace_interface(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    F = Function(space, name="F")

    assert space_id(space) == space_id(function_space(F))
    assert space_id(space) == space_id(F.function_space())

    F_copy = function_copy(F)
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_copy = F.copy(deepcopy=True)
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_0 = F.split()[0]
    assert space_id(function_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(function_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())


@pytest.mark.firedrake
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@seed_test
def test_function_alias(setup_test, test_leaks,
                        dim):
    mesh = UnitIntervalMesh(20)

    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=dim)

    F = Function(space, name="F")
    F.split()
    for i in range(dim):
        F.sub(i)

    F = Function(space, name="F")
    for i in range(dim):
        F.sub(i)
    F.split()

    space = FunctionSpace(mesh, "Lagrange", 1)
    space = FunctionSpace(mesh, ufl.classes.MixedElement(
        *[space.ufl_element() for _ in range(dim)]))

    F = Function(space, name="F")
    F.split()
    for i in range(dim):
        F.sub(i)

    F = Function(space, name="F")
    for i in range(dim):
        F.sub(i)
    F.split()
