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

from tlm_adjoint.firedrake import *

from .test_base import *

import mpi4py.MPI as MPI
import os
import pytest

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@pytest.mark.example
@seed_test
def test_diffusion(setup_test, test_leaks):
    run_example(os.path.join("diffusion", "diffusion.py"))


@pytest.mark.firedrake
@pytest.mark.example
@seed_test
def test_poisson(setup_test, test_leaks):
    run_example(os.path.join("poisson", "poisson.py"))


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
@seed_test
def test_manual_diffusion_forward(setup_test):
    run_example(os.path.join("manual", "diffusion_forward.py"),
                clear_forward_globals=False)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
@seed_test
def test_manual_diffusion_adjoint(setup_test, test_leaks):
    run_example(os.path.join("manual", "diffusion_adjoint.py"))


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
@seed_test
def test_manual_diffusion_hessian(setup_test, test_leaks):
    run_example(os.path.join("manual", "diffusion_hessian.py"))


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
@seed_test
def test_manual_override_forward(setup_test):
    run_example(os.path.join("manual", "override_forward.py"),
                clear_forward_globals=False)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_manual_override_adjoint(setup_test, test_leaks):
    run_example(os.path.join("manual", "override_adjoint.py"))
