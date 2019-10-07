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

from tlm_adjoint_firedrake import *

from test_base import *

import mpi4py.MPI as MPI
import os
import pytest


@pytest.mark.firedrake
@pytest.mark.example
def test_basal(setup_test):
    configure_checkpointing("memory", {"replace": False})
    run_example(os.path.join("basal_sliding", "basal.py"))


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
def test_basal_fp(setup_test):
    configure_checkpointing("memory", {"replace": False})
    run_example(os.path.join("basal_sliding", "basal_fp.py"))


@pytest.mark.firedrake
@pytest.mark.example
def test_diffusion(setup_test, test_leaks):
    run_example(os.path.join("diffusion", "diffusion.py"))
