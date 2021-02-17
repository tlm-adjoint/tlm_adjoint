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

from tlm_adjoint.fenics import *

from test_base import *

import mpi4py.MPI as MPI
import os
import pytest


@pytest.mark.fenics
@pytest.mark.example
def test_basal(setup_test):
    configure_checkpointing("memory", {"drop_references": False})
    run_example(os.path.join("basal_sliding", "basal.py"))


@pytest.mark.fenics
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
def test_basal_fp(setup_test):
    configure_checkpointing("memory", {"drop_references": False})
    run_example(os.path.join("basal_sliding", "basal_fp.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_diffusion(setup_test, test_leaks):
    run_example(os.path.join("diffusion", "diffusion.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_poisson(setup_test, test_leaks):
    run_example(os.path.join("poisson", "poisson.py"))


@pytest.mark.fenics
@pytest.mark.example
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="serial only")
def test_transport(setup_test):
    configure_checkpointing("memory", {"drop_references": False})
    run_example(os.path.join("transport", "transport.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_manual_diffusion_forward(setup_test):
    run_example(os.path.join("manual", "diffusion_forward.py"),
                clear_forward_globals=False)


@pytest.mark.fenics
@pytest.mark.example
def test_manual_diffusion_adjoint(setup_test, test_leaks):
    run_example(os.path.join("manual", "diffusion_adjoint.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_manual_diffusion_hessian(setup_test, test_leaks):
    run_example(os.path.join("manual", "diffusion_hessian.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_manual_override_forward(setup_test):
    run_example(os.path.join("manual", "override_forward.py"),
                clear_forward_globals=False)


@pytest.mark.fenics
@pytest.mark.example
def test_manual_override_adjoint(setup_test, test_leaks):
    start_manager()
    run_example(os.path.join("manual", "override_adjoint.py"))


@pytest.mark.fenics
@pytest.mark.example
def test_manual_diffusion_adjoint_timestepping(setup_test, test_leaks):
    run_example(os.path.join("manual", "diffusion_adjoint_timestepping.py"))
