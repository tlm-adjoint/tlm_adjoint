#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *

from .test_base import *

import os
import pytest

try:
    import mpi4py.MPI as MPI
    pytestmark = pytest.mark.skipif(
        MPI.COMM_WORLD.size > 1, reason="serial only")
except ImportError:
    pass


@pytest.mark.numpy
@pytest.mark.example
@seed_test
def test_diffusion(setup_test, test_leaks, chdir_tmp_path):
    run_example(os.path.join("diffusion", "diffusion.py"))
