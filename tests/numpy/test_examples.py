#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *

from .test_base import *

import os
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.numpy
@pytest.mark.example
@seed_test
def test_diffusion(setup_test, test_leaks, chdir_tmp_path):
    run_example(os.path.join("diffusion", "diffusion.py"))
