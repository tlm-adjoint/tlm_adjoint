#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.firedrake import *

from .test_base import *

import os
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
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
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_0_getting_started(setup_test, tmp_path):
    run_example_notebook("0_getting_started.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_1_time_independent(setup_test, tmp_path):
    reset_manager("memory", {"drop_references": False})
    run_example_notebook("1_time_independent.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_2_verification(setup_test, tmp_path):
    reset_manager("memory", {"drop_references": False})
    run_example_notebook("2_verification.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@pytest.mark.skip  # Long example
@seed_test
def test_3_time_dependent(setup_test, tmp_path):
    run_example_notebook("3_time_dependent.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_4_riesz_maps(setup_test, tmp_path):
    run_example_notebook("4_riesz_maps.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_5_optimization(setup_test, tmp_path):
    run_example_notebook("5_optimization.ipynb", tmp_path)


@pytest.mark.firedrake
@pytest.mark.example
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_6_custom_operations(setup_test, tmp_path):
    run_example_notebook("6_custom_operations.ipynb", tmp_path)
