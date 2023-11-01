#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    DEFAULT_COMM, set_default_float_dtype, set_default_jax_dtype)

from .test_base import seed_test, setup_test, run_example_notebook  # noqa: E501,F401

try:
    import jax
except ImportError:
    jax = None
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1,
    reason="serial only")


@pytest.mark.base
@pytest.mark.example
@seed_test
def test_0_getting_started(setup_test, tmp_path):  # noqa: F811
    run_example_notebook("0_getting_started.ipynb", tmp_path)


@pytest.mark.base
@pytest.mark.example
@pytest.mark.skipif(jax is None, reason="JAX not available")
@seed_test
def test_7_jax_integration(setup_test, tmp_path):  # noqa: F811
    set_default_float_dtype(np.double)
    set_default_jax_dtype(np.double)
    run_example_notebook("7_jax_integration.ipynb", tmp_path)
