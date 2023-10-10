#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    clear_caches, set_default_float_dtype, set_default_jax_dtype,
    reset_manager)

from ..test_base import chdir_tmp_path, jax_tlm_config, seed_test, tmp_path

import logging
import numpy as np
import pytest

__all__ = \
    [
        "chdir_tmp_path",
        "jax_tlm_config",
        "seed_test",
        "setup_test",
        "tmp_path"
    ]


@pytest.fixture
def setup_test():
    try:
        import petsc4py.PETSc as PETSc
        default_dtype = PETSc.ScalarType
    except ImportError:
        default_dtype = np.double
    set_default_float_dtype(default_dtype)
    set_default_jax_dtype(default_dtype)

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()
