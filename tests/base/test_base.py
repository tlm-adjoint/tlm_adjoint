#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import clear_caches, reset_manager

from ..test_base import chdir_tmp_path, seed_test, tmp_path

import logging
import pytest

__all__ = \
    [
        "chdir_tmp_path",
        "seed_test",
        "setup_test",
        "tmp_path"
    ]


@pytest.fixture
def setup_test():
    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()
