#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import DEFAULT_COMM, Float, function_name

from .test_base import seed_test, setup_test  # noqa: F401

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.base
@pytest.mark.parametrize(
    "cls",
    [lambda name: Float(name=name)])
@seed_test
def test_name(setup_test,  # noqa: F811
              cls):
    name = "_tlm_adjoint__test_name"
    F = cls(name=name)
    assert function_name(F) == name
