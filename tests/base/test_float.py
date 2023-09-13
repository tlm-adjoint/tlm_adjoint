#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import DEFAULT_COMM, Float

from .test_base import seed_test, setup_test  # noqa: F401

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.base
@pytest.mark.parametrize("value", [2, 3.0, 4.0 + 5.0j])
@seed_test
def test_Float_new(setup_test,  # noqa: F811
                   value):
    x = Float(name="x")
    assert x.value() == 0.0

    y = x.new(value)
    assert x.value() == 0.0
    assert y.value() == value
