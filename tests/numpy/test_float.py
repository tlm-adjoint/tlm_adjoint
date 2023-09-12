#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *

from .test_base import *

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.numpy
@pytest.mark.parametrize("value", [2, 3.0, 4.0 + 5.0j])
@seed_test
def test_Float_new(setup_test, test_leaks,
                   value):
    x = Float(name="x")
    assert x.value() == 0.0

    y = x.new(value)
    assert x.value() == 0.0
    assert y.value() == value
