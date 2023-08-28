#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *

from .test_base import *

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size > 1, reason="serial only")


@pytest.mark.numpy
@pytest.mark.parametrize(
    "cls",
    [lambda name: Float(name=name),
     lambda name: Function(FunctionSpace(1), name=name)])
@seed_test
def test_name(setup_test, cls):
    name = "_tlm_adjoint__test_name"
    F = cls(name=name)
    assert function_name(F) == name
