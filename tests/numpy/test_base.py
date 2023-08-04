#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *
from tlm_adjoint.numpy import manager as _manager
from tlm_adjoint.alias import gc_disabled
from tlm_adjoint.override import override_method

from ..test_base import chdir_tmp_path, seed_test, tmp_path
from ..test_base import run_example as _run_example

import gc
import logging
import numpy as np
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
import os
import pytest
import weakref

__all__ = \
    [
        "Constant",
        "info",

        "chdir_tmp_path",
        "run_example",
        "seed_test",
        "setup_test",
        "test_default_dtypes",
        "test_leaks",
        "tmp_path"
    ]


@pytest.fixture
def setup_test():
    set_default_dtype(np.float64)

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    reset_manager("memory", {"drop_references": True})
    stop_manager()
    clear_caches()

    yield

    reset_manager("memory", {"drop_references": False})
    clear_caches()


@pytest.fixture(params=[{"default_dtype": np.float64},
                        {"default_dtype": np.complex128}])
def test_default_dtypes(request):
    set_default_dtype(request.param["default_dtype"])


_function_ids = weakref.WeakValueDictionary()


def clear_function_references():
    _function_ids.clear()


@gc_disabled
def referenced_functions():
    return tuple(F_ref for F_ref in map(call, _function_ids.valuerefs())
                 if F_ref is not None)


@override_method(Function, "__init__")
def Function__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    _function_ids[function_id(self)] = self


@pytest.fixture
def test_leaks():
    clear_function_references()

    yield

    gc.collect()

    # Clear some internal storage that is allowed to keep references
    clear_caches()
    manager = _manager()
    manager.drop_references()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    manager._tlm.clear()
    manager._adj_cache.clear()

    gc.collect()

    refs = 0
    for F in referenced_functions():
        info(f"{function_name(F):s} referenced")
        refs += 1
    if refs == 0:
        info("No references")

    clear_function_references()
    assert refs == 0

    manager.reset("memory", {"drop_references": False})


def run_example(example, *,
                add_example_path=True, clear_forward_globals=True):
    if add_example_path:
        filename = os.path.join(os.path.dirname(__file__),
                                os.path.pardir, os.path.pardir,
                                "examples", "numpy", example)
    else:
        filename = example
    _run_example(filename, clear_forward_globals=clear_forward_globals)


def info(message):
    print(message)


class Constant(Function):
    def __init__(self, value=0.0, *, name=None, space_type="primal",
                 static=False, cache=None, checkpoint=None):
        space = FunctionSpace(1)
        super().__init__(space, name=name, space_type=space_type,
                         static=static, cache=cache, checkpoint=checkpoint)
        self.assign(value)

    def assign(self, y):
        function_assign(self, y)
