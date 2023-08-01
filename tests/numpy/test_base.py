#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..test_base import *

from tlm_adjoint.numpy import *
from tlm_adjoint.numpy import manager as _manager
from tlm_adjoint.alias import gc_disabled
from tlm_adjoint.override import override_method

import functools
import gc
import hashlib
import inspect
import logging
import numpy as np
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
from operator import itemgetter
import os
import pytest
import runpy
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
        "test_leaks"
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


def seed_test(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        h_kwargs = dict(kwargs)
        if "tmp_path" in inspect.signature(fn).parameters:
            # Raises an error if tmp_path is a positional argument
            del h_kwargs["tmp_path"]

        h = hashlib.sha256()
        h.update(fn.__name__.encode("utf-8"))
        h.update(str(args).encode("utf-8"))
        h.update(str(sorted(h_kwargs.items(), key=itemgetter(0))).encode("utf-8"))  # noqa: E501
        seed = int(h.hexdigest(), 16)
        seed %= 2 ** 32
        np.random.seed(seed)

        return fn(*args, **kwargs)
    return wrapped_fn


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


@pytest.fixture
def chdir_tmp_path(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    os.chdir(cwd)


def run_example(example, *,
                add_example_path=True, clear_forward_globals=True):
    if add_example_path:
        filename = os.path.join(os.path.dirname(__file__),
                                os.path.pardir, os.path.pardir,
                                "examples", "numpy", example)
    else:
        filename = example

    start_manager()
    gl = runpy.run_path(filename)

    if clear_forward_globals:
        # Clear objects created by the script. Requires the script to define a
        # 'forward' function.
        gl["forward"].__globals__.clear()


def info(message):
    print(message)


class Constant(Function):
    def __init__(self, value=0.0, *, name=None, space_type="primal",
                 static=False, cache=None, checkpoint=None):
        space = FunctionSpace(1)  # , dtype=default_dtype())
        super().__init__(space, name=name, space_type=space_type,
                         static=static, cache=cache, checkpoint=checkpoint)
        self.assign(value)

    def assign(self, y):
        function_assign(self, y)
