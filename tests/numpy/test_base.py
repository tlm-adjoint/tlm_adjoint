#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from tlm_adjoint.numpy import *
from tlm_adjoint.numpy import manager as _manager

import copy
import functools
import gc
import hashlib
import inspect
import logging
import numpy as np
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
    reset_manager("memory", {"drop_references": True})
    clear_caches()
    stop_manager()

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    set_default_dtype(np.float64)

    yield

    reset_manager("memory", {"drop_references": False})


def seed_test(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        h_kwargs = copy.copy(kwargs)
        if "tmp_path" in inspect.signature(fn).parameters:
            # Raises an error if tmp_path is a positional argument
            del h_kwargs["tmp_path"]

        h = hashlib.sha256()
        h.update(fn.__name__.encode("utf-8"))
        h.update(str(args).encode("utf-8"))
        h.update(str(sorted(h_kwargs.items(), key=lambda e: e[0])).encode("utf-8"))  # noqa: E501
        seed = int(h.hexdigest(), 16)
        seed %= 2 ** 32
        np.random.seed(seed)

        return fn(*args, **kwargs)
    return wrapped_fn


@pytest.fixture(params=[{"default_dtype": np.float64},
                        {"default_dtype": np.complex128}])
def test_default_dtypes(request):
    set_default_dtype(request.param["default_dtype"])


function_ids = {}


def _Function__init__(self, *args, **kwargs):
    _Function__init__orig(self, *args, **kwargs)
    function_ids[function_id(self)] = weakref.ref(self)


_Function__init__orig = Function.__init__
Function.__init__ = _Function__init__


def _EquationManager_configure_checkpointing(self, *args, **kwargs):
    if hasattr(self, "_cp_method") \
            and hasattr(self, "_cp_parameters") \
            and hasattr(self, "_cp_manager"):
        if self._cp_method == "multistage" \
                and self._cp_manager.max_n() - self._cp_manager.r() == 0 \
                and "path" in self._cp_parameters:
            self._comm.barrier()
            cp_path = self._cp_parameters["path"]
            assert not os.path.exists(cp_path) or len(os.listdir(cp_path)) == 0

    _EquationManager_configure_checkpointing__orig(self, *args, **kwargs)


_EquationManager_configure_checkpointing__orig = EquationManager.configure_checkpointing  # noqa: E501
EquationManager.configure_checkpointing = _EquationManager_configure_checkpointing  # noqa: E501


@pytest.fixture
def test_leaks():
    function_ids.clear()

    yield

    gc.collect()

    # Clear some internal storage that is allowed to keep references
    manager = _manager()
    manager.drop_references()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    tlm_values = list(manager._tlm.values())  # noqa: F841
    manager._tlm.clear()
    tlm_eqs_values = [list(eq_tlm_eqs.values()) for eq_tlm_eqs in manager._tlm_eqs.values()]  # noqa: E501,F841
    manager._tlm_eqs.clear()

    gc.collect()

    refs = 0
    for F in function_ids.values():
        F = F()
        if F is not None:
            info(f"{function_name(F):s} referenced")
            refs += 1
    if refs == 0:
        info("No references")

    function_ids.clear()
    assert refs == 0

    manager.reset("memory", {"drop_references": False})


@pytest.fixture
def chdir_tmp_path(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    os.chdir(cwd)


def run_example(example, clear_forward_globals=True):
    start_manager()
    filename = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, os.path.pardir,
                            "examples", "numpy", example)
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
