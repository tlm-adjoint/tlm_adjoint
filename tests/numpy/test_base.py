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

import gc
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

        "run_example",
        "setup_test",
        "test_leaks"
    ]


@pytest.fixture
def setup_test():
    reset_manager("memory", {"drop_references": True})
    clear_caches()
    stop_manager()

    logging.getLogger("tlm_adjoint").setLevel(logging.DEBUG)

    np.random.seed(14012313)


function_ids = {}


def _Function__init__(self, *args, **kwargs):
    _Function__init__orig(self, *args, **kwargs)
    function_ids[function_id(self)] = weakref.ref(self)


_Function__init__orig = Function.__init__
Function.__init__ = _Function__init__


@pytest.fixture
def test_leaks():
    function_ids.clear()

    yield

    gc.collect()

    # Clear some internal storage that is allowed to keep references
    manager = _manager()
    manager._cp.clear(clear_refs=True)
    manager._cp_memory.clear()
    tlm_values = list(manager._tlm.values())  # noqa: F841
    manager._tlm.clear()
    tlm_eqs_values = [list(eq_tlm_eqs.values()) for eq_tlm_eqs in manager._tlm_eqs.values()]  # noqa: E501,F841
    manager._tlm_eqs.clear()
    manager.drop_references()

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


def run_example(example, clear_forward_globals=True):
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
    def __init__(self, value=0.0, name=None, static=False, cache=None,
                 checkpoint=None, _data=None):
        super().__init__(FunctionSpace(1), name=name, static=static,
                         cache=cache, checkpoint=checkpoint, _data=_data)
        self.assign(value)

    def assign(self, y):
        if isinstance(y, Constant):
            self.vector()[:] = y.vector()
        else:
            assert isinstance(y, (int, np.integer, float, np.floating))
            self.vector()[:] = backend_ScalarType(y),
