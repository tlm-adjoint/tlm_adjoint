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

from tlm_adjoint_numpy import *  # noqa: F401

from test_base import *          # noqa: F401

import os
import runpy
import pytest


@pytest.mark.numpy
@pytest.mark.examples
@pytest.mark.parametrize("example",
                         [os.path.join("diffusion", "diffusion.py")])
def test_examples(setup_test, test_leaks,
                  example):
    filename = os.path.join(os.path.dirname(__file__),
                            os.path.pardir, os.path.pardir,
                            "examples", "numpy", example)
    gl = runpy.run_path(filename)
    # Clear objects created by the script. Requires the script to define a
    # 'forward' function.
    gl["forward"].__globals__.clear()
