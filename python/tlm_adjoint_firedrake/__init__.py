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

import importlib
import sys

modules = [("backend", "tlm_adjoint_firedrake"),
           ("backend_code_generator_interface", "tlm_adjoint_firedrake"),
           ("function_interface", "tlm_adjoint"),
           ("caches", "tlm_adjoint"),
           ("backend_interface", "tlm_adjoint_firedrake"),
           ("base_equations", "tlm_adjoint"),
           ("equations", "tlm_adjoint"),
           ("tlm_adjoint", "tlm_adjoint"),
           ("firedrake_equations", "tlm_adjoint_firedrake"),
           ("backend_overrides", "tlm_adjoint_firedrake"),
           ("binomial_checkpointing", "tlm_adjoint"),
           ("eigendecomposition", "tlm_adjoint"),
           ("functional", "tlm_adjoint"),
           ("hessian", "tlm_adjoint"),
           ("hessian_optimization", "tlm_adjoint"),
           ("manager", "tlm_adjoint"),
           ("timestepping", "tlm_adjoint")]

tlm_adjoint_module = "tlm_adjoint" in sys.modules

for module_name, package in modules:
    if package == "tlm_adjoint":
        sys.modules[f"tlm_adjoint_firedrake.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint")
    else:
        assert(package == "tlm_adjoint_firedrake")
        sys.modules[f"tlm_adjoint.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint_firedrake")

for module_name, package in modules:
    del(sys.modules[f"tlm_adjoint.{module_name:s}"])

if not tlm_adjoint_module:
    del(sys.modules["tlm_adjoint"])

del(importlib, sys, modules, tlm_adjoint_module, module_name, package)

from .backend import backend        # noqa: F401
from .backend_interface import *    # noqa: F401
from .backend_overrides import *    # noqa: F401
from .base_equations import *       # noqa: F401
from .caches import *               # noqa: F401
from .eigendecomposition import *   # noqa: F401
from .equations import *            # noqa: F401
from .firedrake_equations import *  # noqa: F401
from .functional import *           # noqa: F401
from .hessian import *              # noqa: F401
from .manager import *              # noqa: F401
from .tlm_adjoint import *          # noqa: F401
