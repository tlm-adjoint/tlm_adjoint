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

modules = [("backend", "tlm_adjoint.fenics"),
           ("functions", "tlm_adjoint"),
           ("backend_code_generator_interface", "tlm_adjoint.fenics"),
           ("caches", "tlm_adjoint"),
           ("equations", "tlm_adjoint"),
           ("backend_interface", "tlm_adjoint.fenics"),
           ("backend_overrides", "tlm_adjoint.fenics"),
           ("fenics_equations", "tlm_adjoint.fenics")]

tlm_adjoint_module = "tlm_adjoint" in sys.modules

for module_name, package in modules:
    if package == "tlm_adjoint":
        sys.modules[f"tlm_adjoint.fenics.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint")
    else:
        assert package == "tlm_adjoint.fenics"
        sys.modules[f"tlm_adjoint.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint.fenics")

for module_name, package in modules:
    del sys.modules[f"tlm_adjoint.{module_name:s}"]

if not tlm_adjoint_module:
    del sys.modules["tlm_adjoint"]

del importlib, sys, modules, tlm_adjoint_module, module_name, package

from tlm_adjoint import *  # noqa: E402,F401

from .backend import backend       # noqa: E402,F401
from .backend_code_generator_interface import copy_parameters_dict  # noqa: E402,E501,F401
from .backend_interface import *   # noqa: E402,F401
from .backend_overrides import *   # noqa: E402,F401
from .caches import *              # noqa: E402,F401
from .equations import *           # noqa: E402,F401
from .fenics_equations import *    # noqa: E402,F401
from .functions import *           # noqa: E402,F401
