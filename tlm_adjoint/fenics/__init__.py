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
           ("functions", "tlm_adjoint._code_generator"),
           ("backend_code_generator_interface", "tlm_adjoint.fenics"),
           ("caches", "tlm_adjoint._code_generator"),
           ("equations", "tlm_adjoint._code_generator"),
           ("backend_interface", "tlm_adjoint.fenics"),
           ("backend_overrides", "tlm_adjoint.fenics"),
           ("fenics_equations", "tlm_adjoint.fenics")]

for module_name, package in modules:
    if package == "tlm_adjoint._code_generator":
        sys.modules[f"tlm_adjoint.fenics.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint._code_generator")
    else:
        assert package == "tlm_adjoint.fenics"
        sys.modules[f"tlm_adjoint._code_generator.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint.fenics")

for module_name, package in modules:
    del sys.modules[f"tlm_adjoint._code_generator.{module_name:s}"]
del sys.modules["tlm_adjoint._code_generator"]

del importlib, sys, modules, module_name, package

from .. import *  # noqa: E402,F401

from .backend import backend, backend_ScalarType  # noqa: E402,F401
from .backend_code_generator_interface import copy_parameters_dict  # noqa: E402,E501,F401
from .backend_interface import *   # noqa: E402,F401
from .backend_overrides import *   # noqa: E402,F401
from .caches import *              # noqa: E402,F401
from .equations import *           # noqa: E402,F401
from .fenics_equations import *    # noqa: E402,F401
from .functions import *           # noqa: E402,F401
