#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import sys

modules = [("backend", "tlm_adjoint.firedrake"),
           ("functions", "tlm_adjoint._code_generator"),
           ("backend_code_generator_interface", "tlm_adjoint.firedrake"),
           ("caches", "tlm_adjoint._code_generator"),
           ("equations", "tlm_adjoint._code_generator"),
           ("backend_interface", "tlm_adjoint.firedrake"),
           ("backend_overrides", "tlm_adjoint.firedrake"),
           ("firedrake_equations", "tlm_adjoint.firedrake"),
           ("block_system", "tlm_adjoint._code_generator"),
           ("hessian_system", "tlm_adjoint._code_generator")]

for module_name, package in modules:
    if package == "tlm_adjoint._code_generator":
        sys.modules[f"tlm_adjoint.firedrake.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint._code_generator")
    else:
        assert package == "tlm_adjoint.firedrake"
        sys.modules[f"tlm_adjoint._code_generator.{module_name:s}"] \
            = importlib.import_module(f".{module_name:s}",
                                      package="tlm_adjoint.firedrake")

for module_name, package in modules:
    del sys.modules[f"tlm_adjoint._code_generator.{module_name:s}"]
del sys.modules["tlm_adjoint._code_generator"]

del importlib, sys, modules, module_name, package

from .. import *  # noqa: E402,F401
del adjoint, alias, cached_hessian, caches, checkpointing, \
    eigendecomposition, equation, equations, fixed_point, functional, \
    hessian, instructions, interface, linear_equation, markers, optimization, \
    overloaded_float, override, storage, tangent_linear, tlm_adjoint, \
    verification  # noqa: F821

from .backend import backend, backend_RealType, backend_ScalarType  # noqa: E402,E501,F401
from .backend_code_generator_interface import linear_solver  # noqa: E402,F401
from .backend_interface import *    # noqa: E402,F401
from .backend_overrides import *    # noqa: E402,F401
from .block_system import \
    (ConstantNullspace, DirichletBCNullspace, NoneNullspace, UnityNullspace)  # noqa: E402,E501,F401
from .caches import *               # noqa: E402,F401
from .equations import *            # noqa: E402,F401
from .firedrake_equations import *  # noqa: E402,F401
from .functions import *            # noqa: E402,F401
from .hessian_system import *       # noqa: E402,F401
