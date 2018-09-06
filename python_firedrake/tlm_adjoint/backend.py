#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright(c) 2018 The University of Edinburgh
#
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

backend = "Firedrake"

from firedrake import *

import firedrake

extract_args = firedrake.solving._extract_args

backend_Function = firedrake.Function
backend_LinearSolver = firedrake.LinearSolver
backend_assemble = assemble
backend_project = project
backend_solve = solve

__all__ = \
  [
    "backend",
    
    "backend_Function",
    "backend_LinearSolver",
    "backend_assemble",
    "backend_project",
    "backend_solve",
    
    "Constant",
    "DirichletBC",
    "Function",
    "FunctionSpace",
    "LinearSolver",
    "Parameters",
    "TestFunction",
    "TrialFunction",
    "UnitIntervalMesh",
    "action",
    "adjoint",
    "as_backend_type",
    "assemble",
    "dx",
    "extract_args",
    "firedrake",
    "homogenize",
    "inner",
    "parameters",
    "project",
    "solve"
  ]
