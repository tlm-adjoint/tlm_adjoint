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

base = "Firedrake"

from firedrake import *

import firedrake

base_Constant = firedrake.Constant
base_DirichletBC = firedrake.DirichletBC
base_Function = firedrake.Function

__all__ = \
  [
    "base",
    
    "base_Constant",
    "base_DirichletBC",
    "base_Function",
    
    "Constant",
    "DirichletBC",
    "FunctionSpace",
    "TrialFunction",
    "UnitIntervalMesh",
    "action",
    "adjoint",
    "assemble",
    "firedrake",
    "homogenize",
    "parameters",
    "solve",
    "replace"
  ]
