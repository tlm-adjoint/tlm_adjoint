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

backend = "Firedrake"

from firedrake import *
import firedrake

backend_Matrix = firedrake.matrix.Matrix

extract_args = firedrake.solving._extract_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_LinearSolver = LinearSolver
backend_LinearVariationalSolver = LinearVariationalSolver
backend_NonlinearVariationalProblem = NonlinearVariationalProblem
backend_NonlinearVariationalSolver = NonlinearVariationalSolver
backend_assemble = assemble
backend_project = project
backend_solve = solve

__all__ = \
  [
    "backend",
    
    "backend_Constant",
    "backend_DirichletBC",
    "backend_Function",
    "backend_LinearSolver",
    "backend_LinearVariationalSolver",
    "backend_NonlinearVariationalProblem",
    "backend_NonlinearVariationalSolver",
    "backend_Matrix",
    "backend_assemble",
    "backend_project",
    "backend_solve",
    
    "extract_args",
    
    "FunctionSpace",
    "LinearSolver",
    "Parameters",
    "Projector",
    "Tensor",
    "TestFunction",
    "TrialFunction",
    "UnitIntervalMesh",
    "adjoint",
    "assemble",
    "homogenize",
    "parameters",
    "solve"
  ]
