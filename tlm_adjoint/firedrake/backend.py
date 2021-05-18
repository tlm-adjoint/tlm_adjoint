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

from firedrake import Constant, DirichletBC, Function, FunctionSpace, \
    LinearSolver, LinearVariationalProblem, LinearVariationalSolver, \
    NonlinearVariationalSolver, Parameters, Projector, Tensor, TestFunction, \
    TrialFunction, UnitIntervalMesh, Vector, adjoint, assemble, homogenize, \
    info, parameters, project, solve
import firedrake
import numpy as np
import petsc4py.PETSc as PETSc

backend = "Firedrake"

backend_ScalarType = firedrake.utils.ScalarType.type
backend_RealScalarType = firedrake.utils.RealType.type
backend_ComplexScalarType = PETSc.ComplexType

if not issubclass(backend_ScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")
if not issubclass(backend_RealScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend real scalar type: "
                      f"{backend_RealScalarType}")
if not issubclass(backend_ComplexScalarType, (complex, np.complexfloating)):
    raise ImportError(f"Invalid backend complex scalar type: "
                      f"{backend_ComplexScalarType}")

extract_args = firedrake.solving._extract_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = firedrake.functionspaceimpl.WithGeometry
backend_LinearSolver = LinearSolver
backend_LinearVariationalProblem = LinearVariationalProblem
backend_LinearVariationalSolver = LinearVariationalSolver
backend_Matrix = firedrake.matrix.Matrix
backend_NonlinearVariationalSolver = NonlinearVariationalSolver
backend_Vector = Vector
backend_assemble = assemble
backend_project = project
backend_solve = solve

__all__ = \
    [
        "backend",

        "backend_ComplexScalarType",
        "backend_RealScalarType",
        "backend_ScalarType",

        "extract_args",

        "backend_Constant",
        "backend_DirichletBC",
        "backend_Function",
        "backend_FunctionSpace",
        "backend_LinearSolver",
        "backend_LinearVariationalProblem",
        "backend_LinearVariationalSolver",
        "backend_Matrix",
        "backend_NonlinearVariationalSolver",
        "backend_Vector",
        "backend_assemble",
        "backend_project",
        "backend_solve",

        "FunctionSpace",
        "Parameters",
        "Projector",
        "Tensor",
        "TestFunction",
        "TrialFunction",
        "UnitIntervalMesh",
        "adjoint",
        "homogenize",
        "info",
        "parameters"
    ]
