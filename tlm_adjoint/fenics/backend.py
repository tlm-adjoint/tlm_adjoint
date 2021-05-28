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

# This file previously included a 'vector' function, which followed
# Function::init_vector in dolfin/function/Function.cpp, DOLFIN 2017.2.0.post0
# Code first added 2018-08-03, removed 2018-09-04
#
# Copyright notice from dolfin/function/Function.cpp, DOLFIN 2017.2.0.post0
#
# Copyright (C) 2003-2012 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Garth N. Wells 2005-2010
# Modified by Martin Sandve Alnes 2008-2014
# Modified by Andre Massing 2009

from fenics import Cell, Constant, DirichletBC, Form, Function, \
    FunctionSpace, LocalSolver, Mesh, MeshEditor, KrylovSolver, LUSolver, \
    LinearVariationalSolver, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, Parameters, Point, TensorFunctionSpace, \
    TestFunction, TrialFunction, UnitIntervalMesh, adjoint, as_backend_type, \
    assemble, assemble_system, has_lu_solver_method, info, parameters, \
    project, solve
import fenics
import numpy as np
import petsc4py.PETSc as PETSc

backend = "FEniCS"

backend_ScalarType = PETSc.ScalarType

if not issubclass(backend_ScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")

extract_args = fenics.fem.solving._extract_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = FunctionSpace
backend_KrylovSolver = KrylovSolver
backend_LUSolver = LUSolver
backend_LinearVariationalSolver = LinearVariationalSolver
backend_Matrix = fenics.cpp.la.GenericMatrix
backend_NonlinearVariationalProblem = NonlinearVariationalProblem
backend_NonlinearVariationalSolver = NonlinearVariationalSolver
backend_Vector = fenics.cpp.la.GenericVector
backend_assemble = assemble
backend_assemble_system = assemble_system
backend_project = project
backend_solve = solve

cpp_LinearVariationalProblem = fenics.cpp.fem.LinearVariationalProblem
cpp_NonlinearVariationalProblem = fenics.cpp.fem.NonlinearVariationalProblem
cpp_PETScVector = fenics.cpp.la.PETScVector

__all__ = \
    [
        "backend",

        "backend_ScalarType",

        "extract_args",

        "backend_Constant",
        "backend_DirichletBC",
        "backend_Function",
        "backend_FunctionSpace",
        "backend_KrylovSolver",
        "backend_LUSolver",
        "backend_LinearVariationalSolver",
        "backend_NonlinearVariationalProblem",
        "backend_NonlinearVariationalSolver",
        "backend_Matrix",
        "backend_Vector",
        "backend_assemble",
        "backend_assemble_system",
        "backend_project",
        "backend_solve",

        "cpp_LinearVariationalProblem",
        "cpp_NonlinearVariationalProblem",
        "cpp_PETScVector",

        "Cell",
        "Form",
        "FunctionSpace",
        "LocalSolver",
        "Mesh",
        "MeshEditor",
        "Parameters",
        "Point",
        "TensorFunctionSpace",
        "TestFunction",
        "TrialFunction",
        "UnitIntervalMesh",
        "adjoint",
        "as_backend_type",
        "has_lu_solver_method",
        "info",
        "parameters"
    ]
