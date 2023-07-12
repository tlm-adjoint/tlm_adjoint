#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *
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
backend_Matrix = fenics.cpp.la.GenericMatrix
backend_Vector = fenics.cpp.la.GenericVector
backend_assemble = assemble
backend_assemble_system = assemble_system
backend_project = project
backend_solve = solve

cpp_Assembler = fenics.cpp.fem.Assembler
cpp_PETScVector = fenics.cpp.la.PETScVector
cpp_SystemAssembler = fenics.cpp.fem.SystemAssembler
