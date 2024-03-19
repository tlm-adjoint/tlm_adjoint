from fenics import *
import fenics
import numpy as np
import petsc4py.PETSc as PETSc

backend = "FEniCS"

backend_RealType = PETSc.RealType
backend_ScalarType = PETSc.ScalarType
if not issubclass(backend_RealType, np.floating):
    raise ImportError(f"Invalid backend real type: {backend_RealType}")
if not issubclass(backend_ScalarType, np.floating):
    raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")
complex_mode = False

extract_args = fenics.fem.solving._extract_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = FunctionSpace
backend_LocalSolver = LocalSolver
backend_Matrix = fenics.cpp.la.GenericMatrix
backend_Vector = fenics.cpp.la.GenericVector
backend_action = action
backend_assemble = assemble
backend_assemble_system = assemble_system
backend_project = project
backend_solve = solve

cpp_Assembler = fenics.cpp.fem.Assembler
cpp_Constant = fenics.cpp.function.Constant
cpp_PETScVector = fenics.cpp.la.PETScVector
cpp_SystemAssembler = fenics.cpp.fem.SystemAssembler
