#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .test_base import setup_test  # noqa: F401

import mpi4py.MPI as MPI  # noqa: N817
import numpy as np
import petsc4py.PETSc as PETSc
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")

from tlm_adjoint.firedrake.block_system import (  # noqa: E402
    BlockMatrix, ConstantNullspace, DirichletBCNullspace, System as _System,
    UnityNullspace)

from firedrake import (  # noqa: E402
    Constant, ConvergenceError, DirichletBC, Function, FunctionSpace,
    LinearSolver, SpatialCoordinate, TestFunction, TrialFunction,
    UnitSquareMesh, VectorFunctionSpace, action, adjoint, assemble, ds, dx,
    exp, grad, inner, pi, sin, solve)


class System(_System):
    def solve(self, *args, **kwargs):
        return super().solve(
            *args, correct_initial_guess=False, correct_solution=False,
            **kwargs)


@pytest.mark.firedrake
@pytest.mark.parametrize("pc", ["none", "block_jacobi", "block_chebyshev"])
def test_block_diagonal(setup_test, pc):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 1)
    space_1 = FunctionSpace(mesh, "Lagrange", 2)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    block_00 = inner(trial_0, test_0) * dx
    block_01 = None
    block_10 = None
    block_11 = inner(trial_1, test_1) * dx

    system = System(
        (space_0, space_1), (space_0.dual(), space_1.dual()),
        {(0, 0): block_00, (0, 1): block_01,
         (1, 0): block_10, (1, 1): block_11})

    u_0_ref = Function(space_0)
    u_0_ref.interpolate(X[0] * exp(X[1]))
    u_1_ref = Function(space_1)
    u_1_ref.interpolate(sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    b_0 = assemble(inner(u_0_ref, test_0) * dx)
    b_1 = assemble(inner(u_1_ref, test_1) * dx)

    u_0 = Function(space_0)
    u_1 = Function(space_1)

    if pc == "none":
        pc_fn = None
    elif pc == "block_jacobi":
        def pc_fn(u, b):
            u_0, u_1 = u
            b_0, b_1 = b
            solver_0 = LinearSolver(
                assemble(block_00),
                solver_parameters={"ksp_type": "preonly",
                                   "pc_type": "jacobi",
                                   "ksp_max_it": 1,
                                   "ksp_atol": 0.0,
                                   "ksp_rtol": 0.0})
            solver_1 = LinearSolver(
                assemble(block_11),
                solver_parameters={"ksp_type": "preonly",
                                   "pc_type": "jacobi",
                                   "ksp_max_it": 1,
                                   "ksp_atol": 0.0,
                                   "ksp_rtol": 0.0})
            solver_0.solve(u_0, b_0.copy(deepcopy=True))
            solver_1.solve(u_1, b_1.copy(deepcopy=True))
    else:
        assert pc == "block_chebyshev"

        def pc_fn(u, b):
            u_0, u_1 = u
            b_0, b_1 = b
            # Eigenvalue bounds: 0.0006819477595146459, 0.009730121547766488
            solver_0 = LinearSolver(
                assemble(block_00),
                solver_parameters={"ksp_type": "chebyshev",
                                   "ksp_chebyshev_eigenvalues": "0.0005,0.01",
                                   "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                                   "ksp_chebyshev_esteig_steps": 0,
                                   "ksp_chebyshev_esteig_noisy": False,
                                   "pc_type": "none",
                                   "ksp_max_it": 8,
                                   "ksp_atol": 0.0,
                                   "ksp_rtol": 0.0})
            # Eigenvalue bounds: 0.00015115459477735448, 0.0035778617188465087
            solver_1 = LinearSolver(
                assemble(block_11),
                solver_parameters={"ksp_type": "chebyshev",
                                   "ksp_chebyshev_eigenvalues": "0.0001,0.004",
                                   "ksp_chebyshev_esteig": "0.0,0.0,0.0,0.0",
                                   "ksp_chebyshev_esteig_steps": 0,
                                   "ksp_chebyshev_esteig_noisy": False,
                                   "pc_type": "none",
                                   "ksp_max_it": 8,
                                   "ksp_atol": 0.0,
                                   "ksp_rtol": 0.0})
            try:
                solver_0.solve(u_0, b_0.copy(deepcopy=True))
            except ConvergenceError:
                assert solver_0.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501
            try:
                solver_1.solve(u_1, b_1.copy(deepcopy=True))
            except ConvergenceError:
                assert solver_1.ksp.getConvergedReason() == PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT  # noqa: E501

    ksp_its = system.solve(
        (u_0, u_1), (b_0, b_1),
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14},
        pc_fn=pc_fn)

    u_0_error_norm = np.sqrt(abs(assemble(inner(u_0 - u_0_ref,
                                                u_0 - u_0_ref) * dx)))
    assert u_0_error_norm < 1.0e-12

    u_1_error_norm = np.sqrt(abs(assemble(inner(u_1 - u_1_ref,
                                                u_1 - u_1_ref) * dx)))
    assert u_1_error_norm < 1.0e-12

    if pc == "none":
        assert ksp_its <= 69
    elif pc == "block_jacobi":
        assert ksp_its <= 31
    else:
        assert pc == "block_chebyshev"
        assert ksp_its <= 13


@pytest.mark.firedrake
def test_constant_nullspace(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 1)
    space_1 = FunctionSpace(mesh, "Lagrange", 2)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    block_00 = inner(grad(trial_0), grad(test_0)) * dx
    block_01 = None
    block_10 = None
    block_11 = inner(trial_1, test_1) * dx

    system = System(
        (space_0, space_1), (space_0.dual(), space_1.dual()),
        {(0, 0): block_00, (0, 1): block_01,
         (1, 0): block_10, (1, 1): block_11},
        nullspaces=(ConstantNullspace(), None))

    u_0_ref = Function(space_0)
    u_0_ref.interpolate(X[0] * exp(X[1]))
    u_1_ref = Function(space_1)
    u_1_ref.interpolate(sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    b_0 = assemble(inner(grad(u_0_ref), grad(test_0)) * dx)
    b_1 = assemble(inner(u_1_ref, test_1) * dx)

    u_0 = Function(space_0)
    u_0.assign(Constant(1.0))
    u_1 = Function(space_1)

    _ = system.solve(
        (u_0, u_1), (b_0, b_1),
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    with u_0.dat.vec_ro as u_0_v:
        u_0_sum = u_0_v.sum()
    assert abs(u_0_sum) < 1.0e-13

    u_0_error_norm = np.sqrt(abs(assemble(inner(grad(u_0 - u_0_ref),
                                                grad(u_0 - u_0_ref)) * dx)))
    assert u_0_error_norm < 1.0e-13

    u_1_error_norm = np.sqrt(abs(assemble(inner(u_1 - u_1_ref,
                                                u_1 - u_1_ref) * dx)))
    assert u_1_error_norm < 1.0e-12


@pytest.mark.firedrake
def test_unity_nullspace(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 1)
    space_1 = FunctionSpace(mesh, "Lagrange", 2)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    block_00 = inner(grad(trial_0), grad(test_0)) * dx
    block_01 = None
    block_10 = None
    block_11 = inner(trial_1, test_1) * dx

    system = System(
        (space_0, space_1), (space_0.dual(), space_1.dual()),
        {(0, 0): block_00, (0, 1): block_01,
         (1, 0): block_10, (1, 1): block_11},
        nullspaces=(UnityNullspace(space_0), None))

    u_0_ref = Function(space_0)
    u_0_ref.interpolate(X[0] * exp(X[1]))
    u_1_ref = Function(space_1)
    u_1_ref.interpolate(sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    b_0 = assemble(inner(grad(u_0_ref), grad(test_0)) * dx)
    b_1 = assemble(inner(u_1_ref, test_1) * dx)

    u_0 = Function(space_0)
    u_0.assign(Constant(1.0))
    u_1 = Function(space_1)

    _ = system.solve(
        (u_0, u_1), (b_0, b_1),
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    u_0_int = assemble(u_0 * dx)
    assert abs(u_0_int) < 1.0e-14

    u_0_error_norm = np.sqrt(abs(assemble(inner(grad(u_0 - u_0_ref),
                                                grad(u_0 - u_0_ref)) * dx)))
    assert u_0_error_norm < 1.0e-13

    u_1_error_norm = np.sqrt(abs(assemble(inner(u_1 - u_1_ref,
                                                u_1 - u_1_ref) * dx)))
    assert u_1_error_norm < 1.0e-12


@pytest.mark.firedrake
def test_dirichlet_bc_nullspace(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 1)
    space_1 = FunctionSpace(mesh, "Lagrange", 2)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    block_00 = inner(grad(trial_0), grad(test_0)) * dx
    block_01 = None
    block_10 = None
    block_11 = inner(trial_1, test_1) * dx

    system = System(
        (space_0, space_1), (space_0.dual(), space_1.dual()),
        {(0, 0): block_00, (0, 1): block_01,
         (1, 0): block_10, (1, 1): block_11},
        nullspaces=(DirichletBCNullspace(DirichletBC(space_0, 0.0, "on_boundary")),  # noqa: E501
                    None))

    u_0_ref = Function(space_0)
    u_0_ref.interpolate(X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))  # noqa: E501
    u_1_ref = Function(space_1)
    u_1_ref.interpolate(sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))

    b_0 = assemble(inner(grad(u_0_ref), grad(test_0)) * dx)
    b_1 = assemble(inner(u_1_ref, test_1) * dx)

    u_0 = Function(space_0)
    u_0.assign(Constant(1.0))
    u_1 = Function(space_1)

    _ = system.solve(
        (u_0, u_1), (b_0, b_1),
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    u_0_bc_error_norm = np.sqrt(abs(assemble(inner(u_0, u_0) * ds)))
    assert u_0_bc_error_norm < 1.0e-14

    u_0_error_norm = np.sqrt(abs(assemble(inner(u_0 - u_0_ref,
                                                u_0 - u_0_ref) * dx)))
    assert u_0_error_norm < 1.0e-14

    u_1_error_norm = np.sqrt(abs(assemble(inner(u_1 - u_1_ref,
                                                u_1 - u_1_ref) * dx)))
    assert u_1_error_norm < 1.0e-12


@pytest.mark.firedrake
def test_pressure_projection(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = VectorFunctionSpace(mesh, "Lagrange", 2)
    space_1 = FunctionSpace(mesh, "Lagrange", 1)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

    block_00 = inner(trial_0, test_0) * dx
    block_01 = inner(grad(trial_1), test_0) * dx
    block_10 = adjoint(block_01)
    block_11 = None

    system = System(
        (space_0, space_1), (space_0.dual(), space_1.dual()),
        {(0, 0): block_00, (0, 1): block_01,
         (1, 0): block_10, (1, 1): block_11},
        nullspaces=(DirichletBCNullspace(DirichletBC(space_0, 0.0, "on_boundary")),  # noqa: E501
                    ConstantNullspace()))

    psi_0_ref = Function(space_1)
    psi_0_ref.interpolate(X[0] * exp(X[1]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))  # noqa: E501

    u_s = Function(space_0)
    solve(inner(trial_0, test_0) * dx
          == inner(ufl.perp(grad(psi_0_ref)), test_0) * dx,
          u_s, DirichletBC(space_0, (0.0, 0.0), "on_boundary"),
          solver_parameters={"ksp_type": "cg",
                             "pc_type": "jacobi",
                             "ksp_atol": 1.0e-14,
                             "ksp_rtol": 1.0e-14})

    u_0 = Function(space_0)
    u_1 = Function(space_1)

    b_0 = assemble(inner(u_s, test_0) * dx)
    b_1 = None

    _ = system.solve(
        (u_0, u_1), (b_0, b_1),
        solver_parameters={"linear_solver": "minres",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    u_0_bc_error_norm = np.sqrt(abs(assemble(inner(u_0, u_0) * ds)))
    assert u_0_bc_error_norm == 0.0

    with u_1.dat.vec_ro as u_1_v:
        u_1_sum = u_1_v.sum()
    assert abs(u_1_sum) < 1.0e-13

    u_div = Function(space_1)
    solve(inner(trial_1, test_1) * dx == -action(block_10, u_0),
          u_div,
          solver_parameters={"ksp_type": "cg",
                             "pc_type": "jacobi",
                             "ksp_atol": 1.0e-14,
                             "ksp_rtol": 1.0e-14})

    u_div_error_norm = np.sqrt(abs(assemble(inner(u_div,
                                                  u_div) * dx)))
    assert u_div_error_norm < 1.0e-12

    grad_u_1 = Function(space_0)
    solve(inner(trial_0, test_0) * dx
          == inner(grad(u_1), test_0) * dx,
          grad_u_1, DirichletBC(space_0, (0.0, 0.0), "on_boundary"),
          solver_parameters={"ksp_type": "cg",
                             "pc_type": "jacobi",
                             "ksp_atol": 1.0e-14,
                             "ksp_rtol": 1.0e-14})

    u_orthogonality_error_norm = abs(assemble(inner(grad_u_1, u_0) * dx))
    assert u_orthogonality_error_norm < 1.0e-14

    u_0_error_norm = np.sqrt(abs(assemble(inner(u_s - u_0 - grad_u_1,
                                                u_s - u_0 - grad_u_1) * dx)))
    assert u_0_error_norm < 1.0e-12


@pytest.mark.firedrake
def test_mass(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    bc = DirichletBC(space, 0.0, (1, 2, 3, 4))

    y = Function(space, name="y")
    y.interpolate(exp(X[0]) * sin(pi * X[0]) * sin(2.0 * pi * X[1]))

    u = Function(space, name="u")
    b = assemble(inner(y, test) * dx)

    system = System(
        space, space.dual(),
        inner(trial, test) * dx, nullspaces=DirichletBCNullspace(bc))

    _ = system.solve(
        u, b,
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    u_error_norm = np.sqrt(abs(assemble(inner(u - y, u - y) * dx)))
    assert u_error_norm < 1.0e-13


@pytest.mark.firedrake
def test_sub_block(setup_test):  # noqa: F811
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space_0 = FunctionSpace(mesh, "Lagrange", 1)
    space_1 = FunctionSpace(mesh, "Lagrange", 2)
    space_1 = FunctionSpace(mesh,
                            space_0.ufl_element() * space_1.ufl_element())
    space_2 = FunctionSpace(mesh, "Lagrange", 3)

    test_0, trial_0 = TestFunction(space_0), TrialFunction(space_0)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)
    test_2, trial_2 = TestFunction(space_2), TrialFunction(space_2)

    block_00 = BlockMatrix((space_0, space_1), (space_0, space_1))
    block_00[(0, 0)] = inner(trial_0, test_0) * dx
    block_00[(1, 1)] = inner(trial_1, test_1) * dx
    block_11 = inner(trial_2, test_2) * dx

    u_0_ref = Function(space_0)
    u_0_ref.interpolate(X[0] * exp(X[1]))
    u_1_ref = Function(space_1)
    u_1_ref.sub(0).interpolate(X[1] * exp(X[0]))
    u_1_ref.sub(1).interpolate(X[0] * X[1] * sin(2.0 * pi * X[1]))
    u_2_ref = Function(space_2)
    u_2_ref.interpolate(sin(3.0 * pi * X[0]) * sin(4.0 * pi * X[1]))

    b_0 = assemble(inner(u_0_ref, test_0) * dx)
    b_1 = assemble(inner(u_1_ref, test_1) * dx)
    b_2 = assemble(inner(u_2_ref, test_2) * dx)

    nullspace_2 = DirichletBCNullspace(
        DirichletBC(space_2, 0.0, "on_boundary"))

    system = System(
        ((space_0, space_1), space_2), ((space_0.dual(), space_1.dual()), space_2.dual()),  # noqa: E501
        {(0, 0): block_00, (1, 1): block_11},
        nullspaces=(None, nullspace_2))

    u_0 = Function(space_0)
    u_1 = Function(space_1)
    u_2 = Function(space_2)

    def pc_fn(u, b):
        (u_0, u_1), u_2 = u
        (b_0, b_1), b_2 = b
        assert u_0.function_space() == space_0
        assert u_1.function_space() == space_1
        assert u_2.function_space() == space_2
        assert b_0.function_space() == space_0.dual()
        assert b_1.function_space() == space_1.dual()
        assert b_2.function_space() == space_2.dual()
        u_0.assign(b_0.riesz_representation("l2"))
        u_1.assign(b_1.riesz_representation("l2"))
        u_2.assign(b_2.riesz_representation("l2"))

    _ = system.solve(
        ((u_0, u_1), u_2), ((b_0, b_1), b_2),
        pc_fn=pc_fn,
        solver_parameters={"linear_solver": "cg",
                           "relative_tolerance": 1.0e-14,
                           "absolute_tolerance": 1.0e-14})

    u_2_bc_error_norm = np.sqrt(abs(assemble(inner(u_2, u_2) * ds)))
    assert u_2_bc_error_norm == 0.0

    u_0_error_norm = np.sqrt(abs(assemble(inner(u_0 - u_0_ref,
                                                u_0 - u_0_ref) * dx)))
    assert u_0_error_norm < 1.0e-13

    u_1_error_norm = np.sqrt(abs(assemble(inner(u_1 - u_1_ref,
                                                u_1 - u_1_ref) * dx)))
    assert u_1_error_norm < 1.0e-12

    u_2_error_norm = np.sqrt(abs(assemble(inner(u_2 - u_2_ref,
                                                u_2 - u_2_ref) * dx)))
    assert u_2_error_norm < 1.0e-12
