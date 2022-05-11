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

from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_code_generator_interface import \
    function_vector

from .test_base import *

import mpi4py.MPI as MPI
import numpy as np
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.size not in [1, 4],
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_clear_caches(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = Function(space, name="F", cache=True)

    def cache_item(F):
        form = inner(F, TestFunction(F.function_space())) * dx
        cached_form, _ = assembly_cache().assemble(form)
        return cached_form

    def test_not_cleared(F, cached_form):
        assert len(assembly_cache()) == 1
        assert cached_form() is not None
        assert len(function_caches(F)) == 1

    def test_cleared(F, cached_form):
        assert len(assembly_cache()) == 0
        assert cached_form() is None
        assert len(function_caches(F)) == 0

    assert len(assembly_cache()) == 0

    # Clear default
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    clear_caches()
    test_cleared(F, cached_form)

    # Clear Function caches
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    clear_caches(F)
    test_cleared(F, cached_form)

    # Clear on state update
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    function_update_state(F)
    test_cleared(F, cached_form)

    # Clear on cache update, new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    function_update_caches(F, value=Function(space))
    test_cleared(F, cached_form)

    # Clear on cache update, replacement with new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    function_update_caches(function_replacement(F), value=Function(space))
    test_cleared(F, cached_form)


@pytest.mark.firedrake
@pytest.mark.parametrize("non_static_term", [True, False])
@pytest.mark.parametrize("static_bc", [None, True, False])
@seed_test
def test_cached_rhs(setup_test, test_leaks,
                    non_static_term, static_bc):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)

    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    static_1 = Function(space_1, name="static_1", static=True)
    static_2 = Function(space_2, name="static_2", static=True)
    non_static_1 = Function(space_1, name="non_static_1")
    non_static_2a = Function(space_2, name="non_static_2a")
    non_static_2b = Function(space_2, name="non_static_2b")

    interpolate_expression(static_1, sin(pi * X[0]) * sin(2 * pi * X[1]))
    interpolate_expression(static_2, exp(X[0] * X[1]))
    interpolate_expression(non_static_1, sqrt(1.0 + X[0] * X[0]))
    interpolate_expression(non_static_2a, sqrt(1.0 + X[1] * X[1]))
    interpolate_expression(non_static_2b, 1 / (1 + X[0] * X[0] + X[1] * X[1]))

    bc = DirichletBC(space_1, 1.0, "on_boundary", static=static_bc)

    b = (
        # Static
        inner(static_1, test_1) * dx
        # Static
        + inner(static_1 * static_2, test_1) * dx
        # Static matrix action
        + inner(non_static_1 * static_2, test_1) * dx
        # Static matrix action
        + inner(non_static_2a, test_1) * dx
        # Static matrix actions
        + inner((non_static_1
                 + non_static_2a + non_static_2b), test_1) * dx
        # Static matrix action
        + inner(non_static_2b, test_1) * dx)
    if non_static_term:
        b += inner(non_static_1 * non_static_2a, test_1) * dx

    F = Function(space_1, name="F")
    F_ref = Function(space_1, name="F_ref")
    error = Function(space_1, name="error")

    caches = (assembly_cache(), linear_solver_cache(), local_solver_cache())

    assert tuple(len(cache) for cache in caches) == (0, 0, 0)

    eq = EquationSolver(inner(trial_1, test_1) * dx == b, F, bc,
                        solver_parameters=ls_parameters_cg)
    eq.solve()

    if static_bc in [None, True]:
        assert tuple(len(cache) for cache in caches) == (4, 1, 0)
        assert eq._cache_jacobian
    else:
        assert tuple(len(cache) for cache in caches) == (3, 0, 0)
        assert not eq._cache_jacobian
    assert eq._forward_b_pa[0] is not None
    assert len(eq._forward_b_pa[1]) == 3
    if non_static_term:
        assert eq._forward_b_pa[2] is not None
    else:
        assert eq._forward_b_pa[2] is None

    solve(inner(trial_1, test_1) * dx == b, F_ref, bc,
          solver_parameters=ls_parameters_cg)

    function_assign(error, F_ref)
    function_axpy(error, -1.0, F)
    error_norm = function_linf_norm(error)
    info(f"Error norm = {error_norm:.16e}")
    assert error_norm < 1.0e-13


@pytest.mark.firedrake
@pytest.mark.parametrize("static_control", [True, False])
@seed_test
def test_cached_adjoint(setup_test, test_leaks,
                        static_control):
    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)

    space_1 = FunctionSpace(mesh, "Lagrange", 1)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)

    alpha = Constant(1.0, name="alpha", static=True)
    beta = Function(space_2, name="beta", static=True)
    function_assign(beta, 1.0)
    bc = DirichletBC(space_1, 1.0, "on_boundary")

    def forward(G):
        F = Function(space_1, name="F")
        eq = EquationSolver(
            inner(trial_1, test_1) * dx
            == inner(alpha * beta * G, test_1) * dx, F, bc,
            solver_parameters=ls_parameters_cg)
        eq.solve()

        J = Functional(name="J")
        J.assign(dot(F, F) * dx)
        return J

    G = Function(space_2, name="G", static=static_control)
    interpolate_expression(G, exp(X[0]))

    caches = (assembly_cache(), linear_solver_cache(), local_solver_cache())

    assert tuple(len(cache) for cache in caches) == (0, 0, 0)

    start_manager()
    J = forward(G)
    stop_manager()

    assert tuple(len(cache) for cache in caches) == (2, 1, 0)

    dJ = compute_gradient(J, G)

    assert tuple(len(cache) for cache in caches) == (4, 2, 0)

    assert len(manager()._block) == 0
    ((eq, _),) = manager()._blocks
    adjoint_action, = tuple(eq._adjoint_action_cache.values())
    assert isinstance(adjoint_action, CacheRef)
    assert adjoint_action() is not None
    assert isinstance(eq._adjoint_J_solver, CacheRef)
    assert eq._adjoint_J_solver() is not None

    min_order = taylor_test(forward, G, J_val=J.value(), dJ=dJ)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.parametrize("x_conjugate", [False, True])
@seed_test
def test_mat_terms(setup_test, test_leaks,
                   x_conjugate):
    from tlm_adjoint.firedrake.backend_code_generator_interface import \
        assemble_matrix, matrix_multiply

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)

    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    x = Function(space, name="x")
    x_expr = exp(X[0]) * X[1]
    if issubclass(function_dtype(x), (complex, np.complexfloating)):
        x_expr += X[0] * sin(X[1]) * 1.j
    interpolate_expression(x, x_expr)

    form = inner(ufl.conj(x) if x_conjugate else x, test) * dx

    b_ref = Function(space, name="b_ref", space_type="conjugate_dual")
    assemble(form, tensor=function_vector(b_ref))

    if not complex_mode:
        form = ufl.algorithms.remove_complex_nodes.remove_complex_nodes(form)
    cached_terms, mat_terms, non_cached_terms = split_form(form)

    assert cached_terms.empty()
    if not complex_mode or not x_conjugate:
        assert len(mat_terms) == 1
        assert non_cached_terms.empty()

        assert tuple(mat_terms.keys()) == (function_id(x),)
        A, = tuple(mat_terms.values())
        A, b_bc = assemble_matrix(A)
        b = Function(space, name="b", space_type="conjugate_dual")
        matrix_multiply(A, function_vector(x), tensor=function_vector(b))
        assert b_bc is None
    else:
        assert len(mat_terms) == 0
        assert not non_cached_terms.empty()

        b = Function(space, name="b", space_type="conjugate_dual")
        assemble(non_cached_terms, tensor=function_vector(b))

    b_error = function_copy(b_ref)
    function_axpy(b_error, -1.0, b)
    assert function_linf_norm(b_error) < 1.0e-17
