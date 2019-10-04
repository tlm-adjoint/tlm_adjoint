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
from tlm_adjoint_firedrake import *

from test_base import *

import pytest


@pytest.mark.firedrake
def test_clear_caches(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = Function(space, name="F", cache=True)

    def cache_item(F):
        form = inner(TestFunction(F.function_space()), F) * dx
        cached_form, _ = assembly_cache().assemble(form)
        return cached_form

    def test_not_cleared(F, cached_form):
        assert(len(assembly_cache()) == 1)
        assert(cached_form() is not None)
        assert(len(function_caches(F)) == 1)

    def test_cleared(F, cached_form):
        assert(len(assembly_cache()) == 0)
        assert(cached_form() is None)
        assert(len(function_caches(F)) == 0)

    assert(len(assembly_cache()) == 0)

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

    # Clear on cache update
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    function_update_state(F)
    test_not_cleared(F, cached_form)
    update_caches([F])
    test_cleared(F, cached_form)

    # Clear on cache update, new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([F], [Function(space)])
    test_cleared(F, cached_form)

    # Clear on cache update, Replacement
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([function_replacement(F)])
    test_cleared(F, cached_form)

    # Clear on cache update, Replacement with new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([function_replacement(F)], [Function(space)])
    test_cleared(F, cached_form)


@pytest.mark.firedrake
@pytest.mark.parametrize("non_static_term", [True, False])
@pytest.mark.parametrize("static_bc", [True, False])
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
        inner(test_1, static_1) * dx
        # Static
        + inner(test_1, static_1 * static_2) * dx
        # Static matrix action
        + inner(test_1, non_static_1 * static_2) * dx
        # Static matrix action
        + inner(test_1, non_static_2a) * dx
        # Static matrix actions
        + inner(test_1, (non_static_1
                         + non_static_2a + non_static_2b)) * dx
        # Static matrix action
        + inner(test_1, non_static_2b) * dx)
    if non_static_term:
        b += inner(test_1, non_static_1 * non_static_2a) * dx

    F = Function(space_1, name="F")
    F_ref = Function(space_1, name="F_ref")
    error = Function(space_1, name="error")

    caches = (assembly_cache(), linear_solver_cache(), local_solver_cache())

    assert(tuple(len(cache) for cache in caches) == (0, 0, 0))

    eq = EquationSolver(inner(test_1, trial_1) * dx == b, F, bc,
                        solver_parameters=ls_parameters_cg)
    eq.solve()

    if static_bc:
        assert(tuple(len(cache) for cache in caches) == (4, 1, 0))
        assert(eq._cache_jacobian)
    else:
        assert(tuple(len(cache) for cache in caches) == (3, 0, 0))
        assert(not eq._cache_jacobian)
    assert(eq._forward_b_pa[0] is not None)
    assert(len(eq._forward_b_pa[1]) == 3)
    if non_static_term:
        assert(eq._forward_b_pa[2] is not None)
    else:
        assert(eq._forward_b_pa[2] is None)

    solve(inner(test_1, trial_1) * dx == b, F_ref, bc,
          solver_parameters=ls_parameters_cg)

    function_assign(error, F_ref)
    function_axpy(error, -1.0, F)
    error_norm = function_linf_norm(error)
    info(f"Error norm = {error_norm:.16e}")
    assert(error_norm < 1.0e-13)
