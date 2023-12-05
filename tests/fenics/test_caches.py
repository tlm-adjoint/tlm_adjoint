from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics.caches import split_form
from tlm_adjoint.fenics.functions import bcs_is_static

from .test_base import *

import numpy as np
import pytest
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
@seed_test
def test_clear_caches(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = Function(space, name="F", cache=True)

    def cache_item(F, F_value=None):
        form = inner(F, TestFunction(var_space(F))) * dx
        cached_form, _ = assembly_cache().assemble(
            form, replace_map=None if F_value is None else {F: F_value})
        return cached_form

    def test_not_cleared(F, cached_form, F_value=None):
        assert len(assembly_cache()) == 1
        assert cached_form() is not None
        assert len(var_caches(F)) == 1
        if F_value is not None:
            assert len(var_caches(F_value)) == 1

    def test_cleared(F, cached_form, F_value=None):
        assert len(assembly_cache()) == 0
        assert cached_form() is None
        assert len(var_caches(F)) == 0
        if F_value is not None:
            assert len(var_caches(F_value)) == 0

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
    var_update_state(F)
    test_cleared(F, cached_form)

    # Clear on cache update, new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    var_update_caches(F, value=Function(space))
    test_cleared(F, cached_form)

    # Clear on cache update, replacement with new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    var_update_caches(var_replacement(F), value=Function(space))
    test_cleared(F, cached_form)

    F_value = Function(space, name="F_value")

    # Clear on state update of value
    cached_form = cache_item(F, F_value)
    test_not_cleared(F, cached_form, F_value)
    var_update_state(F_value)
    test_cleared(F, cached_form, F_value)

    # Clear on state update of value, replacement
    cached_form = cache_item(var_replacement(F), F_value)
    test_not_cleared(F, cached_form, F_value)
    var_update_state(F_value)
    test_cleared(F, cached_form, F_value)


@pytest.mark.fenics
@seed_test
def test_static_DirichletBC(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    assert bcs_is_static([DirichletBC(space, 0.0,
                                      "on_boundary")])
    assert bcs_is_static([DirichletBC(space, Function(space, static=True),
                                      "on_boundary")])
    assert not bcs_is_static([DirichletBC(space, Function(space, static=False),
                                          "on_boundary")])


@pytest.mark.fenics
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

    if static_bc in {None, True}:
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

    var_assign(error, F_ref)
    var_axpy(error, -1.0, F)
    error_norm = var_linf_norm(error)
    info(f"Error norm = {error_norm:.16e}")
    assert error_norm < 1.0e-12


@pytest.mark.fenics
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
    beta.interpolate(Constant(1.0))
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

    min_order = taylor_test(forward, G, J_val=J.value, dJ=dJ)
    assert min_order > 1.99


@pytest.mark.fenics
@pytest.mark.parametrize("x_conjugate", [False, True])
@seed_test
def test_mat_terms(setup_test, test_leaks,
                   x_conjugate):
    from tlm_adjoint.fenics.backend_code_generator_interface import (
        assemble_matrix, matrix_multiply)

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)

    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)

    x = Function(space, name="x")
    x_expr = exp(X[0]) * X[1]
    if issubclass(var_dtype(x), np.complexfloating):
        x_expr = x_expr + X[0] * sin(X[1]) * 1.0j
    interpolate_expression(x, x_expr)

    form = inner(ufl.conj(x) if x_conjugate else x, test) * dx

    b_ref = Function(space, name="b_ref", space_type="conjugate_dual")
    assemble(form, tensor=b_ref)

    cached_terms, mat_terms, non_cached_terms = split_form(form)

    assert cached_terms.empty()
    if not complex_mode or not x_conjugate:
        assert len(mat_terms) == 1
        assert non_cached_terms.empty()

        assert tuple(mat_terms.keys()) == (var_id(x),)
        A, = tuple(mat_terms.values())
        A, b_bc = assemble_matrix(A)
        b = Function(space, name="b", space_type="conjugate_dual")
        matrix_multiply(A, x, tensor=b)
        assert b_bc is None
    else:
        assert len(mat_terms) == 0
        assert not non_cached_terms.empty()

        b = Function(space, name="b", space_type="conjugate_dual")
        assemble(non_cached_terms, tensor=b)

    b_error = var_copy(b_ref)
    var_axpy(b_error, -1.0, b)
    assert var_linf_norm(b_error) < 1.0e-16


@pytest.mark.fenics
@seed_test
def test_mass_adjoint_caching(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)

    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)

    M = inner(trial, test) * dx

    cache_ref_0, _ = assembly_cache().assemble(M)
    cache_ref_1, _ = assembly_cache().assemble(adjoint(M))

    assert len(assembly_cache()) == 1
    assert cache_ref_0 is cache_ref_1


@pytest.mark.fenics
@seed_test
def test_stiffness_adjoint_caching(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)

    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)

    K = inner(grad(trial), grad(test)) * dx

    cache_ref_0, _ = assembly_cache().assemble(K)
    cache_ref_1, _ = assembly_cache().assemble(adjoint(K))

    assert len(assembly_cache()) == 1
    assert cache_ref_0 is cache_ref_1
