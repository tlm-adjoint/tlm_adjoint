#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.alias import WeakAlias
from tlm_adjoint.interface import VariableStateChangeError
from tlm_adjoint.firedrake import *

from .test_base import *

import gc
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@pytest.mark.parametrize(
    "cls",
    [lambda name: Float(name=name),
     lambda name: Constant(name=name),
     lambda name: Constant(domain=UnitIntervalMesh(20), name=name),
     lambda name: Function(FunctionSpace(UnitIntervalMesh(20), "Lagrange", 1),
                           name=name)])
@seed_test
def test_name(setup_test,
              cls):
    name = "_tlm_adjoint__test_name"
    F = cls(name=name)
    assert var_name(F) == name


@pytest.mark.firedrake
@seed_test
def test_FunctionSpace_interface(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    F = Function(space, name="F")

    assert space_id(space) == space_id(var_space(F))
    assert space_id(space) == space_id(F.function_space())

    F_copy = var_copy(F)
    assert space_id(space) == space_id(var_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_copy = F.copy(deepcopy=True)
    assert space_id(space) == space_id(var_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_0 = F.subfunctions[0]
    assert space_id(var_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(var_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())


@pytest.mark.firedrake
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@seed_test
def test_Function_alias(setup_test, test_leaks,
                        dim):
    mesh = UnitIntervalMesh(20)

    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=dim)

    F = Function(space, name="F")
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)

    F = Function(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)

    space = FunctionSpace(mesh, "Lagrange", 1)
    space = FunctionSpace(mesh, ufl.classes.MixedElement(
        *[space.ufl_element() for _ in range(dim)]))

    def test_state(F, F_i):
        state = var_state(F_i)
        assert var_state(F_i) == var_state(F)
        var_update_state(F)
        assert var_state(F_i) > state
        assert var_state(F_i) == var_state(F)

        state = var_state(F_i)
        assert var_state(F_i) == var_state(F)
        var_update_state(F_i)
        assert var_state(F_i) > state
        assert var_state(F_i) == var_state(F)

    F = Function(space, name="F")
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
        test_state(F, F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)
        test_state(F, F_i)

    F = Function(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)
        test_state(F, F_i)
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
        test_state(F, F_i)


@pytest.mark.firedrake
@seed_test
def test_default_var_flags(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)

    # Constant, without domain
    c = Constant(0.0)
    assert var_is_static(c) is not None and not var_is_static(c)
    assert var_is_cached(c) is not None and not var_is_cached(c)
    del c

    # Constant, with domain
    c = Constant(0.0, domain=mesh)
    assert var_is_static(c) is not None and not var_is_static(c)
    assert var_is_cached(c) is not None and not var_is_cached(c)
    del c

    # Function
    F = Function(space)
    assert var_is_static(F) is not None and not var_is_static(F)
    assert var_is_cached(F) is not None and not var_is_cached(F)
    del F


@pytest.mark.firedrake
@seed_test
def test_scalar_var(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "R", 0)

    def test_scalar(x, ref):
        val = var_is_scalar(x)
        assert isinstance(val, bool)
        assert val == ref

    test_scalar(Constant(0.0), True)
    test_scalar(Constant((0.0, 0.0)), False)
    test_scalar(Function(space), False)
    test_scalar(Cofunction(space.dual()), False)


@pytest.mark.firedrake
@seed_test
def test_DirichletBC_locking(setup_test, test_leaks):
    def new_bc(space_cls):
        mesh = UnitSquareMesh(10, 10)
        space = space_cls(mesh, "Lagrange", 1)
        bc_value = Function(space, name="bc_value")
        bc = DirichletBC(space, bc_value, "on_boundary", static=True)
        return space, bc

    def new_hbc(space_cls):
        mesh = UnitSquareMesh(10, 10)
        space = space_cls(mesh, "Lagrange", 1)
        bc = HomogeneousDirichletBC(space, "on_boundary")
        return space, bc

    for space_cls in (FunctionSpace,
                      VectorFunctionSpace,
                      TensorFunctionSpace):
        _, bc = new_bc(space_cls)
        try:
            bc.homogenize()
            assert False
        except VariableStateChangeError:
            pass
        del bc

        space, bc = new_bc(space_cls)
        try:
            bc.set_value(Function(space))
            assert False
        except VariableStateChangeError:
            pass
        del space, bc

        space, bc = new_bc(space_cls)
        try:
            bc.function_arg = Function(space)
            assert False
        except VariableStateChangeError:
            pass
        del space, bc

        _, bc = new_hbc(space_cls)
        bc.homogenize()
        del bc

    def new_eq():
        mesh = UnitSquareMesh(10, 10)
        space = FunctionSpace(mesh, "Lagrange", 1)
        bc_value = Function(space, name="bc_value")
        bc = DirichletBC(space, bc_value, "on_boundary")
        test, trial = TestFunction(space), TrialFunction(space)
        u = Function(space, name="u")
        eq = EquationSolver(inner(trial, test) * dx
                            == inner(Constant(1.0), test) * dx, u, bc)
        return space, bc, eq

    _, bc, eq = new_eq()
    try:
        bc.homogenize()
        assert False
    except VariableStateChangeError:
        pass
    del bc, eq

    _, bc, eq = new_eq()
    eq = WeakAlias(eq)
    gc.collect()
    garbage_cleanup()
    try:
        bc.homogenize()
        assert False
    except VariableStateChangeError:
        pass
    del bc, eq
