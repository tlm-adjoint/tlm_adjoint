#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *
from tlm_adjoint.fenics import *

from .test_base import *

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
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


@pytest.mark.fenics
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

    F_0 = Function(F, 0)
    assert space_id(var_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(var_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())

    F_0 = F.split()[0]
    assert space_id(var_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(var_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())

    F_0 = F.split(deepcopy=True)[0]
    assert space_id(var_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(var_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())
    Function(F_0.function_space())


@pytest.mark.fenics
@seed_test
def test_default_var_flags(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)

    # Constant, without domain
    c = Constant(0.0)
    assert var_is_static(c) is not None and not var_is_static(c)
    assert var_is_cached(c) is not None and not var_is_cached(c)
    assert var_is_checkpointed(c) is not None and var_is_checkpointed(c)
    del c

    # Constant, with domain
    c = Constant(0.0, domain=mesh)
    assert var_is_static(c) is not None and not var_is_static(c)
    assert var_is_cached(c) is not None and not var_is_cached(c)
    assert var_is_checkpointed(c) is not None and var_is_checkpointed(c)
    del c

    # Function
    F = Function(space)
    assert var_is_static(F) is not None and not var_is_static(F)
    assert var_is_cached(F) is not None and not var_is_cached(F)
    assert var_is_checkpointed(F) is not None and var_is_checkpointed(F)
    del F
