#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

import pytest
import ufl

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_FunctionSpace_interface(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=2)
    F = Function(space, name="F")

    assert space_id(space) == space_id(function_space(F))
    assert space_id(space) == space_id(F.function_space())

    F_copy = function_copy(F)
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_copy = F.copy(deepcopy=True)
    assert space_id(space) == space_id(function_space(F_copy))
    assert space_id(space) == space_id(F_copy.function_space())

    F_0 = F.subfunctions[0]
    assert space_id(function_space(F_0)) == space_id(F_0.function_space())
    assert space_id(space) != space_id(function_space(F_0))
    assert space_id(space) != space_id(F_0.function_space())


@pytest.mark.firedrake
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@seed_test
def test_function_alias(setup_test, test_leaks,
                        dim):
    mesh = UnitIntervalMesh(20)

    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=dim)

    F = Function(space, name="F")
    for F_i in F.subfunctions:
        assert function_is_alias(F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or function_is_alias(F_i)

    F = Function(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or function_is_alias(F_i)
    for F_i in F.subfunctions:
        assert function_is_alias(F_i)

    space = FunctionSpace(mesh, "Lagrange", 1)
    space = FunctionSpace(mesh, ufl.classes.MixedElement(
        *[space.ufl_element() for _ in range(dim)]))

    F = Function(space, name="F")
    for F_i in F.subfunctions:
        assert function_is_alias(F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or function_is_alias(F_i)

    F = Function(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or function_is_alias(F_i)
    for F_i in F.subfunctions:
        assert function_is_alias(F_i)


@pytest.mark.firedrake
@seed_test
def test_default_function_flags(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)

    # Constant, without domain
    c = Constant(0.0)
    assert function_is_static(c) is not None and not function_is_static(c)
    assert function_is_cached(c) is not None and not function_is_cached(c)
    assert function_is_checkpointed(c) is not None and function_is_checkpointed(c)  # noqa: E501
    del c

    # Constant, with domain
    c = Constant(0.0, domain=mesh)
    assert function_is_static(c) is not None and not function_is_static(c)
    assert function_is_cached(c) is not None and not function_is_cached(c)
    assert function_is_checkpointed(c) is not None and function_is_checkpointed(c)  # noqa: E501
    del c

    # Function
    F = Function(space)
    assert function_is_static(F) is not None and not function_is_static(F)
    assert function_is_cached(F) is not None and not function_is_cached(F)
    assert function_is_checkpointed(F) is not None and function_is_checkpointed(F)  # noqa: E501
    del F
