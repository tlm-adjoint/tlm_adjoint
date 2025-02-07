from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.fixture(params=[{"cls": lambda **kwargs: Constant(**kwargs)},
                        {"cls": lambda **kwargs: Function(FunctionSpace(UnitIntervalMesh(20), "Lagrange", 1), **kwargs)},  # noqa: E501
                        {"cls": lambda **kwargs: Cofunction(FunctionSpace(UnitIntervalMesh(20), "Lagrange", 1).dual(), **kwargs)}])  # noqa: E501
def var_cls(request):
    return request.param["cls"]


@pytest.mark.firedrake
@seed_test
def test_name(setup_test,
              var_cls):
    name = "_tlm_adjoint__test_name"
    F = var_cls(name=name)
    assert var_name(F) == name


@pytest.mark.firedrake
@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("cache", [False, True, None])
@seed_test
def test_replacement(setup_test,
                     var_cls, cache, static):
    name = "_tlm_adjoint__test_name"
    F = var_cls(name=name, static=static, cache=cache)
    F_id = var_id(F)
    F_caches = var_caches(F)

    for var in (F, var_replacement(F)):
        assert var_id(var) == F_id
        assert var_name(var) == name
        assert var_is_static(var) is not None
        assert var_is_static(var) == static
        assert var_is_cached(var) is not None
        assert var_is_cached(var) == (static if cache is None else cache)
        assert var_caches(var) is F_caches


@pytest.mark.firedrake
@seed_test
def test_replacement_eq_hash(setup_test,
                             var_cls):
    F = var_cls()
    F_replacement = var_replacement(F)

    assert F == F
    assert not (F != F)
    assert F_replacement == F_replacement
    assert not (F_replacement != F_replacement)

    assert F != F_replacement
    assert F_replacement != F
    assert not (F == F_replacement)
    assert not (F_replacement == F)

    assert F.count() != F_replacement.count()
    assert hash(F) != hash(F_replacement)
    assert len(set((F, F_replacement))) == 2


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
    assert space_id(space) == space_id(var_space(F_0))
    assert space_id(space) == space_id(F_0.function_space())


@pytest.mark.firedrake
@seed_test
def test_FunctionSpace_space_default_space_type(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)

    assert space_default_space_type(space) == "primal"
    assert space_default_space_type(space.dual()) == "conjugate_dual"


@pytest.mark.firedrake
@seed_test
def test_FunctionSpace_space_id(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space0 = FunctionSpace(mesh, "Lagrange", 1)
    space1 = FunctionSpace(mesh, "Lagrange", 1)
    space2 = FunctionSpace(mesh, "Discontinuous Lagrange", 1)

    assert space_id(space0) == space_id(space1)
    assert space_id(space0) != space_id(space0.dual())
    assert space_id(space0.dual()) == space_id(space1.dual())
    assert space_id(space0) != space_id(space2)
    assert space_id(space0) != space_id(space2.dual())
    assert space_id(space0.dual()) != space_id(space2.dual())


@pytest.mark.firedrake
@pytest.mark.parametrize("cls", [Function,
                                 lambda space, *args, **kwargs: Cofunction(space.dual(), *args, **kwargs)])  # noqa: E501
@pytest.mark.parametrize("dim", [1, 2, 3, 5])
@seed_test
def test_var_alias(setup_test, test_leaks,
                   cls, dim):
    mesh = UnitIntervalMesh(20)

    space = VectorFunctionSpace(mesh, "Lagrange", 1, dim=dim)

    F = cls(space, name="F")
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)

    F = cls(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert dim == 1 or var_is_alias(F_i)
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)

    space = FunctionSpace(mesh, "Lagrange", 1)
    space = FunctionSpace(mesh, MixedElement(
        *[space.ufl_element() for _ in range(dim)]))

    def test_state(F, F_i):
        state_i = var_state(F_i)
        with F.dat.vec as F_v:
            F_v.shift(1.0)
        assert var_state(F_i) > state_i

        state = var_state(F)
        with F_i.dat.vec as F_i_v:
            F_i_v.shift(1.0)
        assert var_state(F) > state

    F = cls(space, name="F")
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
        test_state(F, F_i)
    for i in range(dim):
        F_i = F.sub(i)
        assert var_is_alias(F_i)
        test_state(F, F_i)

    F = cls(space, name="F")
    for i in range(dim):
        F_i = F.sub(i)
        assert var_is_alias(F_i)
        test_state(F, F_i)
    for F_i in F.subfunctions:
        assert var_is_alias(F_i)
        test_state(F, F_i)


@pytest.mark.firedrake
@seed_test
def test_default_var_flags(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)

    # Constant
    c = Constant(0.0)
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
