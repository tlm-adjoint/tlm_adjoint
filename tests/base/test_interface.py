#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    DEFAULT_COMM, Float, VariableStateLockDictionary, Vector, var_lock_state,
    var_name)
from tlm_adjoint.interface import (
    var_decrement_state_lock, var_increment_state_lock, var_state_is_locked)

from .test_base import seed_test, setup_test  # noqa: F401

import itertools
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.base
@pytest.mark.parametrize(
    "cls",
    [lambda name: Float(name=name),
     lambda name: Vector(1, name=name)])
@seed_test
def test_name(setup_test,  # noqa: F811
              cls):
    name = "_tlm_adjoint__test_name"
    F = cls(name=name)
    assert var_name(F) == name


@pytest.mark.base
@seed_test
def test_state_lock(setup_test):  # noqa: F811
    class Test:
        pass

    f = Float(1.0)
    assert not var_state_is_locked(f)

    # Increment 10 times with the same object ...
    t = Test()
    assert not var_state_is_locked(f)
    for _ in range(10):
        var_increment_state_lock(f, t)
        assert var_state_is_locked(f)
    # ... then decrement 10 times with the same object
    for _ in range(9):
        var_decrement_state_lock(f, t)
        assert var_state_is_locked(f)
    var_decrement_state_lock(f, t)
    assert not var_state_is_locked(f)

    # Increment 10 times with the same object ...
    t = Test()
    assert not var_state_is_locked(f)
    for _ in range(10):
        var_increment_state_lock(f, t)
        assert var_state_is_locked(f)
    # ... then destroy the object
    del t
    assert not var_state_is_locked(f)

    # Increment 10 times each with 10 different objects ...
    assert not var_state_is_locked(f)
    T = [Test() for _ in range(10)]
    for t, _ in itertools.product(T, range(10)):
        var_increment_state_lock(f, t)
        assert var_state_is_locked(f)
    # ... then destroy the objects
    t = None
    assert var_state_is_locked(f)
    while len(T) > 1:
        T.pop()
        assert var_state_is_locked(f)
    T.pop()
    assert not var_state_is_locked(f)

    # Increment 10 times using itself ...
    assert not var_state_is_locked(f)
    for _ in range(10):
        var_increment_state_lock(f, f)
        assert var_state_is_locked(f)
    # ... then decrement 10 times using itself
    for _ in range(9):
        var_decrement_state_lock(f, f)
        assert var_state_is_locked(f)
    var_decrement_state_lock(f, f)
    assert not var_state_is_locked(f)

    # Lock
    assert not var_state_is_locked(f)
    var_lock_state(f)
    assert var_state_is_locked(f)


@pytest.mark.base
@seed_test
def test_VariableStateLockDictionary(setup_test):  # noqa: F811
    f = Float(1.0)
    assert not var_state_is_locked(f)

    def test_setitem(d, key, value, s):
        assert len(d) == s
        assert key not in d
        try:
            d[key]
            assert False
        except KeyError:
            pass

        d[key] = value

        assert len(d) == s + 1
        assert key in d
        assert d[key] is value

    def test_delitem(d, key, value, delfn, s):
        assert len(d) == s
        assert key in d
        assert d[key] is value

        delfn(d, key)

        assert len(d) == s - 1
        assert key not in d
        try:
            d[key]
            assert False
        except KeyError:
            pass

    def delfn_del(d, key):
        del d[key]

    def delfn_pop(d, key):
        d.pop(key)

    def test_replaceitem(d, key, oldvalue, newvalue, s):
        assert len(d) == s
        assert key in d
        assert d[key] is oldvalue

        d[key] = newvalue

        assert len(d) == s
        assert key in d
        assert d[key] is newvalue

    keys = list(itertools.chain.from_iterable(
        ([i, (i,), str(i)]) for i in range(10)))

    # Add items, delete items with del and pop
    for delfn in (delfn_del, delfn_pop):
        d = VariableStateLockDictionary()
        assert not var_state_is_locked(f)
        for i, key in enumerate(keys):
            test_setitem(d, key, f, i)
            assert var_state_is_locked(f)
        for i, key in enumerate(keys[1:]):
            test_delitem(d, key, f, delfn, len(keys) - i)
            assert var_state_is_locked(f)
        test_delitem(d, keys[0], f, delfn, 1)
        assert not var_state_is_locked(f)

    # Add items, replace items with None
    d = VariableStateLockDictionary()
    assert not var_state_is_locked(f)
    for i, key in enumerate(keys):
        test_setitem(d, key, f, i)
        assert var_state_is_locked(f)
    for key in keys[1:]:
        test_replaceitem(d, key, f, None, len(keys))
        assert var_state_is_locked(f)
    test_replaceitem(d, keys[0], f, None, len(keys))
    assert not var_state_is_locked(f)

    # Add items, clear
    d = VariableStateLockDictionary()
    assert not var_state_is_locked(f)
    for i, key in enumerate(keys):
        test_setitem(d, key, f, i)
        assert var_state_is_locked(f)
    d.clear()
    assert len(d) == 0
    assert not var_state_is_locked(f)

    # Add items, destroy the VariableStateLockDictionary
    d = VariableStateLockDictionary()
    assert not var_state_is_locked(f)
    for i, key in enumerate(keys):
        test_setitem(d, key, f, i)
        assert var_state_is_locked(f)
    del d
    assert not var_state_is_locked(f)
