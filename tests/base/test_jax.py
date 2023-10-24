#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    DEFAULT_COMM, DotProduct, Float, Hessian, Vector, VectorEquation,
    compute_gradient, new_jax_float, set_default_float_dtype,
    set_default_jax_dtype, start_manager, stop_manager, taylor_test,
    taylor_test_tlm, taylor_test_tlm_adjoint, var_get_values, var_global_size,
    var_is_scalar, var_linf_norm, var_local_size, var_scalar_value, var_sum)

from .test_base import jax_tlm_config, seed_test, setup_test  # noqa: F401

try:
    import jax
except ImportError:
    jax = None
import numpy as np
import operator
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")
pytestmark = pytest.mark.skipif(
    jax is None,
    reason="JAX not available")


@pytest.mark.base
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@seed_test
def test_jax_assignment(setup_test, jax_tlm_config,  # noqa: F811
                        dtype):
    set_default_float_dtype(dtype)
    set_default_jax_dtype(dtype)

    def fn(y):
        return y.copy()

    def forward(y):
        x = Vector(y.space, name="x")
        VectorEquation(x, y, fn=fn).solve()

        c = Float(name="c")
        e = Vector(y.space, name="e", space_type="dual").assign(
            np.ones(x.space.local_size, dtype=x.space.dtype))
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    y = Vector(10, dtype=dtype)
    if issubclass(dtype, (complex, np.complexfloating)):
        y.assign(np.arange(1, 11, dtype=dtype)
                 + 1.0j * np.arange(11, 12, dtype=dtype))
    else:
        y.assign(np.arange(1, 11, dtype=dtype))

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = complex(J)

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, y, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2)
    assert min_order > 2.00


@pytest.mark.base
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@pytest.mark.parametrize("op", [operator.abs,
                                operator.neg,
                                np.sin,
                                np.cos,
                                np.tan,
                                np.arcsin,
                                np.arccos,
                                np.arctan,
                                np.sinh,
                                np.cosh,
                                np.tanh,
                                np.arcsinh,
                                np.arccosh,
                                np.arctanh,
                                np.exp,
                                np.exp2,
                                np.expm1,
                                np.log,
                                np.log2,
                                np.log10,
                                np.log1p,
                                np.sqrt,
                                np.square,
                                np.cbrt,
                                np.reciprocal])
@seed_test
def test_jax_unary_overloading(setup_test, jax_tlm_config,  # noqa: F811
                               op):
    set_default_float_dtype(np.double)
    set_default_jax_dtype(np.double)

    def forward(y):
        x = op(y)
        assert abs(x.vector - op(y.vector)).max() < 1.0e-15

        c = Float(name="c")
        e = Vector(y.space, name="e", space_type="dual").assign(
            np.ones(x.space.local_size, dtype=x.space.dtype))
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    if op in {np.arccosh, np.reciprocal}:
        y = np.array([1.1, 1.2], dtype=np.double)
    else:
        y = np.array([0.1, 0.2], dtype=np.double)
    y = Vector(y)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = float(J)

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 2.93

    min_order = taylor_test_tlm(forward, y, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.base
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@pytest.mark.parametrize("op", [operator.add,
                                operator.sub,
                                operator.mul,
                                operator.truediv,
                                operator.pow,
                                np.arctan2,
                                np.hypot])
@seed_test
def test_jax_binary_overloading(setup_test, jax_tlm_config,  # noqa: F811
                                dtype, op):
    if op in {np.arctan2, np.hypot} \
            and issubclass(dtype, (complex, np.complexfloating)):
        pytest.skip()
    set_default_float_dtype(dtype)
    set_default_jax_dtype(dtype)

    def forward(y):
        x = y * y
        x = op(x, y)
        assert abs(x.vector - op(y.vector ** 2, y.vector)).max() < 1.0e-16

        c = Float(name="c")
        e = Vector(y.space, name="e", space_type="dual").assign(
            np.ones(x.space.local_size, dtype=x.space.dtype))
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    if op is np.arccosh:
        y = np.array([1.1, 1.2], dtype=dtype)
    else:
        y = np.array([0.1, 0.2], dtype=dtype)
    if issubclass(dtype, (complex, np.complexfloating)):
        y += np.array([0.4j, 0.5j], dtype=dtype)
    y = Vector(y)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = complex(J)

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ, seed=1.0e-3)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ, seed=1.0e-3)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, y, tlm_order=1, seed=1.0e-3)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1,
                                        seed=1.0e-3)
    assert min_order > 1.98

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2,
                                        seed=1.0e-3)
    assert min_order > 1.99


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@pytest.mark.parametrize("x_val", [-np.sqrt(2.0),
                                   np.sqrt(3.0),
                                   np.sqrt(5.0) + 1.0j * np.sqrt(7.0)])
@seed_test
def test_jax_float(setup_test, jax_tlm_config,  # noqa: F811
                   dtype, x_val):
    if isinstance(x_val, (complex, np.complexfloating)) \
            and not issubclass(dtype, (complex, np.complexfloating)):
        pytest.skip()
    set_default_jax_dtype(dtype)

    x = new_jax_float().assign(x_val)

    if not issubclass(dtype, (complex, np.complexfloating)):
        assert float(x) == x_val
    assert complex(x) == x_val
    assert var_sum(x) == x_val
    assert abs(var_linf_norm(x) - abs(x_val)) < 1.0e-15
    assert var_local_size(x) == (1 if DEFAULT_COMM.rank == 0 else 0)
    assert var_global_size(x) == 1
    assert (var_get_values(x) == np.array([x_val] if DEFAULT_COMM.rank == 0
                                          else [], dtype=x.space.dtype)).all()
    assert var_is_scalar(x)
    assert var_scalar_value(x) == x_val
