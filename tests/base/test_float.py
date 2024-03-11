from tlm_adjoint import (
    DEFAULT_COMM, DotProduct, Float, FloatEquation, Hessian, compute_gradient,
    set_default_float_dtype, start_manager, stop_manager, taylor_test,
    taylor_test_tlm, taylor_test_tlm_adjoint)

from .test_base import seed_test, setup_test  # noqa: F401

import numpy as np
import operator
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.base
@pytest.mark.parametrize("value", [2, 3.0, 4.0 + 5.0j])
@seed_test
def test_Float_new(setup_test,  # noqa: F811
                   value):
    set_default_float_dtype(np.cdouble)

    x = Float(name="x")
    assert complex(x) == 0.0

    y = x.new(value)
    assert complex(x) == 0.0
    assert complex(y) == value


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@seed_test
def test_Float_assignment(setup_test,  # noqa: F811
                          dtype):
    set_default_float_dtype(dtype)

    def forward(y):
        x = Float(name="x")
        FloatEquation(x, y).solve()

        c = Float(name="c")
        e = Float(name="e", space_type="dual").assign(1.0)
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    if issubclass(dtype, np.complexfloating):
        y = Float(2.0 + 3.0j)
    else:
        y = Float(2.0)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = complex(J)
    if issubclass(dtype, np.floating):
        dm = Float(1.0)
    else:
        dm = None

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ,
                            dM=None if dm is None else dm)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ,
                            dM=None if dm is None else dm)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, y, tlm_order=1,
                                dMs=None if dm is None else (dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1,
                                        dMs=None if dm is None else (dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2,
                                        dMs=None if dm is None else (dm, dm))
    assert min_order > 2.00


@pytest.mark.base
@seed_test
def test_Float_self_assignment(setup_test):  # noqa: F811
    set_default_float_dtype(np.double)

    def forward(y):
        y.assign(y)
        x = Float(name="x")
        x.assign(x + y)
        return x ** 4

    y = Float(2.0)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)
    assert abs(float(dJ) - 4 * float(y) ** 3) == 0.0


@pytest.mark.base
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
def test_Float_unary_overloading(setup_test,  # noqa: F811
                                 op):
    set_default_float_dtype(np.double)

    def forward(y):
        x = op(y)
        assert abs(float(x) - op(float(y))) < 1.0e-15

        c = Float(name="c")
        e = Float(name="e", space_type="dual").assign(1.0)
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    if op in {np.arccosh, np.reciprocal}:
        y = Float(1.1)
    else:
        y = Float(0.1)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = float(J)
    dm = Float(1.0)

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ, seed=1.0e-3, dM=dm)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ, seed=1.0e-3,
                            dM=dm)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, y, tlm_order=1, seed=1.0e-3,
                                dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1,
                                        seed=1.0e-3, dMs=(dm,))
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2,
                                        seed=1.0e-3, dMs=(dm, dm))
    assert min_order > 1.99


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@pytest.mark.parametrize("op", [operator.add,
                                operator.sub,
                                operator.mul,
                                operator.truediv,
                                operator.pow,
                                np.arctan2,
                                np.hypot])
@seed_test
def test_Float_binary_overloading(setup_test,  # noqa: F811
                                  dtype, op):
    if op in {np.arctan2, np.hypot} \
            and issubclass(dtype, np.complexfloating):
        pytest.skip()
    set_default_float_dtype(dtype)

    def forward(y):
        x = y * y
        x = op(x, y)
        if issubclass(dtype, np.complexfloating):
            assert abs(complex(x) - op(complex(y) ** 2, complex(y))) < 1.0e-16
        else:
            assert abs(float(x) - op(float(y) ** 2, float(y))) < 1.0e-16

        c = Float(name="c")
        e = Float(name="e", space_type="dual").assign(1.0)
        DotProduct(c, x, e).solve()

        return (c - 1.0) ** 4

    if op is np.arccosh:
        y = 1.0
    else:
        y = 0.1
    if issubclass(dtype, np.complexfloating):
        y += 0.4j
    y = Float(y)

    start_manager()
    J = forward(y)
    stop_manager()

    dJ = compute_gradient(J, y)

    J_val = complex(J)
    if issubclass(dtype, np.floating):
        dm = Float(1.0)
    else:
        dm = None

    min_order = taylor_test(forward, y, J_val=J_val, dJ=dJ, seed=1.0e-3,
                            dM=None if dm is None else dm)
    assert min_order > 1.98

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, y, J_val=J_val, ddJ=ddJ, seed=1.0e-3,
                            dM=None if dm is None else dm)
    assert min_order > 2.94

    min_order = taylor_test_tlm(forward, y, tlm_order=1, seed=1.0e-3,
                                dMs=None if dm is None else (dm,))
    assert min_order > 1.98

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=1,
                                        seed=1.0e-3,
                                        dMs=None if dm is None else (dm,))
    assert min_order > 1.98

    min_order = taylor_test_tlm_adjoint(forward, y, adjoint_order=2,
                                        seed=1.0e-3,
                                        dMs=None if dm is None else (dm, dm))
    assert min_order > 1.99
