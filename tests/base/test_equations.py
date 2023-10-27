#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import (
    DEFAULT_COMM, Assignment, Axpy, Float, Functional, Hessian,
    compute_gradient, start_manager, stop_manager, taylor_test,
    taylor_test_tlm, taylor_test_tlm_adjoint)

from .test_base import seed_test, setup_test  # noqa: F401

import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.base
@seed_test
def test_Assignment(setup_test):  # noqa: F811
    x = Float(16.0, name="x")

    def forward(x):
        y = [Float(name=f"y_{i:d}") for i in range(9)]
        z = Float(name="z")

        Assignment(y[0], x).solve()
        for i in range(len(y) - 1):
            Assignment(y[i + 1], y[i]).solve()
        # Following line should have no effect on sensitivity
        z.assign(y[-1] ** 2)
        z.assign(y[-1] ** 2)

        x_sq = x * x
        z_sq = z * z

        J = Functional(name="J")
        J.assign(z_sq + 2.0 * x_sq)

        K = Functional(name="K")
        K.assign(z_sq)

        return J, K

    start_manager()
    J, K = forward(x)
    stop_manager()

    assert abs(J.value - 66048.0) == 0.0
    assert abs(K.value - 65536.0) == 0.0

    dJs = compute_gradient([J, K], x)

    dm = Float(1.0, name="dm")

    for forward_J, J_val, dJ in [(lambda x: forward(x)[0], J.value, dJs[0]),
                                 (lambda x: forward(x)[1], K.value, dJs[1])]:
        min_order = taylor_test(forward_J, x, J_val=J_val, dJ=dJ, dM=dm)
        assert min_order > 2.00

        ddJ = Hessian(forward_J)
        min_order = taylor_test(forward_J, x, J_val=J_val, ddJ=ddJ, dM=dm)
        assert min_order > 3.00

        min_order = taylor_test_tlm(forward_J, x, tlm_order=1, dMs=(dm,))
        assert min_order > 2.00

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=1,
                                            dMs=(dm,))
        assert min_order > 2.00

        min_order = taylor_test_tlm_adjoint(forward_J, x, adjoint_order=2,
                                            dMs=(dm, dm))
        assert min_order > 2.00


@pytest.mark.base
@seed_test
def test_Axpy(setup_test):  # noqa: F811
    x = Float(1.0, name="x")

    def forward(x):
        y = [Float(name=f"y_{i:d}") for i in range(5)]
        z = [Float(name=f"z_{i:d}") for i in range(2)]
        z[0].assign(7.0)

        Assignment(y[0], x).solve()
        for i in range(len(y) - 1):
            Axpy(y[i + 1], y[i], i + 1, z[0]).solve()
        z[1].assign(y[-1] * y[-1])

        J = Functional(name="J")
        J.assign(z[1] * z[1])
        return J

    start_manager()
    J = forward(x)
    stop_manager()

    J_val = J.value
    assert abs(J_val - 25411681.0) == 0.0

    dJ = compute_gradient(J, x)

    dm = Float(1.0, name="dm")

    min_order = taylor_test(forward, x, J_val=J_val, dJ=dJ, dM=dm)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, x, J_val=J_val, ddJ=ddJ, dM=dm,
                            seed=2.0e-2)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, x, tlm_order=1, dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, x, adjoint_order=1, dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, x, adjoint_order=2,
                                        dMs=(dm, dm))
    assert min_order > 2.00
