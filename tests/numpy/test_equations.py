#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.numpy import *

from .test_base import *

import numpy as np
import pytest

try:
    import mpi4py.MPI as MPI
    pytestmark = pytest.mark.skipif(
        MPI.COMM_WORLD.size != 1, reason="serial only")
except ImportError:
    pass


@pytest.mark.numpy
@no_space_type_checking
@seed_test
def test_Assignment(setup_test, test_leaks, test_default_dtypes):
    x = Constant(16.0, name="x", static=True)

    def forward(x):
        y = [Constant(name=f"y_{i:d}") for i in range(9)]
        z = Constant(name="z")

        Assignment(y[0], x).solve()
        for i in range(len(y) - 1):
            Assignment(y[i + 1], y[i]).solve()
        # Following line should have no effect on sensitivity
        DotProduct(z, y[-1], y[-1]).solve()
        DotProduct(z, y[-1], y[-1]).solve()

        x_dot_x = Constant(name="x_dot_x")
        DotProduct(x_dot_x, x, x).solve()

        z_dot_z = Constant(name="z_dot_z")
        DotProduct(z_dot_z, z, z).solve()

        J = Functional(name="J")
        Axpy(J.function(), z_dot_z, 2.0, x_dot_x).solve()

        K = Functional(name="K")
        Assignment(K.function(), z_dot_z).solve()

        return J, K

    start_manager()
    J, K = forward(x)
    stop_manager()

    assert abs(J.value() - 66048.0) == 0.0
    assert abs(K.value() - 65536.0) == 0.0

    dJs = compute_gradient([J, K], x)

    dm = Constant(1.0, name="dm", static=True)

    for forward_J, J_val, dJ in [(lambda x: forward(x)[0], J.value(), dJs[0]),
                                 (lambda x: forward(x)[1], K.value(), dJs[1])]:
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


@pytest.mark.numpy
@no_space_type_checking
@seed_test
def test_Axpy(setup_test, test_leaks, test_default_dtypes):
    x = Constant(1.0, name="x", static=True)

    def forward(x):
        y = [Constant(name=f"y_{i:d}") for i in range(5)]
        z = [Constant(name=f"z_{i:d}") for i in range(2)]
        z[0].assign(7.0)

        Assignment(y[0], x).solve()
        for i in range(len(y) - 1):
            Axpy(y[i + 1], y[i], i + 1, z[0]).solve()
        DotProduct(z[1], y[-1], y[-1]).solve()

        J = Functional(name="J")
        DotProduct(J.function(), z[1], z[1]).solve()
        return J

    start_manager()
    J = forward(x)
    stop_manager()

    J_val = J.value()
    assert abs(J_val - 25411681.0) == 0.0

    dJ = compute_gradient(J, x)

    dm = Constant(1.0, name="dm", static=True)

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


@pytest.mark.numpy
@no_space_type_checking
@seed_test
def test_InnerProduct(setup_test, test_leaks):
    space = FunctionSpace(10)

    def forward(F):
        G = Function(space, name="G")
        Assignment(G, F).solve()

        J = Functional(name="J")
        InnerProduct(J.function(), F, G).solve()
        return J

    F = Function(space, name="F", static=True)
    F_arr = np.random.random(function_local_size(F))
    if issubclass(function_dtype(F), (complex, np.complexfloating)):
        F_arr = F_arr + 1.0j * np.random.random(function_local_size(F))
    function_set_values(F, F_arr)
    del F_arr

    start_manager()
    J = forward(F)
    stop_manager()

    dJ = compute_gradient(J, F)
    min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
    assert min_order > 1.99


@pytest.mark.numpy
@no_space_type_checking
@seed_test
def test_Contraction(setup_test, test_leaks, test_default_dtypes):
    dtype = default_dtype()

    space = FunctionSpace(3)
    if issubclass(dtype, (complex, np.complexfloating)):
        A = np.array([[1.0 + 10.0j, 2.0 + 11.0j, 3.0 + 12.0j],
                      [0.0, 4.0 + 13.0j, 5.0 + 14.0j],
                      [0.0, 0.0, 6.0 + 15.0j]],
                     dtype=dtype)
    else:
        A = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
                     dtype=dtype)

    def forward(m):
        x = Function(space, name="x")
        Contraction(x, A, (1,), (m,)).solve()

        x_dot_x = Constant(name="x_dot_x")
        DotProduct(x_dot_x, x, x).solve()

        J = Functional(name="J")
        DotProduct(J.function(), x_dot_x, x_dot_x).solve()
        return x, J

    m = Function(space, name="m", static=True)
    if issubclass(dtype, (complex, np.complexfloating)):
        function_set_values(
            m, np.array([7.0 + 16.0j, 8.0 + 17.0j, 9.0 + 18.0j], dtype=dtype))
    else:
        function_set_values(
            m, np.array([7.0, 8.0, 9.0], dtype=dtype))

    start_manager()
    x, J = forward(m)
    stop_manager()

    A_action = A.dot(m.vector())
    assert abs(A_action - x.vector()).max() == 0.0
    assert abs(A_action.dot(A_action) ** 2 - J.value()) == 0.0

    J_val = J.value()

    dJ = compute_gradient(J, m)

    def forward_J(m):
        return forward(m)[1]

    min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward_J)
    min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward_J, m, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward_J, m, adjoint_order=2)
    assert min_order > 2.00
