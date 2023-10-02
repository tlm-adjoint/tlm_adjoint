#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

try:
    import jax
except ImportError:
    jax = None
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@pytest.mark.skipif(jax is None, reason="JAX not available")
@seed_test
def test_jax_conversion(setup_test):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        m_0 = to_jax(m)
        m_1 = to_firedrake(m_0, space)
        assert np.sqrt(abs(assemble(inner(m - m_1, m - m_1) * dx))) == 0.0

        J = Float(name="J")
        Assembly(J, ((m_1 - Constant(1.0)) ** 4) * dx).solve()
        return J

    m = Function(space, name="m")
    interpolate_expression(m, X[0] * sin(pi * X[0]))

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = complex(J)

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ)
    assert min_order > 1.99

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, m, tlm_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1)
    assert min_order > 1.99

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2)
    assert min_order > 1.99


@pytest.mark.firedrake
@pytest.mark.skipif(jax is None, reason="JAX not available")
@seed_test
def test_jax_integration(setup_test):
    mesh = UnitIntervalMesh(20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    def forward(m):
        def fn(m):
            from jax.numpy import exp, sin
            x = sin(m)
            y = exp(m)
            return (x + y) * y

        m_0 = to_jax(m)
        m_1 = new_jax(m)
        call_jax(m_1, m_0, fn)
        m_2 = to_firedrake(m_1, space)

        m_0_a = var_get_values(m_0)
        assert abs(var_get_values(m_2)
                   - (np.sin(m_0_a) + np.exp(m_0_a)) * np.exp(m_0_a)).max() < 1.0e-14  # noqa: E501

        J = Float(name="J")
        Assembly(J, ((m_2 - Constant(1.0)) ** 4) * dx).solve()
        return J

    m = Function(space, name="m")
    interpolate_expression(m, X[0] * sin(pi * X[0]))

    start_manager()
    J = forward(m)
    stop_manager()

    J_val = complex(J)

    dJ = compute_gradient(J, m)

    min_order = taylor_test(forward, m, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, m, J_val=J_val, ddJ=ddJ)
    assert min_order > 3.00

    min_order = taylor_test_tlm(forward, m, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, m, adjoint_order=2)
    assert min_order > 2.00
