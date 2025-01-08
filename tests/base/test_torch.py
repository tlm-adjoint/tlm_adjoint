from tlm_adjoint import (
    DEFAULT_COMM, Float, from_torch_tensors, set_default_float_dtype,
    to_torch_tensors, torch_wrapped, var_id)

from .test_base import setup_test  # noqa: F401

import numpy as np
import pytest
try:
    import torch
except ModuleNotFoundError:
    torch = None

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")
pytestmark = pytest.mark.skipif(
    torch is None,
    reason="PyTorch not available")


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
def test_torch_tensor_roundtrip(setup_test,  # noqa: F811
                                dtype):
    set_default_float_dtype(dtype)

    if issubclass(dtype, np.complexfloating):
        x = Float(-np.sqrt(2.0) + 1.0j * np.sqrt(3.0))
    else:
        x = Float(-np.sqrt(2.0))
    y = Float()
    from_torch_tensors(y, to_torch_tensors(x))
    assert abs(complex(x) - complex(y)) == 0.0


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
def test_torch_wrapped(setup_test,  # noqa: F811
                       dtype):
    set_default_float_dtype(dtype)

    if issubclass(dtype, np.complexfloating):
        m = Float(-np.sqrt(2.0) + 1.0j * np.sqrt(3.0))
    else:
        m = Float(-np.sqrt(2.0))
    x = Float()

    def forward(m):
        return Float(m)

    x_t = torch_wrapped(forward, m.space)(*to_torch_tensors(m))
    from_torch_tensors(x, x_t)

    assert x is not m
    assert var_id(x) != var_id(m)
    assert abs(complex(x) - complex(m)) == 0.0


@pytest.mark.base
@pytest.mark.parametrize("dtype", [np.double, np.cdouble])
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
def test_torch_vjp(setup_test,  # noqa: F811
                   dtype):
    set_default_float_dtype(dtype)

    if issubclass(dtype, np.complexfloating):
        m = Float(-np.sqrt(2.0) + 1.0j * np.sqrt(3.0))
    else:
        m = Float(-np.sqrt(2.0))
    J = Float(name="J")

    def forward(m):
        return m ** 4

    J_ref = complex(m) ** 4
    forward_t = torch_wrapped(forward, m.space)
    J_t = forward_t(*to_torch_tensors(m))
    from_torch_tensors(J, J_t)
    assert abs(complex(J) - complex(J_ref)) < 1.0e-15

    if issubclass(dtype, np.complexfloating):
        m0 = Float(1.0 - 1.0j)
    else:
        m0 = Float(1.0)
    m0_t = to_torch_tensors(m0, requires_grad=True)

    assert torch.autograd.gradcheck(forward_t, m0_t, eps=1.0e-8,
                                    atol=1.0e-8, rtol=1.0e-7)
