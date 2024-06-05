from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

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


@pytest.mark.firedrake
@seed_test
def test_torch_tensor_roundtrip(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space1 = FunctionSpace(mesh, "Lagrange", 1)
    space2 = FunctionSpace(mesh, "Lagrange", 2)

    u = Function(space1).interpolate(exp(X[0]))
    v = Function(space2).interpolate(sin(pi * X[1]))
    c = Constant(sqrt(2.0))

    for x in u, v, c:
        y = var_new(x)
        from_torch_tensors(y, to_torch_tensors(x))

        err = var_copy(x)
        var_axpy(err, -1.0, y)
        assert var_linf_norm(err) == 0.0


@pytest.mark.firedrake
@seed_test
def test_torch_wrapped(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    m = Function(space).interpolate(X[0])
    x = Function(space)

    def forward(m):
        return m.copy(deepcopy=True)

    x_t = torch_wrapped(forward, m.function_space())(*to_torch_tensors(m))
    from_torch_tensors(x, x_t)

    err = var_copy(x)
    var_axpy(err, -1.0, m)
    assert var_linf_norm(err) == 0.0


@pytest.mark.firedrake
@pytest.mark.skipif(DEFAULT_COMM.size > 1, reason="serial only")
@seed_test
def test_torch_vjp(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)

    if complex_mode:
        m = Function(space).interpolate(X[0] + 1.0j * X[1])
    else:
        m = Function(space).interpolate(X[0])
    J = Float(name="J")

    def forward(m):
        J = Functional(name="J")
        J.assign((m ** 4) * dx)
        return J

    J_ref = assemble((m ** 4) * dx)
    forward_t = torch_wrapped(forward, m.function_space())
    J_t = forward_t(*to_torch_tensors(m))
    from_torch_tensors(J, J_t)
    assert abs(complex(J) - complex(J_ref)) == 0.0

    dm = Function(space, name="dm").interpolate(Constant(1.0))
    dm_t = to_torch_tensors(dm, requires_grad=True)

    assert torch.autograd.gradcheck(forward_t, dm_t, eps=1.0e-8,
                                    atol=1.0e-8, rtol=1.0e-7)
