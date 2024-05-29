"""Interface with PyTorch.

Can be used to embed models, differentiated by tlm_adjoint, within a PyTorch
calculation. Follows the same principles as described in

    - Nacime Bouziani and David A. Ham, 'Physics-driven machine learning models
      coupling PyTorch and Firedrake', 2023, arXiv:2303.06871v3
"""

from .caches import clear_caches
from .interface import (
    Packed, is_var, var_comm, var_dtype, var_get_values, var_id, var_new,
    var_new_conjugate_dual, var_set_values)
from .manager import (
    compute_gradient, manager as _manager, reset_manager, restore_manager,
    set_manager, start_manager, stop_manager)
from .markers import AdjointActionMarker
from .overloaded_float import Float

try:
    import torch
except ModuleNotFoundError:
    torch = None

__all__ = \
    [
        "to_torch_tensors",
        "from_torch_tensors",
        "torch_wrapped"
    ]


def to_torch_tensor(x, *args, **kwargs):
    return torch.tensor(var_get_values(x), *args, **kwargs)


def to_torch_tensors(X, *args, **kwargs):
    """Convert one or more variables to :class:`torch.Tensor` objects.

    :arg X: A variable or :class:`Sequence` or variables.
    :returns: A :class:`torch.Tensor` or :class:`tuple` of
        :class:`torch.Tensor` objects.

    Remaining arguments are passed to :func:`torch.tensor`.
    """

    X = Packed(X)
    X_t = tuple(to_torch_tensor(x, *args, **kwargs) for x in X)
    return X.unpack(X_t)


def from_torch_tensor(x, x_t):
    var_set_values(x, x_t.detach().numpy())
    return x


def from_torch_tensors(X, X_t):
    """Copy data from PyTorch tensors into variables.

    :arg X: A variable or :class:`Sequence` or variables.
    :arg X_t: A :class:`torch.Tensor` or :class:`Sequence` of
        :class:`torch.Tensor` objects.
    """

    X = Packed(X)
    if X.is_packed:
        X_t = (X_t,)
    if len(X) != len(X_t):
        raise ValueError("Invalid length")
    for x, x_t in zip(X, X_t):
        from_torch_tensor(x, x_t)


@restore_manager
def _forward(forward, M, manager):
    set_manager(manager)
    reset_manager()
    clear_caches()

    start_manager()
    X = forward(*M)
    X_packed = Packed(X)
    X = tuple(X_packed)
    J = Float(dtype=var_dtype(X[0]), comm=var_comm(X[0]))
    adj_X = tuple(map(var_new_conjugate_dual, X))
    AdjointActionMarker(J, X, adj_X).solve()
    stop_manager()

    return X_packed.unpack(X), J, X_packed.unpack(adj_X)


class TorchInterface(object if torch is None else torch.autograd.Function):
    @staticmethod
    def forward(ctx, forward, manager, J_id, M, *M_t):
        M = tuple(map(var_new, M))
        from_torch_tensors(M, M_t)

        X, J, adj_X = _forward(forward, M, manager)

        J_id[0] = var_id(J)
        ctx._tlm_adjoint__output_ctx = (forward, manager, J_id, M, J, adj_X)
        return to_torch_tensors(X)

    @staticmethod
    @restore_manager
    def backward(ctx, *adj_X_t):
        forward, manager, J_id, M, J, adj_X = ctx._tlm_adjoint__output_ctx
        if var_id(J) != J_id[0] or manager._cp_schedule.is_exhausted:
            _, J, adj_X = _forward(forward, M, manager)
            J_id[0] = var_id(J)

        from_torch_tensors(adj_X, adj_X_t)
        set_manager(manager)
        dJ = compute_gradient(J, M)

        return (None, None, None, None) + to_torch_tensors(dJ)


def torch_wrapped(forward, M, *, manager=None):
    """Wrap a model, differentiated using tlm_adjoint, so that it can be used
    with PyTorch.

    :arg forward: A callable which accepts one or more variable arguments, and
        returns a variable or :class:`Sequence` of variables.
    :arg M: A variable or :class:`Sequence` of variables defining the input to
        `forward`.
    :arg manager: An :class:`.EquationManager` used to create an internal
        manager via :meth:`.EquationManager.new`. `manager()` is used if not
        supplied.
    :returns: A :class:`tuple` `(M_t, forward_t, X_t)`, where

            - `M_t` is a :class:`torch.Tensor` storing the value of `M`.
            - `forward_t` is a version of `forward` with :class:`torch.Tensor`
              inputs and outputs.
            - `X_t` is a :class:`torch.Tensor` containing the value of
              `forward` evaluated with `M` as input.
    """

    if is_var(M):
        M = (M,)
    if manager is None:
        manager = _manager()
    manager = manager.new()
    J_id = [None]

    M_t = to_torch_tensors(M, requires_grad=True)

    def forward_t(*M_t):
        return TorchInterface.apply(forward, manager, J_id, M, *M_t)

    return M_t, forward_t, forward_t(*M_t)
