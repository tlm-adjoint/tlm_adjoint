"""Interface with PyTorch.

Can be used to embed models, differentiated by tlm_adjoint, within a PyTorch
calculation. Follows the same principles as described in

    - Nacime Bouziani and David A. Ham, 'Physics-driven machine learning models
      coupling PyTorch and Firedrake', 2023, arXiv:2303.06871v3
"""

from .caches import clear_caches as _clear_caches
from .interface import (
    Packed, packed, space_new, var_comm, var_dtype, var_get_values, var_id,
    var_locked, var_new_conjugate_dual, var_set_values)
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


def to_torch_tensor(x, *args, conjugate=False, **kwargs):
    x_a = var_get_values(x)
    if conjugate:
        x_a = x_a.conjugate()
    return torch.tensor(x_a, *args, **kwargs)


def to_torch_tensors(X, *args, conjugate=False, **kwargs):
    """Convert one or more variables to :class:`torch.Tensor` objects.

    Parameters
    ----------

    X : variable or Sequence[variable, ...]
        Variables to be converted.
    conjugate : bool
        Whether to copy the complex conjugate.
    args, kwargs
        Passed to :func:`torch.tensor`.

    Returns
    -------

    tuple[variable, ...]
        The converted variables.
    """

    return tuple(to_torch_tensor(x, *args, conjugate=conjugate, **kwargs)
                 for x in packed(X))


def from_torch_tensor(x, x_t, *, conjugate=False):
    x_a = x_t.detach().numpy()
    if conjugate:
        x_a = x_a.conjugate()
    var_set_values(x, x_a)
    return x


def from_torch_tensors(X, X_t, *, conjugate=False):
    """Copy data from PyTorch tensors into variables.

    Parameters
    ----------

    X : variable or Sequence[variable, ...]
        Output.
    X_t : Sequence[:class:`torch.Tensor`, ...]
        Input.
    conjugate : bool
        Whether to copy the complex conjugate.
    """

    X = packed(X)
    if len(X) != len(X_t):
        raise ValueError("Invalid length")
    for x, x_t in zip(X, X_t):
        from_torch_tensor(x, x_t, conjugate=conjugate)


@restore_manager
def _forward(forward, M, manager, *, clear_caches=False):
    set_manager(manager)
    reset_manager()
    if clear_caches:
        _clear_caches()

    start_manager()
    with var_locked(*M):
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
    def forward(ctx, forward, manager, clear_caches, J_id, space, *M_t):
        M = tuple(map(space_new, space))
        from_torch_tensors(M, M_t)

        X, J, adj_X = _forward(forward, M, manager,
                               clear_caches=clear_caches)

        J_id[0] = var_id(J)
        ctx._tlm_adjoint__output_ctx = (forward, manager, clear_caches,
                                        J_id, M, J, adj_X)
        return to_torch_tensors(X)

    @staticmethod
    @restore_manager
    def backward(ctx, *adj_X_t):
        (forward, manager, clear_caches,
         J_id, M, J, adj_X) = ctx._tlm_adjoint__output_ctx
        if var_id(J) != J_id[0] or manager._cp_schedule.is_exhausted:
            _, J, adj_X = _forward(forward, M, manager,
                                   clear_caches=clear_caches)
            J_id[0] = var_id(J)

        from_torch_tensors(adj_X, adj_X_t, conjugate=True)
        set_manager(manager)
        dJ = compute_gradient(J, M)

        return ((None, None, None, None, None)
                + to_torch_tensors(dJ, conjugate=True))


def torch_wrapped(forward, space, *, manager=None, clear_caches=True):
    """Wrap a model, differentiated using tlm_adjoint, so that it can be used
    with PyTorch.

    Parameters
    ----------

    forward : callable
        Accepts one or more variable arguments, and returns a variable or
        :class:`Sequence` of variables.
    space : space or Sequence[space, ...]
        Defines the spaces for input arguments.
    manager : :class:`.EquationManager`
        Used to create an internal manager via :meth:`.EquationManager.new`.
        `manager()` is used if not supplied.
    clear_caches : bool
        Whether to clear caches before a call of `forward`.

    Returns
    -------

    callable
        A version of `forward` with :class:`torch.Tensor` inputs and outputs.
    """

    space = packed(space)
    if manager is None:
        manager = _manager()
    manager = manager.new()

    J_id = [None]

    def forward_t(*M_t):
        return TorchInterface.apply(
            forward, manager, clear_caches, J_id, space, *M_t)

    return forward_t
