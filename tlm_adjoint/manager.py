"""This module provides a simple :class:`.EquationManager` interface. Functions
defined here access and interact with the default manager.
"""

import functools
import warnings

__all__ = \
    [
        "annotation_enabled",
        "compute_gradient",
        "configure_checkpointing",
        "configure_tlm",
        "manager",
        "manager_disabled",
        "manager_info",
        "paused_manager",
        "new_block",
        "reset_manager",
        "restore_manager",
        "set_manager",
        "start_manager",
        "stop_manager",
        "tlm_enabled",
        "var_tlm",

        "add_tlm",
        "function_tlm"
    ]

_manager = None


def manager():
    """
    :returns: An :class:`.EquationManager`, the current default manager.
    """

    return _manager


def set_manager(manager):
    """Set the default manager.

    :arg manager: An :class:`.EquationManager` to use as the default manager.
    """

    global _manager
    _manager = manager


def restore_manager(fn):
    """Decorator to revert the default manager to the manager used prior to
    calling the decorated callable. A typical use is

    .. code-block:: python

        @restore_manager
        def decorated(*M):
            set_manager(manager().new())
            forward(*M)

    :arg fn: A decorated callable.
    :returns: A callable, where the default manager on entry is restored on
        exit.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        old_manager = manager()
        try:
            return fn(*args, **kwargs)
        finally:
            set_manager(old_manager)
    return wrapped_fn


def configure_checkpointing(cp_method, cp_parameters):
    """See :meth:`.EquationManager.configure_checkpointing`.
    """

    manager().configure_checkpointing(cp_method, cp_parameters=cp_parameters)


def manager_info(*, info=print):
    """See :meth:`.EquationManager.info`.
    """

    manager().info(info=info)


def reset_manager(cp_method=None, cp_parameters=None):
    """See :meth:`.EquationManager.reset`.
    """

    manager().reset(cp_method=cp_method, cp_parameters=cp_parameters)


def annotation_enabled():
    """See :meth:`.EquationManager.annotation_enabled`.
    """

    return manager().annotation_enabled()


def start_manager(*, annotate=True, tlm=True):
    """See :meth:`.EquationManager.start`.
    """

    manager().start(annotate=annotate, tlm=tlm)


def stop_manager(*, annotate=True, tlm=True):
    """See :meth:`.EquationManager.stop`.
    """

    return manager().stop(annotate=annotate, tlm=tlm)


def paused_manager(*, annotate=True, tlm=True):
    """See :meth:`.EquationManager.paused`.
    """

    return manager().paused(annotate=annotate, tlm=tlm)


def manager_disabled(*, annotate=True, tlm=True):
    """Decorator which can be used to disable processing of equations and
    derivation and solution of tangent-linear equations.

    :arg annotate: Whether to disable processing of equations.
    :arg tlm: Whether to disable derivation and solution of tangent-linear
        equations.
    """

    def wrapper(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            with paused_manager(annotate=annotate, tlm=tlm):
                return fn(*args, **kwargs)

        return wrapped_fn

    return wrapper


def configure_tlm(*args, annotate=None, tlm=True):
    """See :meth:`.EquationManager.configure_tlm`.
    """

    manager().configure_tlm(*args, annotate=annotate, tlm=tlm)


def add_tlm(M, dM, max_depth=1):
    warnings.warn("add_tlm is deprecated -- "
                  "use configure_tlm instead",
                  FutureWarning, stacklevel=2)
    manager().add_tlm(M, dM, max_depth=max_depth, _warning=False)


def tlm_enabled():
    """See :meth:`.EquationManager.tlm_enabled`.
    """

    return manager().tlm_enabled()


def var_tlm(x, *args):
    """See :meth:`.EquationManager.var_tlm`.
    """

    return manager().var_tlm(x, *args)


def function_tlm(x, *args):
    ""

    warnings.warn("function_tlm is deprecated -- "
                  "use var_tlm instead",
                  FutureWarning, stacklevel=2)
    return manager().var_tlm(x, *args)


def compute_gradient(Js, M, *, callback=None, prune_forward=True,
                     prune_adjoint=True, prune_replay=True,
                     cache_adjoint_degree=None, store_adjoint=False,
                     adj_ics=None):
    """See :meth:`.EquationManager.compute_gradient`.
    """

    return manager().compute_gradient(
        Js, M, callback=callback, prune_forward=prune_forward,
        prune_adjoint=prune_adjoint, prune_replay=prune_replay,
        cache_adjoint_degree=cache_adjoint_degree, store_adjoint=store_adjoint,
        adj_ics=adj_ics)


def new_block():
    """See :meth:`.EquationManager.new_block`.
    """

    manager().new_block()
