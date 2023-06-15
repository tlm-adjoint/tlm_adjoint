#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module provides a simple
:class:`tlm_adjoint.tlm_adjoint.EquationManager` interface. Functions defined
here access and interact with the default manager, or optionally a different
manager supplied via a `manager` argument.

Documentation provided here indicates the
:class:`tlm_adjoint.tlm_adjoint.EquationManager` methods where more complete
documentation can be found.
"""

import functools
import warnings

__all__ = \
    [
        "annotation_enabled",
        "compute_gradient",
        "configure_checkpointing",
        "configure_tlm",
        "function_tlm",
        "manager",
        "manager_info",
        "paused_manager",
        "new_block",
        "reset_manager",
        "restore_manager",
        "set_manager",
        "start_manager",
        "stop_manager",
        "tlm_enabled",

        "add_tlm",
        "tlm",
        "reset",
        "reset_adjoint",
        "start_annotating",
        "start_tlm",
        "stop_annotating",
        "stop_tlm"
    ]

_manager = None


def manager():
    """
    :returns: A :class:`tlm_adjoint.tlm_adjoint.EquationManager`, the current
        default manager.
    """

    return _manager


def set_manager(manager):
    """Set the default manager.

    :arg manager: A :class:`tlm_adjoint.tlm_adjoint.EquationManager` to use as
        the default manager.
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


def configure_checkpointing(cp_method, cp_parameters, *, manager=None):
    """See
    :meth:`tlm_adjoint.tlm_adjoint.EquationManager.configure_checkpointing`.
    """

    if cp_parameters is None:
        cp_parameters = {}
    if manager is None:
        manager = globals()["manager"]()
    manager.configure_checkpointing(cp_method, cp_parameters=cp_parameters)


def manager_info(*, info=print, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.info`.
    """

    if manager is None:
        manager = globals()["manager"]()
    manager.info(info=info)


def reset_manager(cp_method=None, cp_parameters=None, *, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.reset`.
    """

    if manager is None:
        manager = globals()["manager"]()
    manager.reset(cp_method=cp_method, cp_parameters=cp_parameters)


def reset(cp_method=None, cp_parameters=None, manager=None):
    warnings.warn("reset is deprecated -- use reset_manager instead",
                  DeprecationWarning, stacklevel=2)
    if manager is None:
        manager = globals()["manager"]()
    manager.reset(cp_method=cp_method, cp_parameters=cp_parameters)


def annotation_enabled(*, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.annotation_enabled`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.annotation_enabled()


def start_manager(*, annotate=True, tlm=True, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.start`.
    """

    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=annotate, tlm=tlm)


def start_annotating(manager=None):
    warnings.warn("start_annotating is deprecated -- "
                  "use start_manager instead",
                  DeprecationWarning, stacklevel=2)

    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=True, tlm=False)


def start_tlm(manager=None):
    warnings.warn("start_tlm is deprecated -- "
                  "use start_manager instead",
                  DeprecationWarning, stacklevel=2)

    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=False, tlm=True)


def stop_manager(*, annotate=True, tlm=True, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.stop`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=annotate, tlm=tlm)


def stop_annotating(manager=None):
    warnings.warn("stop_annotating is deprecated -- "
                  "use stop_manager instead",
                  DeprecationWarning, stacklevel=2)

    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=True, tlm=False)


def stop_tlm(manager=None):
    warnings.warn("stop_tlm is deprecated -- "
                  "use stop_manager instead",
                  DeprecationWarning, stacklevel=2)

    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=False, tlm=True)


def paused_manager(*, annotate=True, tlm=True, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.paused`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.paused(annotate=annotate, tlm=tlm)


def configure_tlm(*args, annotate=None, tlm=True, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.configure_tlm`.
    """

    if manager is None:
        manager = globals()["manager"]()
    manager.configure_tlm(*args, annotate=annotate, tlm=tlm)


def add_tlm(M, dM, max_depth=1, manager=None):
    warnings.warn("add_tlm is deprecated -- "
                  "use configure_tlm instead",
                  DeprecationWarning, stacklevel=2)
    if manager is None:
        manager = globals()["manager"]()
    manager.add_tlm(M, dM, max_depth=max_depth, _warning=False)


def tlm_enabled(*, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.tlm_enabled`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.tlm_enabled()


def function_tlm(x, *args, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.function_tlm`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.function_tlm(x, *args)


def tlm(M, dM, x, max_depth=1, manager=None):
    warnings.warn("tlm is deprecated -- "
                  "use function_tlm instead",
                  DeprecationWarning, stacklevel=2)
    if manager is None:
        manager = globals()["manager"]()
    return manager.tlm(M, dM, x, max_depth=max_depth, _warning=False)


def reset_adjoint(manager=None):
    warnings.warn("reset_adjoint is deprecated",
                  DeprecationWarning, stacklevel=2)
    if manager is None:
        manager = globals()["manager"]()
    manager.reset_adjoint(_warning=False)


def compute_gradient(Js, M, *, callback=None, prune_forward=True,
                     prune_adjoint=True, prune_replay=True,
                     cache_adjoint_degree=None, adj_ics=None, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.compute_gradient`.
    """

    if manager is None:
        manager = globals()["manager"]()
    return manager.compute_gradient(Js, M, callback=callback,
                                    prune_forward=prune_forward,
                                    prune_adjoint=prune_adjoint,
                                    prune_replay=prune_replay,
                                    cache_adjoint_degree=cache_adjoint_degree,
                                    adj_ics=adj_ics)


def new_block(*, manager=None):
    """See :meth:`tlm_adjoint.tlm_adjoint.EquationManager.new_block`.
    """

    if manager is None:
        manager = globals()["manager"]()
    manager.new_block()
