#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

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
        "start_annotating",
        "start_manager",
        "start_tlm",
        "stop_annotating",
        "stop_manager",
        "stop_tlm",
        "tlm_enabled",

        "add_tlm",
        "tlm",
        "reset",
        "reset_adjoint"
    ]

_manager = [None]


def manager():
    return _manager[0]


def set_manager(manager):
    _manager[0] = manager


def restore_manager(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        old_manager = manager()
        try:
            return fn(*args, **kwargs)
        finally:
            set_manager(old_manager)
    return wrapped_fn


def configure_checkpointing(cp_method, cp_parameters, manager=None):
    if cp_parameters is None:
        cp_parameters = {}
    if manager is None:
        manager = globals()["manager"]()
    manager.configure_checkpointing(cp_method, cp_parameters=cp_parameters)


def manager_info(info=print, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.info(info=info)


def reset_manager(cp_method=None, cp_parameters=None, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.reset(cp_method=cp_method, cp_parameters=cp_parameters)


def reset(cp_method=None, cp_parameters=None, manager=None):
    warnings.warn("reset is deprecated -- use reset_manager instead",
                  DeprecationWarning, stacklevel=2)
    if manager is None:
        manager = globals()["manager"]()
    manager.reset(cp_method=cp_method, cp_parameters=cp_parameters)


def annotation_enabled(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.annotation_enabled()


def start_manager(*, annotate=True, tlm=True, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=annotate, tlm=tlm)


def start_annotating(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=True, tlm=False)


def start_tlm(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.start(annotate=False, tlm=True)


def stop_manager(*, annotate=True, tlm=True, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=annotate, tlm=tlm)


def stop_annotating(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=True, tlm=False)


def stop_tlm(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.stop(annotate=False, tlm=True)


def paused_manager(*, annotate=True, tlm=True, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.paused(annotate=annotate, tlm=tlm)


def configure_tlm(*args, annotate=None, tlm=True, manager=None):
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


def tlm_enabled(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.tlm_enabled()


def function_tlm(x, *args, manager=None):
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


def compute_gradient(Js, M, callback=None, prune_forward=True,
                     prune_adjoint=True, prune_replay=True,
                     cache_adjoint_degree=None, adj_ics=None, manager=None):
    if manager is None:
        manager = globals()["manager"]()
    return manager.compute_gradient(Js, M, callback=callback,
                                    prune_forward=prune_forward,
                                    prune_adjoint=prune_adjoint,
                                    prune_replay=prune_replay,
                                    cache_adjoint_degree=cache_adjoint_degree,
                                    adj_ics=adj_ics)


def new_block(manager=None):
    if manager is None:
        manager = globals()["manager"]()
    manager.new_block()
