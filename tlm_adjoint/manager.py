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

__all__ = \
    [
        "manager",
        "restore_manager",
        "set_manager"
    ]

_manager = [None]


def manager():
    return _manager[0]


def set_manager(manager):
    _manager[0] = manager


def restore_manager(fn):
    def wrapped_fn(*args, **kwargs):
        old_manager = manager()
        try:
            return fn(*args, **kwargs)
        finally:
            set_manager(old_manager)
    return wrapped_fn
