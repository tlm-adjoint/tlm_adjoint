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
import gc

__all__ = \
    [
        "Alias",
        "AliasException",
        "WeakAlias",
        "gc_disabled"
    ]


class AliasException(Exception):
    pass


def gc_disabled(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            if gc_enabled:
                gc.enable()
    return wrapped_fn


class Alias:
    """
    Alias of obj, holding a reference to obj.
    """

    def __init__(self, obj):
        if isinstance(obj, Alias):
            raise AliasException("Cannot alias Alias")
        super().__setattr__("_tlm_adjoint__alias", obj)

    def __new__(cls, obj):
        obj_cls = type(obj)

        class Alias(cls, obj_cls):
            def __new__(cls, *args, **kwargs):
                return obj_cls(*args, **kwargs)

        Alias.__name__ = f"{obj_cls.__name__:s}Alias"
        return super().__new__(Alias)

    def __getattr__(self, key):
        return getattr(self._tlm_adjoint__alias, key)

    def __setattr__(self, key, value):
        setattr(self._tlm_adjoint__alias, key, value)

    def __delattr__(self, key):
        delattr(self._tlm_adjoint__alias, key)

    def __dir__(self):
        return dir(self._tlm_adjoint__alias)


class WeakAlias:
    """
    Alias of obj, holding no reference to obj, and valid after deallocation of
    obj. Intended to be used in combination with weakref.finalize(obj, ...).
    """

    def __init__(self, obj):
        if hasattr(obj, "__slots__"):
            # Weak references to obj not possible, has attributes not
            # accessible via __dict__ attribute
            raise AliasException("Cannot alias object with __slots__ "
                                 "attribute")
        if isinstance(obj, WeakAlias):
            raise AliasException("Cannot alias WeakAlias")
        super().__setattr__("_tlm_adjoint__alias__dict__", obj.__dict__)

    def __new__(cls, obj):
        obj_cls = type(obj)

        class WeakAlias(cls, obj_cls):
            pass

        WeakAlias.__name__ = f"{obj_cls.__name__:s}WeakAlias"
        return super().__new__(WeakAlias)

    def __getattr__(self, key):
        if key not in self._tlm_adjoint__alias__dict__:
            raise AttributeError(f"No attribute '{key:s}'")
        return self._tlm_adjoint__alias__dict__[key]

    def __setattr__(self, key, value):
        self._tlm_adjoint__alias__dict__[key] = value

    def __delattr__(self, key):
        if key not in self._tlm_adjoint__alias__dict__:
            raise AttributeError(f"No attribute '{key:s}'")
        del self._tlm_adjoint__alias__dict__[key]

    def __dir__(self):
        return list(self._tlm_adjoint__alias__dict__.keys())
