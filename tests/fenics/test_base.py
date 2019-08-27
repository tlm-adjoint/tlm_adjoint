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

from tlm_adjoint_fenics import info, manager as _manager
from tlm_adjoint_fenics.backend import backend_Function

import gc
import weakref

Function_ids = {}
_orig_Function_init = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function_init(self, *args, **kwargs)
    Function_ids[self.id()] = weakref.ref(self)


backend_Function.__init__ = _Function__init__


def leak_check(test):
    def wrapped_test(*args, **kwargs):
        Function_ids.clear()

        test(*args, **kwargs)

        # Clear some internal storage that is allowed to keep references
        manager = _manager()
        manager._cp.clear(clear_refs=True)
        tlm_values = manager._tlm.values()  # noqa: F841
        manager._tlm.clear()
        tlm_eqs_values = manager._tlm_eqs.values()  # noqa: F841
        manager._tlm_eqs.clear()

        gc.collect()

        refs = 0
        for F in Function_ids.values():
            F = F()
            if F is not None:
                info(f"{F.name():s} referenced")
                refs += 1
        if refs == 0:
            info("No references")

        Function_ids.clear()
        assert(refs == 0)
    return wrapped_test
