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

import sys

if "tlm_adjoint.backend" not in sys.modules:
    backend = None

    try:
        import fenics  # noqa: F401
        backend = "FEniCS"
    except ImportError:
        try:
            import firedrake  # noqa: F401
            backend = "Firedrake"
        except ImportError:
            backend = "NumPy"

    if backend == "FEniCS":
        from tlm_adjoint_fenics import *  # noqa: F401
    elif backend == "Firedrake":
        from tlm_adjoint_firedrake import *  # noqa: F401
    else:
        assert(backend == "NumPy")
        from tlm_adjoint_numpy import *  # noqa: F401
