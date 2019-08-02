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

from .backend import backend       # noqa: F401
from .backend_interface import *   # noqa: F401
from .backend_overrides import *   # noqa: F401
from .base_equations import *      # noqa: F401
from .caches import *              # noqa: F401
from .eigendecomposition import *  # noqa: F401
from .equations import *           # noqa: F401
from .fenics_equations import *    # noqa: F401
from .functional import *          # noqa: F401
from .hessian import *             # noqa: F401
from .manager import *             # noqa: F401
from .tlm_adjoint import *         # noqa: F401
