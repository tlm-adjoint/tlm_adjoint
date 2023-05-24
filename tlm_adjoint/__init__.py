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

from .cached_hessian import *      # noqa: F401
from .caches import *              # noqa: F401
from .eigendecomposition import *  # noqa: F401
from .equation import *            # noqa: F401
from .equations import *           # noqa: F401
from .fixed_point import *         # noqa: F401
from .functional import *          # noqa: F401
from .hessian import *             # noqa: F401
from .instructions import *        # noqa: F401
from .interface import *           # noqa: F401
from .linear_equation import *     # noqa: F401
from .manager import *             # noqa: F401
from .optimization import *        # noqa: F401
from .overloaded_float import *    # noqa: F401
from .storage import *             # noqa: F401
from .tangent_linear import *      # noqa: F401
from .tlm_adjoint import *         # noqa: F401
from .verification import *        # noqa: F401

from .hessian import GeneralGaussNewton as GaussNewton  # noqa: F401
from .hessian import GeneralHessian as Hessian          # noqa: F401

del adjoint, alias, cached_hessian, caches, checkpointing, \
    eigendecomposition, equation, equations, fixed_point, functional, \
    hessian, interface, linear_equation, markers, optimization, \
    overloaded_float, storage, tangent_linear, tlm_adjoint, verification  # noqa: E501,F821
