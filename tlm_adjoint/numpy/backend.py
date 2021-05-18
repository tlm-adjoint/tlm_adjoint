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

import numpy as np

backend = "NumPy"

backend_ScalarType = np.float64
backend_RealScalarType = np.float64
backend_ComplexScalarType = np.complex128

if not issubclass(backend_ScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")
if not issubclass(backend_RealScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend real scalar type: "
                      f"{backend_RealScalarType}")
if not issubclass(backend_ComplexScalarType, (complex, np.complexfloating)):
    raise ImportError(f"Invalid backend complex scalar type: "
                      f"{backend_ComplexScalarType}")

__all__ = \
    [
        "backend",

        "backend_ComplexScalarType",
        "backend_RealScalarType",
        "backend_ScalarType"
    ]
