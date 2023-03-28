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
        "get_tangent_linear",
    ]


def get_tangent_linear(x, M, dM, tlm_map):
    """Return a tangent-linear variable associated with a variable `x`.

    This function should be used in place of accessing via the
    :class:`TangentLinearMap`, if the variable `x` may be a control variable.

    :arg x: A function defining the variable for which a tangent-linear
        variable should be returned.
    :arg M: A :class:`Sequence` of functions defining the control.
    :arg dM: A :class:`Sequence` of functions defining the derivative
        direction. The tangent-linear model computes directional derivatives
        with respect to the control defined by `M` and with direction defined
        by `dM`.
    :arg tlm_map: A :class:`TangentLinearMap` storing values for tangent-linear
        variables.
    :returns: If `x` is a control variable then returns the associated
        direction. If `x` is not a control variable then returns a function
        corresponding to a tangent-linear variable if `x` is 'checkpointed'
        (i.e. stored by value by a :class:`CheckpointStorage`), and `None`
        otherwise.
    """

    if x in M:
        return dM[M.index(x)]
    else:
        return tlm_map[x]
