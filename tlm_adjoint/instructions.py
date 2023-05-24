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

from .interface import DEFAULT_COMM, comm_dup_cached, garbage_cleanup

from .equations import EmptyEquation

import gc

__all__ = \
    [
        "Instruction",

        "GarbageCollection"
    ]


class Instruction(EmptyEquation):
    """An adjoint tape record which defines instructions to be performed during
    forward or adjoint calculations.
    """

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        pass


class GarbageCollection(Instruction):
    """An :class:`Instruction` which indicates that garbage collection should
    be performed during forward and adjoint calculations.

    :arg comm: Communicator to use for PETSc garbage cleanup.
    :arg generation: Python garbage collection generation. If a value of `None`
        is provided then Python garbage collection is not performed.
    :arg garbage_cleanup: Whether to perform PETSc garbage cleanup.
    """

    def __init__(self, comm=None, *, generation=2, garbage_cleanup=True):
        if comm is None:
            comm = DEFAULT_COMM

        super().__init__()
        self._comm = comm_dup_cached(comm)
        self._generation = generation
        self._garbage_cleanup = garbage_cleanup

    def _gc(self):
        if self._generation is not None:
            gc.collect(self._generation)
        if self._garbage_cleanup:
            garbage_cleanup(self._comm)

    def forward_solve(self, x, deps=None):
        self._gc()

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        self._gc()

    def tangent_linear(self, M, dM, tlm_map):
        return GarbageCollection(
            self._comm,
            generation=self._generation, garbage_cleanup=self._garbage_cleanup)
