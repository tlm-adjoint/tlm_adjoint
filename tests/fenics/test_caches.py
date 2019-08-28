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

from fenics import *
from tlm_adjoint_fenics import *

from test_base import *

import pytest


@pytest.mark.fenics
def test_clear_caches(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    F = Function(space, name="F", cache=True)

    def cache_item(F):
        form = inner(TestFunction(F.function_space()), F) * dx
        cached_form, _ = assembly_cache().assemble(form)
        return cached_form

    def test_not_cleared(F, cached_form):
        assert(len(assembly_cache()) == 1)
        assert(cached_form() is not None)
        assert(len(function_caches(F)) == 1)

    def test_cleared(F, cached_form):
        assert(len(assembly_cache()) == 0)
        assert(cached_form() is None)
        assert(len(function_caches(F)) == 0)

    assert(len(assembly_cache()) == 0)

    # Clear default
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    clear_caches()
    test_cleared(F, cached_form)

    # Clear Function caches
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    clear_caches(F)
    test_cleared(F, cached_form)

    # Clear on cache update
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    function_update_state(F)
    test_not_cleared(F, cached_form)
    update_caches([F])
    test_cleared(F, cached_form)

    # Clear on cache update, new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([F], [Function(space)])
    test_cleared(F, cached_form)

    # Clear on cache update, ReplacementFunction
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([replaced_function(F)])
    test_cleared(F, cached_form)

    # Clear on cache update, ReplacementFunction with new Function
    cached_form = cache_item(F)
    test_not_cleared(F, cached_form)
    update_caches([replaced_function(F)], [Function(space)])
    test_cleared(F, cached_form)
