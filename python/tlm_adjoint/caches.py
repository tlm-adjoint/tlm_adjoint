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

from .backend import *
from .backend_code_generator_interface import *
from .interface import *

from .alias import gc_disabled
from .functions import eliminate_zeros, function_caches

from collections import defaultdict
import ufl
import warnings
import weakref

__all__ = \
    [
        "AssemblyCache",
        "Cache",
        "CacheException",
        "CacheRef",
        "LinearSolverCache",
        "assembly_cache",
        "form_dependencies",
        "form_neg",
        "is_cached",
        "linear_solver",
        "linear_solver_cache",
        "set_assembly_cache",
        "set_linear_solver_cache",
        "split_form",
        "update_caches"
    ]


class CacheException(Exception):
    pass


def is_cached(e):
    for c in ufl.algorithms.extract_coefficients(e):
        if not is_function(c) or not function_is_cached(c):
            return False
    return True


def form_simplify_sign(form, sign=None):
    integrals = []

    for integral in form.integrals():
        integrand = integral.integrand()

        integral_sign = sign
        while isinstance(integrand, ufl.classes.Product):
            a, b = integrand.ufl_operands
            if isinstance(a, ufl.classes.IntValue) and a == -1:
                if integral_sign is None:
                    integral_sign = -1
                else:
                    integral_sign = -integral_sign
                integrand = b
            elif isinstance(b, ufl.classes.IntValue) and b == -1:
                if integral_sign is None:
                    integral_sign = -1
                else:
                    integral_sign = -integral_sign
                integrand = a
            else:
                break
        if integral_sign is not None:
            if integral_sign < 0:
                integral = integral.reconstruct(integrand=-integrand)
            else:
                integral = integral.reconstruct(integrand=integrand)

        integrals.append(integral)

    return ufl.classes.Form(integrals)


def form_neg(form):
    return form_simplify_sign(form, sign=-1)


def split_arity(form, x, argument):
    if x not in form.coefficients():
        # No dependence on x
        return ufl.classes.Form([]), form

    form_derivative = ufl.derivative(form, x, argument=argument)
    form_derivative = ufl.algorithms.expand_derivatives(form_derivative)
    if x in form_derivative.coefficients():
        # Non-linear
        return ufl.classes.Form([]), form

    arity = len(form.arguments())
    try:
        eq_form = ufl.algorithms.expand_derivatives(
            ufl.replace(form, {x: argument}))
        A = ufl.algorithms.formtransformations.compute_form_with_arity(
            eq_form, arity + 1)
        b = ufl.algorithms.formtransformations.compute_form_with_arity(
            eq_form, arity)
    except ufl.UFLException:
        # UFL error encountered
        return ufl.classes.Form([]), form

    if not is_cached(A):
        # Non-cached higher arity form
        return ufl.classes.Form([]), form

    # Success
    return A, b


def split_terms(terms, base_integral,
                cached_terms=None, mat_terms=None, non_cached_terms=None):
    if cached_terms is None:
        cached_terms = []
    if mat_terms is None:
        mat_terms = defaultdict(lambda: [])
    if non_cached_terms is None:
        non_cached_terms = []

    for term in terms:
        if is_cached(term):
            cached_terms.append(term)
        elif isinstance(term, ufl.classes.Conj):
            x, = term.ufl_operands
            cached_sub, mat_sub, non_cached_sub = split_terms(
                [x], base_integral)
            for term in cached_sub:
                cached_terms.append(ufl.classes.Conj(term))
            for dep_id in mat_sub:
                mat_terms[dep_id].extend(ufl.classes.Conj(mat_term)
                                         for mat_term in mat_sub[dep_id])
            for term in non_cached_sub:
                non_cached_terms.append(ufl.classes.Conj(term))
        elif isinstance(term, ufl.classes.Sum):
            split_terms(term.ufl_operands, base_integral,
                        cached_terms, mat_terms, non_cached_terms)
        elif isinstance(term, ufl.classes.Product):
            x, y = term.ufl_operands
            if is_cached(x):
                cached_sub, mat_sub, non_cached_sub = split_terms(
                    [y], base_integral)
                for term in cached_sub:
                    cached_terms.append(x * term)
                for dep_id in mat_sub:
                    mat_terms[dep_id].extend(
                        x * mat_term for mat_term in mat_sub[dep_id])
                for term in non_cached_sub:
                    non_cached_terms.append(x * term)
            elif is_cached(y):
                cached_sub, mat_sub, non_cached_sub = split_terms(
                    [x], base_integral)
                for term in cached_sub:
                    cached_terms.append(term * y)
                for dep_id in mat_sub:
                    mat_terms[dep_id].extend(
                        mat_term * y for mat_term in mat_sub[dep_id])
                for term in non_cached_sub:
                    non_cached_terms.append(term * y)
            else:
                non_cached_terms.append(term)
        else:
            mat_dep = None
            for dep in ufl.algorithms.extract_coefficients(term):
                if not is_cached(dep):
                    if isinstance(dep, backend_Function) and mat_dep is None:
                        mat_dep = dep
                    else:
                        mat_dep = None
                        break
            if mat_dep is None:
                non_cached_terms.append(term)
            else:
                term_form = ufl.classes.Form(
                    [base_integral.reconstruct(integrand=term)])
                mat_sub, non_cached_sub = split_arity(
                    term_form, mat_dep,
                    argument=TrialFunction(function_space(mat_dep)))
                mat_sub = [integral.integrand()
                           for integral in mat_sub.integrals()]
                non_cached_sub = [integral.integrand()
                                  for integral in non_cached_sub.integrals()]
                if len(mat_sub) > 0:
                    mat_terms[function_id(mat_dep)].extend(mat_sub)
                non_cached_terms.extend(non_cached_sub)

    return cached_terms, mat_terms, non_cached_terms


def split_form(form):
    def add_integral(integrals, base_integral, terms):
        if len(terms) > 0:
            integrand = sum(terms, ufl.classes.Zero())
            integral = base_integral.reconstruct(integrand=integrand)
            integrals.append(integral)

    cached_integrals = []
    mat_integrals = defaultdict(lambda: [])
    non_cached_integrals = []
    for integral in form.integrals():
        cached_terms, mat_terms, non_cached_terms = \
            split_terms([integral.integrand()], integral)
        add_integral(cached_integrals, integral, cached_terms)
        for dep_id in mat_terms:
            add_integral(mat_integrals[dep_id], integral, mat_terms[dep_id])
        add_integral(non_cached_integrals, integral, non_cached_terms)

    cached_form = ufl.classes.Form(cached_integrals)
    mat_forms = {}
    for dep_id in mat_integrals:
        mat_forms[dep_id] = ufl.classes.Form(mat_integrals[dep_id])
    non_cached_forms = ufl.classes.Form(non_cached_integrals)

    return cached_form, mat_forms, non_cached_forms


class CacheRef:
    def __init__(self, value=None):
        self._value = value

    def __call__(self):
        return self._value

    def _clear(self):
        self._value = None


@gc_disabled
def clear_caches(*deps):
    if len(deps) == 0:
        for cache in tuple(Cache._caches.valuerefs()):
            cache = cache()
            if cache is not None:
                cache.clear()
    else:
        for dep in deps:
            function_caches(dep).clear()


def update_caches(eq_deps, deps=None):
    if deps is None:
        for eq_dep in eq_deps:
            function_caches(eq_dep).update(eq_dep)
    else:
        for eq_dep, dep in zip(eq_deps, deps):
            function_caches(eq_dep).update(dep)


class Cache:
    _id_counter = [0]
    _caches = weakref.WeakValueDictionary()

    def __init__(self):
        self._cache = {}
        self._deps_map = {}
        self._dep_caches = {}

        self._id = self._id_counter[0]
        self._id_counter[0] += 1
        self._caches[self._id] = self

    def __del__(self):
        for value in self._cache.values():
            value._clear()

    def __len__(self):
        return len(self._cache)

    def id(self):
        return self._id

    def clear(self, *deps):
        if len(deps) == 0:
            for value in self._cache.values():
                value._clear()
            self._cache.clear()
            self._deps_map.clear()
            for dep_caches in self._dep_caches.values():
                dep_caches = dep_caches()
                if dep_caches is not None:
                    dep_caches.remove(self)
            self._dep_caches.clear()
        else:
            for dep in deps:
                dep_id = dep if isinstance(dep, int) else function_id(dep)
                del dep
                if dep_id in self._deps_map:
                    # Steps in removing cached data associated with dep:
                    #   1. Delete cached items associated with dep -- these are
                    #      given by
                    #        self._cache[key] for key in self._deps_map[dep_id]
                    #   2. Remove the key, and a reference to its associated
                    #      dependency ids, from the keys associated with each
                    #      dependency id associated with each of the keys in 1.
                    #      -- the latter dependency ids are given by
                    #        self._deps_map[dep_id][key]
                    #   3. Remove the (weak) reference to this cache from each
                    #      dependency with no further associated keys
                    for key, dep_ids in self._deps_map[dep_id].items():
                        # Step 1.
                        self._cache[key]._clear()
                        del self._cache[key]
                        for dep_id2 in dep_ids:
                            if dep_id2 != dep_id:
                                # Step 2.
                                del self._deps_map[dep_id2][key]
                                if len(self._deps_map[dep_id2]) == 0:
                                    del self._deps_map[dep_id2]
                                    dep_caches = self._dep_caches[dep_id2]()
                                    if dep_caches is not None:
                                        # Step 3.
                                        dep_caches.remove(self)
                                    del self._dep_caches[dep_id2]
                    # Step 2.
                    del self._deps_map[dep_id]
                    dep_caches = self._dep_caches[dep_id]()
                    if dep_caches is not None:
                        # Step 3.
                        dep_caches.remove(self)
                    del self._dep_caches[dep_id]

    def add(self, key, value, deps=[]):
        if key in self._cache:
            value_ref = self._cache[key]
            value = value_ref()
            if value is None:
                raise CacheException("Unexpected cache value state")
            return value_ref, value

        value = value()
        value_ref = CacheRef(value)
        dep_ids = tuple(function_id(dep) for dep in deps)

        self._cache[key] = value_ref

        for dep, dep_id in zip(deps, dep_ids):
            dep_caches = function_caches(dep)
            dep_caches.add(self)

            if dep_id in self._deps_map:
                self._deps_map[dep_id][key] = dep_ids
                assert dep_id in self._dep_caches
            else:
                self._deps_map[dep_id] = {key: dep_ids}
                self._dep_caches[dep_id] = weakref.ref(dep_caches)

        return value_ref, value

    def get(self, key, default=None):
        return self._cache.get(key, default)


def replaced_form(form):
    replace_map = {}
    for c in form.coefficients():
        if is_function(c):
            replace_map[c] = function_replacement(c)
    return ufl.replace(form, replace_map)


def form_dependencies(form):
    deps = {}
    for dep in form.coefficients():
        if is_function(dep):
            dep_id = function_id(dep)
            if dep_id not in deps:
                deps[dep_id] = dep
    return deps


def form_key(form):
    form = replaced_form(form)
    form = ufl.algorithms.expand_derivatives(form)
    form = ufl.algorithms.expand_compounds(form)
    form = ufl.algorithms.expand_indices(form)
    return form


def assemble_key(form, bcs, assemble_kwargs):
    return (form_key(form), tuple(bcs), parameters_key(assemble_kwargs))


class AssemblyCache(Cache):
    def assemble(self, form, bcs=[], form_compiler_parameters={},
                 solver_parameters=None, linear_solver_parameters=None,
                 replace_map=None):
        if solver_parameters is not None:
            warnings.warn("'solver_parameters' argument is deprecated -- use "
                          "'linear_solver_parameters' instead",
                          DeprecationWarning, stacklevel=2)
            if linear_solver_parameters is not None:
                raise CacheException("Cannot pass both 'solver_parameters' "
                                     "and 'linear_solver_parameters' "
                                     "arguments")
            linear_solver_parameters = solver_parameters
        elif linear_solver_parameters is None:
            linear_solver_parameters = {}

        form = eliminate_zeros(form, non_empty_form=True)
        rank = len(form.arguments())
        assemble_kwargs = assemble_arguments(rank, form_compiler_parameters,
                                             linear_solver_parameters)
        key = assemble_key(form, bcs, assemble_kwargs)

        def value():
            if replace_map is None:
                assemble_form = form
            else:
                assemble_form = ufl.replace(form, replace_map)
            if rank == 0:
                if len(bcs) > 0:
                    raise CacheException("Unexpected boundary conditions for rank 0 form")  # noqa: E501
                b = assemble(assemble_form, **assemble_kwargs)
            elif rank == 1:
                b = assemble(assemble_form, **assemble_kwargs)
                for bc in bcs:
                    bc.apply(b)
            elif rank == 2:
                b = assemble_matrix(assemble_form, bcs, **assemble_kwargs)
            else:
                raise CacheException(f"Unexpected form rank {rank:d}")
            return b

        return self.add(key, value,
                        deps=tuple(form_dependencies(form).values()))


def linear_solver_key(form, bcs, linear_solver_parameters,
                      form_compiler_parameters):
    return (form_key(form), tuple(bcs),
            parameters_key(linear_solver_parameters),
            parameters_key(form_compiler_parameters))


class LinearSolverCache(Cache):
    def linear_solver(self, form, A, bcs=[], form_compiler_parameters={},
                      linear_solver_parameters={}):
        form = eliminate_zeros(form, non_empty_form=True)
        key = linear_solver_key(form, bcs, linear_solver_parameters,
                                form_compiler_parameters)

        def value():
            return linear_solver(A, linear_solver_parameters)

        return self.add(key, value,
                        deps=tuple(form_dependencies(form).values()))


_assembly_cache = [AssemblyCache()]


def assembly_cache():
    return _assembly_cache[0]


def set_assembly_cache(assembly_cache):
    _assembly_cache[0] = assembly_cache


_linear_solver_cache = [LinearSolverCache()]


def linear_solver_cache():
    return _linear_solver_cache[0]


def set_linear_solver_cache(linear_solver_cache):
    _linear_solver_cache[0] = linear_solver_cache
