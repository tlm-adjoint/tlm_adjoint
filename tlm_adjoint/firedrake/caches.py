"""This module implements finite element assembly and linear solver data
caching.
"""

from .backend import (
    Tensor, TrialFunction, backend_assemble, backend_DirichletBC,
    backend_Function, complex_mode)
from ..interface import (
    is_var, var_caches, var_id, var_is_cached, var_is_replacement,
    var_replacement, var_space, var_state, weakref_method)
from .backend_code_generator_interface import linear_solver, parameters_key

from ..caches import Cache

from .assembly import assemble, assemble_matrix, matrix_copy, matrix_multiply
from .expr import (
    derivative, eliminate_zeros, expr_zero, extract_coefficients, form_cached,
    iter_expr, replaced_form)
from .functions import ReplacementFunction

from collections import defaultdict
import itertools
import ufl

__all__ = \
    [
        "AssemblyCache",
        "assembly_cache",
        "set_assembly_cache",

        "LinearSolverCache",
        "linear_solver_cache",
        "set_linear_solver_cache",

        "LocalSolverCache",
        "local_solver_cache",
        "set_local_solver_cache"
    ]


@form_cached("_tlm_adjoint__is_cached")
def is_cached(expr):
    for c in extract_coefficients(expr):
        if not is_var(c) or not var_is_cached(c):
            return False
    return True


def form_simplify_sign(form):
    integrals = []

    for integral in form.integrals():
        integrand = integral.integrand()

        integral_sign = None
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


def form_simplify_conj(form):
    if complex_mode:
        def expr_conj(expr):
            if isinstance(expr, ufl.classes.Conj):
                x, = expr.ufl_operands
                return expr_simplify_conj(x)
            elif isinstance(expr, ufl.classes.Sum):
                return sum(map(expr_conj, expr.ufl_operands),
                           expr_zero(expr))
            elif isinstance(expr, ufl.classes.Product):
                x, y = expr.ufl_operands
                return expr_conj(x) * expr_conj(y)
            else:
                return ufl.conj(expr)

        def expr_simplify_conj(expr):
            if isinstance(expr, ufl.classes.Conj):
                x, = expr.ufl_operands
                return expr_conj(x)
            elif isinstance(expr, ufl.classes.Sum):
                return sum(map(expr_simplify_conj, expr.ufl_operands),
                           expr_zero(expr))
            elif isinstance(expr, ufl.classes.Product):
                x, y = expr.ufl_operands
                return expr_simplify_conj(x) * expr_simplify_conj(y)
            else:
                return expr

        def integral_simplify_conj(integral):
            integrand = integral.integrand()
            integrand = expr_simplify_conj(integrand)
            return integral.reconstruct(integrand=integrand)

        integrals = list(map(integral_simplify_conj, form.integrals()))
        return ufl.classes.Form(integrals)
    else:
        return ufl.algorithms.remove_complex_nodes.remove_complex_nodes(form)


def split_arity(form, x, argument):
    form_arguments = form.arguments()
    arity = len(form_arguments)
    if arity >= 2:
        raise ValueError("Invalid form arity")
    if arity == 1 and form_arguments[0].number() != 0:
        raise ValueError("Invalid form argument")
    if argument.number() < arity:
        raise ValueError("Invalid argument")

    if x not in extract_coefficients(form):
        # No dependence on x
        return ufl.classes.Form([]), form

    form_derivative = derivative(form, x, argument=argument,
                                 enable_automatic_argument=False)
    form_derivative = ufl.algorithms.expand_derivatives(form_derivative)
    if x in extract_coefficients(form_derivative):
        # Non-linear
        return ufl.classes.Form([]), form

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

    try:
        ufl.algorithms.check_arities.check_form_arity(
            A, A.arguments(), complex_mode=complex_mode)
        ufl.algorithms.check_arities.check_form_arity(
            b, b.arguments(), complex_mode=complex_mode)
    except ufl.algorithms.check_arities.ArityMismatch:
        # Arity mismatch
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
        mat_terms = defaultdict(list)
    if non_cached_terms is None:
        non_cached_terms = []

    for term in terms:
        if is_cached(term):
            cached_terms.append(term)
        elif isinstance(term, ufl.classes.Conj):
            term_conj, = term.ufl_operands
            if isinstance(term_conj, ufl.classes.Sum):
                split_terms(
                    tuple(map(ufl.conj, term_conj.ufl_operands)),
                    base_integral,
                    cached_terms, mat_terms, non_cached_terms)
            elif isinstance(term_conj, ufl.classes.Product):
                x, y = term_conj.ufl_operands
                split_terms(
                    (ufl.conj(x) * ufl.conj(y),),
                    base_integral,
                    cached_terms, mat_terms, non_cached_terms)
            else:
                non_cached_terms.append(term)
        elif isinstance(term, ufl.classes.Sum):
            split_terms(term.ufl_operands, base_integral,
                        cached_terms, mat_terms, non_cached_terms)
        elif isinstance(term, ufl.classes.Product):
            x, y = term.ufl_operands
            if is_cached(x):
                cached_sub, mat_sub, non_cached_sub = split_terms(
                    (y,), base_integral)
                for term in cached_sub:
                    cached_terms.append(x * term)
                for dep_id in mat_sub:
                    mat_terms[dep_id].extend(
                        x * mat_term for mat_term in mat_sub[dep_id])
                for term in non_cached_sub:
                    non_cached_terms.append(x * term)
            elif is_cached(y):
                cached_sub, mat_sub, non_cached_sub = split_terms(
                    (x,), base_integral)
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
            for dep in extract_coefficients(term):
                if not is_cached(dep):
                    if mat_dep is not None:
                        mat_dep = None
                        break
                    mat_dep = dep
            if not isinstance(mat_dep, (backend_Function,
                                        ReplacementFunction)):
                non_cached_terms.append(term)
            else:
                term_form = ufl.classes.Form(
                    [base_integral.reconstruct(integrand=term)])
                mat_sub, non_cached_sub = split_arity(
                    term_form, mat_dep,
                    argument=TrialFunction(var_space(mat_dep)))
                mat_sub = [integral.integrand()
                           for integral in mat_sub.integrals()]
                non_cached_sub = [integral.integrand()
                                  for integral in non_cached_sub.integrals()]
                if len(mat_sub) > 0:
                    mat_terms[var_id(mat_dep)].extend(mat_sub)
                non_cached_terms.extend(non_cached_sub)

    return cached_terms, dict(mat_terms), non_cached_terms


def split_form(form):
    forms = ufl.classes.Form([])
    cofunctions = ufl.classes.ZeroBaseForm(form.arguments())
    for weight, comp in iter_expr(form):
        if isinstance(comp, ufl.classes.Form):
            forms = forms + weight * comp
        elif isinstance(comp, ufl.classes.Cofunction):
            cofunctions = cofunctions + weight * comp
        else:
            raise TypeError(f"Unexpected type: {type(comp)}")

    cached_form, mat_forms, non_cached_form = _split_form(forms)
    if non_cached_form.empty():
        non_cached_form = cofunctions
    else:
        non_cached_form = non_cached_form + cofunctions
    return cached_form, mat_forms, non_cached_form


def _split_form(form):
    if form.empty():
        return ufl.classes.Form([]), {}, ufl.classes.Form([])

    if len(form.arguments()) != 1:
        raise ValueError("Arity 1 form required")
    if not complex_mode:
        form = ufl.algorithms.remove_complex_nodes.remove_complex_nodes(form)

    def add_integral(integrals, base_integral, terms):
        if len(terms) > 0:
            integrand = sum(terms, expr_zero(terms[0]))
            integral = base_integral.reconstruct(integrand=integrand)
            integrals.append(integral)

    cached_integrals = []
    mat_integrals = defaultdict(list)
    non_cached_integrals = []
    for integral in form.integrals():
        cached_terms, mat_terms, non_cached_terms = \
            split_terms((integral.integrand(),), integral)
        add_integral(cached_integrals, integral, cached_terms)
        for dep_id in mat_terms:
            add_integral(mat_integrals[dep_id], integral, mat_terms[dep_id])
        add_integral(non_cached_integrals, integral, non_cached_terms)

    cached_form = ufl.classes.Form(cached_integrals)
    mat_forms = {}
    for dep_id in mat_integrals:
        mat_forms[dep_id] = ufl.classes.Form(mat_integrals[dep_id])
    non_cached_form = ufl.classes.Form(non_cached_integrals)

    return cached_form, mat_forms, non_cached_form


def form_key(*forms):
    key = []

    for form in forms:
        deps = [dep for dep in extract_coefficients(form) if is_var(dep)]
        deps = sorted(deps, key=var_id)
        deps_key = tuple((var_id(dep), var_state(dep)) for dep in deps)

        form = replaced_form(form)
        form = ufl.algorithms.expand_derivatives(form)
        form = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(form)  # noqa: E501
        form = ufl.algorithms.expand_indices(form)
        if isinstance(form, ufl.classes.Form):
            form = form_simplify_conj(form)
            form = form_simplify_sign(form)

        key.extend((form, deps_key))

    return tuple(key)


def form_dependencies(*forms):
    deps = {}

    for dep in itertools.chain.from_iterable(map(extract_coefficients, forms)):
        if is_var(dep):
            dep_id = var_id(dep)
            if dep_id in deps:
                dep_old = deps[dep_id]
                assert (dep is dep_old
                        or dep is var_replacement(dep_old)
                        or var_replacement(dep) is dep_old)
                assert var_caches(dep) is var_caches(dep_old)
                if var_is_replacement(dep_old):
                    deps[dep_id] = dep
            else:
                deps[dep_id] = dep

    return tuple(sorted(deps.values(), key=var_id))


class AssemblyCache(Cache):
    """A :class:`.Cache` for finite element assembly data.
    """

    def assemble(self, form, *,
                 bcs=None, form_compiler_parameters=None,
                 linear_solver_parameters=None, replace_map=None):
        """Perform finite element assembly and cache the result, or return a
        previously cached result.

        :arg form: The :class:`ufl.Form` to assemble.
        :arg bcs: Dirichlet boundary conditions.
        :arg form_compiler_parameters: Form compiler parameters.
        :arg linear_solver_parameters: Linear solver parameters. Required for
            assembly parameters which appear in the linear solver parameters.
        :arg replace_map: A :class:`Mapping` defining a map from symbolic
            variables to values.
        :returns: A :class:`tuple` `(value_ref, value)`, where `value` is the
            result of the finite element assembly, and `value_ref` is a
            :class:`.CacheRef` storing a reference to `value`.

                - For an arity zero or arity one form `value_ref` stores the
                  assembled value.
                - For an arity two form `value_ref` is a tuple `(A, b_bc)`. `A`
                  is the assembled matrix, and `b_bc` is a boundary condition
                  right-hand-side term which should be added after assembling a
                  right-hand-side with homogeneous boundary conditions applied.
                  `b_bc` may be `None` to indicate that this term is zero.
        """

        if bcs is None:
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if linear_solver_parameters is None:
            linear_solver_parameters = {}

        form = eliminate_zeros(form)
        if isinstance(form, ufl.classes.ZeroBaseForm):
            raise ValueError("Form cannot be a ZeroBaseForm")
        if replace_map is None:
            assemble_form = form
        else:
            assemble_form = ufl.replace(form, replace_map)
        arity = len(form.arguments())
        assemble_kwargs = {"form_compiler_parameters": form_compiler_parameters,  # noqa: E501
                           "mat_type": linear_solver_parameters.get("mat_type", None)}  # noqa: E501

        key = (form_key(form, assemble_form),
               tuple(bcs),
               parameters_key(assemble_kwargs))

        def value():
            if arity == 0:
                if len(bcs) > 0:
                    raise TypeError("Unexpected boundary conditions for arity "
                                    "0 form")
                b = assemble(assemble_form, **assemble_kwargs)
            elif arity == 1:
                b = assemble(assemble_form, **assemble_kwargs)
                for bc in bcs:
                    bc.apply(b)
            elif arity == 2:
                b = assemble_matrix(assemble_form, bcs=bcs, **assemble_kwargs)
            else:
                raise ValueError(f"Unexpected form arity {arity:d}")
            return b

        return self.add(key, value,
                        deps=form_dependencies(form, assemble_form))


class LinearSolverCache(Cache):
    """A :class:`.Cache` for linear solver data.
    """

    def linear_solver(self, form, *,
                      bcs=None, form_compiler_parameters=None,
                      linear_solver_parameters=None, replace_map=None,
                      assembly_cache=None):
        """Construct a linear solver and cache the result, or return a
        previously cached result.

        :arg form: An arity two :class:`ufl.Form`, defining the matrix.
        :arg bcs: Dirichlet boundary conditions.
        :arg form_compiler_parameters: Form compiler parameters.
        :arg linear_solver_parameters: Linear solver parameters.
        :arg replace_map: A :class:`Mapping` defining a map from symbolic
            variables to values.
        :arg assembly_cache: :class:`.AssemblyCache` to use for finite element
            assembly. Defaults to `assembly_cache()`.
        :returns: A :class:`tuple` `(value_ref, value)`. `value` is a tuple
            `(solver, A, b_bc)`, where `solver` is the linear solver, `A` is
            the assembled matrix, and `b_bc` is a boundary condition
            right-hand-side term which should be added after assembling a
            right-hand-side with homogeneous boundary conditions applied.
            `b_bc` may be `None` to indicate that this term is zero.
            `value_ref` is a :class:`.CacheRef` storing a reference to `value`.
        """

        if bcs is None:
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        if form_compiler_parameters is None:
            form_compiler_parameters = {}
        if linear_solver_parameters is None:
            linear_solver_parameters = {}

        form = eliminate_zeros(form)
        if isinstance(form, ufl.classes.ZeroBaseForm):
            raise ValueError("Form cannot be a ZeroBaseForm")
        if replace_map is None:
            assemble_form = form
        else:
            assemble_form = ufl.replace(form, replace_map)

        key = (form_key(form, assemble_form),
               tuple(bcs),
               parameters_key(form_compiler_parameters),
               parameters_key(linear_solver_parameters))

        if assembly_cache is None:
            assembly_cache = globals()["assembly_cache"]()

        def value():
            _, (A, b_bc) = assembly_cache.assemble(
                form, bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                linear_solver_parameters=linear_solver_parameters,
                replace_map=replace_map)
            solver = linear_solver(matrix_copy(A),
                                   linear_solver_parameters)
            return solver, A, b_bc

        return self.add(key, value,
                        deps=form_dependencies(form, assemble_form))


def LocalSolver(form, *,
                form_compiler_parameters=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    form = eliminate_zeros(form)
    if isinstance(form, ufl.classes.ZeroBaseForm):
        raise ValueError("Form cannot be a ZeroBaseForm")
    local_solver = backend_assemble(
        Tensor(form).inv,
        form_compiler_parameters=form_compiler_parameters)

    def solve_local(self, x, b):
        matrix_multiply(self, b, tensor=x)
    local_solver._tlm_adjoint__solve_local = weakref_method(
        solve_local, local_solver)

    return local_solver


class LocalSolverCache(Cache):
    """A :class:`.Cache` for element-wise local block diagonal linear solver
    data.
    """

    def local_solver(self, form, *,
                     form_compiler_parameters=None, replace_map=None):
        """Compute data for an element-wise local block diagonal linear
        solver and cache the result, or return a previously cached result.

        :arg form: An arity two :class:`ufl.Form`, defining the element-wise
            local block diagonal matrix.
        :arg form_compiler_parameters: Form compiler parameters.
        :arg replace_map: A :class:`Mapping` defining a map from symbolic
            variables to values.
        :returns: A :class:`tuple` `(value_ref, value)`. `value` is a
            :class:`firedrake.matrix.Matrix` storing the assembled inverse
            matrix, and `value_ref` is a :class:`.CacheRef` storing a reference
            to `value`.
        """

        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        form = eliminate_zeros(form)
        if isinstance(form, ufl.classes.ZeroBaseForm):
            raise ValueError("Form cannot be a ZeroBaseForm")
        if replace_map is None:
            assemble_form = form
        else:
            assemble_form = ufl.replace(form, replace_map)

        key = (form_key(form, assemble_form),
               parameters_key(form_compiler_parameters))

        def value():
            return LocalSolver(
                assemble_form,
                form_compiler_parameters=form_compiler_parameters)

        return self.add(key, value,
                        deps=form_dependencies(form, assemble_form))


_assembly_cache = AssemblyCache()


def assembly_cache():
    """
    :returns: The default :class:`.AssemblyCache`.
    """

    return _assembly_cache


def set_assembly_cache(assembly_cache):
    """Set the default :class:`.AssemblyCache`.

    :arg assembly_cache: The new default :class:`.AssemblyCache`.
    """

    global _assembly_cache
    _assembly_cache = assembly_cache


_linear_solver_cache = LinearSolverCache()


def linear_solver_cache():
    """
    :returns: The default :class:`.LinearSolverCache`.
    """

    return _linear_solver_cache


def set_linear_solver_cache(linear_solver_cache):
    """Set the default :class:`.LinearSolverCache`.

    :arg linear_solver_cache: The new default :class:`.LinearSolverCache`.
    """

    global _linear_solver_cache
    _linear_solver_cache = linear_solver_cache


_local_solver_cache = LocalSolverCache()


def local_solver_cache():
    """
    :returns: The default :class:`.LocalSolverCache`.
    """

    return _local_solver_cache


def set_local_solver_cache(local_solver_cache):
    """Set the default :class:`.LocalSolverCache`.

    :arg local_solver_cache: The new default :class:`.LocalSolverCache`.
    """

    global _local_solver_cache
    _local_solver_cache = local_solver_cache
