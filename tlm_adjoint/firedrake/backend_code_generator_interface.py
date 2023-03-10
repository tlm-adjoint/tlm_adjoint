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

from .backend import FunctionSpace, Interpolator, Parameters, TestFunction, \
    backend_Constant, backend_DirichletBC, backend_Function, \
    backend_LinearSolver, backend_Matrix, backend_assemble, backend_solve, \
    complex_mode, extract_args, homogenize, parameters
from ..interface import check_space_type, check_space_types, function_assign, \
    function_axpy, function_copy, function_dtype, function_inner, \
    function_new_conjugate_dual, function_space, function_space_type, space_new

from .functions import eliminate_zeros

from collections.abc import Iterable, Sequence
import copy
import numpy as np
import petsc4py.PETSc as PETSc
import ufl

__all__ = \
    [
        "assemble_arguments",
        "assemble_linear_solver",
        "assemble_matrix",
        "complex_mode",
        "copy_parameters_dict",
        "form_form_compiler_parameters",
        "function_vector",
        "homogenize",
        "interpolate_expression",
        "is_valid_r0_space",
        "linear_solver",
        "matrix_copy",
        "matrix_multiply",
        "parameters_key",
        "process_adjoint_solver_parameters",
        "process_solver_parameters",
        "r0_space",
        "rhs_addto",
        "rhs_copy",
        "update_parameters_dict",
        "verify_assembly",

        "assemble",
        "solve"
    ]


_parameters = parameters.setdefault("tlm_adjoint", {})
_parameters.setdefault("Assembly", {})
_parameters["Assembly"].setdefault("match_quadrature", False)
_parameters.setdefault("EquationSolver", {})
_parameters["EquationSolver"].setdefault("enable_jacobian_caching", True)
_parameters["EquationSolver"].setdefault("cache_rhs_assembly", True)
_parameters["EquationSolver"].setdefault("match_quadrature", False)
_parameters["EquationSolver"].setdefault("defer_adjoint_assembly", False)
_parameters.setdefault("assembly_verification", {})
_parameters["assembly_verification"].setdefault("jacobian_tolerance", np.inf)
_parameters["assembly_verification"].setdefault("rhs_tolerance", np.inf)
# For deprecated AssembleSolver
_parameters.setdefault("AssembleSolver", {})
_parameters["AssembleSolver"].setdefault("match_quadrature", False)
del _parameters


def copy_parameters_dict(parameters):
    new_parameters = parameters.copy()
    for key, value in parameters.items():
        if isinstance(value, (Parameters, dict)):
            value = copy_parameters_dict(value)
        elif isinstance(value, Iterable):
            value = copy.copy(value)  # shallow copy only
        new_parameters[key] = value
    return new_parameters


def update_parameters_dict(parameters, new_parameters):
    for key, value in new_parameters.items():
        if key in parameters \
           and isinstance(parameters[key], (Parameters, dict)) \
           and isinstance(value, (Parameters, dict)):
            update_parameters_dict(parameters[key], value)
        elif isinstance(value, (Parameters, dict)):
            parameters[key] = copy_parameters_dict(value)
        else:
            parameters[key] = value


def process_solver_parameters(solver_parameters, linear):
    solver_parameters = copy_parameters_dict(solver_parameters)
    tlm_adjoint_parameters = solver_parameters.setdefault("tlm_adjoint", {})

    tlm_adjoint_parameters.setdefault("options_prefix", None)
    tlm_adjoint_parameters.setdefault("nullspace", None)
    tlm_adjoint_parameters.setdefault("transpose_nullspace", None)
    tlm_adjoint_parameters.setdefault("near_nullspace", None)

    linear_solver_ic = solver_parameters.setdefault("ksp_initial_guess_nonzero", False)  # noqa: E501

    return (solver_parameters, solver_parameters,
            not linear or linear_solver_ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        adjoint_solver_parameters = copy.copy(linear_solver_parameters)
        tlm_adjoint_parameters = adjoint_solver_parameters["tlm_adjoint"] \
            = copy.copy(linear_solver_parameters["tlm_adjoint"])

        tlm_adjoint_parameters["nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["transpose_nullspace"]
        tlm_adjoint_parameters["transpose_nullspace"] \
            = linear_solver_parameters["tlm_adjoint"]["nullspace"]

        return adjoint_solver_parameters
    else:
        # Copy not required
        return linear_solver_parameters


def assemble_arguments(rank, form_compiler_parameters, solver_parameters):
    kwargs = {"form_compiler_parameters": form_compiler_parameters}
    if rank == 2 and "mat_type" in solver_parameters:
        kwargs["mat_type"] = solver_parameters["mat_type"]
    return kwargs


def strip_terminal_data(form):
    # Replace constants with no domain with constants on the first domain
    domain, = form.ufl_domains()

    replace_map = {}
    replace_map_inverse = {}
    for dep in form.coefficients():
        if isinstance(dep, backend_Constant) and len(dep.ufl_domains()) == 0:
            dep_arr = np.zeros(dep.ufl_shape, dtype=function_dtype(dep))
            replace_map[dep] = backend_Constant(dep_arr, domain=domain)
            replace_map_inverse[replace_map[dep]] = dep

    if len(replace_map) == 0:
        return ufl.algorithms.strip_terminal_data(form)
    else:
        unbound_form, maps = \
            ufl.algorithms.strip_terminal_data(ufl.replace(form, replace_map))

        binding_map = copy.copy(maps[0])
        for replacement_dep, dep in maps[0].items():
            if dep in replace_map_inverse:
                binding_map[replacement_dep] = replace_map_inverse[dep]

        assert len(maps) == 2
        maps = (binding_map, maps[1])

        return (unbound_form, maps)


def bind_form(form):
    if "_tlm_adjoint__bindings" in form._cache:
        bindings = form._cache["_tlm_adjoint__bindings"]
        if hasattr(ufl.algorithms, "replace_terminal_data"):
            if "_tlm_adjoint__unbound_form" not in form._cache:
                form._cache["_tlm_adjoint__unbound_form"] = \
                    strip_terminal_data(eliminate_zeros(form, force_non_empty_form=True))  # noqa: E501
            unbound_form, maps = form._cache["_tlm_adjoint__unbound_form"]

            binding_map = copy.copy(maps[0])
            for replacement_dep, dep in maps[0].items():
                if dep in bindings:
                    binding_map[replacement_dep] = bindings[dep]

            assert len(maps) == 2
            maps = (binding_map, maps[1])

            return ufl.algorithms.replace_terminal_data(unbound_form, maps)
        else:
            return ufl.replace(form, bindings)
    else:
        return form


def _assemble(form, tensor=None, form_compiler_parameters=None,
              *args, **kwargs):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    if tensor is not None and isinstance(tensor, backend_Function):
        check_space_type(tensor, "conjugate_dual")

    form = eliminate_zeros(form, force_non_empty_form=True)
    b = backend_assemble(
        form, tensor=tensor,
        form_compiler_parameters=form_compiler_parameters,
        *args, **kwargs)

    if tensor is None and isinstance(b, backend_Function):
        b._tlm_adjoint__function_interface_attrs.d_setitem("space_type", "conjugate_dual")  # noqa: E501

    return b


def _assemble_system(A_form, b_form=None, bcs=None,
                     form_compiler_parameters=None, *args, **kwargs):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    A_form = bind_form(A_form)
    if b_form is not None:
        b_form = bind_form(b_form)

    A = _assemble(
        A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
        *args, **kwargs)

    if len(bcs) > 0:
        F = backend_Function(A_form.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)

        if b_form is None:
            b = _assemble(
                -ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)

            with b.dat.vec_ro as b_v:
                if b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) == 0.0:
                    b = None
        else:
            b = _assemble(
                b_form - ufl.action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)
    else:
        if b_form is None:
            b = None
        else:
            b = _assemble(
                b_form,
                form_compiler_parameters=form_compiler_parameters,
                *args, **kwargs)

    A._tlm_adjoint__lift_bcs = False

    return A, b


def _LinearSolver_lifted(self, b):
    if getattr(self.A, "_tlm_adjoint__lift_bcs", True):
        return backend_LinearSolver._tlm_adjoint__orig__lifted(self, b)
    else:
        return b


assert not hasattr(backend_LinearSolver, "_tlm_adjoint__orig__lifted")
backend_LinearSolver._tlm_adjoint__orig__lifted = backend_LinearSolver._lifted
backend_LinearSolver._lifted = _LinearSolver_lifted


def assemble_matrix(form, bcs=None, form_compiler_parameters=None,
                    *args, **kwargs):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    return _assemble_system(form, bcs=bcs,
                            form_compiler_parameters=form_compiler_parameters,
                            *args, **kwargs)


def assemble(form, tensor=None, form_compiler_parameters=None,
             *args, **kwargs):
    # Similar interface to assemble in FEniCS 2019.1.0

    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    if tensor is not None and isinstance(tensor, backend_Function):
        check_space_type(tensor, "conjugate_dual")

    form = bind_form(form)
    b = _assemble(
        form, tensor=tensor, form_compiler_parameters=form_compiler_parameters,
        *args, **kwargs)

    if tensor is None and isinstance(b, backend_Function):
        b._tlm_adjoint__function_interface_attrs.d_setitem("space_type", "conjugate_dual")  # noqa: E501

    return b


def assemble_linear_solver(A_form, b_form=None, bcs=None,
                           form_compiler_parameters=None,
                           linear_solver_parameters=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if linear_solver_parameters is None:
        linear_solver_parameters = {}

    A, b = _assemble_system(
        A_form, b_form=b_form, bcs=bcs,
        **assemble_arguments(2, form_compiler_parameters,
                             linear_solver_parameters))

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    if "tlm_adjoint" in linear_solver_parameters:
        linear_solver_parameters = copy.copy(linear_solver_parameters)
        tlm_adjoint_parameters = linear_solver_parameters.pop("tlm_adjoint")
        options_prefix = tlm_adjoint_parameters.get("options_prefix", None)
        nullspace = tlm_adjoint_parameters.get("nullspace", None)
        transpose_nullspace = tlm_adjoint_parameters.get("transpose_nullspace",
                                                         None)
        near_nullspace = tlm_adjoint_parameters.get("near_nullspace", None)
    else:
        options_prefix = None
        nullspace = None
        transpose_nullspace = None
        near_nullspace = None
    return backend_LinearSolver(A, solver_parameters=linear_solver_parameters,
                                options_prefix=options_prefix,
                                nullspace=nullspace,
                                transpose_nullspace=transpose_nullspace,
                                near_nullspace=near_nullspace)


def form_form_compiler_parameters(form, form_compiler_parameters):
    qd = form_compiler_parameters.get("quadrature_degree", "auto")
    if qd in [None, "auto", -1]:
        qd = ufl.algorithms.estimate_total_polynomial_degree(form)
    return {"quadrature_degree": qd}


# def homogenize(bc):


def matrix_copy(A):
    if not isinstance(A, backend_Matrix):
        raise TypeError("Unexpected matrix type")

    options_prefix = A.petscmat.getOptionsPrefix()
    A_copy = backend_Matrix(A.a, A.bcs, A.mat_type,
                            A.M.sparsity, A.M.dtype,
                            options_prefix=options_prefix)

    assert A.petscmat.assembled
    A_copy.petscmat.axpy(1.0, A.petscmat)
    assert A_copy.petscmat.assembled

    # MatAXPY does not propagate the options prefix
    A_copy.petscmat.setOptionsPrefix(options_prefix)

    if hasattr(A, "_tlm_adjoint__lift_bcs"):
        A_copy._tlm_adjoint__lift_bcs = A._tlm_adjoint__lift_bcs

    return A_copy


def matrix_multiply(A, x, *, tensor=None, addto=False,
                    action_type="conjugate_dual"):
    if tensor is None:
        tensor = space_new(
            A.a.arguments()[0].function_space(),
            space_type=function_space_type(x, rel_space_type=action_type))
    else:
        check_space_types(tensor, x, rel_space_type=action_type)

    if addto:
        with x.dat.vec_ro as x_v, tensor.dat.vec as tensor_v:
            A.petscmat.multAdd(x_v, tensor_v, tensor_v)
    else:
        with x.dat.vec_ro as x_v, tensor.dat.vec_wo as tensor_v:
            A.petscmat.mult(x_v, tensor_v)

    return tensor


def is_valid_r0_space(space):
    if not hasattr(space, "_tlm_adjoint__is_valid_r0_space"):
        e = space.ufl_element()
        if e.family() != "Real" or e.degree() != 0:
            valid = False
        elif len(e.value_shape()) == 0:
            r = backend_Function(space)
            r.assign(backend_Constant(1.0), annotate=False, tlm=False)
            with r.dat.vec_ro as r_v:
                r_max = r_v.max()[1]
            valid = (r_max == 1.0)
        else:
            # See Firedrake issue #1456
            valid = False
        space._tlm_adjoint__is_valid_r0_space = valid
    return space._tlm_adjoint__is_valid_r0_space


def r0_space(x):
    if not hasattr(x, "_tlm_adjoint__r0_space"):
        domain, = x.ufl_domains()
        if len(x.ufl_shape) == 0:
            space = FunctionSpace(domain, "R", 0)
        else:
            # See Firedrake issue #1456
            raise NotImplementedError("Rank >= 1 Constant not implemented")
        assert is_valid_r0_space(space)
        x._tlm_adjoint__r0_space = space
    return x._tlm_adjoint__r0_space


def function_vector(x):
    return x


def rhs_copy(x):
    check_space_type(x, "conjugate_dual")
    return function_copy(x)


def rhs_addto(x, y):
    check_space_type(x, "conjugate_dual")
    check_space_type(y, "conjugate_dual")
    function_axpy(x, 1.0, y)


def parameters_key(parameters):
    key = []
    for name in sorted(parameters.keys()):
        sub_parameters = parameters[name]
        if isinstance(sub_parameters, (Parameters, dict)):
            key.append((name, parameters_key(sub_parameters)))
        elif isinstance(sub_parameters, Sequence) \
                and not isinstance(sub_parameters, str):
            key.append((name, tuple(sub_parameters)))
        else:
            key.append((name, sub_parameters))
    return tuple(key)


def verify_assembly(J, rhs, J_mat, b, bcs, form_compiler_parameters,
                    linear_solver_parameters, J_tolerance, b_tolerance):
    if J_mat is not None and not np.isposinf(J_tolerance):
        J_mat_debug = backend_assemble(
            J, bcs=bcs, **assemble_arguments(2,
                                             form_compiler_parameters,
                                             linear_solver_parameters))
        assert J_mat.petscmat.assembled
        J_error = J_mat.petscmat.copy()
        J_error.axpy(-1.0, J_mat_debug.petscmat)
        assert J_error.assembled
        assert J_error.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
            <= J_tolerance * J_mat_debug.petscmat.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501

    if b is not None and not np.isposinf(b_tolerance):
        F = backend_Function(rhs.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)
        b_debug = backend_assemble(
            rhs - ufl.action(J, F), bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        b_error = b.copy(deepcopy=True)
        with b_error.dat.vec as b_error_v, b_debug.dat.vec_ro as b_debug_v:
            b_error_v.axpy(-1.0, b_debug_v)
        with b_error.dat.vec_ro as b_error_v, b_debug.dat.vec_ro as b_v:
            assert b_error_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) \
                <= b_tolerance * b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)  # noqa: E501


def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is None:
        check_space_type(x, "primal")
    else:
        check_space_type(x, "conjugate_dual")
        check_space_type(adj_x, "conjugate_dual")
    for dep in ufl.algorithms.extract_coefficients(expr):
        check_space_type(dep, "primal")

    expr = eliminate_zeros(expr)

    if adj_x is None:
        if isinstance(x, backend_Constant):
            x.assign(expr, annotate=False, tlm=False)
        elif isinstance(x, backend_Function):
            x.interpolate(expr)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
    elif isinstance(x, backend_Constant):
        if len(x.ufl_shape) > 0:
            raise ValueError("Scalar Constant required")
        expr_val = function_new_conjugate_dual(adj_x)
        interpolate_expression(expr_val, expr)
        function_assign(x, function_inner(adj_x, expr_val))
    elif isinstance(x, backend_Function):
        x_space = function_space(x)
        interp = Interpolator(ufl.conj(expr) * TestFunction(x_space), x_space)
        interp.interpolate(adj_x, transpose=True, output=x)
    else:
        raise TypeError(f"Unexpected type: {type(x)}")


def solve(*args, **kwargs):
    if not isinstance(args[0], ufl.classes.Equation):
        return backend_solve(*args, **kwargs)

    eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
        nullspace, transpose_nullspace, near_nullspace, options_prefix = \
        extract_args(*args, **kwargs)
    check_space_type(x, "primal")
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if solver_parameters is None:
        solver_parameters = {}

    if "tlm_adjoint" in solver_parameters:
        solver_parameters = copy.copy(solver_parameters)
        tlm_adjoint_parameters = solver_parameters.pop("tlm_adjoint")

        if "options_prefix" in tlm_adjoint_parameters:
            if options_prefix is not None:
                raise TypeError("Cannot pass both options_prefix argument and "
                                "solver parameter")
            options_prefix = tlm_adjoint_parameters["options_prefix"]

        if "nullspace" in tlm_adjoint_parameters:
            if nullspace is not None:
                raise TypeError("Cannot pass both nullspace argument and "
                                "solver parameter")
            nullspace = tlm_adjoint_parameters["nullspace"]

        if "transpose_nullspace" in tlm_adjoint_parameters:
            if transpose_nullspace is not None:
                raise TypeError("Cannot pass both transpose_nullspace "
                                "argument and solver parameter")
            transpose_nullspace = tlm_adjoint_parameters["transpose_nullspace"]

        if "near_nullspace" in tlm_adjoint_parameters:
            if near_nullspace is not None:
                raise TypeError("Cannot pass both near_nullspace argument and "
                                "solver parameter")
            near_nullspace = tlm_adjoint_parameters["near_nullspace"]

    return backend_solve(eq, x, bcs, J=J, Jp=Jp, M=M,
                         form_compiler_parameters=form_compiler_parameters,
                         solver_parameters=solver_parameters,
                         nullspace=nullspace,
                         transpose_nullspace=transpose_nullspace,
                         near_nullspace=near_nullspace,
                         options_prefix=options_prefix)
