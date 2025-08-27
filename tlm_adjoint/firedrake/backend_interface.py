from .backend import (
    _extract_comm, LinearSolver, Tensor, backend_Cofunction, backend_Function,
    backend_Matrix, backend_assemble, backend_solve, extract_args)
from ..interface import (
    check_space_type, check_space_types_conjugate_dual, packed,
    register_garbage_cleanup, space_eq, space_new)

from .expr import action, eliminate_zeros

import petsc4py.PETSc as PETSc
import pyop2
import ufl

__all__ = [
    "linear_solver"
]


def _assemble(form, tensor=None, bcs=None, *,
              form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    form = eliminate_zeros(form)

    if isinstance(form, ufl.classes.BaseForm) \
            and len(form.arguments()) == 1:
        if tensor is None:
            tensor = backend_Cofunction(form.arguments()[0].function_space().dual())  # noqa: E501
        b = backend_assemble(
            form, tensor=tensor,
            form_compiler_parameters=form_compiler_parameters,
            mat_type=mat_type)
        for bc in bcs:
            bc.apply(backend_Function(b.function_space().dual(), val=b.dat)
                     if isinstance(b, backend_Cofunction) else b)
    else:
        b = backend_assemble(
            form, tensor=tensor, bcs=tuple(bcs),
            form_compiler_parameters=form_compiler_parameters,
            mat_type=mat_type)

    return b


def _assemble_system(A_form, b_form=None, bcs=None, *,
                     form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    A = _assemble(
        A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters,
        mat_type=mat_type)

    if len(bcs) > 0:
        F = backend_Function(A_form.arguments()[0].function_space())
        for bc in bcs:
            bc.apply(F)

        if b_form is None:
            b = _assemble(
                -action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

            with b.dat.vec_ro as b_v:
                if b_v.norm(norm_type=PETSc.NormType.NORM_INFINITY) == 0.0:
                    b = None
        else:
            b = _assemble(
                b_form - action(A_form, F), bcs=bcs,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)
    else:
        if b_form is None:
            b = None
        else:
            b = _assemble(
                b_form,
                form_compiler_parameters=form_compiler_parameters,
                mat_type=mat_type)

    return A, b


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None, mat_type=None):
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    return _assemble_system(form, bcs=bcs,
                            form_compiler_parameters=form_compiler_parameters,
                            mat_type=mat_type)


def assemble(form, tensor=None, bcs=None, *,
             form_compiler_parameters=None, mat_type=None):
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    b = _assemble(
        form, tensor=tensor, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters, mat_type=mat_type)

    return b


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

    return A_copy


def matrix_multiply(A, x, *, tensor=None, addto=False):
    if tensor is None:
        tensor = space_new(
            A.a.arguments()[0].function_space().dual())
    check_space_types_conjugate_dual(tensor, x)

    if addto:
        with x.dat.vec_ro as x_v, tensor.dat.vec as tensor_v:
            A.petscmat.multAdd(x_v, tensor_v, tensor_v)
    else:
        with x.dat.vec_ro as x_v, tensor.dat.vec_wo as tensor_v:
            A.petscmat.mult(x_v, tensor_v)

    return tensor


def assemble_linear_solver(A_form, b_form=None, bcs=None, *,
                           form_compiler_parameters=None,
                           linear_solver_parameters=None):
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if linear_solver_parameters is None:
        linear_solver_parameters = {}

    A, b = _assemble_system(
        A_form, b_form=b_form, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters,
        mat_type=linear_solver_parameters.get("mat_type", None))

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters):
    """Construct a :class:`firedrake.linear_solver.LinearSolver`.

    :arg A: A :class:`firedrake.matrix.Matrix`.
    :arg linear_solver_parameters: Linear solver parameters.
    :returns: The :class:`firedrake.linear_solver.LinearSolver`.
    """

    if "tlm_adjoint" in linear_solver_parameters:
        linear_solver_parameters = dict(linear_solver_parameters)
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
    return LinearSolver(A, solver_parameters=linear_solver_parameters,
                        options_prefix=options_prefix,
                        nullspace=nullspace,
                        transpose_nullspace=transpose_nullspace,
                        near_nullspace=near_nullspace)


def solve(*args, **kwargs):
    if not isinstance(args[0], ufl.classes.Equation):
        return backend_solve(*args, **kwargs)

    extracted_args = extract_args(*args, **kwargs)
    eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
        nullspace, transpose_nullspace, near_nullspace, options_prefix, \
        restrict, pre_apply_bcs, \
        = extracted_args
    check_space_type(x, "primal")
    if bcs is None:
        bcs = ()
    else:
        bcs = packed(bcs)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    if solver_parameters is None:
        solver_parameters = {}

    solver_parameters = dict(solver_parameters)
    tlm_adjoint_parameters = solver_parameters.pop("tlm_adjoint", {})

    def get_parameter(key, value):
        if value is not None:
            if key in tlm_adjoint_parameters:
                raise TypeError(f"Cannot pass both {key:s} argument and "
                                f"solver parameter")
            return value
        else:
            return tlm_adjoint_parameters.get(key, None)

    options_prefix = get_parameter("options_prefix", options_prefix)
    nullspace = get_parameter("nullspace", nullspace)
    transpose_nullspace = get_parameter("transpose_nullspace", transpose_nullspace)  # noqa: E501
    near_nullspace = get_parameter("near_nullspace", near_nullspace)
    pre_apply_bcs = get_parameter("pre_apply_bcs", pre_apply_bcs)

    return backend_solve(eq, x, tuple(bcs), J=J, Jp=Jp, M=M,
                         form_compiler_parameters=form_compiler_parameters,
                         solver_parameters=solver_parameters,
                         nullspace=nullspace,
                         transpose_nullspace=transpose_nullspace,
                         near_nullspace=near_nullspace,
                         options_prefix=options_prefix,
                         restrict=restrict,
                         pre_apply_bcs=pre_apply_bcs)


class LocalSolver:
    def __init__(self, form, *, form_compiler_parameters=None):
        if form_compiler_parameters is None:
            form_compiler_parameters = {}

        arguments = form.arguments()
        test, trial = arguments
        assert test.number() < trial.number()
        b_space = test.function_space().dual()
        x_space = trial.function_space()

        form = eliminate_zeros(form)
        mat = backend_assemble(
            Tensor(form).inv,
            form_compiler_parameters=form_compiler_parameters)

        self._mat = mat
        self._x_space = x_space
        self._b_space = b_space

    def solve(self, x, b):
        if not space_eq(x.function_space(), self._x_space):
            raise ValueError("Invalid space")
        check_space_type(x, "primal")
        if not space_eq(b.function_space(), self._b_space):
            raise ValueError("Invalid space")
        check_space_type(b, "conjugate_dual")

        matrix_multiply(self._mat, b, tensor=x)


def backend_garbage_cleanup(comm):
    if not pyop2.mpi.PYOP2_FINALIZED:
        PETSc.garbage_cleanup(_extract_comm(comm))


register_garbage_cleanup(backend_garbage_cleanup)
