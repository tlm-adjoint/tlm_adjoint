from .backend import (
    KrylovSolver, LUSolver, TestFunction, as_backend_type, backend_Constant,
    backend_DirichletBC, backend_Function, backend_ScalarType,
    backend_assemble, backend_assemble_system, has_lu_solver_method)
from ..interface import (
    DEFAULT_COMM, check_space_type, check_space_types, space_new,
    var_space_type)

from .expr import eliminate_zeros
from .parameters import update_parameters

import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = [
    "linear_solver"
]


def assemble_matrix(form, bcs=None, *,
                    form_compiler_parameters=None):
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    if form_compiler_parameters is None:
        form_compiler_parameters = {}

    if len(bcs) > 0:
        test = TestFunction(form.arguments()[0].function_space())
        if len(test.ufl_shape) == 0:
            zero = backend_Constant(0.0)
        else:
            zero = backend_Constant(np.zeros(test.ufl_shape,
                                             dtype=backend_ScalarType))
        dummy_rhs = ufl.inner(zero, test) * ufl.dx
        A, b_bc = assemble_system(
            form, dummy_rhs, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)
        if b_bc.norm("linf") == 0.0:
            b_bc = None
    else:
        A = assemble(
            form, form_compiler_parameters=form_compiler_parameters)
        b_bc = None

    return A, b_bc


def assemble(form, tensor=None, bcs=None, *,
             form_compiler_parameters=None):
    if tensor is not None and hasattr(tensor, "_tlm_adjoint__function"):
        check_space_type(tensor._tlm_adjoint__function, "conjugate_dual")
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)

    form = eliminate_zeros(form)
    b = backend_assemble(form, tensor=tensor,
                         form_compiler_parameters=form_compiler_parameters)
    for bc in bcs:
        bc.apply(b)
    return b


def assemble_system(A_form, b_form, bcs=None, *,
                    form_compiler_parameters=None):
    A_form = eliminate_zeros(A_form)
    b_form = eliminate_zeros(b_form)
    return backend_assemble_system(
        A_form, b_form, bcs=bcs,
        form_compiler_parameters=form_compiler_parameters)


def matrix_copy(A):
    return A.copy()


def matrix_multiply(A, x, *,
                    tensor=None, addto=False, action_type="conjugate_dual"):
    if isinstance(x, backend_Function):
        x = x.vector()
    if tensor is not None and isinstance(tensor, backend_Function):
        tensor = tensor.vector()
    if tensor is None:
        if hasattr(A, "_tlm_adjoint__form") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
            tensor = space_new(
                A._tlm_adjoint__form.arguments()[0].function_space(),
                space_type=var_space_type(x._tlm_adjoint__function,
                                          rel_space_type=action_type))
            tensor = tensor.vector()
        else:
            return A * x
    elif hasattr(tensor, "_tlm_adjoint__function") and hasattr(x, "_tlm_adjoint__function"):  # noqa: E501
        check_space_types(tensor._tlm_adjoint__function,
                          x._tlm_adjoint__function,
                          rel_space_type=action_type)

    x_v = as_backend_type(x).vec()
    tensor_v = as_backend_type(tensor).vec()
    if addto:
        as_backend_type(A).mat().multAdd(x_v, tensor_v, tensor_v)
    else:
        as_backend_type(A).mat().mult(x_v, tensor_v)
    tensor.apply("insert")

    return tensor


def assemble_linear_solver(A_form, b_form=None, bcs=None, *,
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

    if b_form is None:
        A, b = assemble_matrix(
            A_form, bcs=bcs, form_compiler_parameters=form_compiler_parameters)
    else:
        A, b = assemble_system(
            A_form, b_form, bcs=bcs,
            form_compiler_parameters=form_compiler_parameters)

    solver = linear_solver(A, linear_solver_parameters)

    return solver, A, b


def linear_solver(A, linear_solver_parameters, *, comm=None):
    """Construct a DOLFIN `LUSolver` or `KrylovSolver`.

    :arg A: A DOLFIN matrix.
    :arg linear_solver_parameters: Linear solver parameters.
    :arg comm: A communicator.
    :returns: The DOLFIN `LUSolver` or `KrylovSolver`.
    """

    if comm is None:
        if hasattr(A, "mpi_comm"):
            comm = A.mpi_comm()
        else:
            comm = DEFAULT_COMM

    linear_solver = linear_solver_parameters.get("linear_solver", "default")
    if linear_solver in {"direct", "lu"}:
        linear_solver = "default"
    elif linear_solver == "iterative":
        linear_solver = "gmres"
    is_lu_linear_solver = linear_solver == "default" \
        or has_lu_solver_method(linear_solver)
    if is_lu_linear_solver:
        solver = LUSolver(comm, linear_solver)
        solver.set_operator(A)
        lu_parameters = linear_solver_parameters.get("lu_solver", {})
        update_parameters(solver.parameters, lu_parameters)
    else:
        pc = linear_solver_parameters.get("preconditioner", "default")
        ks_parameters = linear_solver_parameters.get("krylov_solver", {})
        solver = KrylovSolver(comm, linear_solver, pc)
        solver.set_operator(A)
        update_parameters(solver.parameters, ks_parameters)
    return solver
