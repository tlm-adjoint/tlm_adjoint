from .backend import Parameters, parameters

from ..petsc import flattened_options

import ufl

__all__ = []


def copy_parameters(parameters):
    new_parameters = dict(parameters)
    for key, value in parameters.items():
        if isinstance(value, (Parameters, dict)):
            value = copy_parameters(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
        new_parameters[key] = value
    return new_parameters


def update_parameters(parameters, new_parameters):
    for key, value in new_parameters.items():
        if key in parameters \
           and isinstance(parameters[key], (Parameters, dict)) \
           and isinstance(value, (Parameters, dict)):
            update_parameters(parameters[key], value)
        elif isinstance(value, (Parameters, dict)):
            parameters[key] = copy_parameters(value)
        else:
            parameters[key] = value


def process_form_compiler_parameters(form_compiler_parameters):
    params = copy_parameters(parameters["form_compiler"])
    update_parameters(params, form_compiler_parameters)
    return params


def flattened_solver_parameters(solver_parameters):
    solver_parameters = copy_parameters(solver_parameters)
    have_tlm_adjoint_parameters = "tlm_adjoint" in solver_parameters
    if have_tlm_adjoint_parameters:
        tlm_adjoint_parameters = solver_parameters["tlm_adjoint"]
    solver_parameters = dict(flattened_options(solver_parameters))
    if have_tlm_adjoint_parameters:
        if "tlm_adjoint" in solver_parameters:
            raise ValueError("Duplicate key")
        solver_parameters["tlm_adjoint"] = tlm_adjoint_parameters
    return solver_parameters


def form_compiler_quadrature_parameters(form, form_compiler_parameters):
    qd = form_compiler_parameters.get("quadrature_degree", "auto")
    if qd in {None, "auto", -1}:
        qd = ufl.algorithms.estimate_total_polynomial_degree(form)
    return {"quadrature_degree": qd}


def process_solver_parameters(solver_parameters, *, linear):
    solver_parameters = flattened_solver_parameters(solver_parameters)

    tlm_adjoint_parameters = solver_parameters.setdefault("tlm_adjoint", {})
    tlm_adjoint_parameters.setdefault("options_prefix", None)
    tlm_adjoint_parameters.setdefault("nullspace", None)
    tlm_adjoint_parameters.setdefault("transpose_nullspace", None)
    tlm_adjoint_parameters.setdefault("near_nullspace", None)

    linear_solver_ic = solver_parameters.setdefault("ksp_initial_guess_nonzero", False)  # noqa: E501
    ic = not linear or linear_solver_ic

    return (solver_parameters, solver_parameters, ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    tlm_adjoint_parameters = linear_solver_parameters.get("tlm_adjoint", {})
    nullspace = tlm_adjoint_parameters.get("nullspace", None)
    transpose_nullspace = tlm_adjoint_parameters.get("transpose_nullspace", None)  # noqa: E501

    adjoint_solver_parameters = flattened_solver_parameters(linear_solver_parameters)  # noqa: E501
    adjoint_tlm_adjoint_parameters = adjoint_solver_parameters.setdefault("tlm_adjoint", {})  # noqa: E501
    adjoint_tlm_adjoint_parameters["nullspace"] = transpose_nullspace
    adjoint_tlm_adjoint_parameters["transpose_nullspace"] = nullspace

    return adjoint_solver_parameters
