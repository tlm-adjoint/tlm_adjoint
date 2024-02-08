from .backend import Parameters, has_lu_solver_method, parameters

import ffc

__all__ = []


def copy_parameters(parameters):
    if isinstance(parameters, Parameters):
        parameters = dict(parameters)
    new_parameters = {}
    for key in parameters:
        value = parameters[key]
        if isinstance(value, (Parameters, dict)):
            value = copy_parameters(value)
        elif isinstance(value, list):
            value = list(value)
        elif isinstance(value, set):
            value = set(value)
        new_parameters[key] = value
    return new_parameters


def update_parameters(parameters, new_parameters):
    for key in new_parameters:
        value = new_parameters[key]
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


def form_compiler_quadrature_parameters(form, form_compiler_parameters):
    (form_data,), _, _, _ \
        = ffc.analysis.analyze_forms((form,), form_compiler_parameters)
    integral_metadata = tuple(integral_data.metadata
                              for integral_data in form_data.integral_data)
    qr = form_compiler_parameters.get("quadrature_rule", "auto")
    if qr in {None, "auto"}:
        qr = ffc.analysis._extract_common_quadrature_rule(integral_metadata)
    qd = form_compiler_parameters.get("quadrature_degree", "auto")
    if qd in {None, "auto", -1}:
        qd = ffc.analysis._extract_common_quadrature_degree(integral_metadata)
    return {"quadrature_rule": qr, "quadrature_degree": qd}


def process_solver_parameters(solver_parameters, *, linear):
    solver_parameters = copy_parameters(solver_parameters)

    if linear:
        linear_solver_parameters = solver_parameters
    else:
        nl_solver = solver_parameters.setdefault("nonlinear_solver", "newton")
        if nl_solver == "newton":
            linear_solver_parameters = solver_parameters.setdefault("newton_solver", {})  # noqa: E501
        elif nl_solver == "snes":
            linear_solver_parameters = solver_parameters.setdefault("snes_solver", {})  # noqa: E501
        else:
            raise ValueError(f"Unexpected non-linear solver: {nl_solver}")

    linear_solver = linear_solver_parameters.setdefault("linear_solver", "default")  # noqa: E501
    if linear_solver in {"default", "direct", "lu"} \
            or has_lu_solver_method(linear_solver):
        linear_solver_parameters.setdefault("lu_solver", {})
        linear_solver_ic = False
    else:
        ks_parameters = linear_solver_parameters.setdefault("krylov_solver", {})  # noqa: E501
        linear_solver_ic = ks_parameters.setdefault("nonzero_initial_guess", False)  # noqa: E501
    ic = not linear or linear_solver_ic

    return (solver_parameters, linear_solver_parameters, ic, linear_solver_ic)


def process_adjoint_solver_parameters(linear_solver_parameters):
    return copy_parameters(linear_solver_parameters)
