from .backend import (
    FormAssembler, LinearSolver, NonlinearVariationalSolver, Parameters,
    Projector, SameMeshInterpolator, backend_Cofunction, backend_Constant,
    backend_DirichletBC, backend_Function, backend_Vector, backend_assemble,
    backend_project, backend_solve, parameters)
from ..interface import (
    VariableStateChangeError, is_var, space_id, var_comm, var_new, var_space,
    var_state_is_locked, var_update_state)
from .backend_code_generator_interface import (
    copy_parameters_dict, update_parameters_dict)

from ..equation import ZeroAssignment
from ..equations import Assignment, LinearCombination
from ..manager import annotation_enabled, tlm_enabled
from ..patch import (
    add_manager_controls, manager_method, patch_function, patch_method,
    patch_property)

from .backend_interface import Cofunction
from .equations import (
    Assembly, EquationSolver, ExprInterpolation, Projection, expr_new_x,
    linear_equation_new_x)
from .functions import (
    Constant, define_var_alias, expr_zero, extract_coefficients, iter_expr)
from .firedrake_equations import ExprAssignment, LocalProjection

import numbers
import operator
import ufl

__all__ = \
    [
        "assemble",
        "project",
        "solve"
    ]


def parameters_dict_equal(parameters_a, parameters_b):
    if set(parameters_a) != set(parameters_b):
        return False
    for key_a, value_a in parameters_a.items():
        value_b = parameters_b[key_a]
        if isinstance(value_a, (Parameters, dict)):
            if not isinstance(value_b, (Parameters, dict)):
                return False
            elif not parameters_dict_equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False
    return True


def packed_solver_parameters(solver_parameters, *, options_prefix=None,
                             nullspace=None, transpose_nullspace=None,
                             near_nullspace=None):
    solver_parameters = dict(solver_parameters)
    tlm_adjoint_parameters = solver_parameters["tlm_adjoint"] = \
        dict(solver_parameters.get("tlm_adjoint", {}))

    def set_parameter(key, value):
        if value is not None:
            if key in tlm_adjoint_parameters:
                raise TypeError(f"Cannot pass both {key:s} argument and "
                                f"solver parameter")
            tlm_adjoint_parameters[key] = value

    set_parameter("options_prefix", options_prefix)
    set_parameter("nullspace", nullspace)
    set_parameter("transpose_nullspace", transpose_nullspace)
    set_parameter("near_nullspace", near_nullspace)

    return solver_parameters


# Aim for compatibility with Firedrake API


def FormAssembler_assemble_post_call(self, return_value, *args, **kwargs):
    if is_var(return_value):
        var_update_state(return_value)

    if len(self._form.arguments()) > 0:
        form_compiler_parameters = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        update_parameters_dict(form_compiler_parameters,
                               self._form_compiler_params)
        return_value._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


@manager_method(FormAssembler, "assemble",
                post_call=FormAssembler_assemble_post_call)
def FormAssembler_assemble(self, orig, orig_args, *args,
                           annotate, tlm, **kwargs):
    return_value = orig_args()

    if len(self._form.arguments()) == 1:
        eq = Assembly(return_value, self._form,
                      form_compiler_parameters=self._form_compiler_params)
        assert len(eq.initial_condition_dependencies()) == 0
        eq._post_process(annotate=annotate, tlm=tlm)

    return return_value


def DirichletBC_function_arg_fset(self, orig, orig_args, g):
    if getattr(self, "_tlm_adjoint__function_arg_set", False) \
            and is_var(self.function_arg) \
            and var_state_is_locked(self.function_arg):
        raise VariableStateChangeError("Cannot change DirichletBC if the "
                                       "value state is locked")
    return_value = orig_args()
    self._tlm_adjoint__function_arg_set = True
    return return_value


@patch_property(backend_DirichletBC, "function_arg",
                fset=DirichletBC_function_arg_fset)
def DirichletBC_function_arg(self, orig):
    return orig()


def Constant_init_assign(self, value, annotate, tlm):
    if is_var(value):
        eq = Assignment(self, value)
    elif isinstance(value, ufl.classes.Expr):
        eq = ExprAssignment(self, value)
    else:
        eq = None

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
        eq._post_process(annotate=annotate, tlm=tlm)


@manager_method(backend_Constant, "__init__")
def backend_Constant__init__(self, orig, orig_args, value, *args,
                             annotate, tlm, **kwargs):
    orig_args()
    Constant_init_assign(self, value, annotate, tlm)


# Patch the subclass constructor separately so that all variable attributes are
# set before annotation
@manager_method(Constant, "__init__")
def Constant__init__(self, orig, orig_args, value=None, *args,
                     annotate, tlm, **kwargs):
    orig_args()
    if value is not None:
        Constant_init_assign(self, value, annotate, tlm)


def var_update_state_post_call(self, return_value, *args, **kwargs):
    if is_var(self):
        var_update_state(self)
    return return_value


@manager_method(backend_Constant, "assign",
                post_call=var_update_state_post_call)
def Constant_assign(self, orig, orig_args, value, *, annotate, tlm):
    if isinstance(value, numbers.Complex):
        eq = Assignment(self, Constant(value, comm=var_comm(self)))
    elif isinstance(value, backend_Constant):
        if value is not self:
            eq = Assignment(self, value)
        else:
            eq = None
    elif isinstance(value, ufl.classes.Expr):
        eq = ExprInterpolation(
            self, expr_new_x(value, self, annotate=annotate, tlm=tlm))
    else:
        raise TypeError(f"Unexpected type: {type(value)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    return_value = orig_args()
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


def register_in_place(cls, name, op):
    @patch_method(cls, name)
    def wrapped_op(self, orig, orig_args, other):
        annotate = annotation_enabled()
        tlm = tlm_enabled()
        if annotate or tlm:
            return self.assign(op(self, other))
        else:
            return_value = orig_args()
            var_update_state(return_value)
            return return_value


register_in_place(backend_Function, "__iadd__", operator.add)
register_in_place(backend_Function, "__isub__", operator.sub)
register_in_place(backend_Function, "__imul__", operator.mul)
register_in_place(backend_Function, "__itruediv__", operator.truediv)


@manager_method(backend_Function, "assign",
                post_call=var_update_state_post_call)
def Function_assign(self, orig, orig_args, expr, subset=None, *,
                    annotate, tlm):
    expr = ufl.as_ufl(expr)

    def assign(x, y, *,
               subset=None):
        if x is None:
            x = var_new(y)
        if subset is None \
                and isinstance(y, ufl.classes.Zero):
            ZeroAssignment(x).solve(annotate=annotate, tlm=tlm)
        elif subset is None \
                and isinstance(y, backend_Function) \
                and space_id(var_space(y)) == space_id(var_space(x)):
            Assignment(x, y).solve(annotate=annotate, tlm=tlm)
        else:
            ExprAssignment(x, y, subset=subset).solve(annotate=annotate, tlm=tlm)  # noqa: E501
        return x

    if subset is None:
        if isinstance(expr, backend_Function) \
                and space_id(var_space(expr)) == space_id(var_space(self)):
            if expr is not self:
                eq = Assignment(self, expr)
            else:
                eq = None
        elif isinstance(expr, ufl.classes.Expr):
            expr = expr_new_x(expr, self, annotate=annotate, tlm=tlm)
            eq = ExprAssignment(self, expr)
        else:
            raise TypeError(f"Unexpected type: {type(expr)}")
    else:
        if isinstance(expr, ufl.classes.Expr):
            x_0 = assign(None, self)
            expr = ufl.replace(expr, {self: x_0})
            x_1 = assign(None, self, subset=subset)
            assign(self, expr_zero(self))
            eq = ExprAssignment(self, expr, subset=subset)
        else:
            raise TypeError(f"Unexpected type: {type(expr)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    orig(self, expr, subset=subset)
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)

    if subset is not None:
        x_2 = assign(None, self)
        assign(self, x_0 - x_1 + x_2)

    return self


@manager_method(backend_Function, "project",
                post_call=var_update_state_post_call)
def Function_project(self, orig, orig_args, b, bcs=None,
                     solver_parameters=None, form_compiler_parameters=None,
                     use_slate_for_inverse=True, name=None, ad_block_tag=None,
                     *, annotate, tlm):
    if use_slate_for_inverse:
        # Is a local solver actually used?
        projector = Projector(
            b, var_space(self), bcs=bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            use_slate_for_inverse=True)
        use_slate_for_inverse = getattr(projector,
                                        "use_slate_for_inverse", False)

    if use_slate_for_inverse:
        if bcs is not None and (isinstance(bcs, backend_DirichletBC)
                                or len(bcs) > 0):
            raise NotImplementedError("Boundary conditions not supported")
        eq = LocalProjection(
            self, expr_new_x(b, self, annotate=annotate, tlm=tlm),
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)
    else:
        eq = Projection(
            self, expr_new_x(b, self, annotate=annotate, tlm=tlm), bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@manager_method(backend_Function, "copy", patch_without_manager=True)
def Function_copy(self, orig, orig_args, deepcopy=False, *, annotate, tlm):
    if deepcopy:
        F = var_new(self)
        F.assign(self, annotate=annotate, tlm=tlm)
    else:
        F = orig_args()
        define_var_alias(F, self, key=("copy",))
    return F


register_in_place(backend_Cofunction, "__iadd__", operator.add)
register_in_place(backend_Cofunction, "__isub__", operator.sub)
register_in_place(backend_Cofunction, "__imul__",
                  lambda self, other: operator.mul(other, self))


@manager_method(backend_Cofunction, "assign",
                post_call=var_update_state_post_call)
def Cofunction_assign(self, orig, orig_args, expr, subset=None, *,
                      annotate, tlm):
    if subset is not None:
        raise NotImplementedError("subset not supported")

    expr = ufl.as_ufl(expr)

    if isinstance(expr, ufl.classes.Zero):
        eq = ZeroAssignment(self)
    elif isinstance(expr, backend_Cofunction):
        if expr is not self:
            eq = Assignment(self, expr)
        else:
            eq = None
    elif isinstance(expr, ufl.classes.FormSum):
        for weight, comp in iter_expr(expr):
            if len(tuple(c for c in extract_coefficients(weight)
                         if is_var(c))) > 0:
                # See Firedrake issue #3292
                raise NotImplementedError("FormSum weights cannot depend on "
                                          "variables")
            if not isinstance(comp, backend_Cofunction):
                raise TypeError(f"Unexpected type: {type(comp)}")

        expr = expr_new_x(expr, self, annotate=annotate, tlm=tlm)
        # Note: Ignores weight dependencies
        eq = LinearCombination(self, *iter_expr(expr, evaluate_weights=True))
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    orig(self, expr, subset=subset)
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)
    return self


def LinearSolver_solve_post_call(self, return_value, x, b):
    if isinstance(x, backend_Vector):
        x = x.function
    if isinstance(b, backend_Vector):
        b = b.function
    var_update_state(x)
    var_update_state(b)
    return return_value


@manager_method(LinearSolver, "solve",
                post_call=LinearSolver_solve_post_call)
def LinearSolver_solve(self, orig, orig_args, x, b, *, annotate, tlm):
    if self.P is not self.A:
        raise NotImplementedError("Preconditioners not supported")

    if isinstance(x, backend_Vector):
        x = x.function
    if isinstance(b, backend_Vector):
        b = b.function

    A = self.A
    bcs = A.bcs
    solver_parameters = packed_solver_parameters(
        self.parameters, options_prefix=self.options_prefix,
        nullspace=self.nullspace, transpose_nullspace=self.transpose_nullspace,
        near_nullspace=self.near_nullspace)
    form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters

    eq = EquationSolver(
        linear_equation_new_x(A.a == b, x,
                              annotate=annotate, tlm=tlm),
        x, bcs, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@patch_method(NonlinearVariationalSolver, "__init__")
def NonlinearVariationalSolver__init__(
        self, orig, orig_args, problem, *, appctx=None,
        pre_jacobian_callback=None, post_jacobian_callback=None,
        pre_function_callback=None, post_function_callback=None, **kwargs):
    orig_args()
    self._tlm_adjoint__appctx = {} if appctx is None else appctx
    self._tlm_adjoint__pre_jacobian_callback = pre_jacobian_callback
    self._tlm_adjoint__post_jacobian_callback = post_jacobian_callback
    self._tlm_adjoint__pre_function_callback = pre_function_callback
    self._tlm_adjoint__post_function_callback = post_function_callback
    self._tlm_adjoint__transfer_manager = None


@patch_method(NonlinearVariationalSolver, "set_transfer_manager")
def NonlinearVariationalSolver_set_transfer_manager(
        self, orig, orig_args, manager):
    orig_args()
    self._tlm_adjoint__transfer_manager = manager


def NonlinearVariationalSolver_solve_post_call(
        self, return_value, *args, **kwargs):
    var_update_state(self._problem.u)
    return return_value


@manager_method(NonlinearVariationalSolver, "solve",
                post_call=NonlinearVariationalSolver_solve_post_call)
def NonlinearVariationalSolver_solve(
        self, orig, orig_args, bounds=None, *, annotate, tlm):
    if len(set(self._tlm_adjoint__appctx).difference({"state", "form_compiler_parameters"})) > 0:  # noqa: E501
        raise NotImplementedError("appctx not supported")
    if self._tlm_adjoint__pre_jacobian_callback is not None \
            or self._tlm_adjoint__post_jacobian_callback is not None \
            or self._tlm_adjoint__pre_function_callback is not None \
            or self._tlm_adjoint__post_function_callback is not None:
        raise NotImplementedError("Callbacks not supported")
    if self._tlm_adjoint__transfer_manager is not None:
        raise NotImplementedError("Transfer managers not supported")
    if bounds is not None:
        raise NotImplementedError("Bounds not supported")
    if self._problem.Jp is not None:
        raise NotImplementedError("Preconditioners not supported")

    solver_parameters = packed_solver_parameters(
        self.parameters, options_prefix=self.options_prefix,
        nullspace=self._ctx._nullspace,
        transpose_nullspace=self._ctx._nullspace_T,
        near_nullspace=self._ctx._near_nullspace)
    form_compiler_parameters = self._problem.form_compiler_parameters

    eq = EquationSolver(
        self._problem.F == 0, self._problem.u, self._problem.bcs,
        J=self._problem.J, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=self._problem._constant_jacobian,
        cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


def SameMeshInterpolator_interpolate_post_call(
        self, return_value, *args, **kwargs):
    var_update_state(return_value)
    return return_value


@manager_method(SameMeshInterpolator, "_interpolate",
                post_call=SameMeshInterpolator_interpolate_post_call)
def SameMeshInterpolator_interpolate(
        self, orig, orig_args, *function, output=None, transpose=False,
        default_missing_val=None,
        annotate, tlm, **kwargs):
    if transpose:
        raise NotImplementedError("transpose not supported")
    if default_missing_val is not None:
        raise NotImplementedError("default_missing_val not supported")

    return_value = orig_args()

    args = ufl.algorithms.extract_arguments(self.expr)
    if len(args) != len(function):
        raise TypeError("Unexpected number of functions")
    expr = ufl.replace(self.expr, dict(zip(args, function)))
    expr = expr_new_x(expr, return_value, annotate=annotate, tlm=tlm)
    eq = ExprInterpolation(return_value, expr)

    assert len(eq.initial_condition_dependencies()) == 0
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


def fn_globals(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn.__globals__


@patch_function(fn_globals(backend_assemble)["base_form_assembly_visitor"])
def base_form_assembly_visitor(orig, orig_args, expr, tensor, *args, **kwargs):
    annotate = annotation_enabled()
    tlm = tlm_enabled()
    if annotate or tlm:
        if isinstance(expr, ufl.classes.FormSum) \
                and all(isinstance(comp, backend_Cofunction)
                        for comp in args):
            if tensor is None:
                test, = expr.arguments()
                tensor = Cofunction(test.function_space().dual())
            rexpr = expr_zero(expr)
            if len(expr.weights()) != len(args):
                raise ValueError("Invalid args")
            for weight, comp in zip(expr.weights(), args):
                rexpr = rexpr + weight * comp
            return tensor.assign(rexpr)
        elif isinstance(expr, (ufl.classes.Argument,
                               ufl.classes.Coargument,
                               ufl.classes.Coefficient,
                               ufl.classes.Cofunction,
                               ufl.classes.Interpolate,
                               ufl.classes.ZeroBaseForm)):
            if tensor is None:
                return orig_args()
            else:
                return tensor.assign(orig_args())
        elif isinstance(expr, ufl.classes.Form):
            # Handled via FormAssembler.assemble
            pass
        else:
            raise NotImplementedError("Case not implemented")
    return orig_args()


fn_globals(backend_assemble)["base_form_assembly_visitor"] = base_form_assembly_visitor  # noqa: E501


assemble = add_manager_controls(backend_assemble)
solve = add_manager_controls(backend_solve)
project = add_manager_controls(backend_project)