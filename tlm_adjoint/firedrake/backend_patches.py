from .backend import (
    FormAssembler, LinearSolver, NonlinearVariationalSolver, Parameters,
    Projector, SameMeshInterpolator, backend_Cofunction,
    backend_CofunctionSpace, backend_Constant, backend_DirichletBC,
    backend_Function, backend_FunctionSpace, backend_ScalarType,
    backend_Vector, backend_assemble, backend_project, backend_solve,
    homogenize)
from ..interface import (
    DEFAULT_COMM, add_interface, check_space_type, comm_dup_cached,
    comm_parent, is_var, new_space_id, new_var_id, relative_space_type,
    space_id, var_comm, var_is_alias, var_new, var_space, var_update_state)

from ..equation import ZeroAssignment
from ..equations import Assignment, LinearCombination
from ..manager import annotation_enabled, paused_manager, tlm_enabled
from ..patch import (
    add_manager_controls, manager_method, patch_function, patch_method,
    patch_property)

from .assembly import Assembly
from .assignment import ExprAssignment
from .expr import expr_zero, extract_coefficients, iter_expr, new_count
from .interpolation import ExprInterpolation
from .parameters import process_form_compiler_parameters
from .projection import Projection, LocalProjection
from .solve import EquationSolver
from .variables import (
    Cofunction, CofunctionInterface, Constant, ConstantInterface,
    ConstantSpaceInterface, FunctionInterface, FunctionSpaceInterface,
    constant_space, define_var_alias)

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


def expr_new_x(expr, x):
    if x in extract_coefficients(expr):
        x_old = var_new(x)
        x_old.assign(x)
        return ufl.replace(expr, {x: x_old})
    else:
        return expr


def linear_equation_new_x(eq, x):
    lhs, rhs = eq.lhs, eq.rhs
    lhs_x_dep = x in extract_coefficients(lhs)
    rhs_x_dep = x in extract_coefficients(rhs)
    if lhs_x_dep or rhs_x_dep:
        x_old = var_new(x)
        x_old.assign(x)
        if lhs_x_dep:
            lhs = ufl.replace(lhs, {x: x_old})
        if rhs_x_dep:
            rhs = ufl.replace(rhs, {x: x_old})
        return lhs == rhs
    else:
        return eq


# Aim for compatibility with Firedrake API


def FormAssembler_assemble_post_call(self, return_value, *args, **kwargs):
    if is_var(return_value):
        var_update_state(return_value)

    if len(self._form.arguments()) > 0:
        form_compiler_parameters = process_form_compiler_parameters(self._form_compiler_params)  # noqa: E501
        return_value._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


@manager_method(FormAssembler, "assemble",
                post_call=FormAssembler_assemble_post_call)
def FormAssembler_assemble(self, orig, orig_args, *args, **kwargs):
    return_value = orig_args()

    if len(self._form.arguments()) == 1:
        eq = Assembly(return_value, self._form,
                      form_compiler_parameters=self._form_compiler_params)
        assert not eq._pre_process_required
        eq._post_process()

    return return_value


@patch_method(backend_DirichletBC, "__init__")
def DirichletBC__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    if isinstance(self._function_arg, ufl.classes.Zero):
        self._tlm_adjoint__hbc = self
    else:
        self._tlm_adjoint__hbc = homogenize(self)


def Constant_init_assign(self, value):
    if is_var(value):
        eq = Assignment(self, value)
    elif isinstance(value, ufl.classes.Expr) \
            and len(tuple(dep for dep in extract_coefficients(value) if is_var(dep))) > 0:  # noqa: E501
        eq = ExprAssignment(self, value)
    else:
        eq = None

    if eq is not None:
        assert not eq._pre_process_required
        eq._post_process()


@patch_method(backend_Constant, "__init__")
def backend_Constant__init__(self, orig, orig_args, value, domain=None, *,
                             name=None, space=None, comm=None,
                             **kwargs):
    with paused_manager():
        orig(self, value, domain=domain, name=name, **kwargs)

    if name is None:
        name = self.name
    if comm is None:
        if domain is None:
            comm = DEFAULT_COMM
        else:
            comm = domain.comm
    comm = comm_parent(comm)

    if space is None:
        space = constant_space(self.ufl_shape, domain=domain)
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_var_id(), "name": lambda x: name,
                   "state": [0], "space": space,
                   "space_type": "primal", "dtype": self.dat.dtype.type,
                   "static": False, "cache": False,
                   "replacement_count": new_count(self._counted_class)})

    if annotation_enabled() or tlm_enabled():
        Constant_init_assign(self, value)


# Patch the subclass constructor separately so that all variable attributes are
# set before annotation
@manager_method(Constant, "__init__")
def Constant__init__(self, orig, orig_args, value=None, *args, **kwargs):
    orig_args()
    if value is not None:
        Constant_init_assign(self, value)


def var_update_state_post_call(self, return_value, *args, **kwargs):
    if is_var(self):
        var_update_state(self)
    return return_value


@manager_method(backend_Constant, "assign",
                post_call=var_update_state_post_call)
def Constant_assign(self, orig, orig_args, value):
    if isinstance(value, numbers.Complex):
        eq = Assignment(self, Constant(value, comm=var_comm(self)))
    elif isinstance(value, backend_Constant):
        if value is not self:
            eq = Assignment(self, value)
        else:
            eq = None
    elif isinstance(value, ufl.classes.Expr):
        eq = ExprInterpolation(
            self, expr_new_x(value, self))
    else:
        raise TypeError(f"Unexpected type: {type(value)}")

    if eq is not None:
        assert not eq._pre_process_required
    return_value = orig_args()
    if eq is not None:
        eq._post_process()
    return return_value


def new_space_id_cached(space):
    mesh = space.mesh()
    if not hasattr(mesh, "_tlm_adjoint__space_ids"):
        mesh._tlm_adjoint__space_ids = {}
    space_ids = mesh._tlm_adjoint__space_ids

    key = (space, ufl.duals.is_primal(space))
    if key not in space_ids:
        space_ids[key] = new_space_id()
    return space_ids[key]


@patch_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id_cached(self)})


@patch_method(backend_FunctionSpace, "dual")
def FunctionSpace_dual(self, orig, orig_args):
    if "space_dual" not in self._tlm_adjoint__space_interface_attrs:
        self._tlm_adjoint__space_interface_attrs["space_dual"] = orig_args()
    space_dual = self._tlm_adjoint__space_interface_attrs["space_dual"]
    if "space" not in space_dual._tlm_adjoint__space_interface_attrs:
        space_dual._tlm_adjoint__space_interface_attrs["space"] = self
    return space_dual


@patch_method(backend_CofunctionSpace, "__init__")
def CofunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space_dual": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id_cached(self)})


@patch_method(backend_CofunctionSpace, "dual")
def CofunctionSpace_dual(self, orig, orig_args):
    if "space" not in self._tlm_adjoint__space_interface_attrs:
        self._tlm_adjoint__space_interface_attrs["space"] = orig_args()
    space = self._tlm_adjoint__space_interface_attrs["space"]
    if "space_dual" not in space._tlm_adjoint__space_interface_attrs:
        space._tlm_adjoint__space_interface_attrs["space_dual"] = self
    return space


@patch_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, function_space, val=None,
                     *args, **kwargs):
    orig_args()
    add_interface(self, FunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_var_id(),
                   "state": [self.dat, getattr(self.dat, "dat_version", None)],
                   "space_type": "primal", "static": False, "cache": False,
                   "replacement_count": new_count(self._counted_class)})
    if isinstance(val, backend_Function):
        define_var_alias(self, val, key=("Function__init__",))


@patch_method(backend_Function, "__getattr__")
def Function__getattr__(self, orig, orig_args, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return orig_args()


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
def Function_assign(self, orig, orig_args, expr, subset=None):
    expr = ufl.as_ufl(expr)

    def assign(x, y, *,
               subset=None):
        if x is None:
            x = var_new(y)
        if subset is None \
                and isinstance(y, ufl.classes.Zero):
            ZeroAssignment(x).solve()
        elif subset is None \
                and isinstance(y, backend_Function) \
                and space_id(var_space(y)) == space_id(var_space(x)):
            Assignment(x, y).solve()
        else:
            ExprAssignment(x, y, subset=subset).solve()
        return x

    if subset is None:
        if isinstance(expr, backend_Function) \
                and space_id(var_space(expr)) == space_id(var_space(self)):
            if expr is not self:
                eq = Assignment(self, expr)
            else:
                eq = None
        elif isinstance(expr, ufl.classes.Expr):
            expr = expr_new_x(expr, self)
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
        assert not eq._pre_process_required
    orig(self, expr, subset=subset)
    if eq is not None:
        eq._post_process()

    if subset is not None:
        x_2 = assign(None, self)
        assign(self, x_0 - x_1 + x_2)

    return self


@manager_method(backend_Function, "project",
                post_call=var_update_state_post_call)
def Function_project(self, orig, orig_args, b, bcs=None,
                     solver_parameters=None, form_compiler_parameters=None,
                     use_slate_for_inverse=True, name=None, ad_block_tag=None):
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
            self, expr_new_x(b, self),
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)
    else:
        eq = Projection(
            self, expr_new_x(b, self), bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process()
    return_value = orig_args()
    eq._post_process()
    return return_value


@manager_method(backend_Function, "copy", patch_without_manager=True)
def Function_copy(self, orig, orig_args, deepcopy=False):
    if deepcopy:
        F = var_new(self)
        F.assign(self)
    else:
        F = orig_args()
        define_var_alias(F, self, key=("copy",))
    return F


@patch_method(backend_Function, "riesz_representation")
def Function_riesz_representation(self, orig, orig_args,
                                  riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "primal")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


@patch_property(backend_Function, "subfunctions", cached=True)
def Function_subfunctions(self, orig):
    Y = orig()
    for i, y in enumerate(Y):
        define_var_alias(y, self, key=("subfunctions", i))
    return Y


@patch_method(backend_Function, "sub")
def Function_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not var_is_alias(y):
        define_var_alias(y, self, key=("sub", i))
    return y


register_in_place(backend_Cofunction, "__iadd__", operator.add)
register_in_place(backend_Cofunction, "__isub__", operator.sub)
register_in_place(backend_Cofunction, "__imul__",
                  lambda self, other: operator.mul(other, self))


@manager_method(backend_Cofunction, "assign",
                post_call=var_update_state_post_call)
def Cofunction_assign(self, orig, orig_args, expr, subset=None):
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

        expr = expr_new_x(expr, self)
        # Note: Ignores weight dependencies
        eq = LinearCombination(self, *iter_expr(expr, evaluate_weights=True))
    else:
        raise TypeError(f"Unexpected type: {type(expr)}")

    if eq is not None:
        assert not eq._pre_process_required
    orig(self, expr, subset=subset)
    if eq is not None:
        eq._post_process()
    return self


@patch_method(backend_Cofunction, "__init__")
def Cofunction__init__(self, orig, orig_args, function_space, val=None,
                       *args, **kwargs):
    orig_args()
    add_interface(self, CofunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_var_id(),
                   "state": [self.dat, getattr(self.dat, "dat_version", None)],
                   "space_type": "conjugate_dual", "static": False,
                   "cache": False,
                   "replacement_count": new_count(self._counted_class)})
    if isinstance(val, backend_Cofunction):
        define_var_alias(self, val, key=("Cofunction__init__",))


@patch_method(backend_Cofunction, "riesz_representation")
def Cofunction_riesz_representation(self, orig, orig_args,
                                    riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "conjugate_dual")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


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
def LinearSolver_solve(self, orig, orig_args, x, b):
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
        linear_equation_new_x(A.a == b, x),
        x, bcs, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process()
    return_value = orig_args()
    eq._post_process()
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
        self, orig, orig_args, bounds=None):
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

    eq._pre_process()
    return_value = orig_args()
    eq._post_process()
    return return_value


def SameMeshInterpolator_interpolate_post_call(
        self, return_value, *args, **kwargs):
    var_update_state(return_value)
    return return_value


@manager_method(SameMeshInterpolator, "_interpolate",
                post_call=SameMeshInterpolator_interpolate_post_call)
def SameMeshInterpolator_interpolate(
        self, orig, orig_args, *function, output=None, transpose=False,
        default_missing_val=None, **kwargs):
    if transpose:
        raise NotImplementedError("transpose not supported")
    if default_missing_val is not None:
        raise NotImplementedError("default_missing_val not supported")

    return_value = orig_args()

    args = ufl.algorithms.extract_arguments(self.expr)
    if len(args) != len(function):
        raise TypeError("Unexpected number of functions")
    expr = ufl.replace(self.expr, dict(zip(args, function)))
    expr = expr_new_x(expr, return_value)
    eq = ExprInterpolation(return_value, expr)

    assert not eq._pre_process_required
    eq._post_process()
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
        elif isinstance(expr, (ufl.classes.Action,
                               ufl.classes.Argument,
                               ufl.classes.Coargument,
                               ufl.classes.Coefficient,
                               ufl.classes.Cofunction,
                               ufl.classes.Interpolate,
                               ufl.classes.ZeroBaseForm)):
            if isinstance(tensor, (backend_Function, backend_Cofunction)):
                return tensor.assign(orig_args())
            else:
                return orig_args()
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
