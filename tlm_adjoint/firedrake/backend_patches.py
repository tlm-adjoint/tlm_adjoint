from .backend import (
    BaseFormAssembler, LinearVariationalProblem, NonlinearVariationalSolver,
    OneFormAssembler, Projector, SameMeshInterpolator, TwoFormAssembler,
    backend_Cofunction, backend_CofunctionSpace, backend_Constant,
    backend_DirichletBC, backend_Function, backend_FunctionSpace,
    backend_ScalarType, backend_assemble, backend_project, backend_solve,
    homogenize)
from ..interface import (
    DEFAULT_COMM, add_interface, check_space_type, comm_dup_cached,
    comm_parent, conjugate_dual_space_type, is_var, new_space_id, new_var_id,
    space_eq, var_comm, var_is_alias, var_new, var_update_state)

from ..equation import ZeroAssignment
from ..equations import Assignment, Conversion, LinearCombination
from ..manager import annotation_enabled, paused_manager, tlm_enabled
from ..patch import add_manager_controls, manager_method, patch_method

from .assembly import Assembly
from .assignment import ExprAssignment
from .expr import expr_zero, extract_variables, iter_expr, new_count
from .interpolation import ExprInterpolation
from .parameters import process_form_compiler_parameters
from .projection import Projection, LocalProjection
from .solve import EquationSolver
from .variables import (
    Cofunction, CofunctionInterface, Constant, ConstantInterface,
    ConstantSpaceInterface, FunctionInterface, FunctionSpaceInterface,
    constant_space, define_var_alias)

import inspect
import numbers
import operator
import ufl
import warnings

__all__ = \
    [
        "assemble",
        "project",
        "solve"
    ]


def packed_solver_parameters(solver_parameters, *, options_prefix=None,
                             nullspace=None, transpose_nullspace=None,
                             near_nullspace=None, pre_apply_bcs=None):
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
    set_parameter("pre_apply_bcs", pre_apply_bcs)

    return solver_parameters


def expr_new_x(expr, x):
    if x in extract_variables(expr):
        x_old = var_new(x)
        x_old.assign(x)
        return ufl.replace(expr, {x: x_old})
    else:
        return expr


def linear_equation_new_x(eq, x):
    lhs, rhs = eq.lhs, eq.rhs
    lhs_x_dep = x in extract_variables(lhs)
    rhs_x_dep = x in extract_variables(rhs)
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


@patch_method(BaseFormAssembler, "base_form_assembly_visitor")
def BaseFormAssembler_base_form_assembly_visitor(
        self, orig, orig_args, expr, tensor, *args):
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
        elif not isinstance(expr, ufl.classes.Form) \
                and isinstance(tensor, (backend_Function, backend_Cofunction)):
            return tensor.assign(orig_args())
    return orig_args()


def OneFormAssembler_assemble_post_call(self, return_value, *args, **kwargs):
    var_update_state(return_value)

    form_compiler_parameters = process_form_compiler_parameters(self._form_compiler_params)  # noqa: E501
    return_value._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


@manager_method(OneFormAssembler, "assemble",
                post_call=OneFormAssembler_assemble_post_call)
def OneFormAssembler_assemble(self, orig, orig_args, *args, **kwargs):
    return_value = orig_args()

    eq = Assembly(return_value, self._form,
                  form_compiler_parameters=self._form_compiler_params)
    assert not eq._pre_process_required
    eq._post_process()

    return return_value


@patch_method(TwoFormAssembler, "assemble")
def TwoFormAssembler_assemble(self, orig, orig_args, *args, **kwargs):
    return_value = orig_args()

    form_compiler_parameters = process_form_compiler_parameters(self._form_compiler_params)  # noqa: E501
    return_value._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

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
            and len(extract_variables(value)) > 0:
        eq = ExprAssignment(self, value)
    else:
        eq = None

    if eq is not None:
        assert not eq._pre_process_required
        eq._post_process()


@manager_method(backend_Constant, "__init__", patch_without_manager=True)
def backend_Constant__init__(self, orig, orig_args, value, domain=None, *,
                             name=None, space=None, comm=None,
                             **kwargs):
    if domain is not None:
        kwargs["domain"] = domain  # Backwards compatibility
    orig(self, value, name=name, **kwargs)

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

    if space not in space_ids:
        space_ids[space] = new_space_id()
    return space_ids[space]


def space_local_indices_cached(space, cls):
    if not hasattr(space, "_tlm_adjoint__local_indices"):
        space._tlm_adjoint__local_indices = {}
    local_indices = space._tlm_adjoint__local_indices

    if space not in local_indices:
        with cls(space).dat.vec_ro as x_v:
            local_indices[space] = x_v.getOwnershipRange()
    return local_indices[space]


@patch_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    if not ufl.duals.is_primal(self) and not ufl.duals.is_dual(self):
        raise NotImplementedError("Mixed primal/dual spaces not implemented")
    add_interface(self, FunctionSpaceInterface,
                  {"space": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id_cached(self)})
    n0, n1 = space_local_indices_cached(self, backend_Function)
    self._tlm_adjoint__space_interface_attrs["local_indices"] = (n0, n1)


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
    if not ufl.duals.is_primal(self) and not ufl.duals.is_dual(self):
        raise NotImplementedError("Mixed primal/dual spaces not implemented")
    add_interface(self, FunctionSpaceInterface,
                  {"space_dual": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id_cached(self)})
    n0, n1 = space_local_indices_cached(self, backend_Cofunction)
    self._tlm_adjoint__space_interface_attrs["local_indices"] = (n0, n1)


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
                and space_eq(y.function_space(), x.function_space()):
            Assignment(x, y).solve()
        else:
            ExprAssignment(x, y, subset=subset).solve()
        return x

    if subset is None:
        if isinstance(expr, backend_Function) \
                and space_eq(expr.function_space(), self.function_space()):
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


@manager_method(backend_Function, "copy", patch_without_manager=True)
def Function_copy(self, orig, orig_args, deepcopy=False):
    if deepcopy:
        F = var_new(self)
        F.assign(self)
    else:
        F = orig_args()
        define_var_alias(F, self, key=("copy",))
    return F


@manager_method(backend_Function, "project",
                post_call=var_update_state_post_call)
def Function_project(self, orig, orig_args, b, bcs=None,
                     solver_parameters=None, form_compiler_parameters=None,
                     use_slate_for_inverse=True, name=None, ad_block_tag=None):
    if use_slate_for_inverse:
        # Is a local solver actually used?
        projector = Projector(
            b, self.function_space(), bcs=bcs,
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


@patch_method(backend_Function, "riesz_representation")
def Function_riesz_representation(self, orig, orig_args,
                                  riesz_map="L2", *args, **kwargs):
    check_space_type(self, "primal")
    if riesz_map == "l2":
        with paused_manager():
            v = orig_args()
        if v.dat is self.dat:  # Backwards compatibility
            define_var_alias(v, self,
                             key=("riesz_representation", "l2"))
            v._tlm_adjoint__var_interface_attrs.d_setitem(
                "space_type",
                conjugate_dual_space_type(self._tlm_adjoint__var_interface_attrs["space_type"]))  # noqa: E501
        elif annotation_enabled() or tlm_enabled():
            eq = Conversion(v, self)
            assert not eq._pre_process_required
            eq._post_process()
    else:
        v = orig_args()
    return v


@patch_method(backend_Function, "sub")
def Function_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not var_is_alias(y):
        define_var_alias(y, self, key=("sub", i))
    return y


@patch_method(backend_Cofunction, "__init__")
def Cofunction__init__(self, orig, orig_args, function_space, val=None,
                       *args, **kwargs):
    orig_args()
    add_interface(self, CofunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_var_id(),
                   "space_type": "conjugate_dual", "static": False,
                   "cache": False,
                   "replacement_count": new_count(self._counted_class)})
    if isinstance(val, backend_Cofunction):
        define_var_alias(self, val, key=("Cofunction__init__",))


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
            if len(tuple(extract_variables(weight))) > 0:
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


@manager_method(backend_Cofunction, "copy", patch_without_manager=True)
def Cofunction_copy(self, orig, orig_args, deepcopy=False):
    if deepcopy:
        F = var_new(self)
        F.assign(self)
    else:
        F = orig_args()
        define_var_alias(F, self, key=("copy",))
    return F


@patch_method(backend_Cofunction, "riesz_representation")
def Cofunction_riesz_representation(self, orig, orig_args,
                                    riesz_map="L2", *args, **kwargs):
    # Backwards compatibility
    sig = inspect.signature(backend_Cofunction.riesz_representation)
    if "solver_parameters" in kwargs \
            and sig.parameters["solver_options"].kind == inspect.Parameter.KEYWORD_ONLY:  # noqa: E501
        warnings.warn("solver_parameters argument has been renamed to "
                      "solver_options",
                      FutureWarning, stacklevel=3)
        if "solver_options" in kwargs:
            raise TypeError("Cannot supply both solver_options and "
                            "and solver_parameters arguments")
        kwargs["solver_options"] = kwargs.pop("solver_parameters")

    check_space_type(self, "conjugate_dual")
    if riesz_map == "l2":
        with paused_manager():
            v = orig(self, riesz_map, *args, **kwargs)
        if v.dat is self.dat:  # Backwards compatibility
            define_var_alias(v, self,
                             key=("riesz_representation", "l2"))
            v._tlm_adjoint__var_interface_attrs.d_setitem(
                "space_type",
                conjugate_dual_space_type(self._tlm_adjoint__var_interface_attrs["space_type"]))  # noqa: E501
        elif annotation_enabled() or tlm_enabled():
            eq = Conversion(v, self)
            assert not eq._pre_process_required
            eq._post_process()
    else:
        v = orig(self, riesz_map, *args, **kwargs)
    return v


@patch_method(backend_Cofunction, "sub")
def Cofunction_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not var_is_alias(y):
        define_var_alias(y, self, key=("sub", i))
    return y


@patch_method(LinearVariationalProblem, "__init__")
def LinearVariationalProblem__init__(
        self, orig, orig_args, a, L, *args, **kwargs):
    orig_args()
    self._tlm_adjoint__a = a
    self._tlm_adjoint__L = L


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
    u = self._problem.u
    u_restrict = self._problem.u_restrict
    var_update_state(u_restrict)
    if u is not u_restrict:
        var_update_state(u)
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
        near_nullspace=self._ctx._near_nullspace,
        pre_apply_bcs=getattr(self, "pre_apply_bcs", None))
    form_compiler_parameters = self._problem.form_compiler_parameters

    u = self._problem.u
    u_restrict = self._problem.u_restrict

    if isinstance(self._problem, LinearVariationalProblem):
        vp_eq = linear_equation_new_x(
            self._problem._tlm_adjoint__a == self._problem._tlm_adjoint__L,
            u_restrict)
        vp_J = expr_new_x(self._problem.J, u_restrict)
        if u_restrict is not u:
            assert len(vp_eq.lhs.arguments()) == len(vp_J.arguments())
            arg_replace_map = dict(zip(vp_eq.lhs.arguments(), vp_J.arguments()))  # noqa: E501
            vp_eq = (ufl.replace(vp_eq.lhs, arg_replace_map)
                     == ufl.replace(vp_eq.rhs, arg_replace_map))
            del arg_replace_map
    else:
        vp_eq = (self._problem.F == 0)
        vp_J = self._problem.J

    eq = EquationSolver(
        vp_eq, u_restrict, self._problem.bcs,
        J=vp_J, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=self._problem._constant_jacobian,
        cache_rhs_assembly=False)

    eq._pre_process()
    return_value = orig_args()
    eq._post_process()

    if u_restrict is not u:
        eq = ExprInterpolation(u, u_restrict)
        assert not eq._pre_process_required
        eq._post_process()

    return return_value


def SameMeshInterpolator_interpolate_post_call(
        self, return_value, *args, **kwargs):
    var_update_state(return_value)
    return return_value


@manager_method(SameMeshInterpolator, "_interpolate",
                post_call=SameMeshInterpolator_interpolate_post_call)
def SameMeshInterpolator_interpolate(
        self, orig, orig_args, *function, output=None,
        transpose=None, adjoint=False, default_missing_val=None, **kwargs):
    if transpose is not None:
        adjoint = transpose or adjoint
    if default_missing_val is not None:
        raise NotImplementedError("default_missing_val not supported")

    return_value = orig_args()
    check_space_type(return_value, "conjugate_dual" if adjoint else "primal")

    args = ufl.algorithms.extract_arguments(self.expr)
    if len(args) != len(function):
        raise TypeError("Unexpected number of functions")
    expr = ufl.replace(self.expr, dict(zip(args, function)))
    expr = expr_new_x(expr, return_value)
    eq = ExprInterpolation(return_value, expr)

    assert not eq._pre_process_required
    eq._post_process()
    return return_value


assemble = add_manager_controls(backend_assemble)
solve = add_manager_controls(backend_solve)
project = add_manager_controls(backend_project)
