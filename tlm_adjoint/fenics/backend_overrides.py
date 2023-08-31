#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    Form, KrylovSolver, LUSolver, LinearVariationalSolver,
    NonlinearVariationalProblem, NonlinearVariationalSolver, Parameters,
    backend_Constant, backend_DirichletBC, backend_Function, backend_Matrix,
    backend_Vector, backend_project, backend_solve, cpp_Assembler,
    cpp_SystemAssembler, parameters)
from ..interface import (
    function_assign, function_comm, function_new, function_space,
    function_update_state, space_id, space_new)
from .backend_code_generator_interface import (
    copy_parameters_dict, update_parameters_dict)

from ..equations import Assignment
from ..manager import annotation_enabled, tlm_enabled
from ..override import (
    add_manager_controls, manager_method, override_function, override_method)

from .equations import (
    EquationSolver, ExprInterpolation, Projection, expr_new_x,
    linear_equation_new_x)
from .functions import Constant, define_function_alias

import numpy as np
import ufl
import warnings
import weakref

__all__ = \
    [
        "project",
        "solve"
    ]


def parameters_dict_equal(parameters_a, parameters_b):
    for key_a in parameters_a:
        if key_a not in parameters_b:
            return False
        value_a = parameters_a[key_a]
        value_b = parameters_b[key_a]
        if isinstance(value_a, (Parameters, dict)):
            if not isinstance(value_b, (Parameters, dict)):
                return False
            elif not parameters_dict_equal(value_a, value_b):
                return False
        elif value_a != value_b:
            return False
    for key_b in parameters_b:
        if key_b not in parameters_a:
            return False
    return True


class Attributes:
    def __init__(self):
        self._d = {}
        self._keys = {}

    def __getitem__(self, key):
        obj, key = key
        key = (id(obj), key)
        return self._d[key]

    def __setitem__(self, key, value):
        obj, key = key
        key = (id(obj), key)

        if id(obj) not in self._keys:
            self._keys[id(obj)] = []

            def weakref_finalize(obj_id, d, keys):
                for key in keys.pop(obj_id, []):
                    d.pop(key, None)

            weakref.finalize(obj, weakref_finalize,
                             id(obj), self._d, self._keys)

        self._d[key] = value
        self._keys[id(obj)].append(key)


_attrs = Attributes()


def _getattr(self, key):
    return _attrs[(self, key)]


def _setattr(self, key, value):
    _attrs[(self, key)] = value


# Aim for compatibility with FEniCS 2019.1.0 API


@override_method(Form, "__init__")
def Form__init__(self, orig, orig_args, form, *, form_compiler_parameters=None,
                 **kwargs):
    orig_args()

    self._tlm_adjoint__form = form
    if form_compiler_parameters is None:
        form_compiler_parameters = {}
    self._tlm_adjoint__form_compiler_parameters = form_compiler_parameters


@override_method(cpp_Assembler, "assemble")
def Assembler_assemble(self, orig, orig_args, tensor, form):
    if isinstance(tensor, backend_Function):
        tensor = tensor.vector()
    return_value = orig(self, tensor, form)
    if hasattr(tensor, "_tlm_adjoint__function"):
        function_update_state(tensor._tlm_adjoint__function)

    if hasattr(form, "_tlm_adjoint__form") and \
            len(form._tlm_adjoint__form.arguments()) > 0:
        form_compiler_parameters = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
        update_parameters_dict(form_compiler_parameters,
                               form._tlm_adjoint__form_compiler_parameters)

        if self.add_values and hasattr(tensor, "_tlm_adjoint__form"):
            if len(tensor._tlm_adjoint__bcs) > 0:
                warnings.warn("Unexpected boundary conditions",
                              RuntimeWarning)
            if not parameters_dict_equal(
                    tensor._tlm_adjoint_form_compiler_parameters,
                    form_compiler_parameters):
                warnings.warn("Unexpected form compiler parameters",
                              RuntimeWarning)
            tensor._tlm_adjoint__form += form._tlm_adjoint__form
        else:
            tensor._tlm_adjoint__form = form._tlm_adjoint__form
            tensor._tlm_adjoint__bcs = []
            tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


@override_method(cpp_SystemAssembler, "__init__")
def SystemAssembler__init__(self, orig, orig_args, A_form, b_form, bcs=None):
    orig_args()

    _setattr(self, "A_form", A_form)
    _setattr(self, "b_form", b_form)
    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    else:
        bcs = tuple(bcs)
    _setattr(self, "bcs", bcs)


@override_method(cpp_SystemAssembler, "assemble")
def SystemAssembler_assemble(self, orig, orig_args, *args):
    return_value = orig_args()

    A_tensor = None
    b_tensor = None
    x0 = None
    if len(args) == 1:
        if isinstance(args[0], backend_Matrix):
            A_tensor, = args
        else:
            b_tensor, = args
    elif len(args) == 2:
        if isinstance(args[0], backend_Matrix):
            A_tensor, b_tensor = args
        elif isinstance(args[0], backend_Vector):
            b_tensor, x0 = args
        # > 2019.1.0 case here
    else:
        A_tensor, b_tensor, x0 = args

    if b_tensor is not None and hasattr(b_tensor, "_tlm_adjoint__function"):
        function_update_state(b_tensor._tlm_adjoint__function)

    if A_tensor is not None and b_tensor is not None and x0 is None \
            and hasattr(_getattr(self, "A_form"), "_tlm_adjoint__form") \
            and hasattr(_getattr(self, "b_form"), "_tlm_adjoint__form"):
        for tensor, form in ((A_tensor, _getattr(self, "A_form")),
                             (b_tensor, _getattr(self, "b_form"))):
            form_compiler_parameters = copy_parameters_dict(parameters["form_compiler"])  # noqa: E501
            update_parameters_dict(form_compiler_parameters,
                                   form._tlm_adjoint__form_compiler_parameters)

            if self.add_values and hasattr(tensor, "_tlm_adjoint__form"):
                if len(tensor._tlm_adjoint__bcs) > 0:
                    warnings.warn("Unexpected boundary conditions",
                                  RuntimeWarning)
                if not parameters_dict_equal(
                        tensor._tlm_adjoint__form_compiler_parameters,
                        form_compiler_parameters):
                    warnings.warn("Unexpected form compiler parameters",
                                  RuntimeWarning)
                tensor._tlm_adjoint__form += form._tlm_adjoint__form
            else:
                tensor._tlm_adjoint__form = form._tlm_adjoint__form
                tensor._tlm_adjoint__bcs = []
                tensor._tlm_adjoint__form_compiler_parameters = form_compiler_parameters  # noqa: E501

    return return_value


def solve_linear(orig, orig_args, A, x, b,
                 linear_solver="default", preconditioner="default", /):
    annotate = annotation_enabled()
    tlm = tlm_enabled()

    bcs = A._tlm_adjoint__bcs
    if bcs != b._tlm_adjoint__bcs:
        raise ValueError("Non-matching boundary conditions")
    solver_parameters = {"linear_solver": linear_solver,
                         "preconditioner": preconditioner}
    form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
    if not parameters_dict_equal(
            b._tlm_adjoint__form_compiler_parameters,
            form_compiler_parameters):
        raise ValueError("Non-matching form compiler parameters")

    A = A._tlm_adjoint__form
    x = x._tlm_adjoint__function
    b = b._tlm_adjoint__form

    eq = EquationSolver(
        linear_equation_new_x(A == b, x,
                              annotate=annotate, tlm=tlm),
        x, bcs, solver_parameters=solver_parameters,
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    function_update_state(x)
    if hasattr(b, "_tlm_adjoint__function"):
        function_update_state(b._tlm_adjoint__function)
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@add_manager_controls
@override_function(backend_solve)
def solve(orig, orig_args, *args, **kwargs):
    if isinstance(args[0], ufl.classes.Equation):
        return orig_args()
    else:
        return solve_linear(orig, orig_args, *args, **kwargs)


@add_manager_controls
@override_function(backend_project)
def project(orig, orig_args, v, V=None, bcs=None, mesh=None, function=None,
            solver_type="lu", preconditioner_type="default",
            form_compiler_parameters=None, *, solver_parameters=None):
    annotate = annotation_enabled()
    tlm = tlm_enabled()

    if annotate or tlm or solver_parameters is not None:
        if function is None:
            if V is None:
                raise TypeError("V or function required")
            x = space_new(V)
        else:
            x = function

        if bcs is None:
            bcs = ()
        elif isinstance(bcs, backend_DirichletBC):
            bcs = (bcs,)
        else:
            bcs = tuple(bcs)

        solver_parameters_ = {"linear_solver": solver_type,
                              "preconditioner": preconditioner_type}
        if solver_parameters is not None:
            solver_parameters_.update(solver_parameters)
        solver_parameters = solver_parameters_

        eq = Projection(
            x, expr_new_x(v, x, annotate=annotate, tlm=tlm), bcs,
            solver_parameters=solver_parameters,
            form_compiler_parameters=form_compiler_parameters,
            cache_jacobian=False, cache_rhs_assembly=False)

        if solver_parameters is None:
            eq._pre_process(annotate=annotate)
            return_value = orig_args()
            function_update_state(return_value)
            eq._post_process(annotate=annotate, tlm=tlm)
        else:
            eq.solve(annotate=annotate, tlm=tlm)
            return_value = x
    else:
        return_value = orig_args()
        function_update_state(return_value)
    return return_value


@override_method(backend_DirichletBC, "apply")
def _DirichletBC_apply(self, orig, orig_args, *args):
    A = None
    b = None
    x = None
    if len(args) == 1:
        if isinstance(args[0], backend_Matrix):
            A, = args
        else:
            b, = args
            if isinstance(b, backend_Function):
                b = b.vector()
            args = (b,)
    elif len(args) == 2:
        if isinstance(args[0], backend_Matrix):
            A, b = args
            if isinstance(b, backend_Function):
                b = b.vector()
            args = (A, b)
        else:
            b, x = args
            if isinstance(b, backend_Function):
                b = b.vector()
            if isinstance(x, backend_Function):
                x = x.vector()
            args = (b, x)
    else:
        A, b, x = args
        if isinstance(b, backend_Function):
            b = b.vector()
        if isinstance(x, backend_Function):
            x = x.vector()
        args = (A, b, x)

    orig(self, *args)

    if b is not None and hasattr(b, "_tlm_adjoint__function"):
        function_update_state(b._tlm_adjoint__function)

    if x is None:
        if A is not None:
            if not hasattr(A, "_tlm_adjoint__bcs"):
                A._tlm_adjoint__bcs = []
            A._tlm_adjoint__bcs.append(self)

        if b is not None:
            if not hasattr(b, "_tlm_adjoint__bcs"):
                b._tlm_adjoint__bcs = []
            b._tlm_adjoint__bcs.append(self)


def function_update_state_post_call(self, return_value, *args, **kwargs):
    function_update_state(self)
    return return_value


@manager_method(backend_Constant, "assign",
                post_call=function_update_state_post_call)
def Constant_assign(self, orig, orig_args, x, *, annotate, tlm):
    if isinstance(x, (int, np.integer,
                      float, np.floating)):
        eq = Assignment(self, Constant(x, comm=function_comm(self)))
    elif isinstance(x, backend_Constant):
        if x is not self:
            eq = Assignment(self, x)
        else:
            eq = None
    elif isinstance(x, ufl.classes.Expr):
        eq = ExprInterpolation(self, expr_new_x(x, self,
                                                annotate=annotate, tlm=tlm))
    else:
        raise TypeError(f"Unexpected type: {type(x)}")

    if eq is not None:
        assert len(eq.initial_condition_dependencies()) == 0
    return_value = orig_args()
    if eq is not None:
        eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@manager_method(backend_Function, "assign", override_without_manager=True,
                post_call=function_update_state_post_call)
def Function_assign(self, orig, orig_args, rhs, *, annotate, tlm):
    if isinstance(rhs, backend_Function):
        # Prevent a new vector being created

        if space_id(function_space(rhs)) == space_id(function_space(self)):
            if rhs is not self:
                function_assign(self, rhs)

                if annotate or tlm:
                    eq = Assignment(self, rhs)
                    assert len(eq.initial_condition_dependencies()) == 0
                    eq._post_process(annotate=annotate, tlm=tlm)
        else:
            value = function_new(self)
            orig(value, rhs)
            function_assign(self, value)

            if annotate or tlm:
                eq = ExprInterpolation(self, rhs)
                assert len(eq.initial_condition_dependencies()) == 0
                eq._post_process(annotate=annotate, tlm=tlm)
    else:
        orig_args()

        if annotate or tlm:
            eq = ExprInterpolation(self, expr_new_x(rhs, self,
                                                    annotate=annotate, tlm=tlm))  # noqa: E501
            assert len(eq.initial_condition_dependencies()) == 0
            eq._post_process(annotate=annotate, tlm=tlm)


@manager_method(backend_Function, "copy", override_without_manager=True)
def Function_copy(self, orig, orig_args, deepcopy=False, *, annotate, tlm):
    F = orig_args()
    if deepcopy:
        if annotate or tlm:
            eq = Assignment(F, self)
            assert len(eq.initial_condition_dependencies()) == 0
            eq._post_process(annotate=annotate, tlm=tlm)
    else:
        define_function_alias(F, self, key="copy")
    return F


@override_method(backend_Function, "vector")
def Function_vector(self, orig, orig_args):
    vector = orig_args()
    vector._tlm_adjoint__function = self
    if not hasattr(self, "_tlm_adjoint__vector"):
        self._tlm_adjoint__vector = vector
    if self._tlm_adjoint__vector is not vector:
        raise RuntimeError("Vector has changed")
    return vector


@manager_method(backend_Matrix, "__mul__")
def Matrix__mul__(self, orig, orig_args, other, *, annotate, tlm):
    return_value = orig_args()

    if hasattr(self, "_tlm_adjoint__form") \
            and hasattr(other, "_tlm_adjoint__function"):
        if len(self._tlm_adjoint__bcs) > 0:
            raise NotImplementedError("Boundary conditions not supported")

        return_value._tlm_adjoint__form = ufl.action(
            self._tlm_adjoint__form,
            coefficient=other._tlm_adjoint__function)
        return_value._tlm_adjoint__bcs = []
        return_value._tlm_adjoint__form_compiler_parameters \
            = self._tlm_adjoint__form_compiler_parameters

    return return_value


@override_method(LUSolver, "__init__")
def LUSolver__init__(self, orig, orig_args, *args):
    orig_args()

    A = None
    linear_solver = "default"
    if len(args) >= 1 and isinstance(args[0], backend_Matrix):
        A = args[0]
        if len(args) >= 2:
            linear_solver = args[1]
    elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
        A = args[1]
        if len(args) >= 3:
            linear_solver = args[2]
    else:
        if len(args) >= 1:
            linear_solver = args[0]

    _setattr(self, "A", A)
    _setattr(self, "linear_solver", linear_solver)


@override_method(LUSolver, "set_operator")
def LUSolver_set_operator(self, orig, orig_args, A):
    orig_args()
    _setattr(self, "A", A)


def Solver_solve_args(self, *args):
    if isinstance(args[0], backend_Matrix):
        A, x, b = args
    else:
        A = _getattr(self, "A")
        if A is None:
            raise RuntimeError("A not defined")
        x, b = args
    if isinstance(x, backend_Function):
        x = x.vector()
    if isinstance(b, backend_Function):
        b = b.vector()
    return A, x, b


def Solver_solve_pre_call(self, *args):
    A_arg = _getattr(self, "A") is None
    A, x, b = Solver_solve_args(self, *args)
    if A_arg:
        return (A, x, b), {}
    else:
        return (x, b), {}


def Solver_solve_post_call(self, return_value, *args):
    _, x, b = Solver_solve_args(self, *args)
    if hasattr(x, "_tlm_adjoint__function"):
        function_update_state(x._tlm_adjoint__function)
    if hasattr(b, "_tlm_adjoint__function"):
        function_update_state(b._tlm_adjoint__function)

    return return_value


@manager_method(LUSolver, "solve",
                pre_call=Solver_solve_pre_call,
                post_call=Solver_solve_post_call)
def LUSolver_solve(self, orig, orig_args, *args, annotate, tlm):
    A, x, b = Solver_solve_args(self, *args)

    bcs = A._tlm_adjoint__bcs
    if bcs != b._tlm_adjoint__bcs:
        raise ValueError("Non-matching boundary conditions")
    form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
    if not parameters_dict_equal(
            b._tlm_adjoint__form_compiler_parameters,
            form_compiler_parameters):
        raise ValueError("Non-matching form compiler parameters")

    A_form = A._tlm_adjoint__form
    x = x._tlm_adjoint__function
    b_form = b._tlm_adjoint__form
    linear_solver = _getattr(self, "linear_solver")

    eq = EquationSolver(
        linear_equation_new_x(A_form == b_form, x,
                              annotate=annotate, tlm=tlm),
        x, bcs,
        solver_parameters={"linear_solver": linear_solver,
                           "lu_solver": self.parameters},
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)
    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@override_method(KrylovSolver, "__init__")
def KrylovSolver__init__(self, orig, orig_args, *args):
    orig_args()

    A = None
    linear_solver = "default"
    preconditioner = "default"
    if len(args) >= 1 and isinstance(args[0], backend_Matrix):
        A = args[0]
        if len(args) >= 2:
            linear_solver = args[1]
        if len(args) >= 3:
            preconditioner = args[2]
    elif len(args) >= 2 and isinstance(args[1], backend_Matrix):
        A = args[1]
        if len(args) >= 3:
            linear_solver = args[2]
        if len(args) >= 4:
            preconditioner = args[3]
    else:
        if len(args) >= 1:
            linear_solver = args[0]
        if len(args) >= 2:
            preconditioner = args[1]

    _setattr(self, "A", A)
    _setattr(self, "P", None)
    _setattr(self, "linear_solver", linear_solver)
    _setattr(self, "preconditioner", preconditioner)


@override_method(KrylovSolver, "set_operator")
def KrylovSolver_set_operator(self, orig, orig_args, A):
    orig_args()
    _setattr(self, "A", A)


@override_method(KrylovSolver, "set_operators")
def KrylovSolver_set_operators(self, orig, orig_args, A, P):
    orig_args()
    _setattr(self, "A", A)
    _setattr(self, "P", P)


@manager_method(KrylovSolver, "solve",
                pre_call=Solver_solve_pre_call,
                post_call=Solver_solve_post_call)
def KrylovSolver_solve(self, orig, orig_args, *args, annotate, tlm):
    A, x, b = Solver_solve_args(self, *args)
    if _getattr(self, "P") is not None:
        raise NotImplementedError("Preconditioners not supported")

    bcs = A._tlm_adjoint__bcs
    if bcs != b._tlm_adjoint__bcs:
        raise ValueError("Non-matching boundary conditions")
    form_compiler_parameters = A._tlm_adjoint__form_compiler_parameters
    if not parameters_dict_equal(
            b._tlm_adjoint__form_compiler_parameters,
            form_compiler_parameters):
        raise ValueError("Non-matching form compiler parameters")

    A_form = A._tlm_adjoint__form
    x = x._tlm_adjoint__function
    b_form = b._tlm_adjoint__form
    linear_solver = _getattr(self, "linear_solver")
    preconditioner = _getattr(self, "preconditioner")

    eq = EquationSolver(
        linear_equation_new_x(A_form == b_form, x,
                              annotate=annotate, tlm=tlm),
        x, bcs,
        solver_parameters={"linear_solver": linear_solver,
                           "preconditioner": preconditioner,
                           "krylov_solver": self.parameters},
        form_compiler_parameters=form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@override_method(LinearVariationalSolver, "__init__")
def LinearVariationalSolver__init__(self, orig, orig_args, problem):
    orig_args()
    _setattr(self, "problem", problem)


def VariationalSolver_solve_post_call(self, return_value,
                                      *args, **kwargs):
    problem = _getattr(self, "problem")
    function_update_state(problem.u_ufl)
    return return_value


@manager_method(LinearVariationalSolver, "solve",
                post_call=VariationalSolver_solve_post_call)
def LinearVariationalSolver_solve(self, orig, orig_args, *,
                                  annotate, tlm):
    problem = _getattr(self, "problem")

    x = problem.u_ufl
    lhs = problem.a_ufl
    rhs = problem.L_ufl
    eq = EquationSolver(
        linear_equation_new_x(lhs == rhs, x,
                              annotate=annotate, tlm=tlm),
        x, problem.bcs(),
        solver_parameters=self.parameters,
        form_compiler_parameters=problem.form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value


@override_method(NonlinearVariationalProblem, "__init__")
def NonlinearVariationalProblem__init__(
        self, orig, orig_args, F, u, bcs=None, J=None,
        form_compiler_parameters=None):
    orig_args()

    if bcs is None:
        bcs = ()
    elif isinstance(bcs, backend_DirichletBC):
        bcs = (bcs,)
    else:
        bcs = tuple(bcs)
    self._tlm_adjoint__bcs = bcs
    self._tlm_adjoint__has_bounds = False


@override_method(NonlinearVariationalProblem, "set_bounds")
def NonlinearVariationalProblem_set_bounds(self, orig, orig_args,
                                           *args, **kwargs):
    orig_args()
    self._tlm_adjoint__has_bounds = True


@override_method(NonlinearVariationalSolver, "__init__")
def NonlinearVariationalSolver__init__(self, orig, orig_args, problem):
    orig_args()
    _setattr(self, "problem", problem)


@manager_method(NonlinearVariationalSolver, "solve",
                post_call=VariationalSolver_solve_post_call)
def NonlinearVariationalSolver_solve(self, orig, orig_args, *, annotate, tlm):
    problem = _getattr(self, "problem")
    if problem._tlm_adjoint__has_bounds:
        raise NotImplementedError("Bounds not supported")

    eq = EquationSolver(
        problem.F_ufl == 0,
        problem.u_ufl,
        problem._tlm_adjoint__bcs,
        J=problem.J_ufl,
        solver_parameters=self.parameters,
        form_compiler_parameters=problem.form_compiler_parameters,
        cache_jacobian=False, cache_rhs_assembly=False)

    eq._pre_process(annotate=annotate)
    return_value = orig_args()
    eq._post_process(annotate=annotate, tlm=tlm)
    return return_value
