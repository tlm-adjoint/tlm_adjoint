#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module defines types which allow for basic floating point level
algorithmic differentiation. The implementation is intended to be used for a
small number of calculations, for example after the calculation of a functional
obtained from a finite element model.

For example, if annotation and operator overloading is enabled

.. code-block:: python

    import numpy as np

    x = Float(np.pi, name='x')
    y = x * np.sin(x)

will lead to annotation of equations associated with the floating point
calculations.
"""

from .interface import (
    DEFAULT_COMM, SpaceInterface, VariableInterface, add_interface,
    check_space_type, comm_dup_cached, is_var, new_space_id, new_var_id,
    register_subtract_adjoint_derivative_action, space_comm, space_dtype,
    subtract_adjoint_derivative_action_base, var_assign, var_comm, var_id,
    var_is_scalar, var_new, var_new_conjugate_dual, var_scalar_value,
    var_space_type)

from .alias import Alias
from .caches import Caches
from .equation import Equation, ZeroAssignment
from .equations import Assignment, Axpy, Conversion
from .manager import annotation_enabled, tlm_enabled

import contextlib
import functools
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdastr
try:
    from sympy.printing.numpy import NumPyPrinter
except ImportError:
    from sympy.printing.pycode import NumPyPrinter
import warnings


__all__ = \
    [
        "set_default_float_dtype",

        "FloatSpace",

        "OverloadedFloat",
        "SymbolicFloat",

        "Float",
        "FloatEquation",

        "to_float",

        "no_float_overloading",
        "paused_float_overloading"
    ]


_name_counter = 0


def new_symbol_name():
    global _name_counter
    count = _name_counter
    _name_counter += 1
    return f"_tlm_adjoint_symbol__{count:d}"


try:
    import petsc4py.PETSc as PETSc
    _default_dtype = PETSc.ScalarType
except ImportError:
    _default_dtype = np.double


def set_default_float_dtype(dtype):
    """Set the default dtype used by :class:`SymbolicFloat` objects.

    :arg dtype: The default dtype.
    """

    global _default_dtype

    if not issubclass(dtype, (float, np.floating,
                              complex, np.complexfloating)):
        raise TypeError("Invalid dtype")
    _default_dtype = dtype


class FloatSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _dtype(self):
        return self.dtype

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        return self.float_cls(
            name=name, space_type=space_type,
            static=static, cache=cache, checkpoint=checkpoint,
            dtype=space_dtype(self), comm=space_comm(self))


class FloatSpace:
    """Defines the real or complex space.

    :arg float_cls: The :class:`SymbolicFloat` class, in particular used to
        instantiate new variables in :func:`tlm_adjoint.interface.space_new`.
        Defaults to :class:`SymbolicFloat`.
    :arg dtype: The data type associated with the space. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`. Defaults to
        :class:`numpy.cdouble`.
    :arg comm: The communicator associated with the space.
    """

    def __init__(self, float_cls=None, *, dtype=None, comm=None):
        if float_cls is None:
            float_cls = SymbolicFloat
        if dtype is None:
            dtype = _default_dtype
        if comm is None:
            comm = DEFAULT_COMM

        self._comm = comm_dup_cached(comm)
        self._dtype = dtype
        self._float_cls = float_cls

        add_interface(self, FloatSpaceInterface,
                      {"id": new_space_id()})

    @property
    def dtype(self):
        """The dtype associated with the space.
        """

        return self._dtype

    @property
    def comm(self):
        """The communicator associated with the space.
        """

        return self._comm

    @property
    def float_cls(self):
        """The :class:`SymbolicFloat` class associated with the space.
        """

        return self._float_cls


_overloading = True


def no_float_overloading(fn):
    """Decorator to disable :class:`OverloadedFloat` operator overloading.

    :arg fn: A callable for which :class:`OverloadedFloat` operator overloading
        should be disabled.
    :returns: A callable for which :class:`OverloadedFloat` operator
        overloading is disabled.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with paused_float_overloading():
            return fn(*args, **kwargs)
    return wrapped_fn


@contextlib.contextmanager
def paused_float_overloading():
    """Construct a context manager which can be used to temporarily disable
    :class:`OverloadedFloat` operator overloading.

    :returns: A context manager which can be used to temporarily disable
        :class:`OverloadedFloat` operator overloading.
    """

    global _overloading
    overloading = _overloading
    _overloading = False
    yield
    _overloading = overloading


class FloatInterface(VariableInterface):
    def _space(self):
        return self.space

    def _space_type(self):
        return self.space_type

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self._tlm_adjoint__var_interface_attrs["name"]

    def _state(self):
        return self._tlm_adjoint__var_interface_attrs["state"][0]

    def _update_state(self):
        self._tlm_adjoint__var_interface_attrs["state"][0] += 1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__var_interface_attrs["checkpoint"]

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _zero(self):
        var_assign(self, 0.0)

    def _assign(self, y):
        dtype = self.space.dtype

        if isinstance(y, SymbolicFloat):
            y = y.value
        elif isinstance(y, (sp.Integer, sp.Float)):
            y = dtype(y)
        if isinstance(y, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating)):
            if not np.can_cast(y, dtype):
                raise ValueError("Invalid dtype")
            self._value = dtype(y)
        else:
            var_assign(self, var_scalar_value(y))

    def _axpy(self, alpha, x, /):
        var_assign(self, self.value + alpha * var_scalar_value(x))

    def _inner(self, y):
        return var_scalar_value(y).conjugate() * self.value

    def _sum(self):
        return self.value

    def _linf_norm(self):
        return abs(self.value)

    def _local_size(self):
        comm = var_comm(self)
        if comm.rank == 0:
            return 1
        else:
            return 0

    def _global_size(self):
        return 1

    def _local_indices(self):
        comm = var_comm(self)
        if comm.rank == 0:
            return slice(0, 1)
        else:
            return slice(0, 0)

    def _get_values(self):
        comm = var_comm(self)
        dtype = self.space.dtype
        value = dtype(self.value)
        values = np.array([value] if comm.rank == 0 else [], dtype=dtype)
        return values

    def _set_values(self, values):
        comm = var_comm(self)
        if comm.rank == 0:
            if values.shape != (1,):
                raise ValueError("Invalid shape")
            value, = values
        else:
            if values.shape != (0,):
                raise ValueError("Invalid shape")
            value = None
        value = comm.bcast(value, root=0)
        var_assign(self, value)

    def _replacement(self):
        return self

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return True

    def _scalar_value(self):
        # assert var_is_scalar(self)
        return self.value


@no_float_overloading
def expr_dependencies(expr):
    deps = []
    for dep in expr.free_symbols:
        if isinstance(dep, SymbolicFloat):
            deps.append(dep)
        elif is_var(dep):
            raise ValueError("Invalid dependency")
    return sorted(deps, key=lambda dep: var_id(dep))


# Float class name already used by SymPy
class _tlm_adjoint__SymbolicFloat(sp.Symbol):  # noqa: N801
    """A :class:`sympy.core.symbol.Symbol` which is also a 'variable', defining
    a scalar variable.

    If constructing SymPy expressions then the :class:`SymbolicFloat` class
    should be used instead of the :class:`OverloadedFloat` subclass, or else
    :class:`OverloadedFloat` operator overloading should be disabled.

    :arg value: A scalar or :class:`sympy.core.expr.Expr` defining the initial
        value. If a :class:`sympy.core.expr.Expr` then, if annotation or
        derivation and solution of tangent-linear equations is enabled, an
        assignment is processed by the
        :class:`tlm_adjoint.tlm_adjoint.EquationManager` `manager`.
    :arg name: A :class:`str` name for the :class:`SymbolicFloat`.
    :arg space_type: The space type for the :class:`SymbolicFloat`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines the default value for `cache` and `checkpoint`.
    :arg cache: Defines whether results involving this :class:`SymbolicFloat`
        may be cached. Default `static`.
    :arg checkpoint: Defines whether a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage` should store this
        :class:`SymbolicFloat` by value (`checkpoint=True`) or reference
        (`checkpoint=False`). Default `not static`.
    :arg dtype: The data type associated with the :class:`SymbolicFloat`.
        Typically :class:`numpy.double` or :class:`numpy.cdouble`. Defaults to
        :class:`numpy.cdouble`.
    :arg comm: The communicator associated with the :class:`SymbolicFloat`.
    :arg annotate: Whether the :class:`tlm_adjoint.tlm_adjoint.EquationManager`
        should record the solution of equations.
    :arg tlm: Whether tangent-linear equations should be solved.
    """

    def __init__(self, value=0.0, *, name=None, space_type="primal",
                 static=False, cache=None, checkpoint=None,
                 dtype=None, comm=None,
                 annotate=None, tlm=None):
        id = new_var_id()
        if name is None:
            # Following FEniCS 2019.1.0 behaviour
            name = f"f_{id:d}"
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if cache is None:
            cache = static
        if checkpoint is None:
            checkpoint = not static

        super().__init__()
        self._space = FloatSpace(type(self), dtype=dtype, comm=comm)
        self._space_type = space_type
        self._value = self._space.dtype(0.0)
        add_interface(self, FloatInterface,
                      {"cache": cache, "checkpoint": checkpoint, "id": id,
                       "name": name, "state": [0], "static": static})
        self._tlm_adjoint__var_interface_attrs["caches"] = Caches(self)

        if isinstance(value, (int, np.integer, sp.Integer,
                              float, np.floating, sp.Float,
                              complex, np.complexfloating)):
            if value != 0.0:
                var_assign(self, value)
        else:
            self.assign(value, annotate=annotate, tlm=tlm)

    def __new__(cls, value=0.0, *args, **kwargs):
        return super().__new__(cls, new_symbol_name())

    def new(self, value=0.0, *,
            name=None,
            static=False, cache=None, checkpoint=None,
            annotate=None, tlm=None):
        """Return a new object, which same type and space type as this
        :class:`SymbolicFloat`.

        :returns: The new :class:`SymbolicFloat`.

        Arguments are as for the :class:`SymbolicFloat` constructor.
        """

        x = var_new(
            self, name=name, static=static, cache=cache, checkpoint=checkpoint)
        if isinstance(value, (int, np.integer, sp.Integer,
                              float, np.floating, sp.Float,
                              complex, np.complexfloating)):
            if value != 0.0:
                var_assign(x, value)
        else:
            x.assign(
                value,
                annotate=annotate, tlm=tlm)
        return x

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    @property
    def space(self):
        """The :class:`FloatSpace` for the :class:`SymbolicFloat`.
        """

        class CallableProperty(Alias):
            def __call__(self):
                warnings.warn("space is a property and should not be called",
                              DeprecationWarning, stacklevel=2)
                return self

        return CallableProperty(self._space)

    @property
    def space_type(self):
        """The space type for the :class:`SymbolicFloat`.
        """

        return self._space_type

    def assign(self, y, *, annotate=None, tlm=None):
        """:class:`SymbolicFloat` assignment.

        :arg y: A scalar or :class:`sympy.core.expr.Expr` defining the value.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        :returns: The :class:`SymbolicFloat`.
        """

        if annotate is None or annotate:
            annotate = annotation_enabled()
        if tlm is None or tlm:
            tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(y, (int, np.integer, sp.Integer,
                              float, np.floating, sp.Float,
                              complex, np.complexfloating)):
                Assignment(self, self.new(y)).solve(
                    annotate=annotate, tlm=tlm)
            elif isinstance(y, sp.Expr):
                FloatEquation(self, y).solve(
                    annotate=annotate, tlm=tlm)
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        else:
            if isinstance(y, (int, np.integer, sp.Integer,
                              float, np.floating, sp.Float,
                              complex, np.complexfloating)):
                var_assign(self, y)
            elif isinstance(y, sp.Expr):
                deps = expr_dependencies(y)
                var_assign(
                    self,
                    lambdify(y, deps)(*(dep.value for dep in deps)))
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        return self

    def addto(self, y, *, annotate=None, tlm=None):
        """:class:`SymbolicFloat` in-place addition.

        :arg y: A scalar or :class:`sympy.core.expr.Expr` defining the value to
            add.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        x_old = self.new().assign(self, annotate=annotate, tlm=tlm)
        y = self.new().assign(y, annotate=annotate, tlm=tlm)
        Axpy(self, x_old, 1.0, y).solve(annotate=annotate, tlm=tlm)

    @property
    def value(self):
        """Return the current value associated with the :class:`SymbolicFloat`.

        The value may also be accessed by casting using :class:`float` or
        :class:`complex`.

        :returns: The value.
        """

        class CallableProperty(type(self._value)):
            def __call__(self):
                warnings.warn("value is a property and should not be called",
                              DeprecationWarning, stacklevel=2)
                return self

        return CallableProperty(self._value)


# Required by Sphinx
class SymbolicFloat(_tlm_adjoint__SymbolicFloat):
    def assign(self, y, *, annotate=None, tlm=None):
        pass

    def addto(self, y, *, annotate=None, tlm=None):
        pass

    def value(self):
        pass


SymbolicFloat = _tlm_adjoint__SymbolicFloat  # noqa: F811


def operation(op, *args):
    for arg in args:
        if not isinstance(arg, (int, np.integer,
                                float, np.floating,
                                complex, np.complexfloating,
                                sp.Expr)):
            # e.g. we don't want to allow 'Float + str'
            return NotImplemented
    for arg in args:
        if isinstance(arg, SymbolicFloat):
            new = arg.new
            break
    else:
        return NotImplemented

    with paused_float_overloading():
        z = op(*args)
    if _overloading and z is not NotImplemented:
        z = new(z)
    return z


_ops = {}
_op_fns = {}


def register_operation(np_op, *, replace=False):
    def register(sp_op):
        if not replace and np_op in _ops:
            raise RuntimeError("Operation already registered")
        op = _ops[np_op] = functools.partial(operation, sp_op)
        return op
    return register


def register_function(np_op, np_code, *, replace=False):
    def register(cls):
        if not replace and cls.__name__ in _op_fns:
            raise RuntimeError("Function already registered")
        _op_fns[cls.__name__] = np_code
        return cls
    return register


@register_function(np.expm1, "numpy.expm1")
class _tlm_adjoint__expm1(sp.Function):  # noqa: N801
    def fdiff(self, argindex=1):
        if argindex == 1:
            return sp.exp(self.args[0])


class _tlm_adjoint__OverloadedFloat(np.lib.mixins.NDArrayOperatorsMixin,  # noqa: E501,N801
                                    SymbolicFloat):
    """A subclass of :class:`SymbolicFloat` with operator overloading.

    If constructing SymPy expressions then the :class:`SymbolicFloat` class
    should be used instead of the :class:`OverloadedFloat` subclass, or else
    :class:`OverloadedFloat` operator overloading should be disabled.

    For argument documentation see :class:`SymbolicFloat`.
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if len(kwargs) > 0:
            return NotImplemented
        if ufunc not in _ops:
            return NotImplemented
        return _ops[ufunc](*inputs)

    def __eq__(self, other):
        return SymbolicFloat.__eq__(self, other)

    def __ne__(self, other):
        return SymbolicFloat.__ne__(self, other)

    def __hash__(self):
        return SymbolicFloat.__hash__(self)

    @staticmethod
    @register_operation(np.abs)
    def abs(x):
        if not isinstance(x, SymbolicFloat):
            return NotImplemented
        if not issubclass(x.space.dtype, (float, np.floating)):
            return NotImplemented

        if x.value >= 0.0:
            return x
        else:
            return -x

    @staticmethod
    @register_operation(np.negative)
    def negative(x):
        return SymbolicFloat.__neg__(x)

    @staticmethod
    @register_operation(np.add)
    def add(x1, x2):
        if isinstance(x1, SymbolicFloat):
            return SymbolicFloat.__add__(x1, x2)
        else:
            return SymbolicFloat.__radd__(x2, x1)

    @staticmethod
    @register_operation(np.subtract)
    def subtract(x1, x2):
        if isinstance(x1, SymbolicFloat):
            return SymbolicFloat.__sub__(x1, x2)
        else:
            return SymbolicFloat.__rsub__(x2, x1)

    @staticmethod
    @register_operation(np.multiply)
    def multiply(x1, x2):
        if isinstance(x1, SymbolicFloat):
            return SymbolicFloat.__mul__(x1, x2)
        else:
            return SymbolicFloat.__rmul__(x2, x1)

    @staticmethod
    @register_operation(np.divide)
    def divide(x1, x2):
        if isinstance(x1, SymbolicFloat):
            return SymbolicFloat.__truediv__(x1, x2)
        else:
            return SymbolicFloat.__rtruediv__(x2, x1)

    @staticmethod
    @register_operation(np.power)
    def power(x1, x2):
        if isinstance(x1, SymbolicFloat):
            return SymbolicFloat.__pow__(x1, x2)
        else:
            return SymbolicFloat.__rpow__(x2, x1)

    @staticmethod
    @register_operation(np.sin)
    def sin(x):
        return sp.sin(x)

    @staticmethod
    @register_operation(np.cos)
    def cos(x):
        return sp.cos(x)

    @staticmethod
    @register_operation(np.tan)
    def tan(x):
        return sp.tan(x)

    @staticmethod
    @register_operation(np.arcsin)
    def arcsin(x):
        return sp.asin(x)

    @staticmethod
    @register_operation(np.arccos)
    def arccos(x):
        return sp.acos(x)

    @staticmethod
    @register_operation(np.arctan)
    def arctan(x):
        return sp.atan(x)

    @staticmethod
    @register_operation(np.arctan2)
    def arctan2(x1, x2):
        return sp.atan2(x1, x2)

    @staticmethod
    @register_operation(np.sinh)
    def sinh(x):
        return sp.sinh(x)

    @staticmethod
    @register_operation(np.cosh)
    def cosh(x):
        return sp.cosh(x)

    @staticmethod
    @register_operation(np.tanh)
    def tanh(x):
        return sp.tanh(x)

    @staticmethod
    @register_operation(np.arcsinh)
    def arcsinh(x):
        return sp.asinh(x)

    @staticmethod
    @register_operation(np.arccosh)
    def arccosh(x):
        return sp.acosh(x)

    @staticmethod
    @register_operation(np.arctanh)
    def arctanh(x):
        return sp.atanh(x)

    @staticmethod
    @register_operation(np.exp)
    def exp(x):
        return sp.exp(x)

    @staticmethod
    @register_operation(np.expm1)
    def expm1(x):
        return _tlm_adjoint__expm1(x)

    @staticmethod
    @register_operation(np.log)
    def log(x):
        return sp.log(x)

    @staticmethod
    @register_operation(np.log10)
    def log10(x):
        return sp.log(x, 10)

    @staticmethod
    @register_operation(np.sqrt)
    def sqrt(x):
        return sp.sqrt(x)


# Required by Sphinx
class OverloadedFloat(_tlm_adjoint__OverloadedFloat):
    pass


OverloadedFloat = _tlm_adjoint__OverloadedFloat  # noqa: F811


class _tlm_adjoint__Float(OverloadedFloat):  # noqa: N801
    pass


# Required by Sphinx
class Float(_tlm_adjoint__Float):
    pass


Float = _tlm_adjoint__Float  # noqa: F811

_x = sp.Symbol(new_symbol_name())
_F = sp.utilities.lambdify(_x, _x, modules=["numpy"])
global_vars = _F.__globals__
del _x, _F


@no_float_overloading
def lambdify(expr, deps):
    printer = NumPyPrinter(
        settings={"fully_qualified_modules": False,
                  "user_functions": _op_fns})
    code = lambdastr(deps, expr, printer=printer)
    assert "\n" not in code
    local_vars = {}
    exec(f"_tlm_adjoint__F = {code:s}", dict(global_vars), local_vars)
    F = local_vars["_tlm_adjoint__F"]
    F._tlm_adjoint__code = code
    return F


class FloatEquation(Equation):
    r"""Represents an assignment to a :class:`SymbolicFloat` `x`,

    .. math::

        x = \mathcal{G} \left( y_1, y_2, \ldots \right),

    for some :math:`\mathcal{G}` defined by a :class:`sympy.core.expr.Expr`.
    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y_1, y_2, \ldots \right)
            = x - \mathcal{G} \left( y_1, y_2, \ldots \right).

    :arg x: A :class:`SymbolicFloat` defining the forward solution :math:`x`
    :arg expr: A :class:`sympy.core.expr.Expr` defining the right-hand-side.
    """

    @no_float_overloading
    def __init__(self, x, expr):
        check_space_type(x, "primal")
        deps = expr_dependencies(expr)
        for dep in deps:
            check_space_type(dep, "primal")
        if var_id(x) in {var_id(dep) for dep in deps}:
            raise ValueError("Invalid dependency")
        deps.insert(0, x)

        dF_expr = {}
        nl_deps = {}
        for dep_index, dep in enumerate(deps[1:], start=1):
            expr_diff = dF_expr[dep_index] = expr.diff(dep)
            for dep2 in expr_dependencies(expr_diff):
                nl_deps.setdefault(var_id(dep), dep)
                nl_deps.setdefault(var_id(dep2), dep2)
        nl_deps = sorted(nl_deps.values(), key=lambda dep: var_id(dep))
        dF = {dep_index: lambdify(expr_diff, nl_deps)
              for dep_index, expr_diff in dF_expr.items()}

        super().__init__(x, deps, nl_deps=nl_deps,
                         ic=False, adj_ic=False)
        self._F_expr = expr
        self._F = lambdify(expr, deps)
        self._dF_expr = dF_expr
        self._dF = dF

    def forward_solve(self, x, deps=None):
        if deps is None:
            deps = self.dependencies()
        dep_vals = tuple(dep.value for dep in deps)
        x_val = self._F(*dep_vals)
        var_assign(x, x_val)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        deps = self.dependencies()
        nl_dep_vals = tuple(nl_dep.value for nl_dep in nl_deps)
        for dep_index, dep_B in dep_Bs.items():
            dep = deps[dep_index]
            F = var_new_conjugate_dual(dep)
            F_val = (-self._dF[dep_index](*nl_dep_vals).conjugate()
                     * adj_x.value)
            var_assign(F, F_val)
            dep_B.sub(F)

    @no_float_overloading
    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()
        expr = 0
        deps = self.dependencies()
        for dep_index, dF_expr in self._dF_expr.items():
            tau_dep = tlm_map[deps[dep_index]]
            if tau_dep is not None:
                expr = expr + dF_expr * tau_dep
        if isinstance(expr, int) and expr == 0:
            return ZeroAssignment(tlm_map[x])
        else:
            return FloatEquation(tlm_map[x], expr)


def to_float(y, *, name=None, cls=None):
    """Convert a variable to a :class:`Float`.

    :arg y: A scalar-valued variable.
    :arg name: A :class:`str` name.
    :arg cls: Float class. Default :class:`Float`.
    :returns: The :class:`Float`.
    """

    if cls is None:
        cls = Float

    if not var_is_scalar(y):
        raise ValueError("Invalid variable")
    x = cls(name=name, space_type=var_space_type(y))
    Conversion(x, y).solve()
    return x


register_subtract_adjoint_derivative_action(
    SymbolicFloat, object,
    subtract_adjoint_derivative_action_base,
    replace=True)
