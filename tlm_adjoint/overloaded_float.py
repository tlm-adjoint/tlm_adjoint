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
    space_id, subtract_adjoint_derivative_action_base, var_assign, var_comm,
    var_dtype, var_id, var_new, var_new_conjugate_dual, var_scalar_value,
    var_space_type)

from .caches import Caches
from .equation import Equation, ZeroAssignment
from .equations import Assignment, Axpy
from .manager import annotation_enabled, tlm_enabled

import contextlib
import functools
import itertools
import numbers
import numpy as np
import sympy as sp

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


_name_counter = itertools.count()


def new_symbol_name():
    count = next(_name_counter)
    return f"_tlm_adjoint_symbol__{count:d}"


try:
    import petsc4py.PETSc as PETSc
    _default_dtype = PETSc.ScalarType
except ModuleNotFoundError:
    _default_dtype = np.double
_default_dtype = np.dtype(_default_dtype).type
if not issubclass(_default_dtype, (np.floating, np.complexfloating)):
    raise ImportError("Invalid default dtype")


def set_default_float_dtype(dtype):
    """Set the default data type used by :class:`.SymbolicFloat` objects.

    :arg dtype: The default data type.
    """

    global _default_dtype

    dtype = np.dtype(dtype).type
    if not issubclass(dtype, (np.floating, np.complexfloating)):
        raise TypeError("Invalid dtype")
    _default_dtype = dtype


class FloatSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _dtype(self):
        return self.dtype

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _eq(self, other):
        return (space_id(self) == space_id(other)
                or (isinstance(other, type(self))
                    and space_comm(self).py2f() == space_comm(other).py2f()
                    and space_dtype(self) == space_dtype(other)))

    def _global_size(self):
        return 1

    def _local_indices(self):
        if self.comm.rank == 0:
            return slice(0, 1)
        else:
            return slice(0, 0)

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        return self.float_cls(
            name=name, space_type=space_type,
            static=static, cache=cache, dtype=space_dtype(self),
            comm=space_comm(self))


class FloatSpace:
    """Defines the real or complex space.

    :arg float_cls: The :class:`.SymbolicFloat` class, in particular used to
        instantiate new variables in :func:`.space_new`. Defaults to
        :class:`.SymbolicFloat`.
    :arg dtype: The data type associated with the space. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    :arg comm: The communicator associated with the space.
    """

    def __init__(self, float_cls=None, *, dtype=None, comm=None):
        if float_cls is None:
            float_cls = SymbolicFloat
        if dtype is None:
            dtype = _default_dtype
        if comm is None:
            comm = DEFAULT_COMM
        comm = comm_dup_cached(comm)

        dtype = np.dtype(dtype).type
        if not issubclass(dtype, (np.floating, np.complexfloating)):
            raise TypeError("Invalid dtype")

        self._comm = comm
        self._dtype = dtype
        self._float_cls = float_cls

        add_interface(self, FloatSpaceInterface,
                      {"id": new_space_id()})

    @property
    def dtype(self):
        """The data type associated with the space.
        """

        return self._dtype

    @functools.cached_property
    def rdtype(self):
        """The real data type associated with the space.
        """

        return self.dtype(0.0).real.dtype.type

    @property
    def comm(self):
        """The communicator associated with the space.
        """

        return self._comm

    @property
    def float_cls(self):
        """The :class:`.SymbolicFloat` class associated with the space.
        """

        return self._float_cls


_overloading = True


def no_float_overloading(fn):
    """Decorator to disable :class:`.OverloadedFloat` operator overloading.

    :arg fn: A callable for which :class:`.OverloadedFloat` operator
        overloading should be disabled.
    :returns: A callable for which :class:`.OverloadedFloat` operator
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
    :class:`.OverloadedFloat` operator overloading.

    :returns: A context manager which can be used to temporarily disable
        :class:`.OverloadedFloat` operator overloading.
    """

    global _overloading
    overloading = _overloading
    _overloading = False
    try:
        yield
    finally:
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

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _zero(self):
        var_assign(self, 0.0)

    def _assign(self, y):
        if isinstance(y, numbers.Complex):
            self._value = self.space.dtype(y)
        else:
            self._value = self.space.dtype(var_scalar_value(y))

    def _axpy(self, alpha, x, /):
        var_assign(self, self.value + alpha * var_scalar_value(x))

    def _inner(self, y):
        if isinstance(y, SymbolicFloat):
            return y.value.conjugate() * self.value
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _linf_norm(self):
        return self.space.rdtype(abs(self.value))

    def _get_values(self):
        return np.array([self.value] if self.space.comm.rank == 0 else [],
                        dtype=self.space.dtype)

    def _set_values(self, values):
        comm = self.space.comm
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
        return self.space.dtype(self.value)


@no_float_overloading
def expr_dependencies(expr):
    deps = []
    for dep in expr.free_symbols:
        if isinstance(dep, SymbolicFloat):
            deps.append(dep)
        elif is_var(dep):
            raise ValueError("Invalid dependency")
    return sorted(deps, key=var_id)


def expr_new_x(expr, x):
    if x in expr.free_symbols:
        return expr.subs(x, x.new(x))
    else:
        return expr


# Float class name already used by SymPy
class _tlm_adjoint__SymbolicFloat(sp.Symbol):  # noqa: N801
    """A :class:`sympy.core.symbol.Symbol` which is also a 'variable', defining
    a scalar variable.

    If constructing SymPy expressions then the :class:`.SymbolicFloat` class
    should be used instead of the :class:`.OverloadedFloat` subclass, or else
    :class:`.OverloadedFloat` operator overloading should be disabled.

    :arg value: A :class:`numbers.Complex` or :class:`sympy.core.expr.Expr`
        defining the initial value. If a :class:`sympy.core.expr.Expr` then, if
        annotation or derivation and solution of tangent-linear equations is
        enabled, an assignment is processed by the :class:`.EquationManager`
        `manager`.
    :arg name: A :class:`str` name for the :class:`.SymbolicFloat`.
    :arg space_type: The space type for the :class:`.SymbolicFloat`.
        `'primal'`, `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines whether the :class:`.SymbolicFloat` is static, meaning
        that it is stored by reference in checkpointing/replay, and an
        associated tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.SymbolicFloat`
        may be cached. Default `static`.
    :arg dtype: The data type associated with the :class:`.SymbolicFloat`.
        Typically :class:`numpy.double` or :class:`numpy.cdouble`.
    :arg comm: The communicator associated with the :class:`.SymbolicFloat`.
    """

    def __init__(self, value=0.0, *, name=None, space_type="primal",
                 static=False, cache=None,
                 dtype=None, comm=None):
        id = new_var_id()
        if name is None:
            # Following FEniCS 2019.1.0 behaviour
            name = f"f_{id:d}"
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if cache is None:
            cache = static

        super().__init__()
        self._space = FloatSpace(type(self), dtype=dtype, comm=comm)
        self._space_type = space_type
        self._value = self._space.dtype(0.0)
        add_interface(self, FloatInterface,
                      {"cache": cache, "id": id, "name": name, "state": [0],
                       "static": static})
        self._tlm_adjoint__var_interface_attrs["caches"] = Caches(self)

        if isinstance(value, numbers.Complex):
            if value != 0.0:
                var_assign(self, value)
        else:
            self.assign(value)

    def __new__(cls, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = _default_dtype
        if issubclass(dtype, numbers.Real):
            return super().__new__(cls, new_symbol_name(), real=True)
        else:
            return super().__new__(cls, new_symbol_name(), complex=True)

    def new(self, value=0.0, *,
            name=None,
            static=False, cache=None):
        """Return a new object, which same type and space type as this
        :class:`.SymbolicFloat`.

        :returns: The new :class:`.SymbolicFloat`.

        Arguments are as for the :class:`.SymbolicFloat` constructor.
        """

        x = var_new(self, name=name, static=static, cache=cache)
        if isinstance(value, numbers.Complex):
            if value != 0.0:
                var_assign(x, value)
        else:
            x.assign(value)
        return x

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    @property
    def space(self):
        """The :class:`.FloatSpace` for the :class:`.SymbolicFloat`.
        """

        return self._space

    @property
    def space_type(self):
        """The space type for the :class:`.SymbolicFloat`.
        """

        return self._space_type

    def assign(self, y):
        """:class:`.SymbolicFloat` assignment.

        :arg y: A :class:`numbers.Complex` or :class:`sympy.core.expr.Expr`
            defining the value.
        :returns: The :class:`.SymbolicFloat`.
        """

        annotate = annotation_enabled()
        tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(y, numbers.Complex):
                Assignment(self, self.new(y)).solve()
            elif isinstance(y, Float):
                if y is not self:
                    Assignment(self, y).solve()
            elif isinstance(y, sp.Expr):
                if y is not self:
                    FloatEquation(self, expr_new_x(y, self)).solve()
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        else:
            if isinstance(y, (numbers.Complex, Float)):
                var_assign(self, y)
            elif isinstance(y, sp.Expr):
                deps = expr_dependencies(y)
                var_assign(
                    self,
                    lambdify(y, deps)(*(dep.value for dep in deps)))
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        return self

    def addto(self, y):
        """:class:`.SymbolicFloat` in-place addition.

        :arg y: A :class:`numbers.Complex` or :class:`sympy.core.expr.Expr`
            defining the value to add.
        """

        x_old = self.new(self)
        y = self.new(y)
        Axpy(self, x_old, 1.0, y).solve()

    @property
    def value(self):
        """Return the current value associated with the
        :class:`.SymbolicFloat`.

        The value may also be accessed by casting using :class:`float` or
        :class:`complex`.

        :returns: The value.
        """

        return self._value


# Required by Sphinx
class SymbolicFloat(_tlm_adjoint__SymbolicFloat):
    def new(self, value=0.0, *,
            name=None,
            static=False, cache=None):
        pass

    def assign(self, y):
        pass

    def addto(self, y):
        pass

    @property
    def value(self):
        pass


SymbolicFloat = _tlm_adjoint__SymbolicFloat  # noqa: F811


def operation(op, *args):
    for arg in args:
        if not isinstance(arg, (numbers.Complex, sp.Expr)):
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
        op = _ops[np_op] = lambda *args: operation(sp_op, *args)
        return op
    return register


def register_function(np_op, *, replace=False):
    def register(cls):
        if not replace and cls.__name__ in _op_fns:
            raise RuntimeError("Function already registered")
        _op_fns[cls.__name__] = np_op
        return cls
    return register


@register_function(np.expm1)
class _tlm_adjoint__expm1(sp.Function):  # noqa: N801
    def fdiff(self, argindex=1):
        if argindex == 1:
            return sp.exp(self.args[0])


@register_function(np.log1p)
class _tlm_adjoint__log1p(sp.Function):  # noqa: N801
    def fdiff(self, argindex=1):
        if argindex == 1:
            return sp.Integer(1) / (sp.Integer(1) + self.args[0])


@register_function(np.hypot)
class _tlm_adjoint__hypot(sp.Function):  # noqa: N801
    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0] / _tlm_adjoint__hypot(self.args[0],
                                                      self.args[1])
        elif argindex == 2:
            return self.args[1] / _tlm_adjoint__hypot(self.args[0],
                                                      self.args[1])


class _tlm_adjoint__OverloadedFloat(np.lib.mixins.NDArrayOperatorsMixin,  # noqa: E501,N801
                                    SymbolicFloat):
    """A subclass of :class:`.SymbolicFloat` with operator overloading.

    If constructing SymPy expressions then the :class:`.SymbolicFloat` class
    should be used instead of the :class:`.OverloadedFloat` subclass, or else
    :class:`.OverloadedFloat` operator overloading should be disabled.

    For argument documentation see :class:`.SymbolicFloat`.
    """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        out = kwargs.pop("out", None)
        if out is not None and len(out) != 1:
            return NotImplemented
        if len(kwargs) > 0:
            return NotImplemented
        if ufunc not in _ops:
            return NotImplemented
        value = _ops[ufunc](*inputs)
        if out is None:
            out = value
        else:
            out, = out
            out.assign(value)
        return out

    def __eq__(self, other):
        return SymbolicFloat.__eq__(self, other)

    def __ne__(self, other):
        return SymbolicFloat.__ne__(self, other)

    def __hash__(self):
        return SymbolicFloat.__hash__(self)

    @register_operation(np.abs)
    def abs(self):
        return sp.Abs(self)

    @register_operation(np.negative)
    def negative(self):
        return SymbolicFloat.__neg__(self)

    @register_operation(np.add)
    def add(self, other):
        if isinstance(self, SymbolicFloat):
            return SymbolicFloat.__add__(self, other)
        else:
            return SymbolicFloat.__radd__(other, self)

    @register_operation(np.subtract)
    def subtract(self, other):
        if isinstance(self, SymbolicFloat):
            return SymbolicFloat.__sub__(self, other)
        else:
            return SymbolicFloat.__rsub__(other, self)

    @register_operation(np.multiply)
    def multiply(self, other):
        if isinstance(self, SymbolicFloat):
            return SymbolicFloat.__mul__(self, other)
        else:
            return SymbolicFloat.__rmul__(other, self)

    @register_operation(np.divide)
    def divide(self, other):
        if isinstance(self, SymbolicFloat):
            return SymbolicFloat.__truediv__(self, other)
        else:
            return SymbolicFloat.__rtruediv__(other, self)

    @register_operation(np.power)
    def power(self, other):
        if isinstance(self, SymbolicFloat):
            return SymbolicFloat.__pow__(self, other)
        else:
            return SymbolicFloat.__rpow__(other, self)

    @register_operation(np.sin)
    def sin(self):
        return sp.sin(self)

    @register_operation(np.cos)
    def cos(self):
        return sp.cos(self)

    @register_operation(np.tan)
    def tan(self):
        return sp.tan(self)

    @register_operation(np.arcsin)
    def arcsin(self):
        return sp.asin(self)

    @register_operation(np.arccos)
    def arccos(self):
        return sp.acos(self)

    @register_operation(np.arctan)
    def arctan(self):
        return sp.atan(self)

    @register_operation(np.arctan2)
    def arctan2(self, other):
        return sp.atan2(self, other)

    @register_operation(np.hypot)
    def hypot(self, other):
        return _tlm_adjoint__hypot(self, other)

    @register_operation(np.sinh)
    def sinh(self):
        return sp.sinh(self)

    @register_operation(np.cosh)
    def cosh(self):
        return sp.cosh(self)

    @register_operation(np.tanh)
    def tanh(self):
        return sp.tanh(self)

    @register_operation(np.arcsinh)
    def arcsinh(self):
        return sp.asinh(self)

    @register_operation(np.arccosh)
    def arccosh(self):
        return sp.acosh(self)

    @register_operation(np.arctanh)
    def arctanh(self):
        return sp.atanh(self)

    @register_operation(np.exp)
    def exp(self):
        return sp.exp(self)

    @register_operation(np.exp2)
    def exp2(self):
        return 2 ** self

    @register_operation(np.expm1)
    def expm1(self):
        return _tlm_adjoint__expm1(self)

    @register_operation(np.log)
    def log(self):
        return sp.log(self)

    @register_operation(np.log2)
    def log2(self):
        return sp.log(self, 2)

    @register_operation(np.log10)
    def log10(self):
        return sp.log(self, 10)

    @register_operation(np.log1p)
    def log1p(self):
        return _tlm_adjoint__log1p(self)

    @register_operation(np.sqrt)
    def sqrt(self):
        return sp.sqrt(self)

    @register_operation(np.square)
    def square(self):
        return self ** 2

    @register_operation(np.cbrt)
    def cbrt(self):
        return self ** sp.Rational(1, 3)

    @register_operation(np.reciprocal)
    def reciprocal(self):
        return sp.Integer(1) / self


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


@no_float_overloading
def lambdify(expr, deps):
    return sp.lambdify(deps, expr, modules=["numpy", _op_fns])


class FloatEquation(Equation):
    r"""Represents an assignment to a :class:`.SymbolicFloat` `x`,

    .. math::

        x = \mathcal{G} \left( y_1, y_2, \ldots \right),

    for some :math:`\mathcal{G}` defined by a :class:`sympy.core.expr.Expr`.
    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y_1, y_2, \ldots \right)
            = x - \mathcal{G} \left( y_1, y_2, \ldots \right).

    :arg x: A :class:`.SymbolicFloat` defining the forward solution :math:`x`
    :arg expr: A :class:`sympy.core.expr.Expr` defining the right-hand-side.
    """

    @no_float_overloading
    def __init__(self, x, expr):
        check_space_type(x, "primal")
        deps = expr_dependencies(expr)
        for dep in deps:
            check_space_type(dep, "primal")
        if var_id(x) in set(map(var_id, deps)):
            raise ValueError("Invalid dependency")
        deps.insert(0, x)

        rhs = expr
        F = x - expr

        dF_expr = {}
        nl_deps = {}
        for dep_index, dep in enumerate(deps):
            dF = dF_expr[dep_index] = F.diff(dep)
            for dep2 in expr_dependencies(dF):
                nl_deps.setdefault(var_id(dep2), dep2)
        nl_deps = sorted(nl_deps.values(), key=var_id)
        dF = {dep_index: lambdify(dF, nl_deps)
              for dep_index, dF in dF_expr.items()}

        super().__init__(x, deps, nl_deps=nl_deps,
                         ic=False, adj_ic=False)
        self._rhs = lambdify(rhs, deps)
        self._dF_expr = dF_expr
        self._dF = dF

    def forward_solve(self, x, deps=None):
        if deps is None:
            deps = self.dependencies()
        dep_vals = tuple(dep.value for dep in deps)
        x_val = self._rhs(*dep_vals)
        var_assign(x, x_val)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def subtract_adjoint_derivative_actions(self, adj_x, nl_deps, dep_Bs):
        eq_deps = self.dependencies()
        nl_dep_vals = tuple(nl_dep.value for nl_dep in nl_deps)
        for dep_index, dep_B in dep_Bs.items():
            dep = eq_deps[dep_index]
            F = var_new_conjugate_dual(dep).assign(
                self._dF[dep_index](*nl_dep_vals).conjugate() * adj_x.value)
            dep_B.sub(F)

    @no_float_overloading
    def tangent_linear(self, tlm_map):
        x = self.x()
        expr = 0
        deps = self.dependencies()
        for dep_index, dF_expr in self._dF_expr.items():
            dep = deps[dep_index]
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    expr = expr - dF_expr * tau_dep
        if isinstance(expr, int) and expr == 0:
            return ZeroAssignment(tlm_map[x])
        else:
            return FloatEquation(tlm_map[x], expr)


def to_float(y, *, name=None):
    """Convert a variable to a :class:`.Float`.

    :arg y: A scalar variable.
    :arg name: A :class:`str` name.
    :returns: The :class:`.SymbolicFloat`.
    """

    x = Float(name=name, space_type=var_space_type(y), dtype=var_dtype(y),
              comm=var_comm(y))
    Assignment(x, y).solve()
    return x


register_subtract_adjoint_derivative_action(
    SymbolicFloat, object,
    subtract_adjoint_derivative_action_base,
    replace=True)
