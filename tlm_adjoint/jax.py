#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    DEFAULT_COMM, SpaceInterface, VariableInterface, add_interface,
    comm_dup_cached, new_var_id, new_space_id,
    register_subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base, var_axpy, var_caches, var_comm,
    var_dtype, var_id, var_is_cached, var_is_checkpointed, var_is_scalar,
    var_is_static, var_local_size, var_name, var_scalar_value, var_set_values,
    var_space, var_space_type, var_state)

from .caches import Caches
from .equation import Equation
from .equations import Assignment, Conversion
from .manager import (
    annotation_enabled, manager as _manager, manager_disabled, paused_manager,
    tlm_enabled)

from collections.abc import Sequence
import contextlib
import functools
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import numpy as np
import operator

try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None


__all__ = \
    [
        "set_default_jax_dtype",

        "VectorSpace",
        "Vector",
        "VectorEquation",
        "call_jax",

        "new_jax",
        "new_jax_float",
        "to_jax"
    ]

try:
    import petsc4py.PETSc as PETSc
    _default_dtype = PETSc.ScalarType
except ImportError:
    _default_dtype = np.double


def set_default_jax_dtype(dtype):
    global _default_dtype

    if not issubclass(dtype, (float, np.floating,
                              complex, np.complexfloating)):
        raise TypeError("Invalid dtype")
    _default_dtype = dtype


class VectorSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _dtype(self):
        return self.dtype

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        return Vector(self, name=name, space_type=space_type, static=static,
                      cache=cache, checkpoint=checkpoint)


class VectorSpace:
    """A vector space.

    :arg dtype: The data type associated with the space. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    :arg comm: The communicator associated with the space.
    """

    def __init__(self, n, *, dtype=None, comm=None):
        if dtype is None:
            dtype = _default_dtype
        if comm is None:
            comm = DEFAULT_COMM

        self._n = n
        self._dtype = dtype
        self._comm = comm_dup_cached(comm)

        N = self._n
        if MPI is not None:
            N = self._comm.allreduce(N, op=MPI.SUM)
        self._N = N

        if MPI is None:
            self._n1 = self._n
        else:
            self._n1 = self._comm.scan(self._n, op=MPI.SUM)
        self._n0 = self._n1 - self._n

        add_interface(self, VectorSpaceInterface,
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
    def local_size(self):
        """The number of local degrees of freedom.
        """

        return self._n

    @property
    def global_size(self):
        """The global number of degrees of freedom.
        """

        return self._N

    @property
    def ownership_range(self):
        """A tuple `(n0, n1)`, indicating that `slice(n0, n1)` is the range of
        nodes in the global vector owned by this process.
        """

        return (self._n0, self._n1)


class VectorInterface(VariableInterface):
    def _space(self):
        return self.space

    def _space_type(self):
        return self.space_type

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self.name

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
        self._vector = None

    @manager_disabled()
    def _assign(self, y):
        if isinstance(y, (int, np.integer,
                          float, np.floating,
                          complex, np.complexfloating,
                          Vector)):
            self.assign(y)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _axpy(self, alpha, x, /):
        if isinstance(x, Vector):
            if x.space.local_size != self.space.local_size:
                raise ValueError("Invalid shape")
            if self._vector is None:
                if x._vector is not None:
                    self._vector = jax.numpy.array(alpha * x._vector, dtype=self.space.dtype)  # noqa: E501
            else:
                if x._vector is not None:
                    self._vector = jax.numpy.array(self._vector + alpha * x._vector, dtype=self.space.dtype)  # noqa: E501
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    def _inner(self, y):
        if isinstance(y, Vector):
            if y.space.local_size != self.space.local_size:
                raise ValueError("Invalid shape")
            inner = sum(y.vector.conjugate() * self.vector)
            if MPI is not None:
                inner = self.space.comm.allreduce(inner, op=MPI.SUM)
            return inner
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _sum(self):
        s = sum(self.vector)
        if MPI is not None:
            s = self.space.comm.allreduce(s, op=MPI.SUM)
        return s

    def _linf_norm(self):
        norm = abs(self.vector).max(initial=0.0)
        if MPI is not None:
            norm = self.space.comm.allreduce(norm, op=MPI.MAX)
        return norm

    def _local_size(self):
        return self.space.local_size

    def _global_size(self):
        return self.space.global_size

    def _local_indices(self):
        return slice(*self.space.ownership_range)

    def _get_values(self):
        return np.array(self.vector, dtype=self.space.dtype)

    def _set_values(self, values):
        self._vector = jax.numpy.array(values, dtype=self.space.dtype)

    def _replacement(self):
        return ReplacementVector(self)

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return self.space.global_size == 1

    def _scalar_value(self):
        if self.space.comm.rank == 0:
            if self.vector.shape != (1,):
                raise RuntimeError("Invalid parallel decomposition")
            value, = self.vector
        else:
            if self.vector.shape != (0,):
                raise RuntimeError("Invalid parallel decomposition")
            value = None
        return self.space.comm.bcast(value, root=0)


_overloading = True


@contextlib.contextmanager
def paused_vector_overloading():
    global _overloading
    overloading = _overloading
    _overloading = False
    yield
    _overloading = overloading


def unary_operator(x, op):
    annotate = annotation_enabled()
    tlm = tlm_enabled()
    if not _overloading or (not annotate and not tlm):
        with paused_manager():
            return x.new(op(x.vector))
    else:
        z = x.new()
        VectorEquation(z, x, fn=op).solve()
        return z


def binary_operator(x, y, op, *, reverse_args=False):
    op_arg = op
    if reverse_args:
        def op(x, y):
            return op_arg(y, x)
    else:
        op = op_arg

    annotate = annotation_enabled()
    tlm = tlm_enabled()
    if not _overloading or (not annotate and not tlm):
        with paused_manager():
            return x.new(op(x.vector, y.vector))
    elif x is y:
        z = x.new()
        VectorEquation(z, x, fn=lambda x: op(x, x)).solve()
        return z
    elif isinstance(y, Vector):
        z = x.new()
        VectorEquation(z, (x, y), fn=op).solve()
        return z
    else:
        z = x.new()
        VectorEquation(z, x, fn=lambda x: op(x, y)).solve()
        return z


class Vector:
    """Vector, with degrees of freedom stored as a JAX array.

    :arg V: A :class:`VectorSpace`, an :class:`int` defining the number of
        local degrees of freedom, or an ndim 1 array defining the local degrees
        of freedom.
    :arg name: A :class:`str` name for the :class:`Vector`.
    :arg space_type: The space type for the :class:`Vector`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines the default value for `cache` and `checkpoint`.
    :arg cache: Defines whether results involving this :class:`Vector` may be
        cached. Default `static`.
    :arg checkpoint: Defines whether a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage` should store this
        :class:`Vector` by value (`checkpoint=True`) or reference
        (`checkpoint=False`). Default `not static`.
    :arg dtype: The data type. Ignored if `V` is a :class:`VectorSpace`.
    :arg comm: A communicator. Ignored if `V` is a :class:`VectorSpace`.
    """

    def __init__(self, V, *, name=None, space_type="primal", static=False,
                 cache=None, checkpoint=None, dtype=None, comm=None):
        if isinstance(V, VectorSpace):
            vector = None
            n = V.local_size
        elif isinstance(V, int):
            vector = None
            n = V
            V = VectorSpace(n, dtype=dtype, comm=comm)
        elif isinstance(V, (np.ndarray, jax.Array)):
            if dtype is None:
                dtype = V.dtype.type
            if not jax.numpy.can_cast(V, dtype):
                raise ValueError("Invalid dtype")
            vector = jax.numpy.array(V, dtype=dtype)
            n, = V.shape
            V = VectorSpace(n, dtype=dtype, comm=comm)
        else:
            raise TypeError(f"Unexpected type: {type(V)}")

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
        self._name = name
        self._space = V
        self._space_type = space_type
        self._vector = vector
        add_interface(self, VectorInterface,
                      {"cache": cache, "checkpoint": checkpoint, "id": id,
                       "state": [0], "static": static})
        self._tlm_adjoint__var_interface_attrs["caches"] = Caches(self)

    def __float__(self):
        return float(var_scalar_value(self))

    def __complex__(self):
        return complex(var_scalar_value(self))

    def new(self, y=None, *, name=None, static=False, cache=None,
            checkpoint=None):
        """Return a new :class:`Vector`, with the same :class:`VectorSpace` and
        space type as this :class:`Vector`.

        :arg y: Defines a value for the new :class:`Vector`.
        :returns: The new :class:`Vector`.

        Remaining arguments are as for the :class:`Vector` constructor.
        """

        x = Vector(self.space, space_type=self.space_type, name=name,
                   static=static, cache=cache, checkpoint=checkpoint)
        if y is not None:
            x.assign(y)
        return x

    @property
    def name(self):
        """The :class:`str` name of the :class:`Vector`.
        """

        return self._name

    def assign(self, y, *, annotate=None, tlm=None):
        """:class:`Vector` assignment.

        :arg y: A scalar, :class:`Vector`, or ndim 1 array defining the value.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        :returns: The :class:`Vector`.
        """

        if annotate is None or annotate:
            annotate = annotation_enabled()
        if tlm is None or tlm:
            tlm = tlm_enabled()
        if annotate or tlm:
            with paused_manager():
                if isinstance(y, (int, np.integer,
                                  float, np.floating,
                                  complex, np.complexfloating,
                                  np.ndarray, jax.Array)):
                    y = self.new(y)
                elif isinstance(y, Vector):
                    pass
                else:
                    raise TypeError(f"Unexpected type: {type(y)}")

            Assignment(self, y).solve(annotate=annotate, tlm=tlm)
        else:
            if isinstance(y, (int, np.integer,
                              float, np.floating,
                              complex, np.complexfloating)):
                var_set_values(self, jax.numpy.full(self.space.local_size, y,
                                                    dtype=self.space.dtype))
            elif isinstance(y, (np.ndarray, jax.Array)):
                var_set_values(self, y)
            elif isinstance(y, Vector):
                if y.space.local_size != self.space.local_size:
                    raise ValueError("Invalid shape")
                if y._vector is None:
                    self._vector = None
                else:
                    self._vector = jax.numpy.array(y._vector,
                                                   dtype=self.space.dtype)
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        return self

    @property
    def space(self):
        """The :class:`VectorSpace` for the :class:`Vector`.
        """

        return self._space

    @property
    def space_type(self):
        """The space type for the :class:`Vector`.
        """

        return self._space_type

    @property
    def vector(self):
        """A JAX array storing the local degrees of freedom.
        """

        if self._vector is None:
            return jax.numpy.zeros(self.space.local_size,
                                   dtype=self.space.dtype)
        else:
            return self._vector

    def __neg__(self):
        return unary_operator(self, operator.neg)

    def __add__(self, other):
        return binary_operator(self, other, operator.add)

    def __radd__(self, other):
        return binary_operator(self, other, operator.add, reverse_args=True)

    def __sub__(self, other):
        return binary_operator(self, other, operator.sub)

    def __rsub__(self, other):
        return binary_operator(self, other, operator.sub, reverse_args=True)

    def __mul__(self, other):
        return binary_operator(self, other, operator.mul)

    def __rmul__(self, other):
        return binary_operator(self, other, operator.mul, reverse_args=True)

    def __truediv__(self, other):
        return binary_operator(self, other, operator.truediv)

    def __rtruediv__(self, other):
        return binary_operator(self, other, operator.truediv, reverse_args=True)  # noqa: E501

    def __pow__(self, other):
        return binary_operator(self, other, operator.pow)

    def __rpow__(self, other):
        return binary_operator(self, other, operator.pow, reverse_args=True)

    def sin(self):
        return unary_operator(self, jax.numpy.sin)

    def cos(self):
        return unary_operator(self, jax.numpy.cos)

    def tan(self):
        return unary_operator(self, jax.numpy.tan)

    def arcsin(self):
        return unary_operator(self, jax.numpy.arcsin)

    def arccos(self):
        return unary_operator(self, jax.numpy.arccos)

    def arctan(self):
        return unary_operator(self, jax.numpy.arctan)

    def arctan2(self, other):
        return binary_operator(self, other, jax.numpy.arctan2)

    def sinh(self):
        return unary_operator(self, jax.numpy.sinh)

    def cosh(self):
        return unary_operator(self, jax.numpy.cosh)

    def tanh(self):
        return unary_operator(self, jax.numpy.tanh)

    def arcsinh(self):
        return unary_operator(self, jax.numpy.arcsinh)

    def arccosh(self):
        return unary_operator(self, jax.numpy.arccosh)

    def arctanh(self):
        return unary_operator(self, jax.numpy.arctanh)

    def exp(self):
        return unary_operator(self, jax.numpy.exp)

    def expm1(self):
        return unary_operator(self, jax.numpy.expm1)

    def log(self):
        return unary_operator(self, jax.numpy.log)

    def log10(self):
        return unary_operator(self, jax.numpy.log10)

    def sqrt(self):
        return unary_operator(self, jax.numpy.sqrt)


class ReplacementVectorInterface(VariableInterface):
    def _space(self):
        return self.space

    def _space_type(self):
        return self.space_type

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self.name

    def _state(self):
        return -1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__var_interface_attrs["checkpoint"]

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _replacement(self):
        return self

    def _is_replacement(self):
        return True


class ReplacementVector:
    def __init__(self, x):
        self._name = var_name(x)
        self._space = var_space(x)
        self._space_type = var_space_type(x)
        add_interface(self, ReplacementVectorInterface,
                      {"cache": var_is_cached(x),
                       "caches": var_caches(x),
                       "checkpoint": var_is_checkpointed(x),
                       "id": var_id(x),
                       "static": var_is_static(x)})

    @property
    def name(self):
        """The :class:`str` name of the :class:`ReplacementVector`.
        """

        return self._name

    @property
    def space(self):
        """The :class:`VectorSpace` for the :class:`ReplacementVector`.
        """

        return self._space

    @property
    def space_type(self):
        """The space type for the :class:`ReplacementVector`.
        """

        return self._space_type


def jax_forward(fn, X, Y, *, manager=None):
    if manager is None:
        manager = _manager()

    for sub_tlm in manager._tlm.values():
        if len(sub_tlm) > 0:
            raise NotImplementedError("Higher order tangent-linears with JAX "
                                      "not implemented")

    tlm_X = list(X)
    tlm_Y = list(Y)
    for M, dM in manager._tlm:
        tlm_X.extend(manager.var_tlm(x, (M, dM)) for x in X)
        for y in Y:
            tlm_y = manager.var_tlm(y, (M, dM))
            if tlm_y is None:
                tlm_y = y.new()
            tlm_Y.append(tlm_y)
    tlm_X = tuple(tlm_X)
    tlm_Y = tuple(tlm_Y)

    n_X = len(X)
    n_Y = len(Y)
    n_tlm = (len(tlm_Y) // n_Y) - 1
    assert len(tlm_Y) == (n_tlm + 1) * n_Y

    def tlm_fn(*args):
        assert len(args) == (n_tlm + 1) * n_Y

        X, jvp = jax.linearize(fn, *args[:n_Y])

        tlm_X = list(X)
        for i in range(n_Y, len(args), n_Y):
            tlm_X.extend(jvp(*args[i:i + n_Y]))

        assert len(tlm_X) == (n_tlm + 1) * n_X
        return tuple(tlm_X)

    return tlm_X, tlm_Y, tlm_fn


class VectorEquation(Equation):
    """JAX interface. `fn` should be a callable

    .. code-block:: python

        def fn(y0, y1, ...):
            ...
            return x0, x1, ...

    where the `y0`, `y1` are ndim 1 JAX arrays, and the `x_0`, x1`, are scalars
    or ndim 1 JAX arrays.

    :arg X: A :class:`Vector` or a :class:`Sequence` of :class:`Vector objects
        defining outputs, whose value is set by the return value from `fn`.
    :arg Y: A :class:`Vector` or a :class:`Sequence` of :class:`Vector` objects
        defining the inputs, whose values are passed to `fn`.
    :arg fn: A callable.
    """

    def __init__(self, X, Y, fn):
        if not isinstance(X, Sequence):
            X = (X,)
        if not isinstance(Y, Sequence):
            Y = (Y,)
        if len(X) != len(set(X)):
            raise ValueError("Duplicate solution")
        if len(Y) != len(set(Y)):
            raise ValueError("Duplicate dependency")
        if len(set(X).intersection(Y)) > 0:
            raise ValueError("Invalid dependency")

        @functools.wraps(fn)
        def wrapped_fn(*args):
            X = fn(*args)
            if not isinstance(X, Sequence):
                X = X,

            def to_jax_array(x):
                if isinstance(x, jax.Array):
                    if len(x.shape) == 1:
                        return x
                    elif np.prod(x.shape) == 1:
                        return x.flatten()
                    else:
                        raise ValueError("Unexpected shape")
                else:
                    raise TypeError(f"Unexpected type: {type(x)}")

            return tuple(map(to_jax_array, X))

        super().__init__(X, list(X) + list(Y), nl_deps=Y,
                         ic=False, adj_ic=False)
        self._fn = wrapped_fn
        self._annotate = True
        self._vjp = None

    def drop_reference(self):
        super().drop_references()
        self._vjp = None

    def _jax_reverse(self, Y):
        key = tuple((var_id(y), var_state(y)) for y in Y)
        if self._vjp is not None:
            vjp_key, vjp = self._vjp
            if key != vjp_key:
                self._vjp = None

        if self._vjp is None:
            vjp = jax.vjp(self._fn, *(y.vector for y in Y))
            self._vjp = (key, vjp)

        _, vjp = self._vjp
        return vjp

    def solve(self, *, manager=None, annotate=None, tlm=None):
        if manager is None:
            manager = _manager()
        if annotate is None or annotate:
            annotate = manager.annotation_enabled()
        if tlm is None or tlm:
            tlm = manager.tlm_enabled()

        if tlm and len(manager._tlm) > 0:
            X = self.X()
            Y = self.dependencies()[len(X):]
            tlm_X, tlm_Y, tlm_fn = jax_forward(self._fn, X, Y, manager=manager)
            VectorEquation(tlm_X, tlm_Y, tlm_fn).solve(
                manager=manager, annotate=annotate, tlm=False)
        else:
            if not manager._cp.store_data():
                self._annotate = False
                self._vjp = None
            try:
                return super().solve(manager=manager,
                                     annotate=annotate, tlm=tlm)
            finally:
                self._annotate = True

    def forward_solve(self, X, deps=None):
        if not isinstance(X, Sequence):
            X = (X,)
        if deps is None:
            deps = self.dependencies()
        Y = deps[len(X):]

        if self._annotate:
            X_val, _ = self._jax_reverse(Y)
        else:
            X_val = self._fn(*(y.vector for y in Y))

        if len(X) != len(X_val):
            raise ValueError("Invalid solution")
        for x, x_val in zip(X, X_val):
            x.assign(x_val)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        if not isinstance(adj_X, Sequence):
            adj_X = (adj_X,)
        _, vjp = self._jax_reverse(nl_deps)
        dF = vjp(tuple(adj_x.vector.conjugate() for adj_x in adj_X))
        for dep_index, dep_B in dep_Bs.items():
            dep_B.sub((-1.0, dF[dep_index - len(adj_X)].conjugate()))


def call_jax(X, Y, fn):
    """JAX interface. `fn` should be a callable

    .. code-block:: python

        def fn(y0, y1, ...):
            ...
            return x0, x1, ...

    where the `y0`, `y1` are ndim 1 JAX arrays, and the `x_0`, x1`, are scalars
    or ndim 1 JAX arrays.

    :arg X: A :class:`Vector` or a :class:`Sequence` of :class:`Vector objects
        defining outputs, whose value is set by the return value from `fn`.
    :arg Y: A :class:`Vector` or a :class:`Sequence` of :class:`Vector` objects
        defining the inputs, whose values are passed to `fn`.
    :arg fn: A callable.
    """

    VectorEquation(X, Y, fn).solve()


def new_jax(y, space=None, *, name=None):
    """Construct a new zero-valued :class:`Vector`.

    :arg y: A variable.
    :arg space: The :class:`VectorSpace` for the return value.
    :arg name: A :class:`str` name.
    :returns: The :class:`Vector`.
    """

    if space is None:
        space = VectorSpace(
            var_local_size(y), dtype=var_dtype(y), comm=var_comm(y))
    return Vector(space, space_type=var_space_type(y), name=name)


def to_jax(y, space=None, *, name=None):
    """Convert a variable to a :class:`Vector`.

    :arg y: A variable.
    :arg space: The :class:`VectorSpace` for the return value.
    :arg name: A :class:`str` name.
    :returns: The :class:`Vector`.
    """

    x = new_jax(y, space, name=name)
    Conversion(x, y).solve()
    return x


def new_jax_float(space=None, *, name=None, dtype=None, comm=None):
    """Create a new scalar-valued :class:`Vector`.

    :arg space: The :class:`VectorSpace`.
    :arg name: A :class:`str` name.
    :arg dtype: The data type. Ignored if `space` is supplied.
    :arg comm: A communicator. Ignored if `space` is supplied.
    :returns: A scalar-valued :class:`Vector`.
    """

    if comm is None:
        comm = DEFAULT_COMM
    if space is None:
        space = VectorSpace(1 if comm.rank == 0 else 0, dtype=dtype, comm=comm)
    x = Vector(space, name=name)
    if not var_is_scalar(x):
        raise RuntimeError("Vector is not scalar-valued")
    return x


def subtract_adjoint_derivative_action_vector_array(x, alpha, y):
    var_axpy(x, -alpha, x.new(y))


register_subtract_adjoint_derivative_action(
    Vector, object,
    subtract_adjoint_derivative_action_base,
    replace=True)
if jax is not None:
    register_subtract_adjoint_derivative_action(
        Vector, (np.ndarray, jax.Array),
        subtract_adjoint_derivative_action_vector_array)
