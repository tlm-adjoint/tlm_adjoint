from .interface import (
    DEFAULT_COMM, SpaceInterface, VariableInterface, add_interface,
    add_replacement_interface, comm_dup_cached, is_var, new_space_id,
    new_var_id, register_subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base, var_assign, var_axpy, var_comm,
    var_dtype, var_id, var_is_scalar, var_local_size, var_set_values,
    var_space_type, var_state)

from .caches import Caches
from .equation import Equation
from .equations import Assignment, Axpy, Conversion
from .manager import (
    annotation_enabled, manager as _manager, paused_manager, tlm_enabled)

from collections.abc import Sequence
import contextlib
import functools
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import numbers
import numpy as np

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
_default_dtype = np.dtype(_default_dtype).type
if not issubclass(_default_dtype, (np.floating, np.complexfloating)):
    raise ImportError("Invalid default dtype")


def set_default_jax_dtype(dtype):
    """Set the default data type used by :class:`.Vector` objects.

    :arg dtype: The default data type.
    """

    global _default_dtype

    dtype = np.dtype(dtype).type
    if not issubclass(dtype, (np.floating, np.complexfloating)):
        raise TypeError("Invalid dtype")
    _default_dtype = dtype


class VectorSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _dtype(self):
        return self.dtype

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _global_size(self):
        return self.global_size

    def _local_indices(self):
        return slice(*self.ownership_range)

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        return Vector(self, name=name, space_type=space_type, static=static,
                      cache=cache)


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
        comm = comm_dup_cached(comm)

        dtype = np.dtype(dtype).type
        if not issubclass(dtype, (np.floating, np.complexfloating)):
            raise TypeError("Invalid dtype")

        self._n = n
        self._dtype = dtype
        self._comm = comm

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
        """The data type associated with the space.
        """

        return self._dtype

    @functools.cached_property
    def rdtype(self):
        """The real data type associated with the space.
        """

        return self._dtype(0.0).real.dtype.type

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

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _zero(self):
        self._vector = None

    def _assign(self, y):
        if isinstance(y, numbers.Complex):
            self._vector = jax.numpy.full(self.space.local_size, y,
                                          dtype=self.space.dtype)
        elif isinstance(y, Vector):
            if y.space.local_size != self.space.local_size:
                raise ValueError("Invalid shape")
            self._vector = jax.numpy.array(y.vector, dtype=self.space.dtype)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _axpy(self, alpha, x, /):
        if isinstance(x, Vector):
            if x.space.local_size != self.space.local_size:
                raise ValueError("Invalid shape")
            self.assign(jax.numpy.array(self.vector + alpha * x.vector,
                                        dtype=self.space.dtype))
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    def _inner(self, y):
        if isinstance(y, Vector):
            if y.space.local_size != self.space.local_size:
                raise ValueError("Invalid shape")
            inner = self.space.dtype(sum(y.vector.conjugate() * self.vector))
            if MPI is not None:
                inner = self.space.comm.allreduce(inner, op=MPI.SUM)
            return inner
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _linf_norm(self):
        norm = self.space.rdtype(abs(self.vector).max(initial=0.0))
        if MPI is not None:
            norm = self.space.comm.allreduce(norm, op=MPI.MAX)
        return norm

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
        return self.value


_overloading = True


@contextlib.contextmanager
def paused_vector_overloading():
    global _overloading
    overloading = _overloading
    _overloading = False
    try:
        yield
    finally:
        _overloading = overloading


def unary_operation(op, x):
    if not isinstance(x, Vector):
        return NotImplemented

    annotate = annotation_enabled()
    tlm = tlm_enabled()
    if not _overloading or (not annotate and not tlm):
        with paused_manager():
            return x.new(op(x.vector))
    else:
        z = x.new()
        VectorEquation(z, x, fn=op).solve()
        return z


def binary_operation(op, x, y):
    if not isinstance(x, (numbers.Complex, Vector)):
        return NotImplemented
    if not isinstance(y, (numbers.Complex, Vector)):
        return NotImplemented
    if not isinstance(x, Vector) and not isinstance(y, Vector):
        return NotImplemented

    annotate = annotation_enabled()
    tlm = tlm_enabled()
    if not _overloading or (not annotate and not tlm):
        with paused_manager():
            z = op(x.vector if isinstance(x, Vector) else x,
                   y.vector if isinstance(y, Vector) else y)
            return x.new(z) if isinstance(x, Vector) else y.new(z)
    elif isinstance(x, Vector):
        if x is y:
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
    elif isinstance(y, Vector):
        z = y.new()
        VectorEquation(z, y, fn=lambda y: op(x, y)).solve()
        return z
    else:
        raise RuntimeError("Unexpected case encountered")


class Vector(np.lib.mixins.NDArrayOperatorsMixin):
    """Vector, with degrees of freedom stored as a JAX array.

    :arg V: A :class:`.VectorSpace`, an :class:`int` defining the number of
        local degrees of freedom, or an ndim 1 array defining the local degrees
        of freedom.
    :arg name: A :class:`str` name for the :class:`.Vector`.
    :arg space_type: The space type for the :class:`.Vector`. `'primal'`,
        `'dual'`, `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines whether the :class:`.Vector` is static, meaning that
        it is stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`.Vector` may be
        cached. Default `static`.
    :arg dtype: The data type. Ignored if `V` is a :class:`.VectorSpace`.
    :arg comm: A communicator. Ignored if `V` is a :class:`.VectorSpace`.
    """

    def __init__(self, V, *, name=None, space_type="primal", static=False,
                 cache=None, dtype=None, comm=None):
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

        super().__init__()
        self._name = name
        self._space = V
        self._space_type = space_type
        self._vector = vector
        add_interface(self, VectorInterface,
                      {"cache": cache, "id": id, "state": [0],
                       "static": static})
        self._tlm_adjoint__var_interface_attrs["caches"] = Caches(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        if len(kwargs) > 0:
            return NotImplemented
        if ufunc in {np.positive,
                     np.equal, np.not_equal,
                     np.less, np.less_equal,
                     np.greater, np.greater_equal}:
            return NotImplemented
        op = getattr(jax.numpy, ufunc.__name__, None)
        if op is None:
            return NotImplemented
        if ufunc.nargs == 2 and len(inputs) == 1:
            return unary_operation(op, *inputs)
        elif ufunc.nargs == 3 and len(inputs) == 2:
            return binary_operation(op, *inputs)
        else:
            return NotImplemented

    def __eq__(self, other):
        return object.__eq__(self, other)

    def __ne__(self, other):
        return object.__ne__(self, other)

    def __hash__(self):
        return object.__hash__(self)

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    def new(self, y=None, *, name=None, static=False, cache=None):
        """Return a new :class:`.Vector`, with the same :class:`.VectorSpace`
        and space type as this :class:`.Vector`.

        :arg y: Defines a value for the new :class:`.Vector`.
        :returns: The new :class:`.Vector`.

        Remaining arguments are as for the :class:`.Vector` constructor.
        """

        x = Vector(self.space, space_type=self.space_type, name=name,
                   static=static, cache=cache)
        if y is not None:
            x.assign(y)
        return x

    @property
    def name(self):
        """The :class:`str` name of the :class:`.Vector`.
        """

        return self._name

    def assign(self, y):
        """:class:`.Vector` assignment.

        :arg y: A :class:`numbers.Complex`, :class:`.Vector`, or ndim 1 array
            defining the value.
        :returns: The :class:`.Vector`.
        """

        annotate = annotation_enabled()
        tlm = tlm_enabled()
        if annotate or tlm:
            if isinstance(y, (numbers.Complex, np.ndarray, jax.Array)):
                with paused_manager():
                    y = self.new(y)
                Assignment(self, y).solve()
            elif isinstance(y, Vector):
                if y is not self:
                    Assignment(self, y).solve()
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        else:
            if isinstance(y, (numbers.Complex, Vector)):
                var_assign(self, y)
            elif isinstance(y, (np.ndarray, jax.Array)):
                var_set_values(self, y)
            else:
                raise TypeError(f"Unexpected type: {type(y)}")
        return self

    def addto(self, y):
        """:class:`.Vector` in-place addition.

        :arg y: A :class:`numbers.Complex`, :class:`.Vector`, or ndim 1 array
            defining the value to add.
        """

        x_old = self.new(self)
        y = self.new(y)
        Axpy(self, x_old, 1.0, y).solve()

    @property
    def value(self):
        """For a :class:`.Vector` with one element, the value of the element.

        The value may also be accessed by casting using :class:`float` or
        :class:`complex`.

        :returns: The value.
        """

        if self.space.global_size != 1:
            raise ValueError("Invalid variable")

        if self.space.comm.rank == 0:
            if self.vector.shape != (1,):
                raise RuntimeError("Invalid parallel decomposition")
            value, = self.vector
            value = self.space.dtype(value)
        else:
            if self.vector.shape != (0,):
                raise RuntimeError("Invalid parallel decomposition")
            value = None

        return self.space.comm.bcast(value, root=0)

    @property
    def space(self):
        """The :class:`.VectorSpace` for the :class:`.Vector`.
        """

        return self._space

    @property
    def space_type(self):
        """The space type for the :class:`.Vector`.
        """

        return self._space_type

    @property
    def vector(self):
        """A JAX array storing the local degrees of freedom.
        """

        if self._vector is None:
            self._vector = jax.numpy.zeros(self.space.local_size,
                                           dtype=self.space.dtype)
        return self._vector


class ReplacementVector:
    def __init__(self, x):
        add_replacement_interface(self, x)


def jax_forward(fn, X, Y):
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

    where the `y0`, `y1` are ndim 1 JAX arrays, and the `x0`, `x1`, are scalars
    or ndim 1 JAX arrays.

    :arg X: A :class:`.Vector` or a :class:`Sequence` of :class:`.Vector`
        objects defining outputs, whose value is set by the return value from
        `fn`.
    :arg Y: A :class:`.Vector` or a :class:`Sequence` of :class:`.Vector`
        objects defining the inputs, whose values are passed to `fn`.
    :arg fn: A callable.
    :arg with_tlm: Whether to annotate an equation solving for the forward and
        all tangent-linears (`with_tlm=True`), or solving only for the
        forward (`with_tlm=False`).
    """

    def __init__(self, X, Y, fn, *, with_tlm=True, _forward_eq=None):
        if is_var(X):
            X = (X,)
        if is_var(Y):
            Y = (Y,)
        if len(X) != len(set(X)):
            raise ValueError("Duplicate solution")
        if len(Y) != len(set(Y)):
            raise ValueError("Duplicate dependency")
        if len(set(X).intersection(Y)) > 0:
            raise ValueError("Invalid dependency")

        n_X = len(X)
        n_Y = len(Y)

        @functools.wraps(fn)
        def wrapped_fn(*args):
            if len(args) != n_Y:
                raise ValueError("Unexpected number of inputs")
            X_val = fn(*args)
            if not isinstance(X_val, Sequence):
                X_val = X_val,
            if len(X_val) != n_X:
                raise ValueError("Unexpected number of outputs")

            def to_jax_array(x_val):
                if isinstance(x_val, jax.Array):
                    if len(x_val.shape) == 1:
                        return x_val
                    elif np.prod(x_val.shape) == 1:
                        return x_val.flatten()
                    else:
                        raise ValueError("Unexpected shape")
                else:
                    raise TypeError(f"Unexpected type: {type(x_val)}")

            return tuple(map(to_jax_array, X_val))
            return X_val

        super().__init__(X, list(X) + list(Y), nl_deps=Y,
                         ic=False, adj_ic=False)
        self._fn = wrapped_fn
        self._annotate = True
        self._vjp = None

        self._with_tlm = with_tlm
        if _forward_eq is None:
            self._forward_eq = self
        else:
            self._forward_eq = _forward_eq
            self.add_referrer(_forward_eq)

    def drop_reference(self):
        super().drop_references()
        self._vjp = None
        if self._forward_eq is not self:
            self._forward_eq = self._forward_eq._weak_alias

    def _jax_reverse(self, *Y):
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

    def solve(self, *, annotate=None, tlm=None):
        manager = _manager()
        if annotate is None or annotate:
            annotate = annotation_enabled()
        if tlm is None or tlm:
            tlm = tlm_enabled()

        X = self.X()
        Y = self.dependencies()[len(X):]
        if self._with_tlm and tlm and len(manager._tlm) > 0:
            tlm_X, tlm_Y, tlm_fn = jax_forward(self._fn, X, Y)
            VectorEquation(tlm_X, tlm_Y, tlm_fn, _forward_eq=self).solve(
                annotate=annotate, tlm=False)
        else:
            eq = VectorEquation(X, Y, self._fn)
            if not manager._cp.store_data:
                eq._annotate = False
            try:
                return super(type(eq), eq).solve(annotate=annotate, tlm=tlm)
            finally:
                eq._annotate = True

    def forward_solve(self, X, deps=None):
        if is_var(X):
            X = (X,)
        if deps is None:
            deps = self.dependencies()
        Y = deps[len(X):]

        if self._annotate:
            X_val, _ = self._jax_reverse(*Y)
        else:
            X_val = self._fn(*(y.vector for y in Y))

        if len(X) != len(X_val):
            raise ValueError("Invalid solution")
        for x, x_val in zip(X, X_val):
            x.assign(x_val)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def subtract_adjoint_derivative_actions(self, adj_X, nl_deps, dep_Bs):
        if is_var(adj_X):
            adj_X = (adj_X,)
        _, vjp = self._jax_reverse(*nl_deps)
        dF = vjp(tuple(adj_x.vector.conjugate() for adj_x in adj_X))
        N_X = len(self.X())
        for dep_index, dep_B in dep_Bs.items():
            if dep_index < N_X or dep_index >= len(dF) + N_X:
                raise ValueError("Unexpected dep_index")
            dep_B.sub((-1.0, dF[dep_index - N_X].conjugate()))
        self._vjp = None

    def tangent_linear(self, tlm_map):
        X = self._forward_eq.X()
        Y = self._forward_eq.dependencies()[len(X):]
        fn = self._forward_eq._fn

        n_Y = len(Y)

        tau_Y = []
        for y in Y:
            tau_y = tlm_map[y]
            if tau_y is None:
                tau_y = y.new()
            tau_Y.append(tau_y)

        def tlm_fn(*args):
            assert len(args) == 2 * n_Y
            Y_val = args[:n_Y]
            tau_Y_val = args[n_Y:]
            _, jvp = jax.linearize(fn, *Y_val)
            return jvp(*tau_Y_val)

        return VectorEquation([tlm_map[x] for x in X], list(Y) + tau_Y,
                              fn=tlm_fn, with_tlm=False)


def call_jax(X, Y, fn):
    """JAX interface. `fn` should be a callable

    .. code-block:: python

        def fn(y0, y1, ...):
            ...
            return x0, x1, ...

    where the `y0`, `y1` are ndim 1 JAX arrays, and the `x0`, `x1`, are scalars
    or ndim 1 JAX arrays.

    :arg X: A :class:`.Vector` or a :class:`Sequence` of :class:`.Vector`
        objects defining outputs, whose value is set by the return value from
        `fn`.
    :arg Y: A :class:`.Vector` or a :class:`Sequence` of :class:`.Vector`
        objects defining the inputs, whose values are passed to `fn`.
    :arg fn: A callable.
    """

    VectorEquation(X, Y, fn).solve()


def new_jax(y, space=None, *, name=None):
    """Construct a new zero-valued :class:`.Vector`.

    :arg y: A variable.
    :arg space: The :class:`.VectorSpace` for the return value.
    :arg name: A :class:`str` name.
    :returns: The :class:`.Vector`.
    """

    if space is None:
        space = VectorSpace(
            var_local_size(y), dtype=var_dtype(y), comm=var_comm(y))
    return Vector(space, space_type=var_space_type(y), name=name)


def to_jax(y, space=None, *, name=None):
    """Convert a variable to a :class:`.Vector`.

    :arg y: A variable.
    :arg space: The :class:`.VectorSpace` for the return value.
    :arg name: A :class:`str` name.
    :returns: The :class:`.Vector`.
    """

    x = new_jax(y, space, name=name)
    Conversion(x, y).solve()
    return x


def new_jax_float(space=None, *, name=None, dtype=None, comm=None):
    """Create a new :class:`.Vector` with one element.

    :arg space: The :class:`.VectorSpace`.
    :arg name: A :class:`str` name.
    :arg dtype: The data type. Ignored if `space` is supplied.
    :arg comm: A communicator. Ignored if `space` is supplied.
    :returns: A :class:`.Vector` with one element.
    """

    if comm is None:
        comm = DEFAULT_COMM
    if space is None:
        space = VectorSpace(1 if comm.rank == 0 else 0, dtype=dtype, comm=comm)
    x = Vector(space, name=name)
    if not var_is_scalar(x):
        raise RuntimeError("Vector is not a scalar variable")
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
