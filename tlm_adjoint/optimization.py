from .interface import (
    Packed, comm_dup_cached, garbage_cleanup, is_var, packed, space_dtype,
    var_axpy, var_copy, var_dtype, var_get_values, var_is_cached,
    var_is_static, var_linf_norm, var_local_size, var_locked, var_new,
    var_new_conjugate_dual, var_scalar_value, var_set_values, var_space)

from .caches import clear_caches, local_caches
from .hessian import GeneralHessian as Hessian
from .manager import manager as _manager
from .petsc import (
    PETScOptions, PETScVecInterface, attach_destroy_finalizer,
    petsc_option_setdefault)
from .manager import (
    compute_gradient, manager_disabled, reset_manager, restore_manager,
    set_manager, start_manager, stop_manager)

import contextlib
from functools import cached_property, wraps
import numbers
import numpy as np

__all__ = \
    [
        "minimize_scipy",

        "TAOSolver",
        "minimize_tao"
    ]


class ReducedFunctional:
    def __init__(self, forward, *,
                 manager=None):
        if manager is None:
            manager = _manager()
        manager = manager.new()

        forward_arg = forward

        def forward(*M):
            with var_locked(*M):
                return forward_arg(*M)

        self._manager = manager
        self._forward = forward
        self._M = None
        self._M_val = None
        self._J = None

    @property
    def comm(self):
        return self._manager.comm

    @restore_manager
    def objective(self, M, *,
                  force=False):
        M = packed(M)
        if self._M is not None and len(M) != len(self._M):
            raise ValueError("Invalid control")
        for m in M:
            if not issubclass(var_dtype(m), np.floating):
                raise ValueError("Invalid dtype")

        set_manager(self._manager)

        if force or self._M is None or self._M_val is None or self._J is None:
            self._M = None
            self._M_val = None
            self._J = None
        else:
            assert len(M) == len(self._M_val)
            for m, m_val in zip(M, self._M_val):
                m_error = var_copy(m)
                var_axpy(m_error, -1.0, m_val)
                if var_linf_norm(m_error) != 0.0:
                    self._M = None
                    self._M_val = None
                    self._J = None
                    break

        if self._J is None:
            M = tuple(var_copy(m, static=var_is_static(m),
                               cache=var_is_cached(m))
                      for m in M)
            M_val = tuple(map(var_copy, M))

            reset_manager()
            clear_caches()

            start_manager()
            J = self._forward(*M)
            stop_manager()

            self._M = M
            self._M_val = M_val
            self._J = J

        assert self._M is not None
        assert self._M_val is not None
        assert self._J is not None

        J_val = var_scalar_value(self._J)
        if isinstance(J_val, numbers.Integral) \
                or not isinstance(J_val, numbers.Real):
            raise ValueError("Invalid dtype")
        return J_val

    @restore_manager
    def gradient(self, M):
        M_packed = Packed(M)
        M = tuple(M_packed)
        set_manager(self._manager)

        _ = self.objective(M, force=self._manager._cp_schedule.is_exhausted)
        dJ = compute_gradient(self._J, self._M)

        for dJ_i in dJ:
            if not issubclass(var_dtype(dJ_i), np.floating):
                raise ValueError("Invalid dtype")
        return M_packed.unpack(dJ)

    def hessian_action(self, M, dM):
        M_packed = Packed(M)
        M = tuple(M_packed)
        dM = packed(dM)
        for m in M:
            if not issubclass(var_dtype(m), np.floating):
                raise ValueError("Invalid dtype")
        for dm in dM:
            if not issubclass(var_dtype(dm), np.floating):
                raise ValueError("Invalid dtype")

        ddJ = Hessian(self._forward, manager=self._manager)
        _, _, ddJ = ddJ.action(M, dM)

        for ddJ_i in ddJ:
            if not issubclass(var_dtype(ddJ_i), np.floating):
                raise ValueError("Invalid dtype")
        return M_packed.unpack(ddJ)


@contextlib.contextmanager
def duplicated_comm(comm):
    dup_comm = comm.Dup()
    try:
        yield dup_comm
    finally:
        garbage_cleanup(dup_comm)
        dup_comm.Free()


@local_caches
def minimize_scipy(forward, M0, *,
                   manager=None, **kwargs):
    """Minimize a functional using :func:`scipy.optimize.minimize`.

    Warnings
    --------

    Data is gathered onto the root process (process zero) so that the serial
    :func:`scipy.optimize.minimize` can be used.

    Parameters
    ----------

    forward : callable
        Accepts one or more variable arguments, and returns a scalar variable
        defining the functional.
    M0 : variable or Sequence[variable, ...]
        The initial guess.
    manager : :class:`.EquationManager`
        Used to create an internal manager via :meth:`.EquationManager.new`.
        `manager()` is used if not supplied.
    kwargs
        Passed to :func:`scipy.optimize.minimize`.

    Returns
    -------

    M : variable or Sequence[variable, ...]
        The result of the minimization
    minimize_return_value
        The return value from :func:`scipy.optimize.minimize`.
    """

    M0_packed = Packed(M0)
    M0 = tuple(M0_packed)
    comm = (_manager() if manager is None else manager).comm

    with duplicated_comm(comm) as comm:
        N = [0]
        for m0 in M0:
            N.append(N[-1] + var_local_size(m0))
        if comm.rank == 0:
            size_global = comm.gather(np.array(N[-1], dtype=np.int_), root=0)
            N_global = [0]
            for size in size_global:
                N_global.append(N_global[-1] + size)
        else:
            comm.gather(np.array(N[-1], dtype=np.int_), root=0)

        def get(F):
            x = np.full(N[-1], np.nan, dtype=np.double)
            for i, f in enumerate(F):
                f_vals = var_get_values(f)
                if not np.can_cast(f_vals, x.dtype):
                    raise ValueError("Invalid dtype")
                x[N[i]:N[i + 1]] = f_vals

            if comm.rank == 0:
                x_global = comm.gather(x, root=0)
                X = np.full(N_global[-1], np.nan, dtype=np.double)
                for i, x_p in enumerate(x_global):
                    X[N_global[i]:N_global[i + 1]] = x_p
                return X
            else:
                comm.gather(x, root=0)
                return None

        def set(F, x):
            if comm.rank == 0:
                x = comm.scatter([x[N_global[rank]:N_global[rank + 1]]
                                  for rank in range(comm.size)], root=0)
            else:
                assert x is None
                x = comm.scatter(None, root=0)
            for i, f in enumerate(F):
                var_set_values(f, x[N[i]:N[i + 1]])

        M = tuple(var_new(m0, static=var_is_static(m0),
                          cache=var_is_cached(m0))
                  for m0 in M0)
        J_hat = ReducedFunctional(forward, manager=manager)

        def fun(x):
            set(M, x)
            return J_hat.objective(M)

        def fun_bcast(x):
            if comm.rank == 0:
                comm.bcast(("fun", None), root=0)
            return fun(x)

        def jac(x):
            set(M, x)
            dJ = J_hat.gradient(M)
            return get(dJ)

        def jac_bcast(x):
            if comm.rank == 0:
                comm.bcast(("jac", None), root=0)
            return jac(x)

        def hessp(x, p):
            set(M, x)
            P = tuple(var_new(m, static=var_is_static(m),
                              cache=var_is_cached(m))
                      for m in M)
            set(P, p)
            ddJ = J_hat.hessian_action(M, P)
            return get(ddJ)

        def hessp_bcast(x, p):
            if comm.rank == 0:
                comm.bcast(("hessp", None), root=0)
            return hessp(x, p)

        from scipy.optimize import minimize
        if comm.rank == 0:
            x0 = get(M0)
            return_value = minimize(fun_bcast, x0,
                                    jac=jac_bcast, hessp=hessp_bcast, **kwargs)
            comm.bcast(("return", return_value), root=0)
            set(M, return_value.x)
        else:
            get(M0)
            while True:
                action, data = comm.bcast(None, root=0)
                if action == "fun":
                    assert data is None
                    fun(None)
                elif action == "jac":
                    assert data is None
                    jac(None)
                elif action == "hessp":
                    assert data is None
                    hessp(None, None)
                elif action == "return":
                    assert data is not None
                    return_value = data
                    break
                else:
                    raise ValueError(f"Unexpected action '{action:s}'")
            set(M, None)

        if not return_value.success:
            raise RuntimeError("Convergence failure")

        return M0_packed.unpack(M), return_value


def wrapped_action(M):
    M_arg = M

    @wraps(M_arg)
    def M(*X):
        with var_locked(*X):
            M_X = M_arg(*X)
        M_X = packed(M_X)
        if len(M_X) != len(X):
            raise ValueError("Incompatible shape")
        return M_X

    return M


class TAOSolver:
    """Functional minimization using TAO.

    Parameters
    ----------

    forward : callable
        Accepts one or more variable arguments, and returns a scalar variable
        defining the functional.
    space : space or Sequence[space, ...]
        The control space.
    solver_parameters : Mapping
        TAO solver parameters.
    H_0_action : callable
        Defines the initial Hessian inverse approximation. Accepts one or more
        variables as arguments, defining a direction, and returns a variable or
        a :class:`Sequence` of variables defining the (conjugate of) the action
        of an initial Hessian inverse approximation on this direction.
        Arguments should not be modified.
    M_inv_action : callable
        Defines a dual space norm. Accepts one or more variables as arguments,
        defining a direction, and returns a variable or a :class:`Sequence` of
        variables defining the (conjugate of) the action of a Hermitian and
        positive definite matrix on this direction. Arguments should not be
        modified. `H_0_action` is used if not supplied.
    manager : :class:`.EquationManager`
        Used to create an internal manager via :meth:`.EquationManager.new`.
        `manager()` is used if not supplied.
    """

    def __init__(self, forward, space, *, solver_parameters=None,
                 H_0_action=None, M_inv_action=None, manager=None):
        import petsc4py.PETSc as PETSc

        space = tuple(var_space(space) if is_var(space) else space
                      for space in packed(space))
        if solver_parameters is None:
            solver_parameters = {}
        if H_0_action is not None:
            H_0_action = wrapped_action(H_0_action)
        if M_inv_action is None:
            M_inv_action = H_0_action
        else:
            M_inv_action = wrapped_action(M_inv_action)

        for space_i in space:
            if not issubclass(space_dtype(space_i), np.floating):
                raise ValueError("Invalid dtype")

        J_hat = ReducedFunctional(forward, manager=manager)
        comm = comm_dup_cached(J_hat.comm, key="tao")

        vec_interface = PETScVecInterface(
            space, dtype=PETSc.RealType, comm=comm)
        n, N = vec_interface.local_size, vec_interface.global_size
        to_petsc, from_petsc = vec_interface.to_petsc, vec_interface.from_petsc
        new_vec = vec_interface.new_vec

        tao = PETSc.TAO().create(comm=comm)
        options = PETScOptions(f"_tlm_adjoint__{tao.name:s}_")
        options.update(solver_parameters)
        tao.setOptionsPrefix(options.options_prefix)

        M = [None]

        def objective(tao, x):
            from_petsc(x, M[0])
            J_val = J_hat.objective(M[0])
            return J_val

        def gradient(tao, x, g):
            from_petsc(x, M[0])
            dJ = J_hat.gradient(M[0])
            to_petsc(g, dJ)

        def objective_gradient(tao, x, g):
            from_petsc(x, M[0])
            J_val = J_hat.objective(M[0])
            dJ = J_hat.gradient(M[0])
            to_petsc(g, dJ)
            return J_val

        def hessian(tao, x, H, P):
            H.getPythonContext().set_M(x)

        class Hessian:
            def __init__(self):
                self._shift = 0.0

            @cached_property
            def _M_petsc(self):
                return new_vec()

            def set_M(self, x):
                x.copy(result=self._M_petsc.vec)
                self._shift = 0.0

            def shift(self, A, alpha):
                self._shift += alpha

            def mult(self, A, x, y):
                dM = tuple(var_new(m, static=var_is_static(m),
                                   cache=var_is_cached(m))
                           for m in M[0])
                self._M_petsc.from_petsc(M[0])
                from_petsc(x, dM)
                ddJ = J_hat.hessian_action(M[0], dM)
                to_petsc(y, ddJ)
                if self._shift != 0.0:
                    y.axpy(self._shift, x)

        hessian_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                  Hessian(), comm=comm)
        hessian_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        hessian_matrix.setUp()

        tao.setObjective(objective)
        tao.setGradient(gradient, None)
        tao.setObjectiveGradient(objective_gradient, None)
        tao.setHessian(hessian, hessian_matrix)

        if M_inv_action is not None:
            class GradientNorm:
                def mult(self, A, x, y):
                    dJ = tuple(map(var_new_conjugate_dual, M[0]))
                    from_petsc(x, dJ)
                    M_inv_X = M_inv_action(*dJ)
                    to_petsc(y, M_inv_X)

            M_inv_matrix = PETSc.Mat().createPython(((n, N), (n, N)),
                                                    GradientNorm(), comm=comm)
            M_inv_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            M_inv_matrix.setUp()
            tao.setGradientNorm(M_inv_matrix)
        else:
            M_inv_matrix = None

        # Work around obscure change in behaviour after calling
        # TaoSetFromOptions
        with petsc_option_setdefault("tao_lmvm_mat_lmvm_theta", 0.0):
            tao.setFromOptions()

        if tao.getType() in {PETSc.TAO.Type.LMVM, PETSc.TAO.Type.BLMVM} \
                and H_0_action is not None:
            class InitialHessian:
                pass

            class InitialHessianPreconditioner:
                def apply(self, pc, x, y):
                    X = tuple(map(var_new_conjugate_dual, M[0]))
                    from_petsc(x, X)
                    H_0_X = H_0_action(*X)
                    to_petsc(y, H_0_X)

            B_0_matrix = PETSc.Mat().createPython(
                ((n, N), (n, N)), InitialHessian(), comm=comm)
            B_0_matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
            B_0_matrix.setUp()

            B_0_matrix_pc = PETSc.PC().createPython(
                InitialHessianPreconditioner(), comm=comm)
            B_0_matrix_pc.setOperators(B_0_matrix)
            B_0_matrix_pc.setUp()

            tao.setLMVMH0(B_0_matrix)
            ksp = tao.getLMVMH0KSP()
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.setTolerances(rtol=0.0, atol=0.0, divtol=None, max_it=1)
            ksp.setPC(B_0_matrix_pc)
            ksp.setUp()
        else:
            B_0_matrix = None
            B_0_matrix_pc = None

        attach_destroy_finalizer(tao, hessian_matrix, M_inv_matrix,
                                 B_0_matrix_pc, B_0_matrix)

        self._tao = tao
        self._vec_interface = vec_interface
        self._M = M

        attach_destroy_finalizer(self, tao)

    @property
    def tao(self):
        """The :class:`petsc4py.PETSc.TAO` used to solve the optimization
        problem.
        """

        return self._tao

    @local_caches
    @manager_disabled()
    def solve(self, M):
        """Solve the optimization problem.

        Parameters
        ----------

        M : variable or Sequence[variable, ...]
            Defines the initial guess, and stores the result of the
            minimization.
        """

        M = packed(M)
        x = self._vec_interface.new_vec()
        x.to_petsc(M)

        self._M[0] = tuple(var_new(m, static=var_is_static(m),
                                   cache=var_is_cached(m))
                           for m in M)
        try:
            self.tao.solve(x.vec)
        finally:
            self._M[0] = None

        x.from_petsc(M)
        if self.tao.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")


def minimize_tao(forward, M0, *args, **kwargs):
    """Minimize a functional using TAO.

    Parameters
    ----------

    forward : callable
        Accepts one or more variable arguments, and returns a scalar variable
        defining the functional.
    M0 : variable or Sequence[variable, ...]
        The initial guess.
    args, kwargs
        Passed to the :class:`.TAOSolver` constructor.

    Returns
    -------

    variable or Sequence[variable, ...]
        The result of the minimization.
    """

    M0_packed = Packed(M0)
    M0 = tuple(M0_packed)
    M = tuple(var_copy(m0, static=var_is_static(m0), cache=var_is_cached(m0))
              for m0 in M0)
    TAOSolver(forward, M, *args, **kwargs).solve(M)
    return M0_packed.unpack(M)
