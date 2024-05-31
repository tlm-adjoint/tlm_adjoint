from .interface import (
    Packed, VariableStateLockDictionary, packed, var_check_state_lock, var_id,
    var_increment_state_lock, var_new, var_scalar_value)

from .caches import clear_caches
from .hessian import GaussNewton, Hessian
from .manager import manager as _manager
from .manager import (
    compute_gradient, set_manager, reset_manager, restore_manager, var_tlm)
from .tlm_adjoint import AdjointCache, EquationManager

__all__ = \
    [
        "CachedGaussNewton",
        "CachedHessian"
    ]


class HessianOptimization:
    def __init__(self, *, manager=None, cache_adjoint=True):
        if manager is None:
            manager = _manager()
        if manager._alias_eqs:
            raise RuntimeError("Invalid equation manager state")

        comm = manager.comm

        blocks = list(manager._blocks) + [list(manager._block)]

        ics = VariableStateLockDictionary(
            manager._cp.initial_conditions(cp=True, refs=True, copy=False))

        nl_deps = VariableStateLockDictionary()
        for n, block in enumerate(blocks):
            for i, eq in enumerate(block):
                nl_deps[(n, i)] = manager._cp[(n, i)]

        self._comm = comm
        self._blocks = blocks
        self._ics = ics
        self._nl_deps = nl_deps
        self._cache_adjoint = cache_adjoint
        self._adj_cache = AdjointCache()
        if cache_adjoint:
            self._cache_key = None

    def _new_manager(self):
        manager = EquationManager(comm=self._comm,
                                  cp_method="memory",
                                  cp_parameters={"drop_references": False})

        for x_id, value in self._ics.items():
            manager._cp._add_initial_condition(
                x_id=x_id, value=value, refs=False, copy=False)

        return manager

    def _add_forward_equations(self, manager):
        for n, block in enumerate(self._blocks):
            for i, eq in enumerate(block):
                manager._block.append(eq)
                assert ((0, len(manager._blocks), len(manager._block) - 1)
                        not in manager._adj_cache)
                eq_nl_deps = eq.nonlinear_dependencies()
                nl_deps = self._nl_deps[(n, i)]
                manager._cp.update_keys(
                    len(manager._blocks), len(manager._block) - 1,
                    eq)
                manager._cp._add_equation_data(
                    len(manager._blocks), len(manager._block) - 1,
                    eq_nl_deps, nl_deps, eq_nl_deps, nl_deps,
                    refs=False, copy=False)
                yield n, i, eq

    def _tangent_linear(self, manager, eq, M, dM):
        return manager._tangent_linear(eq, M, dM)

    def _add_tangent_linear_equation(self, manager, n, i, eq, M, dM, tlm_eq, *,
                                     annotate=True, solve=True):
        for tlm_dep in tlm_eq.initial_condition_dependencies():
            manager._cp.add_initial_condition(tlm_dep)

        eq_nl_deps = eq.nonlinear_dependencies()
        cp_deps = self._nl_deps[(n, i)]
        assert len(eq_nl_deps) == len(cp_deps)
        eq_deps = {var_id(eq_dep): cp_dep
                   for eq_dep, cp_dep in zip(eq_nl_deps, cp_deps)}
        del eq_nl_deps, cp_deps

        tlm_deps = list(tlm_eq.dependencies())
        for j, tlm_dep in enumerate(tlm_deps):
            tlm_dep_id = var_id(tlm_dep)
            if tlm_dep_id in eq_deps:
                tlm_deps[j] = eq_deps[tlm_dep_id]
        del eq_deps

        if solve:
            tlm_eq.forward(tlm_eq.X(), deps=tlm_deps)

        if annotate:
            manager._block.append(tlm_eq)
            manager._cp.add_equation(
                len(manager._blocks), len(manager._block) - 1, tlm_eq,
                deps=tlm_deps)

    def _setup_manager(self, M, dM, M0=None, *,
                       annotate_tlm=True, solve_tlm=True):
        M = tuple(M)
        dM = tuple(dM)
        if M0 is not None:
            raise TypeError("Cannot supply M0")

        clear_caches(*dM)

        manager = self._new_manager()
        manager.configure_tlm((M, dM), annotate=annotate_tlm)

        if self._cache_adjoint:
            cache_key = (set(map(var_id, M)), annotate_tlm)
            if self._cache_key is None or self._cache_key != cache_key:
                self._adj_cache.clear()
                self._cache_key = cache_key
        manager._adj_cache = self._adj_cache

        for n, i, eq in self._add_forward_equations(manager):
            tlm_eq = self._tangent_linear(manager, eq, M, dM)
            if tlm_eq is not None:
                self._add_tangent_linear_equation(
                    manager, n, i, eq, M, dM, tlm_eq,
                    annotate=annotate_tlm, solve=solve_tlm)

        return manager, M, dM


class CachedHessian(Hessian, HessianOptimization):
    """Represents a Hessian associated with a given forward. Uses a cached
    forward calculation.

    :arg J: A variable defining the Hessian.
    :arg manager: The :class:`.EquationManager` used to record the forward.
        This must have used `'memory'` checkpointing with automatic dropping of
        variable references disabled. `manager()` is used if not supplied.
    :arg cache_adjoint: Whether to cache the first order adjoint calculation.
    """

    def __init__(self, J, *, manager=None, cache_adjoint=True):
        var_increment_state_lock(self, J)

        HessianOptimization.__init__(self, manager=manager,
                                     cache_adjoint=cache_adjoint)
        Hessian.__init__(self)
        self._J = J

    @restore_manager
    def compute_gradient(self, M, M0=None):
        """As for :meth:`Hessian.compute_gradient`, but using a cached forward
        calculation.

        *Important note*: `M` defines the control, but does not define its
        value. The value of the control used is as for the cached forward
        calculation.
        """

        M_packed = Packed(M)
        M = tuple(M_packed)
        if M0 is not None:
            raise TypeError("Cannot supply M0")

        var_check_state_lock(self._J)

        dM = tuple(map(var_new, M))
        manager, M, dM = self._setup_manager(M, dM, M0=M0, solve_tlm=False)
        set_manager(manager)

        dJ = var_tlm(self._J, (M, dM))

        J_val = var_scalar_value(self._J)
        dJ = compute_gradient(
            dJ, dM,
            cache_adjoint_degree=1 if self._cache_adjoint else 0,
            store_adjoint=self._cache_adjoint)

        reset_manager()
        return J_val, M_packed.unpack(dJ)

    @restore_manager
    def action(self, M, dM, M0=None):
        """As for :meth:`Hessian.action`, but using a cached forward
        calculation.

        *Important note*: `M` defines the control, but does not define its
        value. The value of the control used is as for the cached forward
        calculation.
        """

        M_packed = Packed(M)
        M = tuple(M_packed)
        dM = packed(dM)
        if len(set(map(var_id, M)).intersection(map(var_id, dM))) > 0:
            raise ValueError("Direction and controls must be distinct")
        if M0 is not None:
            raise TypeError("Cannot supply M0")

        var_check_state_lock(self._J)

        manager, M, dM = self._setup_manager(M, dM, M0=M0, solve_tlm=True)
        set_manager(manager)

        dJ = var_tlm(self._J, (M, dM))

        J_val = var_scalar_value(self._J)
        dJ_val = var_scalar_value(dJ)
        ddJ = compute_gradient(
            dJ, M,
            cache_adjoint_degree=1 if self._cache_adjoint else 0,
            store_adjoint=self._cache_adjoint)

        reset_manager()
        return J_val, dJ_val, M_packed.unpack(ddJ)


class CachedGaussNewton(GaussNewton, HessianOptimization):
    """Represents a Gauss-Newton approximation to a Hessian associated with a
    given forward. Uses a cached forward calculation.

    :arg X: A variable or a :class:`Sequence` of variables defining the state.
    :arg R_inv_action: See :class:`.GaussNewton`.
    :arg B_inv_action: See :class:`.GaussNewton`.
    :arg manager: The :class:`.EquationManager` used to record the forward.
        This must have used `'memory'` checkpointing with automatic dropping of
        variable references disabled. `manager()` is used if not supplied.
    """

    def __init__(self, X, R_inv_action, B_inv_action=None, *,
                 manager=None):
        X = packed(X)
        var_increment_state_lock(self, *X)

        HessianOptimization.__init__(self, manager=manager,
                                     cache_adjoint=False)
        GaussNewton.__init__(
            self, R_inv_action, B_inv_action=B_inv_action)
        self._X = tuple(X)

    def _setup_manager(self, M, dM, M0=None, *,
                       annotate_tlm=False, solve_tlm=True):
        manager, M, dM = HessianOptimization._setup_manager(
            self, M, dM, M0=M0,
            annotate_tlm=annotate_tlm, solve_tlm=solve_tlm)
        return manager, M, dM, self._X

    def action(self, M, dM, M0=None):
        """As for :meth:`GaussNewton.action`, but using a cached forward
        calculation.

        *Important note*: `M` defines the control, but does not define its
        value. The value of the control used is as used for the cached forward
        calculation.
        """

        if M0 is not None:
            raise TypeError("Cannot supply M0")
        for x in self._X:
            var_check_state_lock(x)

        return GaussNewton.action(self, M, dM, M0=M0)
