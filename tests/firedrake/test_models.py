from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *

import numpy as np
import pytest

try:
    import hrevolve
except ImportError:
    hrevolve = None

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


def oscillator_ref():
    assert not annotation_enabled()
    assert not tlm_enabled()

    nsteps = 20
    dt = Constant(0.01)
    T_0 = Constant((1.0, 0.0))

    mesh = UnitSquareMesh(5, 5)
    space = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test = TestFunction(space)

    T_n = Function(space, name="T_n")
    T_np1 = Function(space, name="T_np1")
    T_s = 0.5 * (T_n + T_np1)

    T_n.interpolate(T_0)
    for n in range(nsteps):
        solve(
            inner((T_np1 - T_n) / dt, test) * dx
            - inner(T_s[1], test[0]) * dx
            + inner(sin(T_s[0]), test[1]) * dx == 0,
            T_np1,
            solver_parameters=ns_parameters_newton_gmres)
        T_n, T_np1 = T_np1, T_n

    return assemble(T_n[0] * dx)


def diffusion_ref():
    assert not annotation_enabled()
    assert not tlm_enabled()

    n_steps = 20
    dt = Constant(0.01)
    kappa = Constant(1.0)

    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    T_n = Function(space, name="T_n")
    T_np1 = Function(space, name="T_np1")

    interpolate_expression(T_n, sin(pi * X[0]) + sin(10.0 * pi * X[0]))
    for n in range(n_steps):
        solve(inner(trial / dt, test) * dx
              + inner(kappa * grad(trial), grad(test)) * dx
              == inner(T_n / dt, test) * dx,
              T_np1,
              DirichletBC(space, 1.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        T_n, T_np1 = T_np1, T_n

    return assemble((dot(T_n, T_n) ** 2) * dx)


@pytest.mark.firedrake
@pytest.mark.parametrize(
    "cp_method, cp_parameters",
    [("memory", {"drop_references": True}),
     ("periodic_disk", {"period": 3, "format": "pickle"}),
     ("periodic_disk", {"period": 3, "format": "hdf5"}),
     ("multistage", {"format": "pickle", "snaps_on_disk": 1,
                     "snaps_in_ram": 2}),
     ("multistage", {"format": "hdf5", "snaps_on_disk": 1,
                     "snaps_in_ram": 2}),
     pytest.param(
         "H-Revolve", {"snapshots_on_disk": 1, "snapshots_in_ram": 2},
         marks=pytest.mark.skipif(hrevolve is None,
                                  reason="H-Revolve not available")),
     ("mixed", {"snapshots": 2, "storage": "disk"})])
@seed_test
def test_oscillator(setup_test, test_leaks,
                    tmp_path, cp_method, cp_parameters):
    n_steps = 20
    cp_parameters = dict(cp_parameters)
    if cp_method != "memory":
        cp_parameters["path"] = str(tmp_path / "checkpoints~")
    if cp_method == "multistage":
        cp_parameters["blocks"] = n_steps
    if cp_method in {"memory", "periodic_disk", "multistage"}:
        configure_checkpointing(cp_method, cp_parameters)
    elif cp_method == "H-Revolve":
        from tlm_adjoint.checkpoint_schedules import HRevolveCheckpointSchedule
        configure_checkpointing(
            lambda **cp_parameters: HRevolveCheckpointSchedule(max_n=n_steps, **cp_parameters),  # noqa: E501
            cp_parameters)
    else:
        assert cp_method == "mixed"
        from tlm_adjoint.checkpoint_schedules import MixedCheckpointSchedule
        configure_checkpointing(
            lambda **cp_parameters: MixedCheckpointSchedule(max_n=n_steps, **cp_parameters),  # noqa: E501
            cp_parameters)

    mesh = UnitSquareMesh(5, 5)
    r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
    space = VectorFunctionSpace(mesh, r0)
    test = TestFunction(space)
    T_0 = Function(space, name="T_0", static=True)
    T_0.interpolate(Constant((1.0, 0.0)))
    dt = Constant(0.01, static=True)

    def forward(T_0):
        clear_caches()

        T_n = Function(space, name="T_n")
        T_np1 = Function(space, name="T_np1")
        T_s = 0.5 * (T_n + T_np1)

        Assignment(T_n, T_0).solve()

        eq = EquationSolver(inner((T_np1 - T_n) / dt, test) * dx
                            - inner(T_s[1], test[0]) * dx
                            + inner(sin(T_s[0]), test[1]) * dx == 0,
                            T_np1,
                            solver_parameters=ns_parameters_newton_gmres)

        for n in range(n_steps):
            eq.solve()
            T_n.assign(T_np1)
            if n < n_steps - 1:
                new_block()

        J = Functional(name="J")
        J.assign(dot(T_n[0], T_n[0]) * dx)
        return J

    start_manager()
    J = forward(T_0)
    stop_manager()

    J_val = J.value
    J_val_ref = oscillator_ref() ** 2
    assert abs(J_val - J_val_ref) < 1.0e-14

    dJ = compute_gradient(J, T_0)

    min_order = taylor_test(forward, T_0, J_val=J_val, dJ=dJ)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, T_0, J_val=J_val, ddJ=ddJ)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, T_0, tlm_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=1)
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=2)
    assert min_order > 2.00


@pytest.mark.firedrake
@pytest.mark.parametrize("n_steps", [1, 2, 5, 20])
@seed_test
def test_diffusion_1d(setup_test, test_leaks,
                      tmp_path, n_steps):
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 2,
                             "snaps_in_ram": 3,
                             "path": str(tmp_path / "checkpoints~")})

    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    const_space = FunctionSpace(mesh, "R", 0)

    @manager_disabled()
    def constant(value, *args, **kwargs):
        F = Function(const_space, *args, **kwargs)
        F.assign(Constant(value))
        return F

    T_0 = Function(space, name="T_0", static=True)
    interpolate_expression(T_0, sin(pi * X[0]) + sin(10.0 * pi * X[0]))
    dt = Constant(0.01, static=True)
    kappa = constant(1.0, name="kappa", static=True)

    def forward(T_0, kappa):
        import sympy as sp
        n = sp.Symbol("n", int=True)
        N = sp.Symbol("N", int=True)

        T = {n: Function(space, name="T_n"),
             n + 1: Function(space, name="T_np1")}
        T[0] = T[n]
        T[N] = T[n]

        eq_T_np1 = EquationSolver(
            inner(trial, test) * dx
            + dt * inner(kappa * grad(trial), grad(test)) * dx
            == inner(T[n], test) * dx,
            T[n + 1],
            DirichletBC(space, 1.0, "on_boundary"),
            solver_parameters=ls_parameters_cg)

        T[0].assign(T_0)
        for n_step in range(n_steps):
            eq_T_np1.solve()
            T[n].assign(T[n + 1])
            if n_step < n_steps - 1:
                new_block()

        J = Functional(name="J")
        J.assign((dot(T[N], T[N]) ** 2) * dx)
        return J

    start_manager()
    J = forward(T_0, kappa)
    stop_manager()

    J_val = J.value
    if n_steps == 20:
        J_val_ref = diffusion_ref()
        assert abs(J_val - J_val_ref) < 1.0e-12

    dJs = compute_gradient(J, [T_0, kappa])

    if issubclass(var_dtype(kappa), np.floating):
        dm_kappa = constant(1.0, name="dm_kappa", static=True)
    else:
        dm_kappa = None
    for m, forward_J, dJ, dm in \
            [(T_0, lambda T_0: forward(T_0, kappa), dJs[0], None),
             (kappa, lambda kappa: forward(T_0, kappa), dJs[1], dm_kappa)]:
        min_order = taylor_test(
            forward_J, m, J_val=J_val, dJ=dJ,
            dM=dm)
        assert min_order > 1.99

        ddJ = Hessian(forward_J)
        min_order = taylor_test(
            forward_J, m, J_val=J_val, ddJ=ddJ, size=3,
            dM=dm)
        assert min_order > 2.98

        min_order = taylor_test_tlm(
            forward_J, m, tlm_order=1,
            dMs=None if dm is None else (dm,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(
            forward_J, m, adjoint_order=1,
            dMs=None if dm is None else (dm,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(
            forward_J, m, adjoint_order=2, seed=1.0e-3,
            dMs=None if dm is None else (dm, dm))
        assert min_order > 1.99


@pytest.mark.firedrake
@seed_test
def test_diffusion_2d(setup_test, test_leaks,
                      tmp_path):
    n_steps = 20
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 2,
                             "snaps_in_ram": 3,
                             "path": str(tmp_path / "checkpoints~")})

    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    const_space = FunctionSpace(mesh, "R", 0)

    @manager_disabled()
    def constant(value, *args, **kwargs):
        F = Function(const_space, *args, **kwargs)
        F.assign(Constant(value))
        return F

    T_0 = Function(space, name="T_0", static=True)
    interpolate_expression(T_0, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    dt = Constant(0.01, static=True)
    kappa = [constant(1.0, name="kappa_00", static=True),
             constant(0.0, name="kappa_10", static=True),
             constant(0.0, name="kappa_01", static=True),
             constant(1.0, name="kappa_11", static=True)]
    bc = DirichletBC(space, 0.0, "on_boundary")

    def forward(kappa_00, kappa_01, kappa_10, kappa_11):
        clear_caches()
        kappa = as_tensor([[kappa_00, kappa_01],
                           [kappa_10, kappa_11]])

        T_n = Function(space, name="T_n")
        T_np1 = Function(space, name="T_np1")

        Assignment(T_n, T_0).solve()

        eq = (inner(trial / dt, test) * dx
              + inner(dot(kappa, grad(trial)), grad(test)) * dx
              == inner(T_n / dt, test) * dx)
        eqs = [EquationSolver(eq, T_np1, bc,
                              solver_parameters=ls_parameters_cg),
               Assignment(T_n, T_np1)]

        for n in range(n_steps):
            for eq in eqs:
                eq.solve()
            if n < n_steps - 1:
                new_block()

        J = Functional(name="J")
        J.assign(dot(T_np1, T_np1) * dx)

        return J

    for tlm_order in range(1, 4):
        min_order = taylor_test_tlm(forward, kappa, tlm_order=tlm_order,
                                    seed=1.0e-3)
        assert min_order > 1.99

    for adjoint_order in range(1, 5):
        min_order = taylor_test_tlm_adjoint(forward, kappa,
                                            adjoint_order=adjoint_order,
                                            seed=1.0e-3)
        assert min_order > 1.99
