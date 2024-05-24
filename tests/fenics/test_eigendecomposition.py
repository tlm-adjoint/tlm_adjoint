from fenics import *
from tlm_adjoint.fenics import *

from .test_base import *

import numpy as np
import pytest
try:
    import slepc4py.SLEPc as SLEPc
except ModuleNotFoundError:
    SLEPc = None

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")
pytestmark = pytest.mark.skipif(
    SLEPc is None,
    reason="SLEPc not available")


@pytest.mark.fenics
@pytest.mark.skipif(complex_mode, reason="real only")
@seed_test
def test_CachedHessian(setup_test):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitIntervalMesh(5)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    zero = Constant(0.0, name="zero")

    def forward(F):
        y = Function(space, name="y")
        EquationSolver(
            inner(grad(trial), grad(test)) * dx
            == inner(F, test) * dx + inner(zero * sin(F), test) * dx,
            y, DirichletBC(space, 0.0, "on_boundary"),
            solver_parameters=ls_parameters_cg).solve()

        J = Functional(name="J")
        J.addto((dot(y, y) ** 2) * dx)
        return J

    F = Function(space, name="F", static=True)
    F.interpolate(Constant(1.0))

    start_manager()
    J = forward(F)
    stop_manager()

    H = Hessian(forward)
    H_opt = CachedHessian(J)

    # Test consistency of matrix action for static direction

    zeta = Function(space, name="zeta", static=True)
    for i in range(5):
        zeta_arr = np.random.random(var_local_size(zeta))
        if issubclass(var_dtype(zeta), np.complexfloating):
            zeta_arr = zeta_arr \
                + 1.0j * np.random.random(var_local_size(zeta))
        var_set_values(zeta, zeta_arr)
        del zeta_arr

        # Leads to an inconsistency if the stored value is not used
        zero.assign(np.NAN)
        _, _, ddJ_opt = H_opt.action(F, zeta)
        zero.assign(0.0)
        _, _, ddJ = H.action(F, zeta)

        error = var_copy(ddJ)
        var_axpy(error, -1.0, ddJ_opt)
        assert var_linf_norm(error) == 0.0

    with paused_space_type_checking():
        lam, V = eigendecompose(space, H.action_fn(F),
                                problem_type=SLEPc.EPS.ProblemType.HEP)

    assert issubclass(lam.dtype.type, np.floating)

    assert len(lam) == len(V)
    for lam_i, v_i in zip(lam, V):
        _, _, v_error = H.action(F, v_i)
        with paused_space_type_checking():
            var_axpy(v_error, -lam_i, v_i)
        assert var_linf_norm(v_error) < 1.0e-19

        _, _, v_error = H_opt.action(F, v_i)
        with paused_space_type_checking():
            var_axpy(v_error, -lam_i, v_i)
        assert var_linf_norm(v_error) < 1.0e-19

    with paused_space_type_checking():
        lam_opt, V_opt = eigendecompose(space, H_opt.action_fn(F),
                                        problem_type=SLEPc.EPS.ProblemType.HEP)

    assert issubclass(lam.dtype.type, np.floating)
    error = (np.array(sorted(lam.real), dtype=float)
             - np.array(sorted(lam_opt.real), dtype=float))
    assert abs(error).max() == 0.0

    assert len(lam) == len(V)
    for lam_i, v_i in zip(lam_opt, V_opt):
        _, _, v_error = H.action(F, v_i)
        with paused_space_type_checking():
            var_axpy(v_error, -lam_i, v_i)
        assert var_linf_norm(v_error) < 1.0e-19

        _, _, v_error = H_opt.action(F, v_i)
        with paused_space_type_checking():
            var_axpy(v_error, -lam_i, v_i)
        assert var_linf_norm(v_error) < 1.0e-19
