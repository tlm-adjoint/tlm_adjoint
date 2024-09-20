from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.block_system import Eigensolver
from tlm_adjoint.hessian_system import HessianMatrix

from .test_base import *

import numbers
import numpy as np
import pytest
try:
    import slepc4py.SLEPc as SLEPc
except ModuleNotFoundError:
    SLEPc = None

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@pytest.mark.skipif(complex_mode, reason="real only")
@pytest.mark.skipif(SLEPc is None, reason="SLEPc not available")
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

    min_order = taylor_test(forward, F, J_val=J.value, ddJ=H)
    assert min_order > 3.00

    min_order = taylor_test(forward, F, J_val=J.value, ddJ=H_opt)
    assert min_order > 3.00

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
        zero.assign(np.nan)
        _, _, ddJ_opt = H_opt.action(F, zeta)
        zero.assign(0.0)
        _, _, ddJ = H.action(F, zeta)

        error = var_copy(ddJ)
        var_axpy(error, -1.0, ddJ_opt)
        assert var_linf_norm(error) == 0.0

    esolver = Eigensolver(
        HessianMatrix(H, F),
        solver_parameters={"eps_type": "krylovschur",
                           "eps_hermitian": None,
                           "eps_largest_magnitude": None,
                           "eps_nev": space.dim(),
                           "eps_conv_rel": None,
                           "eps_tol": 1.0e-12,
                           "eps_purify": False})
    esolver.solve()
    assert len(esolver) == space.dim()
    assert esolver.B_orthonormality_test() < 1.0e-14

    for lam, (v, _) in esolver:
        assert isinstance(lam, numbers.Real)

        _, _, v_error = H.action(F, v)
        var_axpy(v_error, -lam, v.riesz_representation("l2"))
        assert var_linf_norm(v_error) < 1.0e-18

        _, _, v_error = H_opt.action(F, v)
        var_axpy(v_error, -lam, v.riesz_representation("l2"))
        assert var_linf_norm(v_error) < 1.0e-18

    esolver_opt = Eigensolver(
        HessianMatrix(H_opt, F),
        solver_parameters={"eps_type": "krylovschur",
                           "eps_hermitian": None,
                           "eps_largest_magnitude": None,
                           "eps_nev": space.dim(),
                           "eps_conv_rel": None,
                           "eps_tol": 1.0e-12,
                           "eps_purify": False})
    esolver_opt.solve()
    assert len(esolver_opt) == space.dim()
    assert esolver_opt.B_orthonormality_test() < 1.0e-14

    for lam, (v, _) in esolver_opt:
        assert isinstance(lam, numbers.Real)

        _, _, v_error = H.action(F, v)
        var_axpy(v_error, -lam, v.riesz_representation("l2"))
        assert var_linf_norm(v_error) < 1.0e-18

        _, _, v_error = H_opt.action(F, v)
        var_axpy(v_error, -lam, v.riesz_representation("l2"))
        assert var_linf_norm(v_error) < 1.0e-18

    lam, _ = esolver.eigenpairs()
    lam_opt, _ = esolver_opt.eigenpairs()
    assert abs(lam - lam_opt).max() == 0.0
