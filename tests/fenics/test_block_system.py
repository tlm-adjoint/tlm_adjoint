from fenics import *
from tlm_adjoint.fenics import *
from tlm_adjoint.fenics.backend_interface import matrix_multiply
from tlm_adjoint.block_system import Eigensolver, MatrixFreeMatrix

from .test_base import *

import numbers
import numpy as np
import petsc4py.PETSc as PETSc
import pytest
try:
    import slepc4py.SLEPc as SLEPc
except ModuleNotFoundError:
    SLEPc = None

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
@no_space_type_checking
@pytest.mark.skipif(SLEPc is None, reason="SLEPc not available")
@seed_test
def test_HEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    M = assemble(inner(trial, test) * dx)

    def M_action(x, y):
        assemble(inner(x, test) * dx, tensor=y)

    esolver = Eigensolver(MatrixFreeMatrix(space, space, M_action),
                          solver_parameters={"eps_type": "krylovschur",
                                             "eps_hermitian": None,
                                             "eps_largest_magnitude": None,
                                             "eps_nev": space.dim(),
                                             "eps_conv_rel": None,
                                             "eps_tol": 1.0e-12,
                                             "eps_purify": False})
    esolver.solve()
    assert len(esolver) == space.dim()

    error = Function(space)
    for lam, (v_r, v_i) in esolver:
        assert isinstance(lam, numbers.Real)
        assert lam > 0.0
        assert abs(var_inner(v_r, v_r) - 1.0) < 1.0e-14
        matrix_multiply(M, v_r, tensor=error)
        var_axpy(error, -lam, v_r)
        assert var_linf_norm(error) < 1.0e-16
        assert v_i is None


@pytest.mark.fenics
@no_space_type_checking
@pytest.mark.skipif(SLEPc is None, reason="SLEPc not available")
@seed_test
def test_NHEP(setup_test, test_leaks):
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    N = assemble(inner(trial.dx(0), test) * dx)

    def N_action(x, y):
        assemble(inner(x.dx(0), test) * dx, tensor=y)

    esolver = Eigensolver(MatrixFreeMatrix(space, space, N_action),
                          solver_parameters={"eps_type": "krylovschur",
                                             "eps_non_hermitian": None,
                                             "eps_largest_magnitude": None,
                                             "eps_nev": space.dim(),
                                             "eps_conv_rel": None,
                                             "eps_tol": 1.0e-12,
                                             "eps_purify": False})
    esolver.solve()
    assert len(esolver) == space.dim()

    error = Function(space)
    if issubclass(PETSc.ScalarType, np.floating):
        for lam, (v_r, v_i) in esolver:
            assert isinstance(lam, numbers.Complex) and not isinstance(lam, numbers.Real)  # noqa: E501
            assert abs(lam.real) < 1.0e-15
            assert abs(var_inner(v_r, v_r) + var_inner(v_i, v_i) - 1.0) < 1.0e-14  # noqa: E501
            matrix_multiply(N, v_r, tensor=error)
            var_axpy(error, -lam.real, v_r)
            var_axpy(error, +lam.imag, v_i)
            assert var_linf_norm(error) < 1.0e-15
            matrix_multiply(N, v_i, tensor=error)
            var_axpy(error, -lam.real, v_i)
            var_axpy(error, -lam.imag, v_r)
            assert var_linf_norm(error) < 1.0e-15
    else:
        raise ValueError("Unexpected dtype")
