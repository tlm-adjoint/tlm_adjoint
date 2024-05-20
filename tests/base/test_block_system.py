from tlm_adjoint import DEFAULT_COMM, Float, no_float_overloading
from tlm_adjoint.block_system import LinearSolver, Matrix

from .test_base import seed_test, setup_test  # noqa: F401

try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")
pytestmark = pytest.mark.skipif(
    PETSc is None,
    reason="PETSc not available")


@pytest.mark.base
@seed_test
def test_division_solve(setup_test):  # noqa: F811
    class MultMatrix(Matrix):
        def __init__(self, space, alpha):
            super().__init__(space, space)
            self._alpha = alpha

        @no_float_overloading
        def mult_add(self, x, y):
            y.addto(alpha * x.value)

    alpha = 0.5
    b = Float(-2.0)

    solver = LinearSolver(
        MultMatrix(b.space, alpha),
        solver_parameters={"ksp_type": "cg",
                           "ksp_atol": 1.0e-15,
                           "ksp_rtol": 0.0})

    u = Float()
    solver.solve(u, b)
    assert abs(u.value - (b.value / alpha)) == 0.0
    assert solver.ksp.getIterationNumber() == 1
