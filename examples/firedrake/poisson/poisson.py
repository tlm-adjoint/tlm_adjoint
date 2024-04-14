from firedrake import *
from tlm_adjoint.firedrake import *

import numpy as np
# import petsc4py.PETSc as PETSc

# Save relevant citation information
# PETSc.Options().setValue("citations", "petsc.bib")

# Seed the random number generator, to ensure reproducibility of the later
# Taylor verification
np.random.seed(82844243 + DEFAULT_COMM.rank)

# Configure a simple discrete function space
mesh = UnitSquareMesh(20, 20)
X = SpatialCoordinate(mesh)
space = FunctionSpace(mesh, "Lagrange", 1)

# Configure a boundary condition
bc = DirichletBC(space, 1.0, "on_boundary")


def forward(F, x0=None):
    # Clear caches
    clear_caches()

    # Construct and solve a simple equation (the Poisson equation with
    # inhomogeneous Dirichlet boundary conditions)
    x = Function(space, name="x")
    test, trial = TestFunction(space), TrialFunction(space)
    ls_parameters = {"ksp_type": "cg",
                     "pc_type": "sor",
                     "ksp_rtol": 1.0e-14,
                     "ksp_atol": 1.0e-16}
    EquationSolver(inner(grad(trial), grad(test)) * dx
                   == inner(F * F, test) * dx,
                   x, bc, solver_parameters=ls_parameters).solve()

    if x0 is None:
        # If a reference is not provided, create it by copying x
        x0 = var_copy(x, name="x0")

    # The functional
    J = Functional(name="J")
    J.assign(dot(x - x0, x - x0) * dx)

    # Return the reference and the functional
    return x0, J


# Generate a reference solution x0 using F0. The optional "static" flag is used
# for optimization.
F0 = Function(space, name="F0", static=True)
F0.interpolate(
    sin(pi * X[0]) * sin(3.0 * pi * X[1]) * exp(X[0] * X[1]))
x0, _ = forward(F0)

# Set F to one everywhere ...
F = Function(space, name="F", static=True)
F.interpolate(Constant(1.0))
# ... and re-run the forward with this value of F, now obtaining a mis-match
# functional, and processing equations using the manager
start_manager()
_, J = forward(F, x0=x0)
stop_manager()

# Display equation manager information
manager_info()

# Compute a forward model constrained derivative of J with respect to F
dJ = compute_gradient(J, F)
# Taylor verify the forward model constrained derivative
min_order = taylor_test(lambda F: forward(F, x0=x0)[1], F, J_val=J.value,
                        dJ=dJ, size=5)
assert min_order > 1.99

# Taylor verify forward model constrained Hessian actions for the case of two
# equal perturbation directions. Omitting the dJ argument here includes the
# first order tangent-linear in the verification test.
ddJ = Hessian(lambda F: forward(F, x0=x0)[1])
min_order = taylor_test(lambda F: forward(F, x0=x0)[1], F, J_val=J.value,
                        ddJ=ddJ, size=5)
assert min_order > 2.99

# Taylor verify the first order tangent-linear
min_order = taylor_test_tlm(lambda F: forward(F, x0=x0)[1], F,
                            tlm_order=1, size=5)
assert min_order > 1.99

# Taylor verify forward model constrained Hessian actions for the case of two
# possibly different perturbation directions. Assumes validity of the first
# order tangent-linear.
min_order = taylor_test_tlm_adjoint(lambda F: forward(F, x0=x0)[1], F,
                                    adjoint_order=2, size=5)
assert min_order > 1.99
