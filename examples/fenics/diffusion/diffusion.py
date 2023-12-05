from fenics import *
from tlm_adjoint.fenics import *

# import h5py
import numpy as np
# import petsc4py.PETSc as PETSc

# PETSc.Options().setValue("citations", "petsc.bib")
np.random.seed(87838678 + DEFAULT_COMM.rank)

mesh = UnitSquareMesh(50, 50)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)
bc = HomogeneousDirichletBC(space, "on_boundary")

dt = Constant(0.01, static=True)
N = 10
kappa = Function(space, name="kappa", static=True)
kappa.interpolate(Constant(1.0))
Psi_0 = Function(space, name="Psi_0", static=True)
Psi_0.interpolate(Expression("exp(x[0]) * sin(pi * x[0])"
                             + " * sin(10.0 * pi * x[0])"
                             + " * sin(2.0 * pi * x[1])",
                             element=space.ufl_element()))

zeta_1 = Function(space, name="zeta_1", static=True)
zeta_2 = Function(space, name="zeta_2", static=True)
zeta_3 = ZeroFunction(space, name="zeta_3")
var_set_values(zeta_1,
               2.0 * np.random.random(var_local_size(zeta_1)) - 1.0)
var_set_values(zeta_2,
               2.0 * np.random.random(var_local_size(zeta_2)) - 1.0)
# File("zeta_1.pvd", "compressed") << zeta_1
# File("zeta_2.pvd", "compressed") << zeta_2


def forward(kappa, output_filename=None):
    clear_caches()

    Psi_n = Function(space, name="Psi_n")
    Psi_np1 = Function(space, name="Psi_np1")

    eq = EquationSolver(inner(trial / dt, test) * dx
                        + inner(kappa * grad(trial), grad(test)) * dx
                        == inner(Psi_n / dt, test) * dx, Psi_np1,
                        bc, solver_parameters={"linear_solver": "cg",
                                               "preconditioner": "sor",
                                               "krylov_solver": {"absolute_tolerance": 1.0e-16,  # noqa: E501
                                                                 "relative_tolerance": 1.0e-14}})  # noqa: E501
    cycle = Assignment(Psi_n, Psi_np1)

    if output_filename is not None:
        f = File(output_filename, "compressed")

    Assignment(Psi_n, Psi_0).solve()
    if output_filename is not None:
        f << (Psi_n, 0.0)
    for n in range(N):
        eq.solve()
        if n < N - 1:
            cycle.solve()
            new_block()
        else:
            Psi_n = Psi_np1
            Psi_n.rename("Psi_n", "a Function")
            del Psi_np1
        if output_filename is not None:
            f << (Psi_n, (n + 1) * float(dt))

    J = Functional(name="J")
    J.assign(dot(Psi_n, Psi_n) * dx)

    return J


configure_tlm((kappa, zeta_1), ((kappa, zeta_1), (zeta_2, zeta_3)))
start_manager()
# J = forward(kappa, output_filename="forward.pvd")
J = forward(kappa)
dJ_tlm_1 = var_tlm(J, (kappa, zeta_1))
dJ_tlm_2 = var_tlm(J, ((kappa, zeta_1), (zeta_2, zeta_3)))
ddJ_tlm = var_tlm(dJ_tlm_1, ((kappa, zeta_1), (zeta_2, zeta_3)))
stop_manager()

dJ_adj, ddJ_adj, dddJ_adj = compute_gradient(ddJ_tlm, (zeta_3, zeta_2, kappa))


def info_compare(x, y, tol):
    info(f"{x:.16e} {y:.16e} {abs(x - y):.16e}")
    assert abs(x - y) < tol


info("TLM/adjoint consistency, zeta_1")
info_compare(dJ_tlm_1.value, var_inner(zeta_1, dJ_adj), tol=1.0e-18)

info("TLM/adjoint consistency, zeta_2")
info_compare(dJ_tlm_2.value, var_inner(zeta_2, dJ_adj), tol=1.0e-17)

info("Second order TLM/adjoint consistency")
info_compare(ddJ_tlm.value, var_inner(zeta_2, ddJ_adj), tol=1.0e-17)

min_order = taylor_test_tlm(forward, kappa, tlm_order=1, seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm(forward, kappa, tlm_order=2, seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm_adjoint(forward, kappa, adjoint_order=1,
                                    seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm_adjoint(forward, kappa, adjoint_order=2,
                                    seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm_adjoint(forward, kappa, adjoint_order=3,
                                    seed=1.0e-3)
assert min_order > 1.99
