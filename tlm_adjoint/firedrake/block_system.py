r"""This module implements solvers for linear systems defined in mixed spaces.

The :class:`.System` class defines the block structure of the linear system,
and solves the system using an outer Krylov solver. A custom preconditioner can
be defined via the `pc_fn` callback to :meth:`.System.solve`, and this
preconditioner can itself e.g. make use of further Krylov solvers. This
provides a Python interface for custom block preconditioners.

Given a linear problem with a potentially singular matrix :math:`A`

.. math::

    A u = b,

a :class:`.System` instead solves the linear problem

.. math::

    \left[ (I - M U (U^* M U)^{-1} U^*) A (I - V (V^* C V)^{-1} V^* C)
        + M U S V^* C \right] u = (I - M U (U^* M U)^{-1} U^*) b.

Here

    - :math:`U` is a full rank matrix whose columns span the left nullspace for
      a modified system matrix :math:`\tilde{A}`.
    - :math:`V` is a full rank matrix with the same number of columns as
      :math:`U`, whose columns span the nullspace for :math:`\tilde{A}`.
    - :math:`V^* C V` and :math:`S` are invertible matrices.
    - :math:`M` is a Hermitian positive definite matrix.

Here the left nullspace for a matrix is defined to be the nullspace for its
Hermitian transpose, and the modified system matrix :math:`\tilde{A}` is
defined

.. math::

    \tilde{A} = (I - M U (U^* M U)^{-1} U^*) A (I - V (V^* C V)^{-1} V^* C).

This has two primary use cases:

    1. Where a matrix :math:`A` and right-hand-side :math:`b` are constructed
       via finite element assembly on superspaces of the test space and trial
       space. The typical example is in the application of homogeneous
       essential Dirichlet boundary conditions.

    2. Where the matrix :math:`A` is singular and :math:`b` is orthogonal to
       the left nullspace of :math:`A`. Typically one would then choose
       :math:`U` and :math:`V` so that their columns respectively span the left
       nullspace and nullspace of :math:`A`, and the :class:`.System` then
       seeks a solution to the original problem subject to the linear
       constraints :math:`V^* C u = 0`.

Function spaces are defined via Firedrake function spaces, and
:class:`Sequence` objects containing Firedrake function spaces or similar
:class:`Sequence` objects. Similarly functions are defined via
:class:`firedrake.function.Function` or
:class:`firedrake.cofunction.Cofunction` objects, or :class:`Sequence` objects
containing :class:`firedrake.function.Function`,
:class:`firedrake.cofunction.Cofunction`, or similar :class:`Sequence` objects.
This defines a basic tree structure which is useful e.g. when defining block
matrices in terms of sub-block matrices.

Elements of the tree are accessed in a consistent order using a depth first
search. Hence e.g.

.. code-block:: python

    ((u_0, u_1), u_2)

and

.. code-block:: python

    (u_0, u_1, u_2)

where `u_0`, `u_1`, and `u_2` are :class:`firedrake.function.Function` or
:class:`firedrake.cofunction.Cofunction` objects, are both valid
representations of a mixed space solution.
"""

from ..block_system import (
    BlockMatrix as _BlockMatrix, BlockNullspace, Matrix, MixedSpace,
    NoneNullspace, Nullspace, Preconditioner, SystemMatrix, iter_sub, zip_sub)

from firedrake import (
    Constant, DirichletBC, Function, TestFunction, assemble)

import petsc4py.PETSc as PETSc
import ufl

from collections.abc import Sequence
from functools import wraps
import logging
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None

__all__ = \
    [
        "MixedSpace",

        "Nullspace",
        "NoneNullspace",
        "ConstantNullspace",
        "UnityNullspace",
        "DirichletBCNullspace",
        "BlockNullspace",

        "Matrix",
        "PETScMatrix",
        "BlockMatrix",
        "form_matrix",

        "System"
    ]


def apply_bcs(u, bcs):
    if not isinstance(bcs, Sequence):
        bcs = (bcs,)
    if len(bcs) > 0 and not isinstance(u.function_space(), type(bcs[0].function_space())):  # noqa: E501
        u_bc = u.riesz_representation("l2")
    else:
        u_bc = u
    for bc in bcs:
        bc.apply(u_bc)


class ConstantNullspace(Nullspace):
    r"""A nullspace and left nullspace spanned by the vector of ones.

    Here :math:`V = U`, :math:`U` is a single column matrix whose elements are
    ones, :math:`C = M`, and :math:`M` is an identity matrix.

    :arg alpha: Defines the linear constraint matrix :math:`S = \left( \alpha /
        N \right)` where :math:`N` is the length of the vector of ones.
    """

    def __init__(self, *, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    @staticmethod
    def _correct(x, y, *, alpha=1.0):
        with x.dat.vec_ro as x_v:
            x_sum = x_v.sum()
            N = x_v.getSize()

        with y.dat.vec as y_v:
            y_v.shift(alpha * x_sum / float(N))

    def apply_nullspace_transformation_lhs_right(self, x):
        self._correct(x, x, alpha=-1.0)

    def apply_nullspace_transformation_lhs_left(self, y):
        self._correct(y, y, alpha=-1.0)

    def constraint_correct_lhs(self, x, y):
        self._correct(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._correct(b, u, alpha=1.0 / self._alpha)


class UnityNullspace(Nullspace):
    r"""A nullspace and left nullspace defined by the unity-valued function.

    Here :math:`V = U`, :math:`U` is a single column matrix containing the
    degree-of-freedom vector for the unity-valued function, :math:`C = M`,
    and :math:`M` is the mass matrix.

    :arg space: A scalar-valued function space containing the unity-valued
        function.
    :arg alpha: Defines the linear constraint matrix :math:`S = \alpha \left(
        U^* M U \right)^{-1}`.
    """

    def __init__(self, space, *, alpha=1.0):
        U = Function(space, name="U")
        U.interpolate(Constant(1.0))
        MU = assemble(ufl.inner(U, TestFunction(space)) * ufl.dx)
        UMU = assemble(ufl.inner(U, U) * ufl.dx)

        self._alpha = alpha
        self._U = U
        self._MU = MU
        self._UMU = UMU

    @staticmethod
    def _correct(x, y, u, v, *, alpha=1.0):
        with x.dat.vec_ro as x_v, u.dat.vec_ro as u_v:
            u_x = x_v.dot(u_v)

        with y.dat.vec as y_v, v.dat.vec_ro as v_v:
            y_v.axpy(alpha * u_x, v_v)

    def apply_nullspace_transformation_lhs_right(self, x):
        self._correct(
            x, x, self._MU, self._U, alpha=-1.0 / self._UMU)

    def apply_nullspace_transformation_lhs_left(self, y):
        self._correct(
            y, y, self._U, self._MU, alpha=-1.0 / self._UMU)

    def constraint_correct_lhs(self, x, y):
        self._correct(
            x, y, self._MU, self._MU, alpha=self._alpha / self._UMU)

    def pc_constraint_correct_soln(self, u, b):
        self._correct(
            b, u, self._U, self._U, alpha=1.0 / (self._alpha * self._UMU))


class DirichletBCNullspace(Nullspace):
    r"""A nullspace and left nullspace associated with homogeneous Dirichlet
    boundary conditions.

    Here :math:`V = U`, :math:`U` is a zero-one matrix with exactly one
    non-zero per column corresponding to one boundary condition
    degree-of-freedom, :math:`C = M`, and :math:`M` is an identity matrix.

    :arg bcs: A :class:`firedrake.bcs.DirichletBC`, or a :class:`Sequence` of
        :class:`firedrake.bcs.DirichletBC` objects.
    :arg alpha: Defines the linear constraint matrix :math:`S = \alpha M`.
    """

    def __init__(self, bcs, *, alpha=1.0):
        if isinstance(bcs, Sequence):
            bcs = tuple(bcs)
        else:
            bcs = (bcs,)

        space = bcs[0].function_space()
        for bc in bcs:
            if bc.function_space() != space:
                raise ValueError("Invalid space")
            if not isinstance(bc._function_arg, ufl.classes.Zero):
                raise ValueError("Homogeneous boundary conditions required")

        super().__init__()
        self._bcs = bcs
        self._alpha = alpha
        self._c = Function(space)

    def apply_nullspace_transformation_lhs_right(self, x):
        apply_bcs(x, self._bcs)

    def apply_nullspace_transformation_lhs_left(self, y):
        apply_bcs(y, self._bcs)

    def _constraint_correct_lhs(self, x, y, *, alpha=1.0):
        with self._c.dat.vec_wo as c_v:
            c_v.zeroEntries()

        apply_bcs(self._c,
                  tuple(DirichletBC(x.function_space(), x, bc.sub_domain)
                        for bc in self._bcs))

        with self._c.dat.vec_ro as c_v, y.dat.vec as y_v:
            y_v.axpy(alpha, c_v)

    def constraint_correct_lhs(self, x, y):
        self._constraint_correct_lhs(x, y, alpha=self._alpha)

    def pc_constraint_correct_soln(self, u, b):
        self._constraint_correct_lhs(b, u, alpha=1.0 / self._alpha)


class PETScMatrix(Matrix):
    r"""A :class:`tlm_adjoint.block_system.Matrix` associated with a
    :class:`firedrake.matrix.Matrix` :math:`A` mapping :math:`V \rightarrow W`.

    :arg arg_space: Defines the space `V`.
    :arg action_space: Defines the space `W`.
    :arg a: The :class:`firedrake.matrix.Matrix`.
    """

    def __init__(self, arg_space, action_space, a):
        super().__init__(arg_space, action_space)
        self._matrix = a

    def mult_add(self, x, y):
        matrix = self._matrix.petscmat
        with x.dat.vec_ro as x_v, y.dat.vec as y_v:
            matrix.multAdd(x_v, y_v, y_v)


def form_matrix(a, *args, **kwargs):
    """Construct a :class:`.PETScMatrix` associated with a given sesquilinear
    form.

    :arg a: A :class:`ufl.Form` defining the sesquilinear form.
    :returns: The :class:`.PETScMatrix`.

    Remaining arguments are passed to the :func:`firedrake.assemble.assemble`
    function.
    """

    test, trial = a.arguments()
    assert test.number() < trial.number()

    return PETScMatrix(
        trial.function_space(), test.function_space().dual(),
        assemble(a, *args, **kwargs))


class BlockMatrix(_BlockMatrix):
    """A :class:`tlm_adjoint.block_system.BlockMatrix` where blocks may also be
    defined by a :class:`ufl.Form`.
    """

    def __setitem__(self, key, value):
        if isinstance(value, ufl.classes.Form):
            value = form_matrix(value)
        super().__setitem__(key, value)


class System:
    """A linear system

    .. math::

        A u = b.

    :arg arg_spaces: Defines the space for `u`.
    :arg action_spaces: Defines the space for `b`.
    :arg blocks: One of

        - A :class:`tlm_adjoint.firedrake.block_system.Matrix` or
          :class:`ufl.Form` defining :math:`A`.
        - A :class:`Mapping` with items `((i, j), block)` where the matrix
          associated with the block in the `i` th and `j` th column is defined
          by `block`. Each `block` is a
          :class:`tlm_adjoint.firedrake.block_system.Matrix` or
          :class:`ufl.Form`, or `None` to indicate a zero block.

    :arg nullspaces: A :class:`.Nullspace` or a :class:`Sequence` of
        :class:`.Nullspace` objects defining the nullspace and left nullspace
        of :math:`A`. `None` indicates a :class:`.NoneNullspace`.
    :arg comm: Communicator.
    """

    def __init__(self, arg_spaces, action_spaces, blocks, *,
                 nullspaces=None, comm=None):
        if not isinstance(arg_spaces, MixedSpace):
            arg_spaces = MixedSpace(arg_spaces)
        if not isinstance(action_spaces, MixedSpace):
            action_spaces = MixedSpace(action_spaces)

        matrix = BlockMatrix(arg_spaces, action_spaces, blocks)

        nullspace = BlockNullspace(nullspaces)
        if isinstance(nullspace, BlockNullspace):
            if len(nullspace) != len(arg_spaces):
                raise ValueError("Invalid space")
            if len(nullspace) != len(action_spaces):
                raise ValueError("Invalid space")

        if comm is None:
            comm = arg_spaces.comm

        self._comm = comm
        self._arg_space = arg_spaces
        self._action_space = action_spaces
        self._matrix = matrix
        self._nullspace = nullspace

    def solve(self, u, b, *,
              solver_parameters=None, pc_fn=None,
              pre_callback=None, post_callback=None,
              correct_initial_guess=True, correct_solution=True):
        """Solve the linear system.

        :arg u: Defines the solution :math:`u`.
        :arg b: Defines the right-hand-side :math:`b`.
        :arg solver_parameters: A :class:`Mapping` defining outer Krylov solver
            parameters. Parameters (a number of which are based on FEniCS
            solver parameters) are:

            - `'linear_solver'`: The Krylov solver type, default `'fgmres'`.
            - `'pc_side'`: Overrides the PETSc default preconditioning side.
            - `'relative_tolerance'`: Relative tolerance. Required.
            - `'absolute_tolerance'`: Absolute tolerance. Required.
            - `'divergence_limit'`: Overrides the default divergence limit.
            - `'maximum_iterations'`: Maximum number of iterations. Default
              1000.
            - `'norm_type'`: Overrides the default convergence norm definition.
            - `'nonzero_initial_guess'`: Whether to use a non-zero initial
              guess, defined by the input `u`. Default `True`.
            - `'gmres_restart'`: Overrides the default GMRES restart parameter.

        :arg pc_fn: Defines the application of a preconditioner. A callable

            .. code-block:: python

                def pc_fn(u, b):

            The preconditioner is applied to `b`, and the result stored in `u`.
            Defaults to an identity.
        :arg pre_callback: A callable accepting a single
            :class:`petsc4py.PETSc.KSP` argument. Used for detailed manual
            configuration. Called after all other configuration options are
            set, but before the :meth:`petsc4py.PETSc.KSP.setUp` method is
            called.
        :arg post_callback: A callable accepting a single
            :class:`petsc4py.PETSc.KSP` argument. Called after the
            :meth:`petsc4py.PETSc.KSP.solve` method has been called.
        :arg correct_initial_guess: Whether to apply a nullspace correction to
            the initial guess.
        :arg correct_solution: Whether to apply a nullspace correction to
            the solution.
        :returns: The number of Krylov iterations.
        """

        if solver_parameters is None:
            solver_parameters = {}

        if isinstance(u, Sequence):
            u = tuple(u)
        else:
            u = (u,)

            if pc_fn is not None:
                pc_fn_u = pc_fn

                @wraps(pc_fn_u)
                def pc_fn(u, b):
                    u, = tuple(iter_sub(u))
                    return pc_fn_u(u, b)
        u = self._arg_space.tuple_sub(u)

        if isinstance(b, Sequence):
            b = tuple(b)
        else:
            b = (b,)

            if pc_fn is not None:
                pc_fn_b = pc_fn

                @wraps(pc_fn_b)
                def pc_fn(u, b):
                    b, = tuple(iter_sub(b))
                    return pc_fn_b(u, b)
        b = self._action_space.tuple_sub(b)

        if tuple(u_i.function_space() for u_i in iter_sub(u)) \
                != self._arg_space.flattened_space:
            raise ValueError("Invalid space")
        for b_i, space in zip_sub(b, self._action_space.split_space):
            if b_i is not None and b_i.function_space() != space:
                raise ValueError("Invalid space")

        b_c = self._action_space.new_split()
        for b_c_i, b_i in zip_sub(b_c, b):
            if b_i is not None:
                with b_c_i.dat.vec_wo as b_c_i_v, b_i.dat.vec_ro as b_i_v:
                    b_i_v.copy(result=b_c_i_v)

        A = SystemMatrix(self._matrix, self._nullspace)

        mat_A = PETSc.Mat().createPython(
            ((self._action_space.local_size, self._action_space.global_size),
             (self._arg_space.local_size, self._arg_space.global_size)), A,
            comm=self._comm)
        mat_A.setUp()

        if pc_fn is not None:
            A_pc = Preconditioner(self._action_space, self._arg_space,
                                  pc_fn, self._nullspace)
            pc = PETSc.PC().createPython(
                A_pc, comm=self._comm)
            pc.setOperators(mat_A)
            pc.setUp()

        ksp_solver = PETSc.KSP().create(comm=self._comm)
        ksp_solver.setType(solver_parameters.get("linear_solver", "fgmres"))
        if pc_fn is not None:
            ksp_solver.setPC(pc)
        if "pc_side" in solver_parameters:
            ksp_solver.setPCSide(solver_parameters["pc_side"])
        ksp_solver.setOperators(mat_A)
        ksp_solver.setTolerances(
            rtol=solver_parameters["relative_tolerance"],
            atol=solver_parameters["absolute_tolerance"],
            divtol=solver_parameters.get("divergence_limit", None),
            max_it=solver_parameters.get("maximum_iterations", 1000))
        ksp_solver.setInitialGuessNonzero(
            solver_parameters.get("nonzero_initial_guess", True))
        ksp_solver.setNormType(
            solver_parameters.get(
                "norm_type", PETSc.KSP.NormType.DEFAULT))
        if "gmres_restart" in solver_parameters:
            ksp_solver.setGMRESRestart(solver_parameters["gmres_restart"])

        logger = logging.getLogger("tlm_adjoint.System")

        def monitor(ksp_solver, it, r_norm):
            logger.debug(f"KSP: "
                         f"iteration {it:d}, "
                         f"residual norm {r_norm:.16e}")

        ksp_solver.setMonitor(monitor)

        if correct_initial_guess:
            self._nullspace.correct_soln(u)
        self._nullspace.correct_rhs(b_c)

        u_petsc = mat_A.createVecRight()
        self._arg_space.to_petsc(u_petsc, u)
        b_petsc = mat_A.createVecLeft()
        self._action_space.to_petsc(b_petsc, b_c)
        del b_c

        if pre_callback is not None:
            pre_callback(ksp_solver)
        ksp_solver.setUp()
        ksp_solver.solve(b_petsc, u_petsc)
        if post_callback is not None:
            post_callback(ksp_solver)
        del b_petsc

        self._arg_space.from_petsc(u_petsc, u)
        del u_petsc

        if correct_solution:
            # Not needed if the linear problem were to be solved exactly
            self._nullspace.correct_soln(u)

        if ksp_solver.getConvergedReason() <= 0:
            raise RuntimeError("Convergence failure")
        ksp_its = ksp_solver.getIterationNumber()

        ksp_solver.destroy()
        mat_A.destroy()
        if pc_fn is not None:
            pc.destroy()

        return ksp_its
