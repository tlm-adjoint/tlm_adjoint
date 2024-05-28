from .interface import (
    DEFAULT_COMM, Packed, comm_dup_cached, garbage_cleanup, var_assign,
    var_copy, var_id, var_is_replacement, var_is_scalar, var_name)

from .adjoint import AdjointCache, AdjointModelRHS, TransposeComputationalGraph
from .alias import gc_disabled
from .checkpoint_schedules import (
    Clear, Configure, Forward, Reverse, Read, Write, EndForward, EndReverse)
from .checkpoint_schedules import (
    MemoryCheckpointSchedule, MultistageCheckpointSchedule,
    NoneCheckpointSchedule, PeriodicDiskCheckpointSchedule)
from .checkpointing import (
    CheckpointStorage, HDF5Checkpoints, PickleCheckpoints, ReplayStorage)
from .equation import Equation, ZeroAssignment
from .markers import ControlsMarker, FunctionalMarker
from .manager import restore_manager, set_manager
from .tangent_linear import TangentLinear, TangentLinearMap, tlm_key, tlm_keys

from collections import deque
import contextlib
import enum
import functools
import logging
try:
    import MPI
except ModuleNotFoundError:
    MPI = None
import os
import warnings
import weakref

__all__ = \
    [
        "EquationManager"
    ]


class AnnotationState(enum.Enum):
    STOPPED = "stopped"
    ANNOTATING = "annotating"
    FINAL = "final"


class TangentLinearState(enum.Enum):
    STOPPED = "stopped"
    DERIVING = "deriving"
    FINAL = "final"


class EquationManager:
    """Core manager class.

        - Plays the role of an adjoint 'tape'. Records forward equations as
          they are solved.
        - Interacts with checkpointing schedules for adjoint checkpointing and
          forward replay.
        - Derives and manages tangent-linear equations. Tangent-linear
          equations are processed as new forward equations, allowing higher
          order adjoint calculations.
        - Handles variable reference dropping, e.g. handles the dropping of
          references to variables which store values, and their replacement
          with symbolic equivalents, after :class:`.Equation` objects holding
          those references have been destroyed. Internally the manager retains
          a reference to a :class:`.WeakAlias` subclass so that the
          :class:`.Equation` methods may be called after the original
          :class:`.Equation` is destroyed.

    The manager processes forward equations (and tangent-linear equations) as
    they are solved. Equations are collected into 'blocks' of equations,
    corresponding to 'steps' in step-based checkpointing schedules. For
    checkpointing schedule configuration details see
    :meth:`.EquationManager.configure_checkpointing`.

    The configuration of tangent-linears is defined by a tangent-linear tree.
    The root node of this tree corresponds to the forward. Following :math:`n`
    edges from the root node leads to a node associated with an :math:`n` th
    order tangent-linear. For tangent-linear configuration details see
    :meth:`.EquationManager.configure_tlm`.

    On instantiation both equation annotation and tangent-linear derivation and
    solution are *disabled*.

    :arg comm: The communicator associated with the manager.
    :arg cp_method: See :meth:`.EquationManager.configure_checkpointing`.
    :arg cp_parameters: See :meth:`.EquationManager.configure_checkpointing`.
    """

    _id_counter = [0]

    def __init__(self, *, comm=None, cp_method="memory", cp_parameters=None):
        if comm is None:
            comm = DEFAULT_COMM
        comm = comm_dup_cached(comm)
        if cp_parameters is None:
            cp_parameters = {}

        self._comm = comm
        self._to_drop_references = []
        self._finalizes = {}

        @gc_disabled
        def finalize_callback(to_drop_references, finalizes):
            while len(to_drop_references) > 0:
                referrer = to_drop_references.pop()
                referrer._drop_references()
            for finalize in finalizes.values():
                finalize.detach()
            finalizes.clear()
        finalize = weakref.finalize(self, finalize_callback,
                                    self._to_drop_references, self._finalizes)
        finalize.atexit = False

        if MPI is not None:
            self._id_counter[0] = self._comm.allreduce(
                self._id_counter[0], op=MPI.MAX)
        self._id, = self._id_counter
        self._id_counter[0] += 1

        self.reset(cp_method=cp_method, cp_parameters=cp_parameters)

    @property
    def comm(self):
        """The communicator associated with the manager.
        """

        return self._comm

    def info(self, *, info=print):
        """Print manager state information.

        :arg info: A callable which accepts and prints a :class:`str`.
        """

        info("Equation manager status:")
        info(f"Annotation state: {self._annotation_state:s}")
        info(f"Tangent-linear state: {self._tlm_state:s}")
        info("Equations:")
        blocks = list(self._blocks)
        if len(self._block) > 0:
            blocks.append(self._block)
        for n, block in enumerate(blocks):
            info(f"  Block {n:d}")
            for i, eq in enumerate(block):
                eq_X = eq.X()
                if len(eq_X) == 1:
                    X_name = var_name(eq_X[0])
                    X_ids = f"id {var_id(eq_X[0]):d}"
                else:
                    X_name = "(%s)" % (",".join(map(var_name, eq_X)))
                    X_ids = "ids (%s)" % (",".join(f"{var_id(eq_x):d}"
                                                   for eq_x in eq_X))
                info("    Equation %i, %s solving for %s (%s)" %
                     (i, type(eq).__name__, X_name, X_ids))
                nl_dep_ids = set(map(var_id,
                                     eq.nonlinear_dependencies()))
                for j, dep in enumerate(eq.dependencies()):
                    info("      Dependency %i, %s (id %i)%s, %s" %
                         (j, var_name(dep), var_id(dep),
                          ", replaced" if var_is_replacement(dep) else "",
                          "non-linear" if var_id(dep) in nl_dep_ids else "linear"))  # noqa: E501
        info("Storage:")
        info(f'  Storing initial conditions: {"yes" if self._cp.store_ics else "no":s}')  # noqa: E501
        info(f'  Storing equation non-linear dependencies: {"yes" if self._cp.store_data else "no":s}')  # noqa: E501
        info(f"  Initial conditions stored: {len(self._cp._cp):d}")
        info(f"  Initial conditions referenced: {len(self._cp._refs):d}")
        info("Checkpointing:")
        if callable(self._cp_method):
            info("  Method: custom")
        else:
            info(f"  Method: {self._cp_method:s}")

    def new(self, cp_method=None, cp_parameters=None):
        """Construct a new :class:`.EquationManager` sharing the communicator
        with this :class:`.EquationManager`. By default the new
        :class:`.EquationManager` also shares the checkpointing schedule
        configuration with this :class:`.EquationManager`, but this may be
        overridden with the arguments `cp_method` and `cp_parameters`.

        Both equation annotation and tangent-linear derivation and solution are
        *disabled* for the new :class:`.EquationManager`.

        :arg cp_method: See :meth:`.EquationManager.configure_checkpointing`.
        :arg cp_parameters: See
            :meth:`.EquationManager.configure_checkpointing`.
        """

        if cp_method is None:
            if cp_parameters is not None:
                raise TypeError("cp_parameters can only be supplied if "
                                "cp_method is supplied")
            cp_method = self._cp_method
            cp_parameters = self._cp_parameters
        elif cp_parameters is None:
            raise TypeError("cp_parameters must be supplied if cp_method is "
                            "supplied")

        return EquationManager(comm=self._comm, cp_method=cp_method,
                               cp_parameters=cp_parameters)

    @gc_disabled
    def reset(self, cp_method=None, cp_parameters=None):
        """Reset the :class:`.EquationManager`. Clears all recorded equations,
        and all configured tangent-linears.

        By default the :class:`.EquationManager` *retains* its previous
        checkpointing schedule configuration, but this may be overridden with
        the arguments `cp_method` and `cp_parameters`.

        Both equation annotation and tangent-linear derivation and solution are
        *disabled* after calling this method.

        :arg cp_method: See :meth:`.EquationManager.configure_checkpointing`.
        :arg cp_parameters: See
            :meth:`.EquationManager.configure_checkpointing`.
        """

        if cp_method is None:
            if cp_parameters is not None:
                raise TypeError("cp_parameters can only be supplied if "
                                "cp_method is supplied")
            cp_method = self._cp_method
            cp_parameters = self._cp_parameters
        elif cp_parameters is None:
            raise TypeError("cp_parameters must be supplied if cp_method is "
                            "supplied")

        self.drop_references()
        garbage_cleanup(self._comm)

        self._annotation_state = AnnotationState.STOPPED
        self._tlm_state = TangentLinearState.STOPPED
        self._blocks = []
        self._block = []

        self._tlm = TangentLinear()
        self._tlm_map = {}
        self._tlm_eqs = {}

        self._adj_cache = AdjointCache()

        self.configure_checkpointing(cp_method, cp_parameters=cp_parameters)

    def configure_checkpointing(self, cp_method, cp_parameters):
        """Provide a new checkpointing schedule configuration.

        The checkpointing schedule type is defined by the argument `cp_method`,
        and detailed configuration options are provided by the :class:`Mapping`
        argument `cp_parameters`.

        `cp_method` values:

            - `'none'`: No checkpointing. Can be used for tangent-linear only
              calculations. Options defined by `cp_parameters`:

                - `'drop_references'`: Whether to automatically drop references
                  to variables which store values. :class:`bool`, optional,
                  default `False`.

            - `'memory'`: Store all forward restart data and non-linear
              dependency data in memory. Options defined by `cp_parameters`:

                - `'drop_references'`: Whether to automatically drop references
                  to variables which store values. :class:`bool`, optional,
                  default `False`.

            - `'periodic_disk`: Periodically store forward restart data on
              disk. Options defined by `cp_parameters`:

                - `'path'`: Directory in which disk checkpoint data should be
                  stored. :class:`str`, optional, default `'checkpoints~'`.
                - `'format'`: Disk storage format. Either `'pickle'`, for
                  data storage using the pickle module, or `'hdf5'`, for data
                  storage using the h5py library.
                - `'period'`: Interval, in blocks, between storage of forward
                  restart data. :class:`int`, required.

            - `'multistage'`: Forward restart checkpointing with checkpoint
              distribution as described in:

                  - Andreas Griewank and Andrea Walther, 'Algorithm 799:
                    revolve: an implementation of checkpointing for the reverse
                    or adjoint mode of computational differentiation', ACM
                    Transactions on Mathematical Software, 26(1), pp. 19--45,
                    2000, doi: 10.1145/347837.347846

              The memory/disk storage distribution is determined by an
              initial run of the checkpointing schedule, leading to a
              distribution equivalent to that in:

                  - Philipp Stumm and Andrea Walther, 'MultiStage approaches
                    for optimal offline checkpointing', SIAM Journal on
                    Scientific Computing, 31(3), pp. 1946--1967, 2009, doi:
                    10.1137/080718036

              Options defined by `cp_parameters`:

                - `'path'`: Directory in which disk checkpoint data should be
                  stored. :class:`str`, optional, default `'checkpoints~'`.
                - `'format'`: Disk storage format. Either `'pickle'`, for
                  data storage using the pickle module, or `'hdf5'`, for data
                  storage using the h5py library.
                - `'blocks'`: The total number of blocks. :class:`int`,
                  required.
                - `'snaps_in_ram'`: Maximum number of memory checkpoints.
                  :class:`int`, optional, default 0.
                - `'snaps_on_disk'`: Maximum number of disk checkpoints.
                  :class:`int`, optional, default 0.

              The name 'multistage' originates from the corresponding
              `strategy` argument value for the
              `dolfin_adjoint.solving.adj_checkpointing` function in
              dolfin-adjoint (see e.g. version 2017.1.0). The parameter names
              `snaps_in_ram` and `snaps_on_disk` originate from the
              corresponding arguments for the
              `dolfin_adjoint.solving.adj_checkpointing` function in
              dolfin-adjoint (see e.g. version 2017.1.0).

            - A callable: A callable returning a :class:`.CheckpointSchedule`.
              Options defined by `cp_parameters`:

                - `'path'`: Directory in which disk checkpoint data should be
                  stored. :class:`str`, optional, default `'checkpoints~'`.
                - `'format'`: Disk storage format. Either `'pickle'`, for
                  data storage using the pickle module, or `'hdf5'`, for data
                  storage using the h5py library.
                - Other parameters are passed as keyword arguments to the
                  callable.
        """

        if len(self._block) != 0 or len(self._blocks) != 0:
            raise RuntimeError("Cannot configure checkpointing after "
                               "equations have been recorded")

        cp_parameters = dict(cp_parameters)

        if not callable(cp_method) and cp_method in {"none", "memory"}:
            alias_eqs = cp_parameters.get("drop_references", False)
        else:
            alias_eqs = True

        if callable(cp_method):
            cp_schedule_kwargs = dict(cp_parameters)
            cp_schedule_kwargs.pop("path", None)
            cp_schedule_kwargs.pop("format", None)
            cp_schedule = cp_method(**cp_schedule_kwargs)
        elif cp_method == "none":
            cp_schedule = NoneCheckpointSchedule()
        elif cp_method == "memory":
            cp_schedule = MemoryCheckpointSchedule()
        elif cp_method == "periodic_disk":
            cp_schedule = PeriodicDiskCheckpointSchedule(
                cp_parameters["period"])
        elif cp_method == "multistage":
            cp_schedule = MultistageCheckpointSchedule(
                cp_parameters["blocks"],
                cp_parameters.get("snaps_in_ram", 0),
                cp_parameters.get("snaps_on_disk", 0),
                trajectory="maximum")
        else:
            raise ValueError(f"Unrecognized checkpointing method: "
                             f"{cp_method:s}")

        if cp_schedule.uses_disk_storage:
            cp_path = cp_parameters.get("path", "checkpoints~")
            cp_format = cp_parameters.get("format", "hdf5")

            self._comm.barrier()
            if self._comm.rank == 0:
                if not os.path.exists(cp_path):
                    os.makedirs(cp_path)
            self._comm.barrier()

            if cp_format == "pickle":
                cp_disk = PickleCheckpoints(
                    os.path.join(cp_path, f"checkpoint_{self._id:d}_"),
                    comm=self._comm)
            elif cp_format == "hdf5":
                cp_disk = HDF5Checkpoints(
                    os.path.join(cp_path, f"checkpoint_{self._id:d}_"),
                    comm=self._comm)
            else:
                raise ValueError(f"Unrecognized checkpointing format: "
                                 f"{cp_format:s}")
        else:
            cp_path = None
            cp_disk = None

        self._cp_method = cp_method
        self._cp_parameters = cp_parameters
        self._alias_eqs = alias_eqs
        self._cp_schedule = cp_schedule
        self._cp_memory = {}
        self._cp_path = cp_path
        self._cp_disk = cp_disk

        self._cp = CheckpointStorage(store_ics=False,
                                     store_data=False)
        assert len(self._blocks) == 0 and len(self._block) == 0
        self._checkpoint()

    def configure_tlm(self, *args, annotate=None, tlm=True):
        """Configure the tangent-linear tree.

        :arg args: A :class:`tuple` of `(M_i, dM_i)` pairs. `M_i` is a variable
            or a :class:`Sequence` of variables defining a control. `dM_i` is a
            variable or a :class:`Sequence` of variables defining a derivative
            direction. Identifies a node in the tree (and hence identifies a
            tangent-linear) corresponding to differentation, in order, with
            respect to the each control defined by `M_i` and with each
            direction defined by `dM_i`.
        :arg annotate: If `True` then enable annotation for the identified
            tangent-linear, and enable annotation for all tangent-linears on
            which it depends. If `False` then disable annotation for the
            identified tangent-linear, all tangent-linears which depend on it,
            and any newly added tangent-linears. Defaults to `tlm`.
        :arg tlm: If `True` then add (or retain) the identified tangent-linear,
            and add all tangent-linears on which it depends. If `False` then
            remove the identified tangent-linear, and remove all
            tangent-linears which depend on it.
        """

        if self._tlm_state == TangentLinearState.FINAL:
            raise RuntimeError("Cannot configure tangent-linears after "
                               "finalization")

        if annotate is None:
            annotate = tlm
        if annotate and not tlm:
            raise ValueError("Invalid annotate/tlm combination")

        if tlm:
            # Could be optimized to avoid encountering parent nodes multiple
            # times
            for M_dM_keys in tlm_keys(*args):
                node = self._tlm
                for (M, dM), key in M_dM_keys:
                    if (M, dM) in node:
                        if annotate:
                            node[(M, dM)].set_is_annotated(True)
                    else:
                        node.add(M, dM, annotate=annotate)
                        if key not in self._tlm_map:
                            self._tlm_map[key] = TangentLinearMap(M, dM)
                    node = node[(M, dM)]

        if not annotate or not tlm:
            def depends(keys_a, keys_b):
                j = 0
                for i, key_a in enumerate(keys_a):
                    if j >= len(keys_b):
                        return True
                    elif key_a == keys_b[j]:
                        j += 1
                return j >= len(keys_b)

            keys = tuple(key
                         for _, key in map(lambda arg: tlm_key(*arg), args))
            remaining_nodes = [(self._tlm,
                                (tlm_key(*child_M_dM)[1],),
                                child_M_dM,
                                child)
                               for child_M_dM, child in self._tlm.items()]
            while len(remaining_nodes) > 0:
                parent, node_keys, node_M_dM, node = remaining_nodes.pop()
                if depends(node_keys, keys):
                    if not tlm:
                        parent.remove(*node_M_dM)
                    elif not annotate:
                        node.set_is_annotated(False)
                if node_M_dM in parent:
                    remaining_nodes.extend(
                        (node,
                         tuple(list(node_keys) + [tlm_key(*child_M_dM)[1]]),
                         child_M_dM,
                         child)
                        for child_M_dM, child in node.items())

    def add_tlm(self, M, dM, max_depth=1, *, _warning=True):
        if _warning:
            warnings.warn("EquationManager.add_tlm method is deprecated -- "
                          "use EquationManager.configure_tlm instead",
                          DeprecationWarning, stacklevel=2)

        if self._tlm_state == TangentLinearState.FINAL:
            raise RuntimeError("Cannot configure tangent-linear after "
                               "finalization")

        (M, dM), key = tlm_key(M, dM)

        for depth in range(max_depth):
            remaining_nodes = [self._tlm]
            while len(remaining_nodes) > 0:
                node = remaining_nodes.pop()
                remaining_nodes.extend(node.values())
                node.add(M, dM, annotate=True)

        if key not in self._tlm_map:
            self._tlm_map[key] = TangentLinearMap(M, dM)

    def tlm_enabled(self):
        """
        :returns: Whether derivation and solution of tangent-linear equations
            is enabled.
        """

        return self._tlm_state == TangentLinearState.DERIVING

    def var_tlm(self, x, *args):
        """Return a tangent-linear variable.

        :arg x: A variable whose tangent-linear variable should be returned.
            Cannot not be a replacement.
        :arg args: Identifies the tangent-linear. See
            :meth:`.EquationManager.configure_tlm`.
        :returns: The tangent-linear variable.
        """

        if var_is_replacement(x):
            raise ValueError("x cannot be a replacement")

        tau = x
        for _, key in map(lambda arg: tlm_key(*arg), args):
            tau = self._tlm_map[key][tau]
        return tau

    def function_tlm(self, x, *args):
        """
        """
        warnings.warn("function_tlm method is deprecated -- "
                      "use var_tlm instead",
                      DeprecationWarning, stacklevel=2)
        return self.var_tlm(x, *args)

    def annotation_enabled(self):
        """
        :returns: Whether recording of equations is enabled.
        """

        return self._annotation_state == AnnotationState.ANNOTATING

    def start(self, *, annotate=True, tlm=True):
        """Start recording of equations and derivation and solution of
        tangent-linear equations.

        :arg annotate: Whether recording of equations should be enabled.
        :arg tlm: Whether derivation and solution of tangent-linear equations
            should be enabled.
        """

        if annotate:
            self._annotation_state \
                = {AnnotationState.STOPPED: AnnotationState.ANNOTATING,
                   AnnotationState.ANNOTATING: AnnotationState.ANNOTATING}[self._annotation_state]  # noqa: E501

        if tlm:
            self._tlm_state \
                = {TangentLinearState.STOPPED: TangentLinearState.DERIVING,
                   TangentLinearState.DERIVING: TangentLinearState.DERIVING}[self._tlm_state]  # noqa: E501

    def stop(self, *, annotate=True, tlm=True):
        """Stop recording of equations and derivation and solution of
        tangent-linear equations.

        :arg annotate: Whether recording of equations should be disabled.
        :arg tlm: Whether derivation and solution of tangent-linear equations
            should be disabled.
        :returns: A :class:`tuple` `(annotation_state, tlm_state)`.
            `annotation_state` is a :class:`bool` indicating whether recording
            of equations was enabled prior to the call to
            :meth:`.EquationManager.stop`. `tlm_state` is a :class:`bool`
            indicating whether derivation and solution of tangent-linear
            equations was enabled prior to the call to
            :meth:`.EquationManager.stop`.
        """

        state = (self.annotation_enabled(), self.tlm_enabled())

        if annotate:
            self._annotation_state \
                = {AnnotationState.STOPPED: AnnotationState.STOPPED,
                   AnnotationState.ANNOTATING: AnnotationState.STOPPED,
                   AnnotationState.FINAL: AnnotationState.FINAL}[self._annotation_state]  # noqa: E501

        if tlm:
            self._tlm_state \
                = {TangentLinearState.STOPPED: TangentLinearState.STOPPED,
                   TangentLinearState.DERIVING: TangentLinearState.STOPPED,
                   TangentLinearState.FINAL: TangentLinearState.FINAL}[self._tlm_state]  # noqa: E501

        return state

    @contextlib.contextmanager
    def paused(self, *, annotate=True, tlm=True):
        """Construct a context manager which can be used to temporarily disable
        recording of equations and derivation and solution of tangent-linear
        equations.

        :arg annotate: Whether recording of equations should be temporarily
            disabled.
        :arg tlm: Whether derivation and solution of tangent-linear equations
            should be temporarily disabled.
        :returns: A context manager which can be used to temporarily disable
            recording of equations and derivation and solution of
            tangent-linear equations.
        """

        annotate, tlm = self.stop(annotate=annotate, tlm=tlm)
        try:
            yield
        finally:
            self.start(annotate=annotate, tlm=tlm)

    def add_initial_condition(self, x):
        """Process an 'initial condition' -- a variable whose associated value
        is needed prior to solving an equation.

        :arg x: A variable defining the initial condition.
        """

        annotate = self.annotation_enabled()

        if annotate:
            if self._annotation_state == AnnotationState.FINAL:
                raise RuntimeError("Cannot add initial conditions after "
                                   "finalization")

            self._cp.add_initial_condition(x)

    @restore_manager
    def add_equation(self, eq):
        """Process an :class:`.Equation` after it has been solved.

        :arg eq: The :class:`.Equation`.
        """

        set_manager(self)
        self.drop_references()

        annotate = self.annotation_enabled()
        tlm = self.tlm_enabled()

        if annotate:
            if self._annotation_state == AnnotationState.FINAL:
                raise RuntimeError("Cannot add equations after finalization")

            if self._alias_eqs:
                self._add_equation_finalizes(eq)
                self._block.append(eq._weak_alias)
            else:
                self._block.append(eq)
            self._cp.add_equation(
                len(self._blocks), len(self._block) - 1, eq)

        if tlm:
            if self._tlm_state == TangentLinearState.FINAL:
                raise RuntimeError("Cannot add tangent-linear equations after "
                                   "finalization")

            remaining_eqs = deque((eq, child_M_dM, child)
                                  for child_M_dM, child in self._tlm.items())
            while len(remaining_eqs) > 0:
                parent_eq, (node_M, node_dM), node = remaining_eqs.popleft()

                node_eq = self._tangent_linear(parent_eq, node_M, node_dM)
                if node_eq is not None:
                    node_eq.solve(
                        annotate=annotate and node.is_annotated(), tlm=False)
                    remaining_eqs.extend(
                        (node_eq, child_M_dM, child)
                        for child_M_dM, child in node.items())

    def _tangent_linear(self, eq, M, dM):
        (M, dM), key = tlm_key(M, dM)

        X = eq.X()
        X_ids = set(map(var_id, X))
        if not X_ids.isdisjoint(set(key[0])):
            raise ValueError("Invalid tangent-linear parameter")
        if not X_ids.isdisjoint(set(key[1])):
            raise ValueError("Invalid tangent-linear direction")

        eq_id = eq.id
        eq_tlm_eqs = self._tlm_eqs.setdefault(eq_id, {})

        tlm_map = self._tlm_map[key]
        tlm_eq = eq_tlm_eqs.get(key, None)
        if tlm_eq is None:
            for dep in eq.dependencies():
                if dep in tlm_map:
                    tlm_eq = eq.tangent_linear(tlm_map)
                    if tlm_eq is None:
                        warnings.warn("Equation.tangent_linear should return "
                                      "an Equation",
                                      DeprecationWarning)
                        tlm_eq = ZeroAssignment([tlm_map[x] for x in X])
                    tlm_eq._tlm_adjoint__tlm_root_id = getattr(
                        eq, "_tlm_adjoint__tlm_root_id", eq_id)
                    tlm_eq._tlm_adjoint__tlm_key = tuple(
                        list(getattr(eq, "_tlm_adjoint__tlm_key", ()))
                        + [key])

                    eq_tlm_eqs[key] = tlm_eq
                    break

        return tlm_eq

    @gc_disabled
    def _add_equation_finalizes(self, eq):
        for referrer in eq.referrers():
            referrer_id = referrer.id
            if referrer_id not in self._finalizes:
                @gc_disabled
                def finalize_callback(self_ref, referrer_alias, referrer_id):
                    self = self_ref()
                    if self is not None:
                        self._to_drop_references.append(referrer_alias)
                        del self._finalizes[referrer_id]
                finalize = weakref.finalize(
                    referrer, finalize_callback,
                    weakref.ref(self), referrer._weak_alias, referrer_id)
                finalize.atexit = False
                self._finalizes[referrer_id] = finalize

    @gc_disabled
    def drop_references(self):
        """Drop references to variables which store values, referenced by
        objects which have been destroyed, and replace them symbolic
        equivalents.
        """

        while len(self._to_drop_references) > 0:
            referrer = self._to_drop_references.pop()
            referrer._drop_references()
            if isinstance(referrer, Equation):
                referrer_id = referrer.id
                if referrer_id in self._tlm_eqs:
                    del self._tlm_eqs[referrer_id]

    def _write_memory_checkpoint(self, n, *, ics=True, data=True):
        if n in self._cp_memory or \
                (self._cp_disk is not None and n in self._cp_disk):
            raise RuntimeError("Duplicate checkpoint")

        self._cp_memory[n] = self._cp.checkpoint_data(
            ics=ics, data=data, copy=True)

    def _read_memory_checkpoint(self, n, *, ic_ids=None, ics=True, data=True,
                                delete=False):
        read_cp, read_data, read_storage = self._cp_memory[n]
        if delete:
            del self._cp_memory[n]

        if ics or data:
            if ics:
                read_cp = tuple(key for key in read_cp
                                if ic_ids is None or key[0] in ic_ids)
            else:
                read_cp = ()
            if not data:
                read_data = {}

            keys = set(read_cp)
            for eq_data in read_data.values():
                keys.update(eq_data)
            read_storage = {key: read_storage[key] for key in read_storage
                            if key in keys}

            self._cp.update(read_cp, read_data, read_storage,
                            copy=not delete)

    def _write_disk_checkpoint(self, n, *, ics=True, data=True):
        if n in self._cp_memory or n in self._cp_disk:
            raise RuntimeError("Duplicate checkpoint")

        self._cp_disk.write(
            n, *self._cp.checkpoint_data(ics=ics, data=data, copy=False))

    def _read_disk_checkpoint(self, n, *, ic_ids=None, ics=True, data=True,
                              delete=False):
        if ics or data:
            read_cp, read_data, read_storage = \
                self._cp_disk.read(n, ics=ics, data=data, ic_ids=ic_ids)

            self._cp.update(read_cp, read_data, read_storage,
                            copy=False)

        if delete:
            self._cp_disk.delete(n)

    def _checkpoint(self, *, final=False):
        assert len(self._block) == 0
        n = len(self._blocks)
        if final:
            self._cp_schedule.finalize(n)
        if n < self._cp_schedule.n:
            return

        logger = logging.getLogger("tlm_adjoint.checkpointing")

        @functools.singledispatch
        def action(cp_action):
            raise TypeError(f"Unexpected checkpointing action: {cp_action}")

        @action.register(Clear)
        def action_clear(cp_action):
            self._cp.clear(clear_ics=cp_action.clear_ics,
                           clear_data=cp_action.clear_data)

        @action.register(Configure)
        def action_configure(cp_action):
            self._cp.configure(store_ics=cp_action.store_ics,
                               store_data=cp_action.store_data)

        @action.register(Forward)
        def action_forward(cp_action):
            logger.debug(f"forward: forward advance to {cp_action.n1:d}")
            if cp_action.n0 != n:
                raise RuntimeError("Invalid checkpointing state")
            if cp_action.n1 <= n:
                raise RuntimeError("Invalid checkpointing state")

        @action.register(Write)
        def action_write(cp_action):
            if cp_action.n >= n:
                raise RuntimeError("Invalid checkpointing state")
            if cp_action.storage == "disk":
                logger.debug(f"forward: save checkpoint data at "
                             f"{cp_action.n:d} on disk")
                self._write_disk_checkpoint(cp_action.n)
            elif cp_action.storage == "RAM":
                logger.debug(f"forward: save checkpoint data at "
                             f"{cp_action.n:d} in RAM")
                self._write_memory_checkpoint(cp_action.n)
            else:
                raise ValueError(f"Unrecognized checkpointing storage: "
                                 f"{cp_action.storage:s}")

        @action.register(EndForward)
        def action_end_forward(cp_action):
            if self._cp_schedule.max_n is None \
                    or n != self._cp_schedule.max_n:
                raise RuntimeError("Invalid checkpointing state")

        while True:
            cp_action = next(self._cp_schedule)
            action(cp_action)
            if isinstance(cp_action, (Forward, EndForward)):
                break

        for cls in action.registry:
            @action.register(cls)
            def action_pass(cp_action):
                pass
        del action

    def _restore_checkpoint(self, n, *, transpose_deps=None):
        if self._cp_schedule.max_n is None:
            raise RuntimeError("Invalid checkpointing state")
        if n > self._cp_schedule.max_n - self._cp_schedule.r - 1:
            return
        elif n != self._cp_schedule.max_n - self._cp_schedule.r - 1:
            raise RuntimeError("Invalid checkpointing state")

        logger = logging.getLogger("tlm_adjoint.checkpointing")

        storage = None
        initialize_storage_cp = False
        cp_n = None

        @functools.singledispatch
        def action(cp_action):
            raise TypeError(f"Unexpected checkpointing action: {cp_action}")

        @action.register(Clear)
        def action_clear(cp_action):
            nonlocal initialize_storage_cp

            if initialize_storage_cp:
                storage.update(self._cp.initial_conditions(cp=True,
                                                           refs=False,
                                                           copy=False),
                               copy=not cp_action.clear_ics or not cp_action.clear_data)  # noqa: E501
                initialize_storage_cp = False
            self._cp.clear(clear_ics=cp_action.clear_ics,
                           clear_data=cp_action.clear_data)

        @action.register(Configure)
        def action_configure(cp_action):
            self._cp.configure(store_ics=cp_action.store_ics,
                               store_data=cp_action.store_data)

        @action.register(Forward)
        def action_forward(cp_action):
            nonlocal initialize_storage_cp, cp_n

            if storage is None or cp_n is None:
                raise RuntimeError("Invalid checkpointing state")
            if initialize_storage_cp:
                storage.update(self._cp.initial_conditions(cp=True,
                                                           refs=False,
                                                           copy=False),
                               copy=True)
                initialize_storage_cp = False

            logger.debug(f"reverse: forward advance to {cp_action.n1:d}")
            if cp_action.n0 != cp_n:
                raise RuntimeError("Invalid checkpointing state")
            if cp_action.n1 > n + 1:
                raise RuntimeError("Invalid checkpointing state")

            for n1 in cp_action:
                for i, eq in enumerate(self._blocks[n1]):
                    if storage.is_active(n1, i):
                        X = tuple(storage[eq_x] for eq_x in eq.X())
                        deps = tuple(storage[eq_dep]
                                     for eq_dep in eq.dependencies())

                        for eq_dep in eq.initial_condition_dependencies():
                            self._cp.add_initial_condition(
                                eq_dep, value=storage[eq_dep])
                        eq.forward(X, deps=deps)
                        self._cp.add_equation(n1, i, eq, deps=deps)
                    elif transpose_deps.any_is_active(n1, i):
                        nl_deps = tuple(storage[eq_dep]
                                        for eq_dep in eq.nonlinear_dependencies())  # noqa: E501

                        self._cp.add_equation_data(
                            n1, i, eq, nl_deps=nl_deps)
                    else:
                        self._cp.update_keys(
                            n1, i, eq)

                    storage_state = storage.popleft()
                    assert storage_state == (n1, i)
                garbage_cleanup(self._comm)
            cp_n = cp_action.n1
            assert cp_n <= n + 1
            assert cp_n < n + 1 or len(storage) == 0

        @action.register(Reverse)
        def action_reverse(cp_action):
            logger.debug(f"reverse: adjoint step back to {cp_action.n0:d}")
            if cp_action.n1 != n + 1:
                raise RuntimeError("Invalid checkpointing state")
            if cp_action.n0 > n:
                raise RuntimeError("Invalid checkpointing state")

        @action.register(Read)
        def action_read(cp_action):
            nonlocal storage, initialize_storage_cp, cp_n

            if storage is not None or cp_n is not None:
                raise RuntimeError("Invalid checkpointing state")

            cp_n = cp_action.n
            logger.debug(f'reverse: load checkpoint data at {cp_n:d} from '
                         f'{cp_action.storage:s} and '
                         f'{"delete" if cp_action.delete else "keep":s}')

            storage = ReplayStorage(self._blocks, cp_n, n + 1,
                                    transpose_deps=transpose_deps)
            garbage_cleanup(self._comm)
            initialize_storage_cp = True
            storage.update(self._cp.initial_conditions(cp=False,
                                                       refs=True,
                                                       copy=False),
                           copy=False)

            if cp_action.storage == "disk":
                self._read_disk_checkpoint(cp_n, ic_ids=set(storage),
                                           delete=cp_action.delete)
            elif cp_action.storage == "RAM":
                self._read_memory_checkpoint(cp_n, ic_ids=set(storage),
                                             delete=cp_action.delete)
            else:
                raise ValueError(f"Unrecognized checkpointing storage: "
                                 f"{cp_action.storage:s}")

        @action.register(Write)
        def action_write(cp_action):
            if cp_action.n >= n:
                raise RuntimeError("Invalid checkpointing state")
            if cp_action.storage == "disk":
                logger.debug(f"reverse: save checkpoint data at "
                             f"{cp_action.n:d} on disk")
                self._write_disk_checkpoint(cp_action.n)
            elif cp_action.storage == "RAM":
                logger.debug(f"reverse: save checkpoint data at "
                             f"{cp_action.n:d} in RAM")
                self._write_memory_checkpoint(cp_action.n)
            else:
                raise ValueError(f"Unrecognized checkpointing storage: "
                                 f"{cp_action.storage:s}")

        while True:
            cp_action = next(self._cp_schedule)
            action(cp_action)
            if isinstance(cp_action, Reverse):
                break

        for cls in action.registry:
            @action.register(cls)
            def action_pass(cp_action):
                pass
        del action

    def new_block(self):
        """End the current block of equations, and begin a new block. The
        blocks of equations correspond to the 'steps' in step-based
        checkpointing schedules.
        """

        self.drop_references()
        garbage_cleanup(self._comm)

        if self._annotation_state in {AnnotationState.STOPPED,
                                      AnnotationState.FINAL}:
            return

        if self._cp_schedule.max_n is not None \
                and len(self._blocks) == self._cp_schedule.max_n - 1:
            # Wait for the finalize
            warnings.warn(
                "Attempting to end the final block without finalizing -- "
                "ignored", RuntimeWarning, stacklevel=2)
            return

        self._blocks.append(tuple(self._block))
        self._block = []
        self._checkpoint(final=False)

    def finalize(self):
        """End the final block of equations. Equations cannot be recorded, and
        new tangent-linear equations cannot be derived and solved, after a call
        to this method.

        Called by :meth:`.EquationManager.compute_gradient`, and typically need
        not be called manually.
        """

        self.drop_references()
        garbage_cleanup(self._comm)

        if self._annotation_state == AnnotationState.FINAL:
            return

        self._annotation_state = AnnotationState.FINAL
        self._tlm_state = TangentLinearState.FINAL

        self._blocks.append(tuple(self._block))
        self._blocks = tuple(self._blocks)
        self._block = ()
        if self._cp_schedule.max_n is not None \
                and len(self._blocks) < self._cp_schedule.max_n:
            warnings.warn(
                "Insufficient number of blocks -- empty block(s) added",
                RuntimeWarning, stacklevel=2)
            while len(self._blocks) < self._cp_schedule.max_n:
                self._checkpoint(final=False)
                self._blocks.append([])
        self._checkpoint(final=True)

    @restore_manager
    def compute_gradient(self, Js, M, *, callback=None, prune_forward=True,
                         prune_adjoint=True, prune_replay=True,
                         cache_adjoint_degree=None, store_adjoint=False,
                         adj_ics=None):
        """Core adjoint driver method.

        Compute the derivative of one or more functionals with respect to one
        or more controls, using an adjoint approach.

        :arg Js: A variable or a sequence of variables defining the functionals
            to differentiate.
        :arg M: A variable or a :class:`Sequence` of variables defining the
            controls. Derivatives with respect to the controls are computed.
        :arg callback: Diagnostic callback. A callable of the form

            .. code-block:: python

                def callback(J_i, n, i, eq, adj_X):

            with

                - `J_i`: An :class:`int` defining the index of the functional.
                - `n`: An :class:`int` definining the index of the block of
                  equations.
                - `i`: An :class:`int` defining the index of the considered
                  equation in block `n`.
                - `eq`: The :class:`.Equation`, equation `i` in block `n`.
                - `adj_X`: The adjoint solution associated with equation `i` in
                  block `n` for the `J_i` th functional. `None` indicates that
                  the solution is zero or is not computed (due to an activity
                  analysis). Otherwise a variable if `eq` has a single solution
                  component, and a :class:`Sequence` of variables otherwise.

        :arg prune_forward: Controls the activity analysis. Whether a forward
            traversal of the computational graph, tracing variables which
            depend on the controls, should be applied.
        :arg prune_adjoint: Controls the activity analysis. Whether a reverse
            traversal of the computational graph, tracing variables on which
            the functionals depend, should be applied.
        :arg prune_replay: Controls the activity analysis. Whether an activity
            analysis should be applied when solving forward equations during
            checkpointing/replay.
        :arg cache_adjoint_degree: Adjoint solutions can be cached and reused
            across adjoints where the solution is the same -- e.g. first order
            adjoint solutions associated with the same functional and same
            block and equation indices are equal. A value of `None` indicates
            that caching should be applied at all degrees, a value of 0
            indicates that no caching should be applied, and any positive
            :class:`int` indicates that caching should be applied for adjoints
            up to and including degree `cache_adjoint_degree`.
        :arg store_adjoint: Whether cached adjoint solutions should be retained
            after the call to this method. Can be used to cache and reuse first
            order adjoint solutions in multiple calls to this method.
        :arg adj_ics: A :class:`Mapping`. Items are `(x, value)` where `x` is a
            variable or variable ID identifying a forward variable. The adjoint
            variable associated with the final equation solving for `x` is
            initialized to the value stored by the variable `value`.
        :returns: The conjugate of the derivatives. The return type depends on
            the type of `Js` and `M`.

              - If `Js` is a variable and `M` is a variable, returns a variable
                storing the conjugate of the derivative.
              - If `Js` is a :class:`Sequence`, and `M` is a variable, returns
                a variable whose :math:`i` th component stores the conjugate of
                the derivative of the :math:`i` th functional.
              - If `Js` is a variable and `M` is a :class:`Sequence`, returns a
                :class:`Sequence` of variables whose :math:`j` th component
                stores the conjugate of the derivative with respect to the
                :math:`j` th control.
              - If both `Js` and `M` are :class:`Sequence` objects, returns a
                :class:`Sequence` whose :math:`i` th component stores the
                conjugate of the derivatives of the :math:`i` th functional.
                Each of these is a :class:`Sequence` of variables whose
                :math:`j` th component stores the conjugate of the derivative
                with respect to the :math:`j` th control.
        """

        set_manager(self)
        self.finalize()

        if self._cp_schedule.is_exhausted:
            raise RuntimeError("Invalid checkpointing state")

        # Functionals
        Js_packed = Packed(Js)
        Js = tuple(Js_packed)
        if not all(map(var_is_scalar, Js)):
            raise ValueError("Functional must be a scalar variable")

        # Controls
        M_packed = Packed(M)
        M = tuple(M_packed)

        # Derivatives
        dJ = [None for J in Js]

        # Add two additional blocks, one at the start and one at the end of the
        # forward:
        #   Control block   :  Represents the equation "controls = inputs"
        #   Functional block:  Represents the equations "outputs = functionals"
        blocks_N = len(self._blocks)
        blocks = {-1: [ControlsMarker(M)]}
        blocks.update(enumerate(self._blocks))
        blocks[blocks_N] = list(map(FunctionalMarker, Js))
        J_markers = tuple(eq.x() for eq in blocks[blocks_N])

        # Adjoint equation right-hand-sides
        Bs = tuple(AdjointModelRHS(blocks) for _ in Js)
        # Adjoint initial condition
        for J_i in range(len(Js)):
            var_assign(Bs[J_i][blocks_N][J_i].b(), 1.0)

        # Transpose computational graph
        transpose_deps = TransposeComputationalGraph(
            J_markers, M, blocks,
            prune_forward=prune_forward, prune_adjoint=prune_adjoint)

        # Initialize the adjoint cache
        self._adj_cache.initialize(J_markers, blocks, transpose_deps,
                                   cache_degree=cache_adjoint_degree)

        # Adjoint variables
        adj_Xs = tuple({} for _ in Js)
        if adj_ics is not None:
            for J_i in range(len(Js)):
                for x_id, adj_x in adj_ics[J_i].items():
                    if not isinstance(x_id, int):
                        x_id = var_id(x_id)
                    if transpose_deps.has_adj_ic(J_i, x_id):
                        adj_Xs[J_i][x_id] = var_copy(adj_x)

        # Reverse (blocks)
        for n in range(blocks_N, -2, -1):
            block = blocks[n]

            cp_block = n >= 0 and n < blocks_N
            if cp_block:
                # Load/restore forward data
                self._restore_checkpoint(
                    n, transpose_deps=transpose_deps if prune_replay else None)

            # Reverse (equations in block n)
            for i in range(len(block) - 1, -1, -1):
                eq = block[i]
                eq_X = eq.X()

                for J_i, J in enumerate(Js):
                    # Adjoint right-hand-side associated with this equation
                    B_state, eq_B = Bs[J_i].pop()
                    assert B_state == (n, i)

                    # Extract adjoint initial condition
                    adj_X_ic = tuple(adj_Xs[J_i].pop(var_id(x), None)
                                     for x in eq_X)
                    if transpose_deps.is_solved(J_i, n, i):
                        adj_X_ic_ids = set(map(var_id,
                                               eq.adjoint_initial_condition_dependencies()))  # noqa: E501
                        assert len(eq_X) == len(adj_X_ic)
                        for x, adj_x_ic in zip(eq_X, adj_X_ic):
                            if var_id(x) not in adj_X_ic_ids:
                                assert adj_x_ic is None
                        del adj_X_ic_ids
                    else:
                        for adj_x_ic in adj_X_ic:
                            assert adj_x_ic is None

                    if transpose_deps.is_solved(J_i, n, i):
                        assert (J_i, n, i) not in self._adj_cache

                        # Construct adjoint initial condition
                        if len(eq.adjoint_initial_condition_dependencies()) == 0:  # noqa: E501
                            adj_X = None
                        else:
                            adj_X = []
                            for m, adj_x_ic in enumerate(adj_X_ic):
                                if adj_x_ic is None:
                                    adj_X.append(eq.new_adj_X(m))
                                else:
                                    adj_X.append(adj_x_ic)

                        # Non-linear dependency data
                        nl_deps = self._cp[(n, i)] if cp_block else ()

                        # Solve adjoint equation, add terms to adjoint
                        # equations
                        adj_X = eq.adjoint(
                            adj_X, nl_deps,
                            eq_B.B(),
                            transpose_deps.adj_Bs(J_i, n, i, eq, Bs[J_i]))
                    elif transpose_deps.is_active(J_i, n, i):
                        # Extract adjoint solution from the cache
                        if store_adjoint:
                            adj_X = self._adj_cache.get(J_i, n, i,
                                                        copy=False)
                        else:
                            adj_X = self._adj_cache.pop(J_i, n, i,
                                                        copy=False)

                        # Non-linear dependency data
                        nl_deps = self._cp[(n, i)] if cp_block else ()

                        # Add terms to adjoint equations
                        eq.adjoint_cached(
                            adj_X, nl_deps,
                            transpose_deps.adj_Bs(J_i, n, i, eq, Bs[J_i]))
                    else:
                        if not store_adjoint \
                                and (J_i, n, i) in self._adj_cache:
                            self._adj_cache.remove(J_i, n, i)

                        # Adjoint solution has no effect on sensitivity
                        adj_X = None

                    if adj_X is not None:
                        # Store adjoint initial conditions
                        assert len(eq_X) == len(adj_X)
                        for m, (x, adj_x) in enumerate(zip(eq_X, adj_X)):
                            if transpose_deps.is_stored_adj_ic(J_i, n, i, m):
                                adj_Xs[J_i][var_id(x)] = var_copy(adj_x)

                        # Store adjoint solution in the cache
                        self._adj_cache.cache(J_i, n, i, adj_X,
                                              copy=True, store=store_adjoint)

                    # Diagnostic callback
                    if callback is not None:
                        callback(J_i, n, i, eq,
                                 None if adj_X is None else eq._unpack(adj_X))

                    if n == -1:
                        assert i == 0
                        # A requested derivative
                        if adj_X is None:
                            dJ[J_i] = eq.new_adj_X()
                        else:
                            dJ[J_i] = tuple(map(var_copy, adj_X))

                self.drop_references()
            garbage_cleanup(self._comm)

        for B in Bs:
            assert B.is_empty()
        for J_i in range(len(adj_Xs)):
            assert len(adj_Xs[J_i]) == 0
        if not store_adjoint:
            assert len(self._adj_cache) == 0

        if self._cp_schedule.max_n is None \
                or self._cp_schedule.r != self._cp_schedule.max_n:
            raise RuntimeError("Invalid checkpointing state")

        @functools.singledispatch
        def action(cp_action):
            raise TypeError(f"Unexpected checkpointing action: {cp_action}")

        @action.register(Clear)
        def action_clear(cp_action):
            self._cp.clear(clear_ics=cp_action.clear_ics,
                           clear_data=cp_action.clear_data)

        @action.register(EndReverse)
        def action_end_reverse(cp_action):
            pass

        while True:
            cp_action = next(self._cp_schedule)
            action(cp_action)
            if isinstance(cp_action, EndReverse):
                break

        for cls in action.registry:
            @action.register(cls)
            def action_pass(cp_action):
                pass
        del action

        garbage_cleanup(self._comm)
        return Js_packed.unpack(tuple(M_packed.unpack(dJ_) for dJ_ in dJ))


set_manager(EquationManager())
