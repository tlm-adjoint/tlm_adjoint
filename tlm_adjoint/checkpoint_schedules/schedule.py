from abc import ABC, abstractmethod
import functools
import warnings

__all__ = \
    [
        "CheckpointAction",
        "Clear",
        "Configure",
        "Forward",
        "Reverse",
        "Read",
        "Write",
        "EndForward",
        "EndReverse",

        "CheckpointSchedule"
    ]


class CheckpointAction:
    """A checkpointing action.
    """

    def __init__(self, *args):
        self._args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return type(self) is type(other) and self.args == other.args

    @property
    def args(self):
        """Action parameters."""

        return self._args


class Clear(CheckpointAction):
    """A checkpointing action which clears the intermediate storage.

    :arg clear_ics: Whether to clear stored forward restart data.
    :arg clear_data: Whether to clear stored non-linear dependency data.
    """

    def __init__(self, clear_ics, clear_data):
        super().__init__(clear_ics, clear_data)

    @property
    def clear_ics(self):
        """Whether to clear stored forward restart data."""

        return self.args[0]

    @property
    def clear_data(self):
        """Whether to clear stored non-linear dependency data."""

        return self.args[1]


class Configure(CheckpointAction):
    """A checkpointing action which configures the intermediate storage.

    :arg store_ics: Whether to store forward restart data.
    :arg store_data: Whether to store non-linear dependency data.
    """

    def __init__(self, store_ics, store_data):
        super().__init__(store_ics, store_data)

    @property
    def store_ics(self):
        """Whether to store forward restart data."""

        return self.args[0]

    @property
    def store_data(self):
        """Whether to store non-linear dependency data."""

        return self.args[1]


class Forward(CheckpointAction):
    """A checkpointing action which indicates forward advancement.

    :arg n0: The forward should advance from the start of this step.
    :arg n1: The forward should advance to the start of this step.
    """

    def __init__(self, n0, n1):
        super().__init__(n0, n1)

    def __iter__(self):
        yield from range(self.n0, self.n1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """The forward should advance from the start of this step."""

        return self.args[0]

    @property
    def n1(self):
        """The forward should advance to the start of this step."""

        return self.args[1]


class Reverse(CheckpointAction):
    """A checkpointing action which indicates adjoint advancement.

    :arg n1: The adjoint should advance from the start of this step.
    :arg n0: The adjoint should advance to the start of this step.
    """

    def __init__(self, n1, n0):
        super().__init__(n1, n0)

    def __iter__(self):
        yield from range(self.n1 - 1, self.n0 - 1, -1)

    def __len__(self):
        return self.n1 - self.n0

    def __contains__(self, step):
        return self.n0 <= step < self.n1

    @property
    def n0(self):
        """The adjoint should advance to the start of this step."""

        return self.args[1]

    @property
    def n1(self):
        """The adjoint should advance from the start of this step."""

        return self.args[0]


class Read(CheckpointAction):
    """A checkpointing action which indicates loading of data from a
    checkpointing unit.

    :arg n: The step with which the loaded data is associated.
    :arg storage: The storage from which the data should be loaded. Either
        `'RAM'` or `'disk'`.
    :arg delete: Whether the data should be deleted from the indicated storage
        after it has been loaded.
    """

    def __init__(self, n, storage, delete):
        super().__init__(n, storage, delete)

    @property
    def n(self):
        """The step with which the loaded data is associated."""

        return self.args[0]

    @property
    def storage(self):
        """The storage from which the data should be loaded. Either `'RAM'` or
        `'disk'`."""

        return self.args[1]

    @property
    def delete(self):
        """Whether the data should be deleted from the indicated storage after
        it has been loaded."""

        return self.args[2]


class Write(CheckpointAction):
    """A checkpointing action which indicates saving of data to a checkpointing
    unit.

    :arg n: The step with which the saved data is associated.
    :arg storage: The storage to which the data should be saved. Either `'RAM'`
        or `'disk'`.
    """

    def __init__(self, n, storage):
        super().__init__(n, storage)

    @property
    def n(self):
        """The step with which the saved data is associated."""

        return self.args[0]

    @property
    def storage(self):
        """The storage to which the data should be saved. Either `'RAM'` or
        `'disk'`."""

        return self.args[1]


class EndForward(CheckpointAction):
    """A checkpointing action which indicates the end of the initial forward
    calculation.
    """

    def __init__(self):
        super().__init__()


class EndReverse(CheckpointAction):
    """A checkpointing action which indicates the end of an adjoint
    calculation.

    :arg exhausted: Indicates whether the schedule has concluded. If `True`
        then this action should be the last action in the schedule.
    """

    def __init__(self, exhausted):
        super().__init__(exhausted)

    @property
    def exhausted(self):
        """Indicates whether the schedule has concluded. If `True` then this
        action should be the last action in the schedule."""

        return self.args[0]


class CheckpointSchedule(ABC):
    """A checkpointing schedule.

    Actions in the schedule are accessed by iterating over elements, and
    actions may be implemented using single-dispatch functions. e.g.

    .. code-block:: python

        @functools.singledispatch
        def action(cp_action):
            raise TypeError(f"Unexpected checkpointing action: {cp_action}")

        @action.register(Forward)
        def action_forward(cp_action):
            logger.debug(f"forward: forward advance to {cp_action.n1:d}")

        # ...

        for cp_action in cp_schedule:
            action(cp_action)
            if isinstance(cp_action, EndReverse):
                break

    Schedules control an intermediate storage, which buffers forward restart
    data for forward restart checkpoints, and which stores non-linear
    dependency data either for storage in checkpointing units or for immediate
    use by the adjoint. For details see

        - James R. Maddison, 'On the implementation of checkpointing with
          high-level algorithmic differentiation',
          https://arxiv.org/abs/2305.09568v1, 2023

    In 'offline' schedules, where the number of steps in the forward
    calculation is initially known, this should be provided using the `max_n`
    argument on instantiation. In 'online' schedules, where the number of steps
    in the forward calculation is initially unknown, the number of forward
    steps should later be provided using the
    :meth:`.CheckpointSchedule.finalize` method.

    :arg max_n: The number of steps in the initial forward calculation. If not
        supplied then this should later be provided by calling the
        :meth:`.CheckpointSchedule.finalize` method.
    """

    def __init__(self, max_n=None):
        if max_n is not None and max_n < 1:
            raise ValueError("max_n must be positive")

        self._n = 0
        self._r = 0
        self._max_n = max_n

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls_iter = cls.iter

        @functools.wraps(cls_iter)
        def iter(self):
            if not hasattr(self, "_iter"):
                self._iter = cls_iter(self)
            return self._iter

        cls.iter = iter

        class CallableBool:
            def __init__(self, value):
                self._value = value

            def __bool__(self):
                return bool(self._value)

            def __call__(self):
                warnings.warn("is_exhausted is a property and should not "
                              "be called",
                              DeprecationWarning, stacklevel=2)
                return bool(self)

        @property
        def is_exhausted(self):
            value = orig_is_exhausted.__get__(self, type(self))
            return CallableBool(value)

        orig_is_exhausted = cls.is_exhausted
        cls.is_exhausted = is_exhausted

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter())

    @abstractmethod
    def iter(self):
        """A generator which should be overridden in derived classes in order
        to define a checkpointing schedule.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def is_exhausted(self):
        """Whether the schedule has concluded. Note that some schedules permit
        multiple adjoint calculation, and may never conclude.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def uses_disk_storage(self):
        """Whether the schedule may use disk storage. If `False` then no disk
        storage is required.
        """

        raise NotImplementedError

    @property
    def n(self):
        """The forward location. After executing all actions defined so far in
        the schedule the forward is at the start of this step.
        """

        return self._n

    @property
    def r(self):
        """The number of adjoint steps advanced in the current adjoint
        calculation after executing all actions defined so far in the schedule.
        """

        return self._r

    @property
    def max_n(self):
        """The number of forward steps in the initial forward calculation. May
        return `None` if this has not yet been provided to the scheduler.
        """

        return self._max_n

    @property
    def is_running(self):
        """Whether the schedule is 'running' -- i.e. at least one action has
        been defined so far in the schedule.
        """

        return hasattr(self, "_iter")

    def finalize(self, n):
        """Indicate the number of forward steps in the initial forward
        calculation.

        :arg n: The number of steps in the initial forward calculation.
        """

        if n < 1:
            raise ValueError("n must be positive")
        if self._max_n is None:
            if self._n >= n:
                self._n = n
                self._max_n = n
            else:
                raise RuntimeError("Invalid checkpointing state")
        elif self._n != n or self._max_n != n:
            raise RuntimeError("Invalid checkpointing state")
