#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import functools

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

    Parameters can be accessed via the `args` attribute.
    """

    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return f"{type(self).__name__}{self.args!r}"

    def __eq__(self, other):
        return type(self) is type(other) and self.args == other.args


class Clear(CheckpointAction):
    """A checkpointing action which clears the intermediate storage.

    :arg clear_ics: Whether to clear stored forward restart data. Accessed
        via the `clear_ics` attribute.
    :arg clear_data: Whether to clear stored non-linear dependency data.
        Accessed via the `clear_data` attribute.
    """

    def __init__(self, clear_ics, clear_data):
        super().__init__(clear_ics, clear_data)

    @property
    def clear_ics(self):
        return self.args[0]

    @property
    def clear_data(self):
        return self.args[1]


class Configure(CheckpointAction):
    """A checkpointing action which configures the intermediate storage.

    :arg store_ics: Whether to store forward restart data. Accessed via the
        `store_ics` attribute.
    :arg store_data: Whether to store non-linear dependency data. Accessed
        via the `store_data` attribute.
    """

    def __init__(self, store_ics, store_data):
        super().__init__(store_ics, store_data)

    @property
    def store_ics(self):
        return self.args[0]

    @property
    def store_data(self):
        return self.args[1]


class Forward(CheckpointAction):
    """A checkpointing action which indicates forward advancement.

    :arg n0: The forward should start from the start of this step. Accessed via
        the `n0` attribute.
    :arg n1: The forward should advance to the start of this step. Accessed via
        the `n1` attribute.
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
        return self.args[0]

    @property
    def n1(self):
        return self.args[1]


class Reverse(CheckpointAction):
    """A checkpointing action which indicates adjoint advancement.

    :arg n1: The adjoint should advance from the start of this step. Accessed
        via the `n1` attribute.
    :arg n0: The adjoint should to the start of this step. Accessed via the
        `n0` attribute.
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
        return self.args[1]

    @property
    def n1(self):
        return self.args[0]


class Read(CheckpointAction):
    """A checkpointing action which indicates loading of data from a
    checkpointing unit.

    :arg n: The step with which the loaded data is associated. Accessed via the
        `n` attribute.
    :arg storage: The storage from which the data should be loaded. Either
        `'RAM'` or `'disk'`. Accessed via the `storage` attribute.
    :arg delete: Whether the data should be deleted from the indicated storage
        after it has been loaded. Accessed via the `delete` attribute.
    """

    def __init__(self, n, storage, delete):
        super().__init__(n, storage, delete)

    @property
    def n(self):
        return self.args[0]

    @property
    def storage(self):
        return self.args[1]

    @property
    def delete(self):
        return self.args[2]


class Write(CheckpointAction):
    """A checkpointing action which indicates saving of data to a checkpointing
    unit.

    :arg n: The step with which the saved data is associated. Accessed via the
        `n` attribute.
    :arg storage: The storage to which the data should be saved. Either `'RAM'`
        or `'disk'`. Accessed via the `storage` attribute.
    """

    def __init__(self, n, storage):
        super().__init__(n, storage)

    @property
    def n(self):
        return self.args[0]

    @property
    def storage(self):
        return self.args[1]


class EndForward(CheckpointAction):
    """A checkpointing action which indicates the end of the initial forward
    calculation.
    """

    pass


class EndReverse(CheckpointAction):
    """A checkpointing action which indicates the end of an adjoint
    calculation.

    :arg exhausted: Indicates whether the schedule has concluded. If `True`
        then this action should be the last action in the schedule. Accessed
        via the `exhausted` attribute.
    """

    def __init__(self, exhausted):
        super().__init__(exhausted)

    @property
    def exhausted(self):
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
    steps should later be provided using the :meth:`finalize` method.

    :arg max_n: The number of steps in the initial forward calculation. If not
        supplied then this should later be provided by calling the
        :meth:`finalize` method.
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

    @abstractmethod
    def is_exhausted(self):
        """Return whether the schedule has concluded. Note that some schedules
        permit multiple adjoint calculation, and may never conclude.
        """

        raise NotImplementedError

    @abstractmethod
    def uses_disk_storage(self):
        """Return whether the schedule may use disk storage. If `False` then no
        disk storage is required.
        """

        raise NotImplementedError

    def n(self):
        """Return the forward location. After executing all actions defined so
        far in the schedule the forward is at the start of this step.
        """

        return self._n

    def r(self):
        """Return the number of adjoint steps advanced in the current adjoint
        calculation after executing all actions defined so far in the schedule.
        """

        return self._r

    def max_n(self):
        """Return the number of forward steps in the initial forward
        calculation. May return `None` if this has not yet been provided to the
        scheduler.
        """

        return self._max_n

    def is_running(self):
        """Return whether the schedule is 'running' -- i.e. at least one action
        has been defined so far in the schedule.
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
