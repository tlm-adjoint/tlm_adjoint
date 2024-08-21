r"""Translation between checkpointing schedules provided by the
checkpoint_schedules library and a tlm_adjoint :class:`.CheckpointSchedule`.

Wrapped :class:`checkpoint_schedule.CheckpointSchedule` classes can be
imported from this module and then passed to :func:`.configure_checkpointing`,
e.g.

.. code-block:: python

    from tlm_adjoint import configure_checkpointing
    from tlm_adjoint.checkpoint_schedules.checkpoint_schedules \
        import MultistageCheckpointSchedule

    configure_checkpointing(
        MultistageCheckpointSchedule,
        {"max_n": 30, "snapshots_in_ram": 0, "snapshots_on_disk": 3})
"""

try:
    import checkpoint_schedules
except ModuleNotFoundError:
    checkpoint_schedules = None
if checkpoint_schedules is not None:
    from checkpoint_schedules import (
        CheckpointSchedule as _CheckpointSchedule, Forward as _Forward,
        Reverse as _Reverse, Copy as _Copy, Move as _Move,
        EndForward as _EndForward, EndReverse as _EndReverse)
    from checkpoint_schedules import StorageType

from .schedule import (
    CheckpointSchedule, Clear, Configure, Forward, Reverse, Read, Write,
    EndForward, EndReverse)

from functools import singledispatch, wraps
import itertools


def translation(cls):
    class Translation(CheckpointSchedule):
        def __init__(self, *args, **kwargs):
            self._cp_schedule = cls(*args, **kwargs)
            super().__init__(self._cp_schedule.max_n)
            self._is_exhausted = self._cp_schedule.is_exhausted

        def iter(self):
            # Used to ensure that we do not finalize the wrapped scheduler
            # while yielding actions associated with a single wrapped action.
            # Prevents multiple finalization of the wrapped schedule.
            def finalizer(fn):
                @wraps(fn)
                def wrapped_fn(cp_action):
                    if self.max_n != self._cp_schedule.max_n:
                        self._cp_schedule.finalize(self.max_n)
                    yield from fn(cp_action)
                    if self.max_n != self._cp_schedule.max_n:
                        self._cp_schedule.finalize(self.max_n)
                return wrapped_fn

            @singledispatch
            @finalizer
            def action(cp_action):
                raise TypeError(f"Unexpected action type: {type(cp_action)}")
                yield None

            ics = (0, 0)
            data = (0, 0)
            replay = None
            checkpoints = {StorageType.RAM: {}, StorageType.DISK: {}}

            def clear():
                nonlocal ics, data

                ics = (0, 0)
                data = (0, 0)
                yield Clear(True, True)

            def read(n, storage, *, delete):
                nonlocal ics, data, replay

                if replay != (0, 0):
                    raise RuntimeError("Invalid checkpointing state")
                replay, _ = ics, data = checkpoints[storage][n]
                if delete:
                    del checkpoints[storage][n]
                self._n = n
                yield Read(n, {StorageType.RAM: "RAM",
                               StorageType.DISK: "disk"}[storage], delete)

            def write(n, storage):
                checkpoints[storage][n] = (ics, data)
                yield Write(n, {StorageType.RAM: "RAM",
                                StorageType.DISK: "disk"}[storage])

            def input_output(n, from_storage, to_storage, *, delete):
                if to_storage in {StorageType.RAM, StorageType.DISK}:
                    yield from clear()
                    yield from read(n, from_storage, delete=delete)
                    yield from write(n, to_storage)
                    yield from clear()
                elif to_storage == StorageType.WORK:
                    yield from clear()
                    yield from read(n, from_storage, delete=delete)
                else:
                    raise ValueError(f"Unexpected storage type: "
                                     f"{to_storage}")

            @action.register(_Forward)
            @finalizer
            def action_forward(cp_action):
                nonlocal ics, data

                yield from clear()
                yield Configure(cp_action.write_ics, cp_action.write_adj_deps)

                if cp_action.write_ics:
                    ics = (cp_action.n0, cp_action.n1)
                if cp_action.write_adj_deps:
                    data = (cp_action.n0, cp_action.n1)
                if replay is not None and (cp_action.n0 < replay[0] or cp_action.n1 > replay[1]):  # noqa: E501
                    raise RuntimeError("Invalid checkpointing state")
                self._n = cp_action.n1
                yield Forward(cp_action.n0, cp_action.n1)

                if cp_action.storage == StorageType.NONE:
                    if cp_action.write_ics or cp_action.write_adj_deps:
                        raise ValueError("Unexpected action parameters")
                elif cp_action.storage in {StorageType.RAM, StorageType.DISK}:
                    yield from write(cp_action.n0, cp_action.storage)
                    yield from clear()
                elif cp_action.storage == StorageType.WORK:
                    if cp_action.write_ics:
                        raise ValueError("Unexpected action parameters")
                else:
                    raise ValueError(f"Unexpected storage type: "
                                     f"{cp_action.storage}")

            @action.register(_Reverse)
            @finalizer
            def action_reverse(cp_action):
                nonlocal replay

                if self.max_n is None:
                    raise RuntimeError("Invalid checkpointing state")
                if cp_action.n0 < data[0] or cp_action.n1 > data[1]:
                    raise RuntimeError("Invalid checkpointing state")
                replay = (0, 0)
                self._r = self._cp_schedule.r
                yield Reverse(cp_action.n1, cp_action.n0)
                yield from clear()

            @action.register(_Copy)
            @finalizer
            def action_copy(cp_action):
                yield from input_output(
                    cp_action.n, cp_action.from_storage, cp_action.to_storage,
                    delete=False)

            @action.register(_Move)
            @finalizer
            def action_move(cp_action):
                yield from input_output(
                    cp_action.n, cp_action.from_storage, cp_action.to_storage,
                    delete=True)

            @action.register(_EndForward)
            @finalizer
            def action_end_forward(cp_action):
                nonlocal replay

                replay = (0, 0)
                self._is_exhausted = self._cp_schedule.is_exhausted
                yield EndForward()

            @action.register(_EndReverse)
            @finalizer
            def action_end_reverse(cp_action):
                if self._cp_schedule.is_exhausted:
                    yield from clear()
                self._r = self._cp_schedule.r
                self._is_exhausted = self._cp_schedule.is_exhausted
                yield EndReverse(self._cp_schedule.is_exhausted)

            yield from clear()
            yield from itertools.chain.from_iterable(
                map(action, self._cp_schedule))

        @property
        def is_exhausted(self):
            return self._is_exhausted

        @property
        def uses_disk_storage(self):
            return self._cp_schedule.uses_storage_type(StorageType.DISK)

    return Translation


def _init():
    if checkpoint_schedules is not None:
        __all__.append("StorageType")

        for name in dir(checkpoint_schedules):
            obj = getattr(checkpoint_schedules, name)
            if isinstance(obj, type) \
                    and issubclass(obj, _CheckpointSchedule) \
                    and obj is not _CheckpointSchedule:
                globals()[name] = cls = translation(obj)
                cls.__doc__ = (f"Wrapper for the checkpoint_schedules "
                               f":class:`checkpoint_schedules.{name}` class.")
                __all__.append(name)

        __all__.sort()


__all__ = []
_init()
del _init
