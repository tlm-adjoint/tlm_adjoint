from checkpoint_schedules import StorageType
from checkpoint_schedules import (
    Forward as _Forward, Reverse as _Reverse, Copy as _Copy, Move as _Move,
    EndForward as _EndForward, EndReverse as _EndReverse)

from .schedule import (
    CheckpointSchedule, Configure, Clear, Forward, Reverse, Read, Write,
    EndForward, EndReverse)

import functools

__all__ = \
    [
    ]


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
            def locked(fn):
                @functools.wraps(fn)
                def wrapped_fn(cp_action):
                    max_n = self._max_n
                    try:
                        yield from fn(cp_action)
                    finally:
                        if self._max_n != max_n:
                            self._cp_schedule.finalize(self._max_n)
                return wrapped_fn

            @functools.singledispatch
            @locked
            def action(cp_action):
                raise TypeError(f"Unexpected action type: {type(cp_action)}")
                yield None

            @action.register(_Forward)
            @locked
            def action_forward(cp_action):
                yield Clear(True, True)
                yield Configure(cp_action.write_ics, cp_action.write_adj_deps)
                self._n = self._cp_schedule.n
                yield Forward(cp_action.n0, cp_action.n1)
                if cp_action.storage not in {StorageType.NONE,
                                             StorageType.WORK}:
                    yield Write(cp_action.n0,
                                {StorageType.RAM: "RAM",
                                 StorageType.DISK: "disk"}[cp_action.storage])
                    yield Clear(True, False)

            @action.register(_Reverse)
            @locked
            def action_reverse(cp_action):
                if self._max_n is None:
                    raise RuntimeError("Invalid checkpointing state")
                self._r = self._cp_schedule.r
                yield Reverse(cp_action.n1, cp_action.n0)

            @action.register(_Copy)
            @locked
            def action_copy(cp_action):
                if cp_action.to_storage == StorageType.NONE:
                    pass
                elif cp_action.to_storage in {StorageType.RAM, StorageType.DISK}:  # noqa: E501
                    yield Clear(True, True)
                    self._n = self._cp_schedule.n
                    yield Read(cp_action.n,
                               {StorageType.RAM: "RAM",
                                StorageType.DISK: "disk"}[cp_action.from_storage],  # noqa: E501
                               False)
                    yield Write(cp_action.n0,
                                {StorageType.RAM: "RAM",
                                 StorageType.DISK: "disk"}[cp_action.to_storage])  # noqa: E501
                    yield Clear(True, True)
                elif cp_action.to_storage == StorageType.WORK:
                    yield Clear(True, True)
                    self._n = self._cp_schedule.n
                    yield Read(cp_action.n,
                               {StorageType.RAM: "RAM",
                                StorageType.DISK: "disk"}[cp_action.from_storage],  # noqa: E501
                               False)
                else:
                    raise ValueError(f"Unexpected storage type: "
                                     f"{cp_action.to_storage}")

            @action.register(_Move)
            @locked
            def action_move(cp_action):
                if cp_action.to_storage == StorageType.NONE:
                    pass
                elif cp_action.to_storage in {StorageType.RAM, StorageType.DISK}:  # noqa: E501
                    yield Clear(True, True)
                    self._n = self._cp_schedule.n
                    yield Read(cp_action.n,
                               {StorageType.RAM: "RAM",
                                StorageType.DISK: "disk"}[cp_action.from_storage],  # noqa: E501
                               True)
                    yield Write(cp_action.n0,
                                {StorageType.RAM: "RAM",
                                 StorageType.DISK: "disk"}[cp_action.to_storage])  # noqa: E501
                    yield Clear(True, True)
                elif cp_action.to_storage == StorageType.WORK:
                    yield Clear(True, True)
                    self._n = self._cp_schedule.n
                    yield Read(cp_action.n,
                               {StorageType.RAM: "RAM",
                                StorageType.DISK: "disk"}[cp_action.from_storage],  # noqa: E501
                               True)
                else:
                    raise ValueError(f"Unexpected storage type: "
                                     f"{cp_action.to_storage}")

            @action.register(_EndForward)
            @locked
            def action_end_forward(cp_action):
                self._is_exhausted = self._cp_schedule.is_exhausted
                yield EndForward()

            @action.register(_EndReverse)
            @locked
            def action_end_reverse(cp_action):
                if self._cp_schedule.is_exhausted:
                    yield Clear(True, True)
                self._r = self._cp_schedule.r
                self._is_exhausted = self._cp_schedule.is_exhausted
                yield EndReverse(self._cp_schedule.is_exhausted)

            yield Clear(True, True)
            while not self._cp_schedule.is_exhausted:
                yield from action(next(self._cp_schedule))

        @property
        def is_exhausted(self):
            return self._is_exhausted

        @property
        def uses_disk_storage(self):
            return self._cp_schedule.uses_storage_type(StorageType.DISK)

    return Translation
