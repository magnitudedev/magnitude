"""Logical expert residency, independently testable from byte transport and math."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Transfer:
    expert: int
    slot: int
    generation: int


@dataclass
class _Slot:
    expert: int | None = None
    transfer: Transfer | None = None
    readers: int = 0
    touched: int = 0


class ResidentLease:
    def __init__(self, directory: Residency, slots: tuple[int, ...]):
        self.directory, self.slots = directory, slots
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            for slot in self.slots:
                self.directory._slots[slot].readers -= 1
            self.closed = True


class Residency:
    """Publish after all bytes arrive; no eviction may overwrite a leased consumer.

    An old mapping is withdrawn before transport starts. Failure leaves its slot
    absent, because the original bytes may already have been partially overwritten.
    Generation-tagged transfers reject delayed completion from a cancelled read.
    """

    def __init__(self, experts: int, slots: int):
        if not 0 < slots <= experts:
            raise ValueError("residency requires 0 < slots <= logical experts")
        self.experts = experts
        self._slots = [_Slot() for _ in range(slots)]
        self._resident: dict[int, int] = {}
        self._loading: dict[int, Transfer] = {}
        self._generation = 0
        self._clock = 0

    def _check(self, expert: int) -> None:
        if not 0 <= expert < self.experts:
            raise ValueError("logical expert ID is out of bounds")

    def resolve(self, expert: int) -> int | None:
        self._check(expert)
        return self._resident.get(expert)

    def touch(self, experts: tuple[int, ...]) -> None:
        self._clock += 1
        for expert in experts:
            address = self.resolve(expert)
            if address is not None:
                self._slots[address].touched = self._clock

    def pin(self, experts: tuple[int, ...]) -> ResidentLease:
        resolved = tuple(self.resolve(expert) for expert in dict.fromkeys(experts))
        if any(slot is None for slot in resolved):
            raise ValueError("cannot lease an absent expert")
        slots = tuple(slot for slot in resolved if slot is not None)
        for slot in slots:
            self._slots[slot].readers += 1
        self.touch(experts)
        return ResidentLease(self, slots)

    def reserve(self, expert: int) -> Transfer | None:
        if self.resolve(expert) is not None:
            self.touch((expert,))
            return None
        if expert in self._loading:
            return self._loading[expert]
        choices = [
            index
            for index, slot in enumerate(self._slots)
            if slot.readers == 0 and slot.transfer is None
        ]
        if not choices:
            raise MemoryError("all expert slots are leased or being filled")
        address = min(
            choices,
            key=lambda index: (
                self._slots[index].expert is not None,
                self._slots[index].touched,
                index,
            ),
        )
        slot = self._slots[address]
        if slot.expert is not None:
            del self._resident[slot.expert]
        slot.expert = None
        self._generation += 1
        transfer = Transfer(expert, address, self._generation)
        slot.transfer = transfer
        self._loading[expert] = transfer
        return transfer

    def finish(self, transfer: Transfer, *, publish: bool) -> None:
        if self._loading.get(transfer.expert) != transfer:
            raise ValueError("stale expert transfer")
        slot = self._slots[transfer.slot]
        assert slot.transfer == transfer and slot.expert is None and slot.readers == 0
        slot.transfer = None
        del self._loading[transfer.expert]
        if publish:
            slot.expert = transfer.expert
            self._resident[transfer.expert] = transfer.slot
            self.touch((transfer.expert,))

    def validate(self) -> None:
        assert len(set(self._resident.values())) == len(self._resident)
        for address, slot in enumerate(self._slots):
            assert slot.readers >= 0
            if slot.expert is not None:
                assert self._resident[slot.expert] == address
                assert slot.transfer is None
            if slot.transfer is not None:
                assert self._loading[slot.transfer.expert] == slot.transfer
                assert slot.expert is None and slot.readers == 0
            if slot.readers:
                assert slot.expert is not None
