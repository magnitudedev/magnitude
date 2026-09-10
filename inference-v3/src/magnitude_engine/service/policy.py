"""Pure service selection over residency, waiting age and completed batch cost."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from pydantic import Field

from magnitude_engine.data import Record

RequestId = NewType("RequestId", int)


class Phase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class Limits(Record):
    max_requests: int = Field(default=128, gt=0)
    max_batch: int = Field(default=8, gt=0)
    prefill_tokens: int = Field(default=512, gt=0)
    decode_share: float = Field(default=0.5, gt=0, lt=1)
    locality_seconds: float = Field(default=0.05, ge=0, allow_inf_nan=False)


@dataclass(frozen=True)
class Candidate:
    identity: RequestId
    phase: Phase
    active: bool
    resident: bool
    waiting_since_ns: int
    service_ns: int
    preemption_debt: int


@dataclass(frozen=True)
class Selection:
    phase: Phase
    requests: tuple[RequestId, ...]
    contended: bool


class Scheduler:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.decode_debt_ns = 0.0
        self.contended = False
        self.completed_service_ns = 0

    def priority(self, candidate: Candidate, now_ns: int) -> tuple[float, int, int]:
        locality = self.limits.locality_seconds * 1e9
        return (
            now_ns
            - candidate.waiting_since_ns
            + locality * (candidate.active + candidate.resident + candidate.preemption_debt),
            -candidate.service_ns,
            -candidate.identity,
        )

    def select(self, candidates: tuple[Candidate, ...], now_ns: int) -> Selection | None:
        if not candidates:
            self.contended = False
            self.decode_debt_ns = 0.0
            return None
        decoding = any(c.phase == Phase.DECODE for c in candidates)
        prefill = any(c.phase == Phase.PREFILL for c in candidates)
        contended = decoding and prefill
        first = contended and not self.contended
        if not contended or first:
            self.decode_debt_ns = 0.0
        self.contended = contended
        phase = (
            Phase.DECODE
            if decoding and (not prefill or first or self.decode_debt_ns > 0)
            else Phase.PREFILL
        )
        ordered = sorted(
            (c for c in candidates if c.phase == phase),
            key=lambda c: self.priority(c, now_ns),
            reverse=True,
        )
        return Selection(
            phase, tuple(c.identity for c in ordered[: self.limits.max_batch]), contended
        )

    def completed(self, selection: Selection, elapsed_ns: int) -> None:
        if elapsed_ns < 0:
            raise ValueError("completed service duration cannot be negative")
        self.completed_service_ns += elapsed_ns
        if selection.contended:
            if selection.phase == Phase.DECODE:
                self.decode_debt_ns = max(0.0, self.decode_debt_ns - elapsed_ns)
            else:
                share = self.limits.decode_share
                self.decode_debt_ns += elapsed_ns * share / (1 - share)
