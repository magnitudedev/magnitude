"""Backend-free engine policy contracts, independent of their concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

type Phase = Literal["decode", "prefill"]


@dataclass(frozen=True)
class Runnable:
    identity: str
    prefill_remaining: int
    output_credit: int
    # Equal non-None groups describe currently compatible prompt operations.
    # This is live eligibility evidence, never a model or architecture identifier.
    prefill_group: object | None = None


@dataclass(frozen=True)
class Service:
    identity: str
    tokens: int


@dataclass(frozen=True)
class Schedule:
    """One bounded phase; execution feedback precedes the next selection."""

    phase: Phase
    services: tuple[Service, ...]
    budget_ns: int | None = None


@dataclass(frozen=True)
class CompletedService:
    """Elapsed phase service counted once, regardless of physical batch width."""

    phase: Phase
    elapsed_ns: int
    input_tokens: int = 0
    preparation_ns: int = 0


class Scheduler(ABC):
    max_active: int
    max_queued: int
    prefill_tokens: int

    @abstractmethod
    def select(self, rows: tuple[Runnable, ...]) -> Schedule | None: ...

    @abstractmethod
    def observe(self, service: CompletedService) -> None: ...

    @abstractmethod
    def reset(self) -> None:
        """Start an independent workload without learned rates or service credit."""
        ...
