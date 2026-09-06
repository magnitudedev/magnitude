"""Experiments declare their claim and operating point before collecting samples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Literal, Protocol

from magnitude_engine.composition import Blueprint


@dataclass(frozen=True, kw_only=True)
class Experiment:
    identity: str
    subject: SubjectBlueprint
    characteristic: str
    measurement_width: Literal["component", "integrated", "workload", "product"]
    claim: str
    warmup: int = 2
    repetitions: int = 7
    timeout_seconds: float = 120
    comparison: str = "characterization only; no improvement claim"
    invariants: tuple[str, ...] = ()
    run_class: Literal["diagnostic", "smoke", "development", "acceptance", "product", "soak"] = (
        "development"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.subject, SubjectBlueprint):
            raise TypeError("experiment subject must be a typed blueprint")
        if not all((self.identity, self.characteristic, self.claim)):
            raise ValueError("benchmark identity, subject, characteristic and claim are required")
        if self.measurement_width not in ("component", "integrated", "workload", "product"):
            raise ValueError("invalid measurement width")
        if self.run_class not in (
            "diagnostic",
            "smoke",
            "development",
            "acceptance",
            "product",
            "soak",
        ):
            raise ValueError("invalid benchmark run class")
        if self.warmup < 0 or self.repetitions < 1 or self.timeout_seconds <= 0:
            raise ValueError("invalid warmup, repetition count or timeout")
        self.subject.validate_measurement(self.warmup, self.repetitions)
        if not math.isfinite(self.timeout_seconds):
            raise ValueError("timeout must be finite")

    def record(self) -> dict:
        return {
            f.name: self.subject.describe() if f.name == "subject" else getattr(self, f.name)
            for f in fields(self)
        }


@dataclass(frozen=True)
class Observation:
    output_digest: str
    counters: dict[str, int | float] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)


class Subject(Protocol):
    """Reset and validation are outside timing; completion is inside it.

    reset must drain prior work and establish the declared starting state.
    invoke builds/submits exactly the declared operation. complete waits for all
    work counted by that operation, without inserting per-kernel synchronization.
    observe validates results and returns counters; it must raise on invalid work.
    close drains and frees owned resources even after a failed measurement.
    """

    def reset(self) -> None: ...
    def invoke(self) -> None: ...
    def complete(self) -> None: ...
    def observe(self) -> Observation: ...
    def close(self) -> None: ...


class SubjectBlueprint(Blueprint[Subject]):
    def validate_measurement(self, warmup: int, repetitions: int) -> None:
        """Subjects with one-shot semantics reject incompatible measurement schedules."""
