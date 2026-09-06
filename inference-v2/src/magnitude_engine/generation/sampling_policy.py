"""Request-local sampling configuration, independent of the tensor backend."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    history_window: int = 64
    seed: int | None = None

    def __post_init__(self) -> None:
        values = (
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        )
        if (
            not all(type(v) in (int, float) and math.isfinite(v) for v in values)
            or type(self.top_k) is not int
            or type(self.history_window) is not int
            or (self.seed is not None and type(self.seed) is not int)
            or self.temperature < 0
            or self.top_k < 0
            or not 0 < self.top_p <= 1
            or not 0 <= self.min_p <= 1
            or self.repetition_penalty <= 0
            or self.history_window < 1
        ):
            raise ValueError("invalid sampling policy")

    @property
    def uses_history(self) -> bool:
        return (
            self.repetition_penalty != 1
            or self.presence_penalty != 0
            or self.frequency_penalty != 0
        )
