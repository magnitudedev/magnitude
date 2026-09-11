"""A Metric preserves its observation, model, and comparison separately."""

import statistics
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

type Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
type Bytes = Annotated[int, Field(ge=0)]


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TimeLowerBound(Record):
    seconds: Seconds
    formula: str
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]


class TimingPass(StrEnum):
    COMPLETED = "completed"
    DEVICE_EVENTS = "device_events"


class TimingBoundary(StrEnum):
    PREPARE_THROUGH_SUBMISSION = "prepare_through_submission"
    PREPARE_THROUGH_COMPLETION = "prepare_through_completion"
    DEVICE_EXECUTION = "device_execution"
    HTTP_FIRST_TOKEN = "http_first_token"
    HTTP_COMPLETION = "http_completion"


class TrafficInterface(StrEnum):
    LOGICAL_OPERANDS = "logical_operands"
    DEVICE_MEMORY = "device_memory"


class Latency(Record):
    samples_seconds: tuple[Seconds, ...]
    boundary: TimingBoundary
    lower_bound: TimeLowerBound | None = None
    unavailable_observations: tuple[str, ...] = ()
    missing_model_evidence: tuple[str, ...] = ()

    @property
    def median_seconds(self) -> float | None:
        return statistics.median(self.samples_seconds) if self.samples_seconds else None

    @property
    def bound_ratio(self) -> float | None:
        observed = self.median_seconds
        if self.lower_bound is None or observed is None or observed == 0:
            return None
        return self.lower_bound.seconds / observed


class ReadTraffic(Record):
    minimum_bytes: Bytes
    interface: TrafficInterface
    formula: str
    assumptions: tuple[str, ...]
    observed_bytes: tuple[Bytes, ...] = ()
    unavailable: tuple[str, ...] = ()


class Realized(Record):
    """Which kernel a measured operation's table actually chose, and on what.

    Two runs with the same composition digest on different hardware may realize
    different candidates. Recording the name and the resident representation is
    what lets a regression be attributed to a selection change rather than to a
    kernel change.
    """

    operation: str
    candidate: str
    representation: str | None = None
    detail: tuple[tuple[str, int], ...] = ()


class LinearMetrics(Record):
    completed_latency: Latency
    device_latency: Latency
    reads: ReadTraffic
    selection: tuple[Realized, ...] = ()


class Validation(Record):
    passed: bool
    method: str
    maximum_absolute_error: float | None = Field(ge=0, allow_inf_nan=False)
    maximum_bound_fraction: float | None = Field(ge=0, allow_inf_nan=False)
    failure: str | None = None


class Sample(Record):
    pass_kind: TimingPass
    completed_seconds: Seconds
    device_seconds: Seconds | None
