"""Versioned, validated evidence contracts shared by Lab, history and the TUI."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_serializer, model_validator

from ..binding import SourceInfo
from ..compiler.dependencies import CodeDependency
from ..formula import FormulaRef, Unit, units
from ..performance.resources import Resource
from ..runtime.observation import Activity, ObservationStatus, RuntimeObservation
from ..tensor.types import DType


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class CacheCondition(StrEnum):
    RESIDENT = "resident"
    SOURCE_CACHE_UNCONTROLLED = "source-cache-uncontrolled"
    SOURCE_CACHE_WARM = "source-cache-warm"
    SOURCE_CACHE_COLD = "source-cache-cold"


class MeasurementProtocol(Record):
    version: Literal[1, 2] = 2
    boundary: Literal["complete-operation"] = "complete-operation"
    samples: int = Field(default=3, ge=1, le=100)
    warmups: int = Field(default=1, ge=0, le=10)
    minimum_warmup_seconds: float = Field(default=0, ge=0, le=5)
    # Included in series identity; changing a numerical check is not an invisible
    # way to turn an incorrect optimization into a historical improvement.
    absolute_tolerance: float = Field(default=0, ge=0)
    relative_tolerance: float = Field(default=0, ge=0)
    kernel_limit: int | None = Field(default=1024, ge=1, le=65536)
    measure_invalid: bool = False
    inputs: Literal["reference", "production"] = "reference"

    @model_validator(mode="before")
    @classmethod
    def historical_host_protocol(cls, value):
        # Retained measurements must keep their original protocol/key. Version 1
        # never requested native instrumentation; do not silently add it on read.
        if isinstance(value, dict) and value.get("version") == 1:
            if value.get("kernel_limit") is not None:
                raise ValueError("protocol version 1 does not contain native kernel timing")
            return {**value, "kernel_limit": None}
        return value

    @model_serializer(mode="wrap")
    def serialized_protocol(self, handler):
        value = handler(self)
        # No extra conditioning preserves the established ordinary series key.
        if not self.minimum_warmup_seconds:
            value.pop("minimum_warmup_seconds", None)
        if not self.measure_invalid:
            value.pop("measure_invalid", None)
        if self.inputs == "reference":
            value.pop("inputs", None)
        if self.version == 1:
            value.pop("kernel_limit", None)
        return value


class SourceCondition(Record):
    binding: str = Field(min_length=1)
    source: SourceInfo
    cache: CacheCondition
    # A cold label requires a real conditioning mechanism and its version.
    conditioning: str | None = None

    @model_validator(mode="after")
    def explicit_conditioning(self):
        if self.cache in (CacheCondition.SOURCE_CACHE_COLD, CacheCondition.SOURCE_CACHE_WARM) and not self.conditioning:
            raise ValueError("controlled-source measurement requires a declared conditioning mechanism")
        return self


class Series(Record):
    schema_version: Literal[1] = 1
    formula: FormulaRef
    semantics: str = Field(min_length=1)
    fixture: str = Field(min_length=1)
    device: str = Field(min_length=1)
    geometry: str = Field(min_length=1)
    precision: str = Field(min_length=1)
    sources: tuple[SourceCondition, ...] = ()
    protocol: MeasurementProtocol = MeasurementProtocol()

    @property
    def identity(self) -> str:
        # No operation source hash, kernel decomposition or display path here.
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class Implementation(Record):
    fingerprint: str = Field(min_length=1)
    compiler: str = Field(min_length=1)
    dependencies: tuple[str, ...]
    authored: tuple[CodeDependency, ...] = ()


class Phase(StrEnum):
    REFRESH = "refresh"
    PREPARE = "prepare"
    CHARACTERIZE = "characterize"
    COMPILE = "compile"
    CHECK = "check"
    CONDITION = "condition"
    SAMPLE = "sample"
    PUBLISH = "publish"


class PhaseTime(Record):
    phase: Phase
    elapsed_ns: int = Field(ge=0)


class UsefulQuantity(Record):
    name: str = Field(min_length=1)
    amount: float = Field(ge=0)
    unit: Unit
    # E.g. conventional contraction FLOPs, not a claim that every possible
    # algorithm must perform this many hardware instructions.
    basis: str = Field(min_length=1)


class CeilingKind(StrEnum):
    THEORETICAL = "theoretical"
    EMPIRICAL = "empirical"


class ResourceDemand(Record):
    resource: Resource
    dtype: DType
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)
    unit: Unit
    basis: str
    source: SourceInfo | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.lower > self.upper:
            raise ValueError("resource demand bounds are reversed")
        unit = {Resource.MATRIX_ARITHMETIC: units.flop, Resource.VECTOR_ARITHMETIC: units.flop,
                Resource.INTEGER_ARITHMETIC: units.integer_op, Resource.SPECIAL_FUNCTIONS: units.special_op,
                Resource.COMPARISONS: units.comparison, Resource.EXECUTION_COPY: units.byte,
                Resource.SOURCE_IMPORT: units.byte}[self.resource]
        if self.unit != unit:
            raise ValueError("resource demand has incompatible units")
        if self.source is not None and self.resource != Resource.SOURCE_IMPORT:
            raise ValueError("source provenance belongs to source-path demand")
        return self


class ResourceLimit(Record):
    demand: ResourceDemand
    rate: float = Field(gt=0)
    rate_dtype: DType
    measurement: str
    assumptions: tuple[str, ...]

    @model_validator(mode="after")
    def compatible_precision(self):
        if self.demand.resource not in {Resource.EXECUTION_COPY, Resource.SOURCE_IMPORT} and self.rate_dtype != self.demand.dtype and not (
            self.demand.resource == Resource.MATRIX_ARITHMETIC and
            self.demand.dtype == DType.BF16 and self.rate_dtype == DType.F32
        ):
            raise ValueError("resource reference uses an incompatible numerical precision")
        return self

    @property
    def lower_seconds(self) -> float:
        return self.demand.lower / self.rate

    @property
    def upper_seconds(self) -> float:
        return self.demand.upper / self.rate


class Roofline(Record):
    revision: str = Field(min_length=1)
    characterization: str = Field(min_length=1)
    limits: tuple[ResourceLimit, ...]
    assumptions: tuple[str, ...]

    @property
    def seconds(self) -> float:
        # Precision variants share a resource pool. Independent constraints
        # overlap ideally; do not impose kernel/child publication barriers.
        pools = {}
        for limit in self.limits:
            resource = (limit.demand.resource, limit.demand.source)
            pools[resource] = pools.get(resource, 0) + limit.lower_seconds
        return max(pools.values(), default=0)

    @property
    def bottleneck(self) -> str:
        pools = {}
        for limit in self.limits:
            resource = (limit.demand.resource, limit.demand.source)
            pools[resource] = pools.get(resource, 0) + limit.lower_seconds
        return max(pools, key=pools.get)[0].value if pools else "no-device-work"


class Direction(StrEnum):
    HIGHER = "higher-is-better"
    LOWER = "lower-is-better"


class Metric(Record):
    name: str = Field(min_length=1)
    value: float = Field(ge=0)
    unit: Unit
    basis: str = Field(min_length=1)


class UnavailableMetric(Record):
    name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class Ceiling(Record):
    metric: str = Field(min_length=1)
    unit: Unit
    value: float = Field(ge=0)
    direction: Direction
    kind: CeilingKind
    resource: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    assumptions: tuple[str, ...]
    provenance: str = Field(min_length=1)


class Outcome(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Measurement(Record):
    schema_version: Literal[1] = 1
    identity: str = Field(min_length=1)
    created: datetime
    series: Series
    implementation: Implementation | None
    outcome: Outcome
    checked: bool
    samples: tuple[RuntimeObservation, ...] = ()
    quantities: tuple[UsefulQuantity, ...] = ()
    ceilings: tuple[Ceiling, ...] = ()
    phases: tuple[PhaseTime, ...] = ()
    error: str | None = None
    unavailable: tuple[UnavailableMetric, ...] = ()
    roofline: Roofline | None = None
    preparation: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_evidence(self):
        if self.created.utcoffset() is None:
            raise ValueError("measurement timestamp must identify its timezone")
        if self.outcome == Outcome.COMPLETE:
            if self.implementation is None:
                raise ValueError("successful measurements require executable provenance")
            if not self.checked or len(self.samples) != self.series.protocol.samples:
                raise ValueError("successful measurements require checking and the declared samples")
            if any(sample.status != ObservationStatus.COMPLETE for sample in self.samples):
                raise ValueError("unfinished runtime observations cannot qualify a measurement")
            if self.error is not None:
                raise ValueError("successful measurements cannot contain an error")
            if len({sample.kernels is None for sample in self.samples}) != 1:
                raise ValueError("native kernel timing cannot be partially available across samples")
            if len({sample.kernels.clock for sample in self.samples if sample.kernels is not None}) > 1:
                raise ValueError("native samples must use the same clock method")
            if any(sample.kernels is not None and (
                self.series.protocol.kernel_limit is None or
                len(sample.kernels.activities) > self.series.protocol.kernel_limit
            ) for sample in self.samples):
                raise ValueError("native samples exceed the declared measurement protocol")
        if len({item.name for item in self.quantities}) != len(self.quantities):
            raise ValueError("useful quantity names must be unique")
        if len({item.phase for item in self.phases}) != len(self.phases):
            raise ValueError("phase durations are accumulated once per phase")
        metrics = {item.name: item for item in self.metrics}
        for ceiling in self.ceilings:
            if self.outcome == Outcome.COMPLETE and (
                ceiling.metric not in metrics or metrics[ceiling.metric].unit != ceiling.unit
            ):
                raise ValueError("ceiling must refer to a recorded metric with matching units")
        return self

    @property
    def median_seconds(self) -> float | None:
        if self.outcome != Outcome.COMPLETE:
            return None
        return median(sample.elapsed_ns for sample in self.samples) / 1_000_000_000

    @property
    def observed_seconds(self) -> float | None:
        """Timing evidence can survive a numerical failure without qualifying it."""
        if len(self.samples) != self.series.protocol.samples or any(
            s.status != ObservationStatus.COMPLETE for s in self.samples
        ):
            return None
        return median(sample.elapsed_ns for sample in self.samples) / 1_000_000_000

    @property
    def work_seconds(self) -> float:
        """Reported phases, not a substitute for request-to-visible turnaround."""
        return sum(phase.elapsed_ns for phase in self.phases) / 1_000_000_000

    @property
    def metrics(self) -> tuple[Metric, ...]:
        duration = self.median_seconds
        if duration is None:
            return ()
        values = [Metric(name="elapsed", value=duration, unit=units.second,
                         basis="median complete-operation wall duration")]
        kernel_seconds = None
        busy = [s.kernels.busy_ns for s in self.samples if s.kernels is not None]
        if len(busy) == len(self.samples) and all(value is not None for value in busy):
            values.append(Metric(name="kernel-busy-time", value=median(v for v in busy if v is not None) / 1e9,
                                 unit=units.second, basis="median union of native intervals per sample"))
        native = tuple(sample.kernels for sample in self.samples if sample.kernels is not None)
        if len(native) == len(self.samples):
            kernel_seconds = median(sample.elapsed_ns for sample in native) / 1e9
            values.append(Metric(name="kernel-device-time", value=kernel_seconds, unit=units.second,
                                 basis="median sum of native compute-pass durations; excludes host I/O and submission gaps"))
            values.append(Metric(name="kernel-count",
                                 value=median(len(sample.activities) for sample in native),
                                 unit=units.kernel,
                                 basis="median native dispatches observed inside the operation boundary"))
        for name, amounts, basis in (
            ("reserved-baseline", (sample.memory.baseline_bytes for sample in self.samples),
             "median runtime reservation baseline; includes other prepared resources"),
            ("reserved-peak", (sample.memory.peak_bytes for sample in self.samples),
             "median peak unique reserved backing in the runtime during invocation"),
            ("reserved-increase", (sample.memory.peak_bytes - sample.memory.baseline_bytes for sample in self.samples),
             "median reservation increase above each invocation's baseline"),
        ):
            values.append(Metric(name=name, value=median(amounts), unit=units.byte, basis=basis))
        for activity in (Activity.SOURCE_READ, Activity.UPLOAD, Activity.DOWNLOAD):
            amount = median(sample.completed_bytes(activity) for sample in self.samples)
            values.append(Metric(
                name=f"{activity.value}-bytes",
                value=amount,
                unit=units.byte, basis="median bytes crossing the API; not hardware traffic",
            ))
            if duration > 0:
                values.append(Metric(name=f"{activity.value}-rate", value=amount / duration,
                                     unit=Unit("byte/s", "storage/time"),
                                     basis="API bytes divided by complete-operation wall time, not isolated bus bandwidth"))
        for activity, unit in ((Activity.ALLOCATE, units.allocation), (Activity.SUBMIT, units.submission)):
            values.append(Metric(
                name=f"{activity.value}-count", unit=unit,
                value=median(sum(item.kind == activity and item.error is None for item in sample.activities)
                             for sample in self.samples),
                basis="median successful runtime API calls; a native submission can contain multiple kernels",
            ))
        for name, attribute in (("reserved-during", "reserved_bytes"), ("released-during", "released_bytes")):
            values.append(Metric(name=name, value=median(getattr(sample.memory, attribute) for sample in self.samples),
                                 unit=units.byte, basis="total reservation traffic during the complete invocation, not peak backing"))
        for quantity in self.quantities:
            values.append(Metric(name=f"useful:{quantity.name}", value=quantity.amount,
                                 unit=quantity.unit, basis=quantity.basis))
            if duration > 0:
                values.append(Metric(
                    name=f"rate:{quantity.name}", value=quantity.amount / duration,
                    unit=Unit(f"{quantity.unit.name}/s", f"{quantity.unit.dimension}/time"),
                    basis=f"{quantity.basis}; divided by median complete-operation wall duration",
                ))
            if kernel_seconds is not None and kernel_seconds > 0:
                values.append(Metric(
                    name=f"kernel-rate:{quantity.name}", value=quantity.amount / kernel_seconds,
                    unit=Unit(f"{quantity.unit.name}/s", f"{quantity.unit.dimension}/time"),
                    basis=f"{quantity.basis}; divided by median summed native compute-pass time, not operation wall time",
                ))
        if self.roofline is not None:
            values.append(Metric(name="modeled-time-floor", value=self.roofline.seconds,
                                 unit=units.second, basis="formula demands / empirical resource references; ideal overlap and reuse"))
            for limit in self.roofline.limits:
                demand = limit.demand
                name = f"{demand.resource.value}:{demand.dtype.value}"
                if demand.source is not None:
                    name += f":{demand.source.fingerprint}"
                values.append(Metric(name=f"demand:{name}", value=demand.lower, unit=demand.unit,
                                     basis=demand.basis + (f"; obligation range {demand.lower:g}–{demand.upper:g}"
                                                         if demand.lower != demand.upper else "")))
                if duration > 0:
                    values.append(Metric(name=f"throughput:{name}", value=demand.lower / duration,
                                         unit=Unit(f"{demand.unit.name}/s", f"{demand.unit.dimension}/time"),
                                         basis="useful resource demand / measured complete-operation wall time"))
        return tuple(values)


class History(Record):
    series: Series
    latest: Measurement | None
    latest_success: Measurement | None
    best: Measurement | None
    observations: tuple[Measurement, ...]

    def stale(self, implementation: Implementation) -> bool:
        return (self.latest_success is None or
                self.latest_success.implementation != implementation)


class JobResult(Record):
    """Request outcome, including failures before a formula series can be prepared.

    Queue and publication time are not folded into operation latency. A client
    may separately acknowledge visibility; worker completion is not UI rendering.
    """

    schema_version: Literal[1] = 1
    identity: str = Field(min_length=1)
    requested: datetime
    finished: datetime
    formula: FormulaRef
    semantics: str = Field(min_length=1)
    outcome: Outcome
    measurement: str | None = None
    queue_ns: int = Field(ge=0)
    active_ns: int = Field(ge=0)
    publication_ns: int = Field(ge=0)
    phases: tuple[PhaseTime, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def valid_result(self):
        if self.requested.utcoffset() is None or self.finished.utcoffset() is None:
            raise ValueError("job timestamps must identify their timezone")
        if self.outcome == Outcome.COMPLETE and (self.measurement is None or self.error is not None):
            raise ValueError("completed jobs require a published measurement and no error")
        return self


class JobStart(Record):
    identity: str = Field(min_length=1)
    worker: str = Field(min_length=1)
    requested: datetime
    formula: FormulaRef
    semantics: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_request(self):
        if self.requested.utcoffset() is None:
            raise ValueError("job request timestamp must identify its timezone")
        return self


class Visibility(Record):
    job: str = Field(min_length=1)
    client: str = Field(min_length=1)
    request_to_visible_ns: int = Field(ge=0)
    completed_to_visible_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_interval(self):
        if self.completed_to_visible_ns > self.request_to_visible_ns:
            raise ValueError("visibility delay cannot exceed the complete request interval")
        return self
