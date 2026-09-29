"""Model identity and execution context shared by benchmarks and formula evidence.

Engine adapters supply their existing workload recipe verbatim. This module does
not generate requests, assign model aliases, or infer equivalence from labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from statistics import median
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from ..formula import FormulaRef, Unit
from ..runtime.observation import ObservationStatus, RuntimeObservation
from .records import Record


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Model(Record):
    identity: str = Field(min_length=1)
    label: str = Field(min_length=1)


class Workload(Record):
    """A benchmark recipe, or an explicitly synthetic component fixture recipe."""

    kind: Literal["benchmark", "synthetic"]
    recipe: dict[str, JsonValue]
    realization: str = Field(min_length=1)

    @property
    def identity(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class ExecutionContext(Record):
    model: Model
    workload: Workload
    engine: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    numerical_contract: str = Field(min_length=1)
    hardware: str = Field(min_length=1)
    host: str | None = None
    implementation: str = Field(min_length=1)
    conditions: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def comparison_key(self) -> str:
        value = self.model_dump(mode="json")
        value.pop("implementation")
        value["model"] = self.model.identity
        return fingerprint(value)


class Scope(Record):
    kind: Literal["request", "prefill", "decode", "formula"]
    formula: FormulaRef | None = None
    semantics: str | None = None
    occurrence: int | None = Field(default=None, ge=0)
    coordinates: tuple[tuple[str, int], ...] = ()

    @model_validator(mode="after")
    def valid_scope(self):
        if (self.kind == "formula") != (self.formula is not None):
            raise ValueError("only formula scopes carry a formula contract")
        if self.kind == "formula" and (not self.semantics or self.occurrence is None):
            raise ValueError("formula scope requires semantics and an occurrence")
        if self.kind != "formula" and (self.semantics is not None or self.occurrence is not None):
            raise ValueError("only formula scopes carry semantics and occurrences")
        if len(dict(self.coordinates)) != len(self.coordinates):
            raise ValueError("dynamic coordinate names must be unique")
        return self


class ExternalMapping(Record):
    """Evidence-backed correspondence; never an inferred distribution of time."""

    region: str = Field(min_length=1)
    scopes: tuple[Scope, ...] = Field(min_length=1)
    relationship: Literal["equivalent", "corresponding", "enclosing"]
    revision: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    differences: tuple[str, ...] = ()

    @model_validator(mode="after")
    def qualified(self):
        if self.relationship == "equivalent" and self.differences:
            raise ValueError("equivalent mapping cannot hide contract differences")
        if self.relationship == "corresponding" and not self.differences:
            raise ValueError("corresponding mapping must identify differences")
        return self


class ObservedMetric(Record):
    name: str = Field(min_length=1)
    unit: Unit
    samples: tuple[float, ...] = Field(min_length=1)
    boundary: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    # Exact numerator for rates; source timers are not silently reinterpreted.
    counts: tuple[int, ...] = ()

    @model_validator(mode="after")
    def nonnegative(self):
        if any(not math.isfinite(value) or value < 0 for value in self.samples) or any(
            value < 0 for value in self.counts
        ):
            raise ValueError("observations and counts must be nonnegative")
        if self.counts and len(self.counts) != len(self.samples):
            raise ValueError("each rate observation requires its own count")
        return self

    @property
    def median(self) -> float:
        return median(self.samples)


class RunEvidence(Record):
    """Enclosing execution record; formula samples remain in Measurement."""

    schema_version: Literal[1] = 1
    identity: str = Field(min_length=1)
    created: datetime
    context: ExecutionContext
    scope: Scope
    protocol: dict[str, JsonValue]
    status: Literal["complete", "failed", "incomplete", "cancelled"]
    correctness: Literal["passed", "failed", "unchecked"]
    configuration: str | None = None
    measurements: tuple[str, ...] = ()
    metrics: tuple[ObservedMetric, ...] = ()
    observations: tuple[RuntimeObservation, ...] = ()
    mappings: tuple[ExternalMapping, ...] = ()
    source_runs: tuple[str, ...] = ()
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    attachments: dict[str, JsonValue] = Field(default_factory=dict)
    unavailable: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_run(self):
        if self.created.utcoffset() is None:
            raise ValueError("run timestamp requires a timezone")
        if self.status == "complete" and any(
            item.status != ObservationStatus.COMPLETE for item in self.observations
        ):
            raise ValueError("complete runs cannot contain unfinished runtime observations")
        if len(set(self.measurements)) != len(self.measurements):
            raise ValueError("a measurement is attached once per run")
        if len({(m.name, m.boundary) for m in self.metrics}) != len(self.metrics):
            raise ValueError("metric names and boundaries identify unique observations")
        return self

    @property
    def comparison_key(self) -> str:
        protocol = {k: v for k, v in self.protocol.items() if k not in ("pair_ids", "orders")}
        return fingerprint(
            (self.context.comparison_key, self.scope.model_dump(mode="json"), protocol)
        )


class PairedComparison(Record):
    """Pairs are ordered A/B observations from the same host and protocol."""

    baseline: str
    candidate: str
    metric: str
    boundary: str
    deltas: tuple[float, ...]

    @classmethod
    def from_runs(cls, baseline: RunEvidence, candidate: RunEvidence, metric: str, boundary: str):
        if baseline.comparison_key != candidate.comparison_key:
            raise ValueError("paired comparison requires matching workload, hardware, and protocol")
        if any(run.status != "complete" for run in (baseline, candidate)):
            raise ValueError("paired comparison requires complete execution")
        if baseline.context.host is None:
            raise ValueError("paired comparison requires an identified execution host")
        a = next(m for m in baseline.metrics if (m.name, m.boundary) == (metric, boundary))
        b = next(m for m in candidate.metrics if (m.name, m.boundary) == (metric, boundary))
        if a.unit != b.unit or len(a.samples) != len(b.samples):
            raise ValueError("paired observations require matching units and pair counts")
        # The caller must supply an explicit shared pair-order protocol.
        pairs = baseline.protocol.get("pair_ids")
        if (
            not isinstance(pairs, list)
            or not all(isinstance(p, str) for p in pairs)
            or len(set(pairs)) != len(pairs)
            or len(pairs) != len(a.samples)
            or candidate.protocol.get("pair_ids") != pairs
            or candidate.protocol.get("orders") != baseline.protocol.get("orders")
        ):
            raise ValueError("paired observations require recorded pair identities")
        return cls(
            baseline=baseline.identity,
            candidate=candidate.identity,
            metric=metric,
            boundary=boundary,
            deltas=tuple(x - y for x, y in zip(a.samples, b.samples, strict=True)),
        )
