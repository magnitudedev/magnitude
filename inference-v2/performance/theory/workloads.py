"""Typed formula inputs; unrelated benchmark provenance stays in the raw record."""

from pydantic import BaseModel, ConfigDict, Field

from performance.facts import AttentionGeometry, RecurrentGeometry
from performance.theory.resources import DependentPhases, Extent


class Workload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    information_domain: str = "component"
    kv_information_domain: str | None = None
    conventional_arithmetic: bool = False
    required_inputs: tuple[Extent, ...] | None = None
    required_outputs: tuple[Extent, ...] = ()
    dependent_phases: DependentPhases | None = None


class AttentionWorkload(Workload):
    histories: tuple[int, ...] = Field(min_length=1)
    query_tokens: int = Field(gt=0)
    geometry: AttentionGeometry | None = None


class RecurrentWorkload(Workload):
    batch_size: int = Field(gt=0)
    query_tokens: int = Field(gt=0)
    geometry: RecurrentGeometry | None = None


class NeuralWorkload(Workload):
    batch_size: int = Field(default=1, gt=0)
    query_tokens: int = Field(default=1, gt=0)
    mode: str = "execute"
    measured_tokens: int = Field(default=1, gt=0)
    distinct_input_tokens: int = Field(default=1, gt=0)
    distinct_experts: int | None = Field(default=None, gt=0)


class RetainedShape(BaseModel):
    identity: str
    shape: tuple[int, ...]
    element_bytes: int = Field(gt=0)


class StateWorkload(Workload):
    retained_shapes: tuple[RetainedShape, ...] | None = None
    retained_rows: int | None = Field(default=None, ge=0)
    retained_positions: int | None = Field(default=None, ge=0)
    append_tokens: int | None = Field(default=None, ge=0)
    restore_mode: str | None = None
    reconstruction_inputs: tuple[Extent, ...] | None = None
    reconstruction_operations: dict[str, float] = Field(default_factory=dict)


class Statistics(BaseModel):
    TTFT: str | None = None


class ServiceWorkload(Workload):
    output_tokens: int | None = Field(default=None, ge=0)
    rows: int = Field(default=1, gt=0)
    waves: int = Field(default=1, gt=0)
    statistics: Statistics = Field(default_factory=Statistics)


class ControlWorkload(Workload):
    eligible_prefix_tokens: int | None = Field(default=None, ge=0)
    vocabulary: int | None = Field(default=None, gt=0)
    positions: int | None = Field(default=None, ge=0)
    element_bytes: int | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, ge=0)
    rounds: int | None = Field(default=None, ge=0)
    elements: int | None = Field(default=None, ge=0)
