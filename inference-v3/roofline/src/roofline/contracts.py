"""Versioned wire and durable evidence, independent of an engine installation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import uuid4

from formula_performance.records import Capacity
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


def encoded(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def identity() -> str:
    return uuid4().hex


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"expected a relative contained path: {value!r}")
    if path.as_posix() != value or value == ".":
        raise ValueError(f"noncanonical path: {value!r}")
    return value


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Model(Record):
    sha256: Hash
    locations: dict[str, str]

    @model_validator(mode="after")
    def absolute_locations(self):
        for target, path in self.locations.items():
            if not PurePosixPath(path).is_absolute():
                raise ValueError(f"model location for {target} must be an absolute file path")
        return self


class Connection(Record):
    kind: Literal["local", "ssh"]
    host: str | None = None

    @model_validator(mode="after")
    def host_required(self):
        if (self.kind == "ssh") != (self.host is not None):
            raise ValueError("only SSH connections require a host")
        if self.host and (self.host.startswith("-") or any(c.isspace() for c in self.host)):
            raise ValueError("invalid SSH host")
        return self


class Device(Record):
    backend: Literal["metal", "cuda", "cpu"]
    index: int = Field(default=0, ge=0)
    maximum_bytes: int | None = Field(default=None, gt=0)


class Target(Record):
    connection: Connection
    device: Device
    worker_root: str | None = None
    capacities: tuple[Capacity, ...] = ()

    @model_validator(mode="after")
    def worker_directory(self):
        if self.worker_root and not (
            PurePosixPath(self.worker_root).is_absolute() or self.worker_root.startswith("~/")
        ):
            raise ValueError("worker_root must be absolute or start with ~/")
        return self


class Models(Record):
    version: Literal[1] = 1
    models: dict[str, Model]

    @model_validator(mode="after")
    def valid_ids(self):
        for key in self.models:
            parts = key.split(":")
            if len(parts) != 3 or any(not s or any(c.isspace() for c in s) for s in parts):
                raise ValueError(f"model ID must be model:format:quantization: {key}")
        return self


class Targets(Record):
    version: Literal[1] = 1
    targets: dict[str, Target]


class Protocol(Record):
    samples: int = Field(default=3, ge=1, le=100)
    warmups: int = Field(default=1, ge=0, le=10)
    deadline_seconds: int = Field(default=3600, ge=1, le=86400)


class Experiment(Record):
    model: str
    workload: Literal["prose"] = "prose"
    context: int = Field(default=2048, ge=1)
    steps: int = Field(default=128, ge=1, le=4096)
    scope: str = "decode"
    step: int | None = Field(default=None, ge=0)
    targets: tuple[str, ...] = ("local",)
    engine: Literal["magnitude", "llama.cpp"] = "magnitude"
    source: str | None = None
    against_source: str | None = None
    input_source: str | None = None
    input_target: str | None = None
    protocol: Protocol = Protocol()

    @model_validator(mode="after")
    def valid_selection(self):
        if self.scope.split("/", 1)[0] not in ("prefill", "decode"):
            raise ValueError("scope must start with prefill or decode")
        if self.step is not None and (
            self.step >= self.steps or not self.scope.startswith("decode/")
        ):
            raise ValueError("--step selects a component occurrence within the decode workload")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("select unique, nonempty targets")
        if (self.input_source is None) != (self.input_target is None):
            raise ValueError("shared inputs require both --input-source and --input-target")
        if self.input_source is not None and (
            self.engine != "magnitude"
            or not self.scope.startswith("decode/")
            or self.step is None
            or self.against_source is not None
        ):
            raise ValueError(
                "shared inputs require one Magnitude decode component --step without source pairing"
            )
        return self


class SourceFile(Record):
    path: str
    blob: Hash
    executable: bool = False

    @model_validator(mode="after")
    def contained(self):
        safe_path(self.path)
        return self


class Source(Record):
    version: Literal[1] = 1
    files: tuple[SourceFile, ...]

    @property
    def source_id(self):
        return digest(self)


class Attempt(Record):
    attempt_id: str
    target_name: str
    target: Target
    role: Literal["measure", "inputs"] = "measure"
    status: Literal[
        "queued", "accepted", "running", "complete", "failed", "cancelled", "unavailable"
    ] = "queued"
    dispatched: bool = False
    measurement_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    error: str | None = None


class Request(Record):
    version: Literal[1] = 1
    request_id: str
    created: str
    experiment: Experiment
    model: Model | None
    operation: Literal["measure", "characterize", "prepare-inputs"] = "measure"
    attempts: tuple[Attempt, ...]
    cancelled: bool = False
    inputs: dict[str, Hash] = Field(default_factory=dict)


class Scope(Record):
    selector: str
    parent: str | None = None
    contract: str
    semantics: str | None = None
    occurrence: int | None = None
    complete: bool = True
    reason: str | None = None
    contribution_seconds: float | None = Field(default=None, ge=0)


class Measurement(Record):
    version: Literal[1] = 1
    measurement_id: str
    request_id: str
    attempt_id: str
    created: str
    model: str
    artifact: Hash
    source_id: Hash | None
    target: str
    engine: str
    workload: dict[str, JsonValue]
    scope: str
    hardware: dict[str, JsonValue]
    protocol: dict[str, JsonValue]
    status: Literal["complete", "failed", "incomplete", "unavailable"]
    correctness: Literal["passed", "failed", "unchecked"]
    samples_seconds: tuple[float, ...] = ()
    scopes: tuple[Scope, ...] = ()
    artifacts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    costs: dict[str, float] = Field(default_factory=dict)
    unavailable: tuple[str, ...] = ()
    error: str | None = None
    pair_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def qualified_samples(self):
        if any(s < 0 for s in self.samples_seconds):
            raise ValueError("negative sample duration")
        if self.status == "complete" and not self.samples_seconds:
            raise ValueError("complete measurements require samples")
        if self.pair_ids and len(self.pair_ids) != len(self.samples_seconds):
            raise ValueError("each paired sample needs an identity")
        return self

    @property
    def comparison_key(self):
        return digest(
            {
                "component_contract": self.details.get("comparison")
                or (
                    {"unknown_inputs": self.measurement_id}
                    if self.scope.startswith(("decode/", "prefill/"))
                    else None
                ),
                **{
                    k: getattr(self, k)
                    for k in (
                        "model",
                        "artifact",
                        "engine",
                        "workload",
                        "scope",
                        "hardware",
                        "protocol",
                    )
                },
            }
        )
