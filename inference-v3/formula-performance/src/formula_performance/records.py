"""Immutable analytical contracts; importing them never imports an execution runtime."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def identity(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class Expression(Record):
    """A finite arithmetic expression, never executable archived Python."""

    op: Literal["constant", "parameter", "add", "multiply", "divide", "maximum", "minimum"]
    value: float | None = None
    name: str | None = None
    arguments: tuple[Expression, ...] = ()

    @model_validator(mode="after")
    def well_formed(self):
        if self.op == "constant":
            valid = self.value is not None and self.name is None and not self.arguments
        elif self.op == "parameter":
            valid = bool(self.name) and self.value is None and not self.arguments
        else:
            valid = self.value is None and self.name is None and len(self.arguments) >= 2
            if self.op == "divide":
                valid = valid and len(self.arguments) == 2
        if not valid:
            raise ValueError("invalid performance expression")
        return self

    @classmethod
    def constant(cls, value):
        return cls(op="constant", value=value)

    @classmethod
    def parameter(cls, name):
        return cls(op="parameter", name=name)

    def evaluate(self, bindings: dict[str, float]) -> float:
        if self.op == "constant":
            result = self.value
        elif self.op == "parameter":
            result = bindings[self.name]
        else:
            terms = [arg.evaluate(bindings) for arg in self.arguments]
            if self.op == "add":
                result = sum(terms)
            elif self.op == "multiply":
                result = math.prod(terms)
            elif self.op == "divide":
                result = terms[0] / terms[1]
            else:
                result = (max if self.op == "maximum" else min)(terms)
        if result is None or not math.isfinite(result):
            raise ValueError("non-finite performance expression")
        return result


class Unit(Record):
    name: str = Field(min_length=1)
    dimension: str = Field(min_length=1)


class Quantity(Record):
    name: str = Field(min_length=1)
    unit: Unit
    expression: Expression
    meaning: str = Field(min_length=1)


class Obligation(Record):
    """A unique necessary demand in a declared domain; alternatives are capacities."""

    identity: str
    origins: tuple[str, ...] = Field(min_length=1)
    amount: Expression
    unit: Unit
    resource: str
    mappings: tuple[str, ...] = Field(min_length=1)
    rule: str
    conditions: tuple[str, ...] = ()
    kind: Literal["necessary", "conditional"] = "necessary"
    boundary: str | None = None


class Formula(Record):
    occurrence: int | None = None
    component: str
    parent: str | None
    definition: str
    version: int = Field(ge=1)
    semantics: str
    primary: Quantity
    quantities: tuple[Quantity, ...] = ()
    nodes: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    label: str


class SerialStages(Record):
    """A declared unavoidable barrier, stronger than an ordinary tensor dependency."""

    identity: str
    component: str
    stages: tuple[tuple[str, ...], ...] = Field(min_length=2)
    rule: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()


class Manifest(Record):
    version: Literal[1] = 1
    graph: str
    formulas: tuple[Formula, ...]
    obligations: tuple[Obligation, ...]
    parameters: dict[str, float]
    conditions: tuple[str, ...] = ()
    unresolved: dict[str, str] = Field(default_factory=dict)
    numerical_graph: dict[str, JsonValue] = Field(default_factory=dict)
    serial_stages: tuple[SerialStages, ...] = ()

    @model_validator(mode="after")
    def references(self):
        components = {f.component for f in self.formulas}
        if len(components) != len(self.formulas):
            raise ValueError("duplicate component in a realization")
        ids = [o.identity for o in self.obligations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate obligation identity")
        for serial in self.serial_stages:
            ids_in_stages = [item for stage in serial.stages for item in stage]
            if serial.component not in components or not set(ids_in_stages) <= set(ids):
                raise ValueError("serial certificate references unknown component or demand")
            if len(ids_in_stages) != len(set(ids_in_stages)):
                raise ValueError("serial certificate counts the same obligation twice")
            if any(not stage for stage in serial.stages):
                raise ValueError("empty serial stage")
        nodes = {n for f in self.formulas for n in f.nodes}
        for obligation in self.obligations:
            if not set(obligation.origins) <= nodes:
                raise ValueError("obligation has unknown numerical origins")
            try:
                amount = obligation.amount.evaluate(self.parameters)
            except KeyError as error:
                if error.args[0] not in self.unresolved:
                    raise ValueError("unbound obligation without an explanation") from error
            else:
                if amount < 0:
                    raise ValueError("negative obligation")
        by_id = {f.component: f for f in self.formulas}
        for formula in self.formulas:
            if formula.parent is not None and formula.parent not in components:
                raise ValueError("unknown formula parent")
            if not set(formula.dependencies) <= components:
                raise ValueError("unknown formula dependency")
            if formula.primary.expression.evaluate(self.parameters) < 0:
                raise ValueError("negative useful quantity")
            seen = {formula.component}
            parent = formula.parent
            while parent is not None:
                if parent in seen:
                    raise ValueError("cyclic formula containment")
                seen.add(parent)
                parent = by_id[parent].parent
            if formula.parent is not None and not set(formula.nodes) <= set(
                by_id[formula.parent].nodes
            ):
                raise ValueError("child numerical work lies outside parent")
        return self


class Capacity(Record):
    parameter: str
    pool: str
    unit: Unit
    value: float = Field(gt=0)
    kind: Literal["upper-bound", "conditional-upper-bound", "achieved"]
    provenance: str = Field(min_length=1)
    conditions: tuple[str, ...] = ()


class Hardware(Record):
    identity: str
    label: str
    facts: dict[str, JsonValue] = Field(default_factory=dict)
    capacities: tuple[Capacity, ...] = ()

    @model_validator(mode="after")
    def distinct_parameters(self):
        keys = [c.parameter for c in self.capacities]
        if len(set(keys)) != len(keys):
            raise ValueError("ambiguous hardware capacity parameter")
        return self


class Region(Record):
    identity: str
    owners: tuple[str, ...] = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    dependencies: tuple[str, ...] = ()

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("negative execution interval")
        return self


class Capture(Record):
    component: str = ""
    identity: str
    manifest: str
    clock: str
    regions: tuple[Region, ...]
    coverage: Literal["complete", "partial"]
    # Host observations without a native-clock correspondence are separate captures.
    boundary_seconds: float | None = Field(default=None, ge=0)
    execution_graph: str | None = None

    @model_validator(mode="after")
    def distinct_events(self):
        by_id = {r.identity: r for r in self.regions}
        if len(by_id) != len(self.regions):
            raise ValueError("physical event is recorded twice")
        for r in self.regions:
            if not set(r.dependencies) <= by_id.keys():
                raise ValueError("unknown execution dependency")
            if any(by_id[d].end > r.start for d in r.dependencies):
                raise ValueError("execution dependency overlaps its successor")
            pending, seen = list(r.dependencies), set()
            while pending:
                predecessor = pending.pop()
                if predecessor == r.identity:
                    raise ValueError("cyclic execution dependency")
                if predecessor not in seen:
                    seen.add(predecessor)
                    pending.extend(by_id[predecessor].dependencies)
        return self


class Observation(Record):
    identity: str
    manifest: str
    component: str
    hardware: str
    implementation: str
    created: str
    coordinates: dict[str, JsonValue]
    samples: tuple[float, ...]
    boundary: str
    correctness: Literal["passed", "failed", "unchecked"]
    status: Literal["complete", "failed", "incomplete", "unavailable"]
    captures: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    @model_validator(mode="after")
    def durations(self):
        if any(s < 0 for s in self.samples):
            raise ValueError("negative observation duration")
        if self.status == "complete" and not self.samples:
            raise ValueError("complete observation needs samples")
        return self


class Transfer(Record):
    """Explicit measured/conditional relation between isolated and parent regions."""

    identity: str
    child: str
    parent: str
    baseline_child: str
    baseline_parent: str
    # The actual time attributable to the child in this parent, in parent's domain.
    contribution: float = Field(ge=0)
    scale: float = Field(gt=0, default=1)
    relationship: Literal["same-execution", "conditional-serial"]
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...] = Field(min_length=1)


class Publication(Record):
    version: Literal[1] = 1
    manifests: tuple[Manifest, ...] = ()
    hardware: tuple[Hardware, ...] = ()
    observations: tuple[Observation, ...] = ()
    captures: tuple[Capture, ...] = ()
    transfers: tuple[Transfer, ...] = ()

    @model_validator(mode="after")
    def closure(self):
        manifests = {identity(m): m for m in self.manifests}
        hardware = {identity(h) for h in self.hardware}
        captures = {c.identity: c for c in self.captures}
        if len(captures) != len(self.captures):
            raise ValueError("duplicate capture")
        for capture in self.captures:
            if capture.manifest not in manifests:
                raise ValueError("capture manifest is missing")
            formulas = {f.component: f for f in manifests[capture.manifest].formulas}
            if capture.component not in formulas:
                raise ValueError("capture boundary is not a formula")
            known = set()
            for component in formulas:
                ancestor = component
                while ancestor is not None:
                    if ancestor == capture.component:
                        known.add(component)
                        break
                    ancestor = formulas[ancestor].parent
            if any(not set(r.owners) <= known for r in capture.regions):
                raise ValueError("capture references unknown formula")
        ids = set()
        for observation in self.observations:
            if observation.identity in ids:
                raise ValueError("duplicate observation")
            ids.add(observation.identity)
            if observation.manifest not in manifests or observation.hardware not in hardware:
                raise ValueError("observation needs graph and hardware binding")
            if observation.component not in {
                f.component for f in manifests[observation.manifest].formulas
            }:
                raise ValueError("observation needs a declared formula boundary")
            if not set(observation.captures) <= captures.keys():
                raise ValueError("observation capture is missing")
            if any(captures[c].manifest != observation.manifest for c in observation.captures):
                raise ValueError("observation/capture graph mismatch")
        # Transfers may reference earlier publications; the evaluator resolves the closure.
        return self
