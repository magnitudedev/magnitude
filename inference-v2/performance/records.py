"""Serializable facts; no constructors, device objects or executable configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any

from magnitude_engine.components import ComponentId
from performance.facts import Configuration, Facts


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    binding: ComponentId
    source: str
    parameters: Facts | None = None
    children: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    execution: str = "joint"
    configuration: Configuration = field(default_factory=Configuration)

    def __post_init__(self):
        from performance.theory.catalog import parameter_type

        if not isinstance(self.binding, ComponentId):
            raise TypeError("nodes require a ComponentId from a declaration or decoded record")
        if not self.source:
            raise ValueError("implementation source fingerprint is required")
        if self.parameters is not None and not isinstance(
            self.parameters, parameter_type(self.binding.kind)
        ):
            raise TypeError("node parameters do not satisfy its contract")
        if self.execution not in ("joint", "serial", "parallel"):
            raise ValueError("invalid execution composition")

    @property
    def implementation(self) -> str:
        return str(self.binding)

    @property
    def component(self):
        return self.binding.kind

    def record(self) -> dict:
        return dict(
            implementation=self.implementation,
            source=self.source,
            parameters=self.parameters.model_dump(mode="json")
            if self.parameters is not None
            else None,
            children=self.children,
            dependencies=self.dependencies,
            execution=self.execution,
            configuration=self.configuration.model_dump(mode="json"),
        )

    @classmethod
    def read(cls, value: dict):
        from performance.theory.catalog import read_parameters

        binding = ComponentId(value["implementation"])
        parameters = value["parameters"]
        return cls(
            binding,
            value["source"],
            None if parameters is None else read_parameters(binding.kind, parameters),
            value.get("children", {}),
            value.get("dependencies", {}),
            value.get("execution", "joint"),
            Configuration.model_validate(value.get("configuration", {})),
        )


@dataclass(frozen=True)
class CompositionOrigin:
    definition: str
    scope: str
    configuration: str = "default"
    selection: str = "candidate"

    def __post_init__(self):
        if self.selection not in ("default", "candidate", "historical"):
            raise ValueError("invalid composition selection")
        if not self.definition or not self.scope or not self.configuration:
            raise ValueError("composition requires a production identity and scope")


@dataclass(frozen=True)
class Assembly:
    root: str
    nodes: dict[str, Node]
    label: str
    artifacts: dict = field(default_factory=dict)
    origin: CompositionOrigin | None = None

    def __post_init__(self):
        if self.root not in self.nodes:
            raise ValueError("assembly root is absent")
        for node in self.nodes.values():
            if any(
                p not in self.nodes for p in (*node.children.values(), *node.dependencies.values())
            ):
                raise ValueError("dangling component reference")
        self._keys(False)  # Also checks cycles in dependency composition.

    def _keys(self, revision: bool) -> dict[str, str]:
        # Canonical references preserve aliasing: two uses of one allocation differ
        # from two equal allocations, without incorporating occurrence paths.
        def rooted(root):
            indices, records, pending = {}, [], set()

            def visit(path):
                if path in pending:
                    raise ValueError("cyclic component dependencies")
                if path in indices:
                    return indices[path]
                index = indices[path] = len(records)
                records.append(None)
                pending.add(path)
                node = self.nodes[path]
                records[index] = {
                    "implementation": node.implementation,
                    "parameters": node.parameters.model_dump(mode="json")
                    if node.parameters is not None
                    else None,
                    "execution": node.execution,
                    "configuration": node.configuration.model_dump(mode="json"),
                    "source": node.source if revision else None,
                    "children": {k: visit(v) for k, v in sorted(node.children.items())},
                    "dependencies": {k: visit(v) for k, v in sorted(node.dependencies.items())},
                }
                pending.remove(path)
                return index

            visit(root)
            return digest(records)

        return {path: rooted(path) for path in self.nodes}

    @property
    def identity(self) -> str:
        if self.origin is None:
            # Standalone components have a stable contract entry, with variants in history.
            return digest(
                {"component": self.nodes[self.root].component, "artifacts": self.artifacts}
            )
        return digest(
            {
                "definition": self.origin.definition,
                "scope": self.origin.scope,
                "configuration": self.origin.configuration,
                "artifacts": self.artifacts,
            }
        )

    @cached_property
    def revision(self) -> str:
        return self._component_keys[self.root]

    @cached_property
    def _component_keys(self) -> dict[str, str]:
        return self._keys(True)

    def component_keys(self) -> dict[str, str]:
        return self._component_keys

    def record(self) -> dict:
        return dict(
            root=self.root,
            nodes={k: v.record() for k, v in self.nodes.items()},
            label=self.label,
            artifacts=self.artifacts,
            origin=asdict(self.origin) if self.origin else None,
        )

    @classmethod
    def read(cls, value: dict) -> Assembly:
        return cls(
            value["root"],
            {k: Node.read(v) for k, v in value["nodes"].items()},
            value["label"],
            value.get("artifacts", {}),
            CompositionOrigin(**value["origin"]) if value.get("origin") else None,
        )


@dataclass(frozen=True)
class Profile:
    hardware: dict
    runtime: dict
    capacities: dict[str, float] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, value in self.capacities.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid capacity {name}")
        encoded(asdict(self))

    @property
    def identity(self) -> str:
        return digest(asdict(self))

    def record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Observation:
    output_digest: str = ""
    counters: dict[str, int | float] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    # Dimension metrics are explicit; generic counters are not performance scores.
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.metrics.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid observed metric {key}")
        encoded(asdict(self))
