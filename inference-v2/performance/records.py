"""Serializable facts; no constructors, device objects or executable configuration."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


@dataclass(frozen=True)
class Node:
    implementation: str
    source: str
    parameters: dict = field(default_factory=dict)
    children: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    # Execution composition must be declared, never inferred from a display tree.
    execution: str = "joint"

    def __post_init__(self):
        if not re.fullmatch(
            r"[A-Z0-9_.]+:[A-Z0-9_.]+:(MLX|LM|VLM|MAG):[A-Z0-9_]+", self.implementation
        ):
            raise ValueError(f"invalid implementation ID: {self.implementation}")
        if not self.source:
            raise ValueError("implementation source fingerprint is required")
        if self.execution not in ("joint", "serial", "parallel"):
            raise ValueError("invalid execution composition")
        encoded(asdict(self))

    @property
    def component(self) -> str:
        return ":".join(self.implementation.split(":")[:2])


@dataclass(frozen=True)
class Assembly:
    root: str
    nodes: dict[str, Node]
    label: str
    artifacts: dict = field(default_factory=dict)

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
                    "parameters": node.parameters,
                    "execution": node.execution,
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
        return digest({"root": self._keys(False)[self.root], "artifacts": self.artifacts})

    @property
    def revision(self) -> str:
        return self._keys(True)[self.root]

    def component_keys(self) -> dict[str, str]:
        return self._keys(True)

    def record(self) -> dict:
        return asdict(self)

    @classmethod
    def read(cls, value: dict) -> Assembly:
        return cls(**{**value, "nodes": {k: Node(**v) for k, v in value["nodes"].items()}})


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
