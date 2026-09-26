"""Optimistic resource bounds and identity-aware composition of required information."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from performance.records import Profile


@dataclass(frozen=True, order=True)
class Extent:
    identity: str
    start: int
    end: int

    def __post_init__(self):
        if not self.identity or self.start < 0 or self.end < self.start:
            raise ValueError("invalid information extent")


def union(extents: tuple[Extent, ...]) -> tuple[Extent, ...]:
    result: list[Extent] = []
    for extent in sorted(extents):
        if result and result[-1].identity == extent.identity and extent.start <= result[-1].end:
            old = result.pop()
            result.append(Extent(old.identity, old.start, max(old.end, extent.end)))
        else:
            result.append(extent)
    return tuple(result)


def subtract(values: tuple[Extent, ...], removed: tuple[Extent, ...]) -> tuple[Extent, ...]:
    result = []
    for value in union(values):
        pieces = [(value.start, value.end)]
        for cut in union(removed):
            if cut.identity != value.identity:
                continue
            next_pieces = []
            for start, end in pieces:
                if cut.end <= start or cut.start >= end:
                    next_pieces.append((start, end))
                else:
                    if start < cut.start:
                        next_pieces.append((start, cut.start))
                    if end > cut.end:
                        next_pieces.append((cut.end, end))
            pieces = next_pieces
        result.extend(Extent(value.identity, start, end) for start, end in pieces)
    return union(tuple(result))


@dataclass(frozen=True)
class Demands:
    inputs: tuple[Extent, ...] = ()
    outputs: tuple[Extent, ...] = ()
    operations: dict[str, float] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in self.operations.values()):
            raise ValueError("resource demands must be finite and nonnegative")


def join(*children: Demands, retained: tuple[Extent, ...] = ()) -> Demands:
    """Internal producer/consumer information need not cross the parent boundary."""
    inputs = union(tuple(x for c in children for x in c.inputs))
    outputs = union(tuple(x for c in children for x in c.outputs))
    operations: dict[str, float] = {}
    for child in children:
        for resource, count in child.operations.items():
            operations[resource] = operations.get(resource, 0) + count
    return Demands(
        subtract(inputs, outputs),
        union((*subtract(outputs, inputs), *retained)),
        operations,
        tuple(sorted({a for c in children for a in c.assumptions})),
        tuple(sorted({m for c in children for m in c.missing})),
    )


@dataclass(frozen=True)
class Bound:
    value: float | None
    unit: str
    direction: str = "lower"
    terms: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    kind: str | None = None
    derivation: str = "CONTRACT:RELAXATION"

    def __post_init__(self):
        if self.kind is None:
            object.__setattr__(
                self,
                "kind",
                "missing" if self.value is None else "zero" if self.value == 0 else "bounded",
            )
        if self.kind not in ("bounded", "missing", "zero", "unbounded"):
            raise ValueError("invalid bound kind")
        if self.value is not None and (not math.isfinite(self.value) or self.value < 0):
            raise ValueError("invalid bound value")


def time_bound(demand: Demands, profile: Profile) -> Bound:
    terms: dict[str, float] = {}
    missing = list(demand.missing)
    incoming = sum(e.end - e.start for e in union(demand.inputs))
    if incoming:
        for key in ("dram_bytes_per_second", "fast_storage_bytes"):
            if key not in profile.capacities:
                missing.append(key)
        if not any(
            k not in profile.capacities for k in ("dram_bytes_per_second", "fast_storage_bytes")
        ):
            rate = profile.capacities["dram_bytes_per_second"]
            if rate <= 0:
                missing.append("positive dram_bytes_per_second")
            else:
                terms["dram_read"] = (
                    max(0, incoming - profile.capacities["fast_storage_bytes"]) / rate
                )
    for resource, count in demand.operations.items():
        key = f"{resource}_per_second"
        if count and profile.capacities.get(key, 0) > 0:
            terms[resource] = count / profile.capacities[key]
        elif count:
            missing.append(key)
    # Returning an MLX array does not prove its bytes were written through DRAM.
    return Bound(
        None if demand.missing or (missing and not terms) else max(terms.values(), default=0),
        "seconds",
        terms=terms,
        missing=tuple(sorted(set(missing))),
        assumptions=demand.assumptions
        + ("capacity inputs are upper bounds, not measured sustainable rates",),
        derivation="RESOURCE:ROOFLINE",
    )


@dataclass(frozen=True)
class DependentPhases:
    """Contract certificate, not an inference from an autoregressive benchmark label.

    Each phase must access the declared incompressible information *after* its
    prerequisite becomes available. No precomputation, persistent summary, replay
    or speculative execution can discharge that access before the phase barrier.
    The proof reference must establish these premises for this workload contract.
    """

    demands: tuple[Demands, ...]
    proof: str

    def __post_init__(self):
        if not self.proof or not self.demands:
            raise ValueError("dependent phases require a contract proof and nonempty demands")


def dependent_time_bound(phases: DependentPhases, profile: Profile) -> Bound:
    """At phase i at most F bytes are resident: T >= sum_i max(0,W_i-F)/B.

    Uses only read traffic; arithmetic and transfer may overlap. The phase
    barriers must exclude overlap of these reads across phases. Shared weights
    are intentionally counted again; F is credited anew at every phase.
    """
    reads = [
        time_bound(Demands(inputs=d.inputs, missing=d.missing), profile) for d in phases.demands
    ]
    return Bound(
        None if any(b.value is None for b in reads) else sum(b.value or 0 for b in reads),
        "seconds",
        terms={"dependent_reads": sum(b.value or 0 for b in reads)},
        missing=tuple(sorted({m for b in reads for m in b.missing})),
        assumptions=(
            "strict post-barrier information access; no precomputation or replay",
            phases.proof,
        ),
        derivation="RESOURCE:DEPENDENT_READS",
    )


def tightened_time_bound(
    demand: Demands, profile: Profile, phases: DependentPhases | None = None
) -> Bound:
    base = time_bound(demand, profile)
    if phases is None:
        return base
    extra = dependent_time_bound(phases, profile)
    if extra.value is None or (base.value is not None and base.value >= extra.value):
        return base
    return extra
