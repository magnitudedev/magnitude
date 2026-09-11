"""Declarative kernel selection: a table of candidates and one way to read it.

An operation declares which schedules can serve a shape, when each applies, how
it ranks against the others, and what scratch it needs. ``realize`` evaluates
that table once per shape — behind each operation's plan cache, which is itself
behind the program's invocation-geometry cache — and never on the per-step host
path. Nothing inspects a capability, a precision or a representation at
``prepare`` time: a ``Plan`` is a fixed tuple of executables.

This is what replaces an if-chain per operation. A table can be read, tested
against the kernel it must pick, and printed into a run record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.precision import Precision
from magnitude_engine.platform.execution import DeviceContext, Executable, TensorSpec
from magnitude_engine.weights.representation import Representation

if TYPE_CHECKING:
    from magnitude_engine.operations.preparation import Preparation
    from magnitude_engine.platform.execution import Tensor


class ScratchArena(Protocol):
    """The bound program's one scratch allocation, seen by its operations.

    Regions are named, so every plan that declares the same name shares one
    allocation and the largest declaration wins. An operation never allocates.
    """

    context: DeviceContext

    def region(self, preparation: Preparation, name: str, spec: TensorSpec) -> Tensor: ...

    def reserve(self, regions: tuple[Scratch, ...]) -> None: ...

    def available(self, regions: tuple[str, ...]) -> int:
        """Budget a plan may claim: what is free plus what these regions hold."""
        ...


@dataclass(frozen=True)
class Scratch:
    """An arena region a plan needs, named so plans can share one allocation."""

    name: str
    spec: TensorSpec


@dataclass(frozen=True)
class Plan:
    """The compiled result of one realized operation instance."""

    candidate: str
    executables: tuple[Executable, ...]
    scratch: tuple[Scratch, ...] = ()
    detail: tuple[tuple[str, int], ...] = ()
    """Schedule-owned policy worth recording, such as a partition count."""

    representation: str | None = None
    """What the weights were resident in, stamped by ``realize`` for run records."""


@dataclass(frozen=True)
class Selection[S]:
    """Everything a candidate is allowed to look at."""

    shape: S
    precision: Precision
    capability: Capability
    representation: Representation | None = None


@dataclass(frozen=True)
class Candidate[S]:
    name: str
    applies: Callable[[Selection[S]], bool]
    build: Callable[[DeviceContext, Selection[S]], Plan]
    rank: int = 0
    scratch: Callable[[Selection[S]], tuple[Scratch, ...]] = field(
        default=lambda selection: ()
    )
    """Declared before anything is allocated, so a plan that cannot fit is skipped."""


class NoCandidate(ValueError):
    def __init__(self, operation: str, selection: Selection):
        super().__init__(
            f"no {operation} candidate applies to {selection.shape} "
            f"at {selection.precision.rounding} on {selection.capability}"
        )
        self.operation, self.selection = operation, selection


def select[S](
    operation: str,
    table: tuple[Candidate[S], ...],
    selection: Selection[S],
    *,
    available_bytes: int,
) -> tuple[Candidate[S], tuple[Scratch, ...]]:
    """The highest-ranked applicable candidate whose declared scratch fits.

    Separate from building so the choice can be read — and tested against the
    kernel it must pick — without compiling anything.
    """
    for candidate in sorted(table, key=lambda row: -row.rank):
        if not candidate.applies(selection):
            continue
        scratch = candidate.scratch(selection)
        if sum(region.spec.nbytes for region in scratch) > available_bytes:
            continue
        return candidate, scratch
    raise NoCandidate(operation, selection)


def realize[S](
    operation: str,
    table: tuple[Candidate[S], ...],
    context: DeviceContext,
    selection: Selection[S],
    *,
    available_bytes: int | None = None,
) -> Plan:
    budget = (
        context.budget_bytes - context.allocated_bytes
        if available_bytes is None
        else available_bytes
    )
    candidate, scratch = select(operation, table, selection, available_bytes=budget)
    plan = candidate.build(context, selection)
    if plan.candidate != candidate.name:
        raise ValueError(f"{candidate.name} built a plan named {plan.candidate}")
    return replace(
        plan,
        scratch=plan.scratch or scratch,
        representation=_named(selection.representation),
    )


def _named(representation: Representation | None) -> str | None:
    if representation is None:
        return None
    fields = ", ".join(
        f"{key}={getattr(value, 'name', value)}"
        for key, value in vars(representation).items()
    )
    return f"{type(representation).__name__}({fields})"
