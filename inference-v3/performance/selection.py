"""What a measured component's tables actually chose.

A plan carries the candidate's name, the representation it read, and any
schedule-owned policy worth recording. Collecting those is how a run record
distinguishes a selection change from a kernel change.
"""

from __future__ import annotations

from magnitude_engine.operations.candidates import Plan
from performance.metrics import Realized

_OPERATIONS = {
    "ResidentProjections": "projection",
    "GatedLinear": "gated",
    "CausalAttention": "attention",
    "DeltaRecurrence": "recurrence",
    "RMSNorm": "norm",
    "ResidentEmbedding": "embedding",
}


def realized(component: object) -> tuple[Realized, ...]:
    """Every plan reachable from a component, deduplicated and ordered."""
    found: dict[tuple[str, str, str | None, tuple], None] = {}
    for owner in _reachable(component, set()):
        operation = _OPERATIONS.get(type(owner).__name__)
        if operation is None:
            continue
        for plan in getattr(owner, "_plans", {}).values():
            if isinstance(plan, Plan):
                found[(operation, plan.candidate, plan.representation, plan.detail)] = None
    return tuple(
        Realized(operation=operation, candidate=candidate, representation=layout, detail=detail)
        for operation, candidate, layout, detail in found
    )


def _reachable(value: object, seen: set[int]):
    """Walk the component graph without assuming a particular model's shape."""
    if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None))):
        return
    seen.add(id(value))
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _reachable(item, seen)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _reachable(item, seen)
        return
    try:
        attributes = getattr(value, "__dict__", None)
    except (AttributeError, RuntimeError, TypeError):
        # Compiled runtime handles are deliberately opaque and some reject
        # Python state introspection instead of returning no ``__dict__``.
        return
    if attributes is None:
        return
    yield value
    for name, item in tuple(attributes.items()):
        if name.startswith("__") or name in (
            "context",
            "driver",
            "weights",
            "format",
            "_plans",
        ):
            continue
        yield from _reachable(item, seen)
