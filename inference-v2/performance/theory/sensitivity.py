"""Finite counterfactuals for a declared, observation-checked execution composition."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sensitivity:
    predicted_seconds_saved: float | None
    issue: str | None = None


def predict(
    *,
    execution: str,
    children: dict[str, float],
    parent_seconds: float,
    child: str,
    seconds_saved: float,
    invocations: dict[str, int] | None = None,
    absolute_tolerance: float = 0,
    relative_tolerance: float = 0,
) -> Sensitivity:
    """Conditional prediction, not a proof that observed agreement establishes a contract."""
    if execution not in ("serial", "parallel"):
        return Sensitivity(None, "joint execution requires a joint model")
    numbers = [
        *children.values(),
        parent_seconds,
        seconds_saved,
        absolute_tolerance,
        relative_tolerance,
    ]
    if child not in children or any(not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError("invalid sensitivity inputs")
    if seconds_saved > children[child]:
        raise ValueError("child saving exceeds child duration")
    invocations = invocations or {}
    if any(not isinstance(n, int) or n < 1 for n in invocations.values()):
        raise ValueError("invocations must be positive integers")
    before = {k: v * invocations.get(k, 1) for k, v in children.items()}
    combine = sum if execution == "serial" else max
    predicted = combine(before.values())
    if abs(parent_seconds - predicted) > max(
        absolute_tolerance, relative_tolerance * parent_seconds
    ):
        return Sensitivity(None, "parent observation contradicts declared composition")
    after = {**before, child: before[child] - seconds_saved * invocations.get(child, 1)}
    return Sensitivity(predicted - combine(after.values()))
