"""Readiness partitions that preserve the qualified grouped-QMM dispatch geometry."""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True)
class AssignmentGroup:
    start: int
    end: int
    first_expert: int
    expert_limit: int

    @property
    def grouped_matrix(self) -> bool:
        return self.end - self.start >= max(16, 4 * (self.expert_limit - self.first_expert))


def grouped_ranges(
    sorted_experts: np.ndarray, target_span: int
) -> tuple[AssignmentGroup, ...] | None:
    """Merge sparse neighboring spans until the exact grouped dispatch remains legal.

    Span width includes unassigned logical experts between its endpoints. Counting
    only distinct assignments would change backend dispatch and bfloat16 results.
    None means the operator must retain its original whole-batch computation.
    """
    ids = np.asarray(sorted_experts)
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or target_span < 1
        or np.any(ids < 0)
        or np.any(ids[1:] < ids[:-1])
    ):
        raise ValueError("grouping requires sorted nonnegative expert IDs and a positive span")
    if not ids.size:
        return ()
    stops = (np.flatnonzero(ids[1:] // target_span != ids[:-1] // target_span) + 1).tolist()
    stops.append(len(ids))
    groups = []
    start = 0
    for stop in stops:
        candidate = AssignmentGroup(start, stop, int(ids[start]), int(ids[stop - 1]) + 1)
        if candidate.grouped_matrix:
            groups.append(candidate)
            start = stop
    if start == len(ids):
        return tuple(groups)
    while groups:
        previous = groups.pop()
        combined = AssignmentGroup(
            previous.start, len(ids), previous.first_expert, int(ids[-1]) + 1
        )
        if combined.grouped_matrix:
            return (*groups, combined)
    return None


def support_missing(
    sorted_experts: np.ndarray, ready: np.ndarray, *, target_span: int, storage_slots: int
) -> tuple[np.ndarray, np.ndarray, tuple[AssignmentGroup, ...]] | None:
    """Borrow whole ready runs to keep both early and missing phases compute-eligible."""
    ids = np.asarray(sorted_experts)
    ready = np.asarray(ready)
    groups = grouped_ranges(ids, target_span)
    if (
        storage_slots < 1
        or ready.dtype != np.bool_
        or ready.shape != ids.shape
        or (ids.size and int(ids[-1]) >= storage_slots)
        or np.any((ids[1:] == ids[:-1]) & (ready[1:] != ready[:-1]))
    ):
        raise ValueError("readiness must describe complete expert records within the bank")
    if not groups or ready.all() or not ready.any():
        return None
    missing = ~ready.copy()
    for group in groups:
        needed = int(missing[group.start : group.end].sum())
        if needed == 0:
            continue
        threshold = max(16, 4 * (group.expert_limit - group.first_expert))
        changes = np.flatnonzero(
            ids[group.start + 1 : group.end] != ids[group.start : group.end - 1]
        )
        bounds = [group.start, *(int(x) + group.start + 1 for x in changes), group.end]
        candidates = [(b - a, a, b) for a, b in pairwise(bounds) if ready[a]]
        for size, start, stop in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if needed >= threshold:
                break
            missing[start:stop] = True
            needed += size
    initial = ~missing
    if initial.sum() < max(16, 4 * storage_slots):
        return None
    stages = grouped_ranges(ids[missing], target_span)
    return (initial, missing, stages) if stages else None
