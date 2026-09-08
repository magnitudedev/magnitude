"""Expert computation: selected dots, scalar activation and ordered route sum."""

from typing import TYPE_CHECKING

import mlx.core as mx

from ..core.computation import computation
from .selection import RouteSum, SelectedAffine

if TYPE_CHECKING:
    from .weights import ExpertWeights


def supported(weights: "ExpertWeights", hidden: mx.array, assignments: mx.array) -> bool:
    gate, up, down = weights.gate, weights.up, weights.down
    short = hidden.ndim == 3 and hidden.shape[1] <= 8
    if (
        (not short and hidden.size // hidden.shape[-1] > 8)
        or assignments.shape[-1] > 16
        or hidden.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or gate.encoding != up.encoding
        or gate.weight.shape != up.weight.shape
    ):
        return False
    for projection in (gate, up, down):
        if (
            projection.encoding.bits not in (4, 8)
            or projection.encoding.group_size % 8
            or projection.scales.dtype != hidden.dtype
            or projection.biases.dtype != hidden.dtype
            or projection.weight.dtype != mx.uint32
            or projection.weight.ndim != 3
        ):
            return False
        width = projection.weight.shape[-1] * 32 // projection.encoding.bits
        per_lane = 64 // projection.encoding.bits
        if (
            width % (32 * per_lane)
            or projection.encoding.group_size % per_lane
            or width % projection.encoding.group_size
            or projection.weight.shape[-2] % 8
        ):
            return False
    return True


def shared_supported(
    weights: "ExpertWeights", shared: "ExpertWeights", hidden: mx.array, assignments: mx.array
) -> bool:
    return supported(weights, hidden, assignments) and all(
        a.encoding == b.encoding
        and a.weight.shape[1:] == b.weight.shape
        and a.scales.shape[1:] == b.scales.shape
        and a.biases.shape[1:] == b.biases.shape
        and a.weight.dtype == b.weight.dtype
        and a.scales.dtype == b.scales.dtype
        and a.biases.dtype == b.biases.dtype
        for a, b in zip(
            (weights.gate, weights.up, weights.down),
            (shared.gate, shared.up, shared.down),
            strict=True,
        )
    )


def _assignments(
    weights: "ExpertWeights",
    assignments: mx.array,
    rows: int,
    shared: bool,
) -> tuple[mx.array, mx.array]:
    ids = assignments.reshape(rows, -1)
    if shared:
        ids = mx.concatenate(
            [ids, mx.full((rows, 1), weights.gate.weight.shape[0], ids.dtype)], axis=-1
        )
    ids = ids.reshape(-1)
    experts = weights.gate.weight.shape[0] + shared
    # Grouping adds a sort and a materialized down result. Sparse assignments
    # use direct tiles; dense assignments can amortize those costs through reuse.
    reuse = rows > 1 and ids.size >= 2 * experts
    order = mx.argsort(ids) if reuse and experts > 1 else mx.arange(ids.size, dtype=mx.uint32)
    return ids, order


@computation
def _gate_up(hidden, gate, up, shared_gate, shared_up, ids, order, *, slots, shared):
    dot = SelectedAffine(gate.encoding.bits, gate.encoding.group_size, slots, shared)
    g = dot(
        hidden,
        ids,
        order,
        gate.weight,
        gate.scales,
        gate.biases,
        shared_gate.weight,
        shared_gate.scales,
        shared_gate.biases,
    )[0]
    u = dot(
        hidden,
        ids,
        order,
        up.weight,
        up.scales,
        up.biases,
        shared_up.weight,
        shared_up.scales,
        shared_up.biases,
    )[0]
    return (g * mx.sigmoid(g)).astype(hidden.dtype) * u


@computation
def _down(activation, down, shared_down, ids, order, scores, shared_score, *, slots, shared):
    dot = SelectedAffine(down.encoding.bits, down.encoding.group_size, slots, shared, per_slot=True)
    projected = dot(
        activation,
        ids,
        order,
        down.weight,
        down.scales,
        down.biases,
        shared_down.weight,
        shared_down.scales,
        shared_down.biases,
    )[0]
    return RouteSum(shared)(projected, scores, shared_score)[0]


def _activate(weights, hidden, assignments, shared, ids, order):
    gate = _gate_up
    shared_gate, shared_up = (
        (shared.gate, shared.up) if shared is not None else (weights.gate, weights.up)
    )
    return gate(
        hidden,
        weights.gate,
        weights.up,
        shared_gate,
        shared_up,
        ids,
        order,
        slots=assignments.shape[-1] + (shared is not None),
        shared=shared is not None,
    )


def activate(weights, hidden, assignments, shared=None):
    rows = hidden.size // hidden.shape[-1]
    ids, order = _assignments(weights, assignments, rows, shared is not None)
    return _activate(weights, hidden, assignments, shared, ids, order)


def apply(weights, hidden, assignments, scores, *, shared=None, shared_score=None):
    rows = hidden.size // hidden.shape[-1]
    if shared is not None and shared_score is None:
        raise ValueError("shared expert requires its coefficient")
    ids, order = _assignments(weights, assignments, rows, shared is not None)
    activation = _activate(weights, hidden, assignments, shared, ids, order)
    down = _down
    return down(
        activation,
        weights.down,
        shared.down if shared is not None else weights.down,
        ids,
        order,
        scores.reshape(rows, -1),
        shared_score if shared_score is not None else scores,
        slots=assignments.shape[-1] + (shared is not None),
        shared=shared is not None,
    ).reshape(hidden.shape)
