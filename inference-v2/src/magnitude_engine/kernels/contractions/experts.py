"""Selected-expert execution with reusable affine inputs and fused epilogues.

Gate/up writes native activations; down consumes them directly. Each projection
keeps affine input preparation local while preserving MLX's BF16 boundaries.
"""

from typing import TYPE_CHECKING

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

from .tiles import TILE, row_tile

if TYPE_CHECKING:
    from .weights import ExpertWeights


GATE_UP = Program(
    "magnitude_expert_gate_up",
    Source("contractions/gate_up.metal", (TILE,)),
)


DOWN = Program(
    "magnitude_expert_down_sum",
    Source("contractions/down.metal", (TILE,)),
)


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


SELECTED = Program("magnitude_selected_down", Source("contractions/selected.metal", (TILE,)))
COMBINE = Program("magnitude_expert_combine", Source("contractions/combine.metal"))


def _assignments(
    weights: "ExpertWeights",
    assignments: mx.array,
    rows: int,
    shared: bool,
) -> tuple[mx.array, mx.array, int]:
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
    # Paired gate/up carries twice the live accumulators of a linear projection.
    return ids, order, row_tile(ids.size, maximum=4) if reuse else 1


def _activate(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    shared: "ExpertWeights | None",
    ids: mx.array,
    order: mx.array,
    tile: int,
) -> mx.array:
    gate, up = weights.gate, weights.up
    rows, width = hidden.size // hidden.shape[-1], hidden.shape[-1]
    top_k, intermediate = assignments.shape[-1], gate.weight.shape[1]
    slots = top_k + (shared is not None)
    sg, su = (shared.gate, shared.up) if shared is not None else (gate, up)
    return KernelPlan(
        program=GATE_UP,
        inputs=(
            Input("x", hidden),
            Input("ids", ids),
            Input("order", order),
            Input("wg", gate.weight),
            Input("sg_", gate.scales),
            Input("bg_", gate.biases),
            Input("wu", up.weight),
            Input("su_", up.scales),
            Input("bu_", up.biases),
            Input("wsg", sg.weight),
            Input("ssg", sg.scales),
            Input("bsg", sg.biases),
            Input("wsu", su.weight),
            Input("ssu", su.scales),
            Input("bsu", su.biases),
        ),
        outputs=(Output("out", (rows, slots, intermediate), hidden.dtype),),
        launch=Launch((64, intermediate // 8, (ids.size + tile - 1) // tile), (64, 1, 1)),
        template=(
            ("T", hidden.dtype),
            ("K", width),
            ("N", intermediate),
            ("TOPK", top_k),
            ("SHARED", shared is not None),
            ("E", gate.weight.shape[0]),
            ("M", ids.size),
            ("R", tile),
            ("BITS", gate.encoding.bits),
            ("GROUP", gate.encoding.group_size),
            ("PACK", 64 // gate.encoding.bits),
        ),
    ).run()[0]


def activate(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    shared: "ExpertWeights | None" = None,
) -> mx.array:
    rows = hidden.size // hidden.shape[-1]
    ids, order, tile = _assignments(weights, assignments, rows, shared is not None)
    return _activate(weights, hidden, assignments, shared, ids, order, tile)


def apply(
    weights: "ExpertWeights",
    hidden: mx.array,
    assignments: mx.array,
    scores: mx.array,
    *,
    shared: "ExpertWeights | None" = None,
    shared_score: mx.array | None = None,
) -> mx.array:
    down = weights.down
    rows, width = hidden.size // hidden.shape[-1], hidden.shape[-1]
    top_k, intermediate = assignments.shape[-1], weights.gate.weight.shape[1]
    if shared is not None and shared_score is None:
        raise ValueError("shared expert requires its coefficient")
    ids, order, tile = _assignments(weights, assignments, rows, shared is not None)
    activation = _activate(weights, hidden, assignments, shared, ids, order, tile)
    sd = shared.down if shared is not None else down
    operands = (
        Input("x", activation),
        Input("w", down.weight),
        Input("scales", down.scales),
        Input("biases", down.biases),
        Input("wsh", sd.weight),
        Input("ssh", sd.scales),
        Input("bsh", sd.biases),
    )
    coefficients = (
        Input("scores", scores.reshape(rows, top_k)),
        Input("shared_score", shared_score if shared_score is not None else scores),
    )
    template = (
        ("T", hidden.dtype),
        ("K", intermediate),
        ("N", width),
        ("SHARED", shared is not None),
        ("BITS", down.encoding.bits),
        ("GROUP", down.encoding.group_size),
        ("PACK", 64 // down.encoding.bits),
    )
    if tile == 1:
        # Direct tiles keep projection and ordered combination in one launch.
        return KernelPlan(
            program=DOWN,
            inputs=operands + (Input("inds", assignments.reshape(rows, top_k)),) + coefficients,
            outputs=(Output("out", hidden.shape, hidden.dtype),),
            launch=Launch((64, width // 8, rows), (64, 1, 1)),
            template=template + (("TOPK", top_k),),
        ).run()[0]
    projected = KernelPlan(
        program=SELECTED,
        inputs=operands + (Input("ids", ids), Input("order", order)),
        outputs=(Output("out", (rows, ids.size // rows, width), hidden.dtype),),
        launch=Launch((64, width // 8, (ids.size + tile - 1) // tile), (64, 1, 1)),
        template=template + (("E", down.weight.shape[0]), ("M", ids.size), ("R", tile)),
    ).run()[0]
    return KernelPlan(
        program=COMBINE,
        inputs=(Input("x", projected),) + coefficients,
        outputs=(Output("out", hidden.shape, hidden.dtype),),
        launch=Launch((rows * width, 1, 1), (256, 1, 1)),
        template=(
            ("T", hidden.dtype),
            ("M", rows),
            ("N", width),
            ("TOPK", top_k),
            ("SHARED", shared is not None),
        ),
    ).run()[0]
