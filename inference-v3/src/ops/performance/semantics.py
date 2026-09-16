"""Useful mathematical work, distinct from the work of an implementation."""

from __future__ import annotations

from dataclasses import dataclass

from ..formula import FormulaCall
from .bounds import Bounds, ZERO


@dataclass(frozen=True, slots=True)
class UsefulWork:
    floating: Bounds = ZERO
    integer: Bounds = ZERO
    special: Bounds = ZERO
    comparisons: Bounds = ZERO
    issues: tuple[str, ...] = ()
    # Subset of floating work that is a conventional contraction. Keeping the
    # split here (at the semantic rule) avoids rediscovering it from kernel names.
    matrix: Bounds = ZERO

    def __add__(self, other):
        return UsefulWork(self.floating + other.floating, self.integer + other.integer,
                          self.special + other.special, self.comparisons + other.comparisons,
                          tuple(dict.fromkeys((*self.issues, *other.issues))), self.matrix + other.matrix)

    def scale(self, count):
        return UsefulWork(self.floating * count, self.integer * count, self.special * count,
                          self.comparisons * count, self.issues, self.matrix * count)


def work(*, floating=0, integer=0, special=0, comparisons=0, issues=(), matrix=0):
    def bound(value):
        return value if isinstance(value, Bounds) else Bounds.exact(value)
    return UsefulWork(bound(floating), bound(integer), bound(special), bound(comparisons), tuple(issues), bound(matrix))


def elementwise(*, floating=0, integer=0, special=0, comparisons=0):
    def derive(inputs, attrs, outputs, *, values=None):
        # Binary primitives also accept integer tensors. Their units follow the
        # numerical contract, not the spelling of the shared arithmetic helper.
        if not outputs[0].dtype.floating:
            return work(integer=integer + floating, special=special,
                        comparisons=comparisons).scale(outputs[0].elements)
        return work(floating=floating, integer=integer, special=special,
                    comparisons=comparisons).scale(outputs[0].elements)
    return derive


def no_arithmetic(inputs, attrs, outputs, *, values=None):
    return UsefulWork()


def contraction(inputs, attrs, outputs, *, values=None):
    bias = outputs[0].elements if len(inputs) == 3 else 0
    matrix = 2 * outputs[0].elements * inputs[0].shape[-1]
    return work(floating=matrix + bias, matrix=matrix)


def normalization(inputs, attrs, outputs, *, values=None):
    rows = inputs[0].elements // inputs[0].shape[-1]
    return work(floating=inputs[0].elements * (4 if len(inputs) == 2 else 3) + rows,
                special=rows)


def softmax(inputs, attrs, outputs, *, values=None):
    elements = inputs[0].elements
    return work(floating=3 * elements, special=elements, comparisons=elements)


def attention(inputs, attrs, outputs, *, values=None):
    queries, history = inputs[:2]
    rows, heads, width = queries.shape
    capacity = history.shape[1]
    pairs = rows * heads * capacity
    if len(inputs) == 3:
        visible = values[2] if values is not None else None
        if visible is None:
            # Shape-only descriptions cannot substitute capacity for actual work.
            pairs = Bounds(0, pairs)
        else:
            counts = visible[:, 1] if visible.ndim == 2 else visible
            starts = visible[:, 0] if visible.ndim == 2 else counts * 0
            if any(int(start) < 0 or int(count) < 0 or int(start) + int(count) > capacity
                   for start, count in zip(starts, counts, strict=True)):
                raise ValueError("attention visibility lies outside cache capacity")
            pairs = heads * sum(map(int, counts))
    return work(floating=pairs * (4 * width + 4), matrix=pairs * (4 * width), special=pairs, comparisons=pairs,
                issues=("attention useful work requires concrete visible ranges",) if isinstance(pairs, Bounds) else ())


def recurrence(inputs, attrs, outputs, *, values=None):
    return work(floating=inputs[0].elements * (2 if len(inputs) == 3 else 0))


def persistent_attention(inputs, attrs, outputs, *, values=None):
    query, history, keys, value, visible = inputs
    rows, heads, width = query.shape
    if values is None:
        pairs = Bounds(0, rows * heads * (history.shape[0] + keys.shape[0]))
    else:
        ranges = values[4]
        for history_start, history_count, current_start, current_count in ranges:
            if (min(history_start, history_count, current_start, current_count) < 0
                    or history_start + history_count > history.shape[0]
                    or current_start + current_count > keys.shape[0]):
                raise ValueError("persistent attention visibility lies outside its sources")
        pairs = heads * sum(int(row[1]) + int(row[3]) for row in ranges)
    contractions = pairs * 2 * (width + value.shape[-1])
    return work(floating=contractions + pairs * 4, matrix=contractions,
                special=pairs, comparisons=pairs,
                issues=("persistent attention useful work requires concrete visible ranges",)
                if isinstance(pairs, Bounds) else ())


def gated_delta(inputs, attrs, outputs, *, values=None):
    query, key, value = inputs[:3]
    rows, heads, value_width = value.shape
    key_width = query.shape[-1]
    # Decay, remembered value, beta residual, outer update, query projection.
    return work(floating=rows * heads * (7 * value_width * key_width + 2 * value_width))


def recurrent_prepare(inputs, attrs, outputs, *, values=None):
    rows, channels = inputs[0].shape
    heads, width = attrs["key_heads"], attrs["width"]
    value_heads = attrs["value_heads"]
    return work(floating=2 * rows * channels * attrs["convolution_width"]
                + 2 * rows * heads * width * 3 + rows * value_heads * 5,
                special=rows * channels + rows * value_heads * 4 + 2 * rows * heads)


def attention_prepare(inputs, attrs, outputs, *, values=None):
    rows = inputs[0].shape[0]
    heads = attrs["query_heads"] + attrs["kv_heads"]
    width = attrs["width"]
    rotary_width = attrs["rotary_width"]
    return work(floating=rows * heads * (4 * width + 3 * rotary_width),
                special=rows * heads * (1 + rotary_width))


def routing(inputs, attrs, outputs, *, values=None):
    rows, experts = inputs[0].shape
    k = attrs["k"]
    # Top-k comparison count is an algorithm-dependent obligation interval.
    # Report this honestly rather than equating a particular selection loop with
    # the mathematical minimum. Softmax scoring has separate fixed obligations.
    normalized = attrs["scoring"] == "softmax"
    return work(floating=rows * experts * 3 if normalized else rows * experts * 2,
                special=rows * experts, comparisons=Bounds(rows * max(0, experts - 1), rows * experts * k))


def experts(inputs, attrs, outputs, *, values=None):
    hidden, routes = inputs[:2]
    rows, width = hidden.shape
    choices = routes.shape[1]
    intermediate = inputs[3].shape[1]
    matrix = 6 * rows * choices * width * intermediate
    return work(floating=matrix
                + rows * choices * (2 * intermediate + 2 * width),
                matrix=matrix, special=rows * choices * intermediate)


def sampling(inputs, attrs, outputs, *, values=None):
    rows, vocabulary = inputs[0].shape
    if values is None or any(value is None for value in values):
        raise ValueError("sampling work requires concrete logits and draw policies")
    import numpy as np

    logits, draws = values
    valid = ~np.any(np.isnan(logits) | np.isposinf(logits), axis=1)
    finite = np.sum(np.isfinite(logits), axis=1)
    randomized = int(np.sum(finite[valid & (draws[:, 0] == 1)]))
    comparisons = sum(max(0, int(count) - 1) for count in finite[valid])
    # Conventional Philox4x32-10 word: per round two wide products, two
    # high-word extractions, four XORs and two key additions, then one shift.
    # This is semantic work, not a count of emitted GPU instructions.
    return work(floating=5 * randomized, special=2 * randomized,
                comparisons=comparisons, integer=101 * randomized)


def constrained_sampling(inputs, attrs, outputs, *, values=None):
    if values is None or any(value is None for value in values):
        raise ValueError("constrained sampling requires concrete distributions and masks")
    import numpy as np

    logits, draws, masks, mask_rows = values
    masked = logits.copy()
    for row, mask_row in enumerate(mask_rows):
        if mask_row < -1 or mask_row >= len(masks):
            masked[row] = np.nan
            continue
        if mask_row >= 0:
            for token in range(logits.shape[1]):
                if not int(masks[mask_row, token // 32]) & (1 << (token % 32)):
                    masked[row, token] = -np.inf
    invalid = np.any(np.isnan(logits) | np.isposinf(logits), axis=1)
    masked[invalid] = np.nan
    membership_tests = int(np.sum((mask_rows >= 0) & (mask_rows < len(masks)))) * logits.shape[1]
    return (sampling(inputs[:2], attrs, outputs, values=(masked, draws))
            + work(integer=membership_tests, comparisons=membership_tests))


@dataclass(frozen=True, slots=True)
class FormulaWork:
    work: UsefulWork
    input_bytes: int
    output_bytes: int
    quantity_values: tuple


def node_work(graph, node, *, values=None) -> UsefulWork:
    from ..tensor.primitive import primitives

    primitive = primitives.get(node.operation)
    inputs = tuple(graph.value(value).spec for value in node.inputs)
    outputs = tuple(graph.value(value).spec for value in node.outputs)
    if primitive.work is None:
        raise ValueError(f"missing useful-work rule: {node.operation}")
    return primitive.work(inputs, node.attributes, outputs,
                          values=tuple(values.get(value) for value in node.inputs) if values is not None else None)


def formula_work(graph, call: FormulaCall, *, values=None) -> FormulaWork:
    result = UsefulWork()
    for node_id in call.nodes:
        result += node_work(graph, graph.node(node_id), values=values)
    inputs = {port.value for port in call.inputs}
    outputs = {port.value for port in call.outputs}
    return FormulaWork(result, sum(graph.value(value).spec.storage_nbytes for value in inputs),
                       sum(graph.value(value).spec.storage_nbytes for value in outputs), call.quantities)
