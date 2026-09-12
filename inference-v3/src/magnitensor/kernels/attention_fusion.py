"""Attention preparation fused with KV-cache append."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph


@T.macro
def _prepare_append(
    query_gate, keys, values, query_norm, key_norm, coordinates, cache, positions,
    query_out, gate_out, next_cache, rows, query_heads, kv_heads, width,
    rotary_width, base, sections, epsilon, dtype, threads,
):
    half = rotary_width // 2
    with T.Kernel(query_heads + kv_heads, rows, threads=threads) as (head, row):
        lane = T.get_thread_binding()
        square = T.alloc_local((1,), "float32")
        shared = T.alloc_shared((threads,), "float32")
        square[0] = 0.0
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                raw = T.if_then_else(
                    head < query_heads,
                    query_gate[row, head * 2 * width + channel],
                    keys[row, (head - query_heads) * width + channel],
                )
                square[0] += T.cast(raw, "float32") * T.cast(raw, "float32")
        shared[lane] = square[0]
        T.sync_threads()
        for step in T.unroll(int(math.log2(threads))):
            distance = threads >> (step + 1)
            if lane < distance:
                shared[lane] += shared[lane + distance]
            T.sync_threads()
        inverse = T.rsqrt(shared[0] / width + epsilon)
        destination = T.cast(positions[row], "int32")
        for chunk in T.serial(T.ceildiv(width, threads)):
            channel = chunk * threads + lane
            if channel < width:
                source = T.cast(
                    T.if_then_else(
                        head < query_heads,
                        query_gate[row, head * 2 * width + channel],
                        keys[row, (head - query_heads) * width + channel],
                    ), "float32",
                )
                weight = T.cast(
                    T.if_then_else(head < query_heads, query_norm[channel], key_norm[channel]),
                    "float32",
                )
                prepared = T.alloc_local((1,), "float32")
                prepared[0] = source * inverse * weight
                if channel < rotary_width:
                    index = channel % half
                    axis = T.if_then_else(
                        index % 3 == 1 and index < sections[1] * 3, 1,
                        T.if_then_else(index % 3 == 2 and index < sections[2] * 3, 2, 0),
                    )
                    pair = (channel + half) % rotary_width
                    paired = T.cast(
                        T.if_then_else(
                            head < query_heads,
                            query_gate[row, head * 2 * width + pair],
                            keys[row, (head - query_heads) * width + pair],
                        ), "float32",
                    ) * inverse * T.cast(
                        T.if_then_else(head < query_heads, query_norm[pair], key_norm[pair]),
                        "float32",
                    )
                    angle = T.cast(coordinates[row, axis], "float32") / T.pow(
                        base, T.cast(index * 2, "float32") / rotary_width
                    )
                    prepared[0] = prepared[0] * T.cos(angle) + T.if_then_else(
                        channel < half, -paired, paired
                    ) * T.sin(angle)
                if head < query_heads:
                    query_out[row, head, channel] = T.cast(prepared[0], dtype)
                    gate_out[row, head, channel] = query_gate[
                        row, head * 2 * width + width + channel
                    ]
                elif destination >= 0:
                    kv_head = head - query_heads
                    next_cache[0, destination, kv_head, channel] = T.cast(prepared[0], dtype)
                    next_cache[1, destination, kv_head, channel] = values[
                        row, kv_head * width + channel
                    ]


class _PrepareAppendEmitter:
    def __init__(self, attrs, rows, threads, dtype):
        self.attrs, self.rows, self.threads, self.dtype = attrs, rows, threads, dtype

    def __call__(self, operands: tuple[Any, ...]) -> None:
        # Inputs are explicitly ordered by the rule; outputs follow them.
        _prepare_append(
            *operands[:8], operands[8], operands[9], operands[10], self.rows,
            self.attrs["query_heads"], self.attrs["kv_heads"], self.attrs["width"],
            self.attrs["rotary_width"], self.attrs["base"], self.attrs["sections"],
            self.attrs["epsilon"], self.dtype, self.threads,
        )


class AttentionPrepareAppendRule:
    name = "attention-prepare-append"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        prepare = graph.nodes[root]
        if prepare.operation != "attention_prepare" or root + 2 >= len(graph.nodes):
            return ()
        reshape, append = graph.nodes[root + 1], graph.nodes[root + 2]
        if reshape.operation != "reshape" or append.operation != "kv_append":
            return ()
        if prepare.outputs[1] not in append.inputs or reshape.outputs[0] not in append.inputs:
            return ()
        cache = append.inputs[0]
        positions = append.inputs[3]
        raw_values = reshape.inputs[0]
        inputs = (
            prepare.inputs[0], prepare.inputs[1], raw_values,
            prepare.inputs[2], prepare.inputs[3], prepare.inputs[4], cache, positions,
        )
        outputs = (prepare.outputs[0], prepare.outputs[2], append.outputs[0])
        specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
        if any(not spec.static for spec in specs):
            return ()
        rows = cast(int, specs[0].shape[0])
        width = prepare.attributes["width"]
        threads = 1 << (width - 1).bit_length()
        if threads > context.capabilities.threads_per_group or "shared" not in context.capabilities.memory_scopes:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"attention.prepare-append@{root}:{root + 2}",
            frozenset({root, root + 1, root + 2}), inputs, outputs,
            _PrepareAppendEmitter(prepare.attributes, rows, threads, specs[0].dtype.value),
            4e-7 + moved / 200e9,
            aliases=((append.outputs[0], cache),), priority=80,
        ),)


__all__ = ["AttentionPrepareAppendRule"]
