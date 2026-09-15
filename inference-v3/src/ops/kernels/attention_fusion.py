"""Attention preparation fused with KV-cache append."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from .normalization import _reduction_threads


@T.macro
def _prepare_append(
    query_gate,
    keys,
    values,
    query_norm,
    key_norm,
    coordinates,
    cache,
    positions,
    query_out,
    gate_out,
    next_cache,
    key_out,
    rows,
    query_heads,
    kv_heads,
    width,
    rotary_width,
    base,
    sections,
    epsilon,
    dtype,
    subgroup_width,
    subgroups,
    publish_cache,
):
    half = rotary_width // 2
    heads = query_heads + kv_heads
    channels_per_lane = (width + subgroup_width - 1) // subgroup_width
    with T.Kernel(T.ceildiv(heads, subgroups), rows, threads=subgroup_width * subgroups) as (
        head_block,
        row,
    ):
        thread = T.get_thread_binding()
        lane = thread % subgroup_width
        subgroup = thread // subgroup_width
        head = head_block * subgroups + subgroup
        square = T.alloc_local((1,), "float32")
        cosine = T.alloc_shared((half,), "float32")
        sine = T.alloc_shared((half,), "float32")
        for chunk in T.serial(T.ceildiv(half, subgroup_width)):
            index = chunk * subgroup_width + lane
            if subgroup == 0 and index < half:
                axis = T.if_then_else(
                    index % 3 == 1 and index < sections[1] * 3,
                    1,
                    T.if_then_else(index % 3 == 2 and index < sections[2] * 3, 2, 0),
                )
                angle = T.cast(coordinates[row, axis], "float32") / T.pow(
                    base, T.cast(index * 2, "float32") / rotary_width
                )
                cosine[index] = T.cos(angle)
                sine[index] = T.sin(angle)
        T.sync_threads()
        square[0] = 0.0
        for item in T.serial(channels_per_lane):
            channel = lane * channels_per_lane + item
            if head < heads and channel < width:
                raw = T.alloc_local((1,), "float32")
                if head < query_heads:
                    raw[0] = T.cast(query_gate[row, head * 2 * width + channel], "float32")
                else:
                    raw[0] = T.cast(keys[row, (head - query_heads) * width + channel], "float32")
                square[0] += raw[0] * raw[0]
        inverse = T.rsqrt(T.warp_reduce_sum(square[0]) / width + epsilon)
        destination = T.cast(positions[row], "int32") if publish_cache else 0
        for item in T.serial(channels_per_lane):
            channel = lane * channels_per_lane + item
            if head < heads and channel < width:
                source = T.alloc_local((1,), "float32")
                weight = T.alloc_local((1,), "float32")
                if head < query_heads:
                    source[0] = T.cast(query_gate[row, head * 2 * width + channel], "float32")
                    weight[0] = T.cast(query_norm[channel], "float32")
                else:
                    source[0] = T.cast(keys[row, (head - query_heads) * width + channel], "float32")
                    weight[0] = T.cast(key_norm[channel], "float32")
                prepared = T.alloc_local((1,), "float32")
                prepared[0] = source[0] * inverse * weight[0]
                if channel < rotary_width:
                    index = channel % half
                    pair = (channel + half) % rotary_width
                    paired = T.alloc_local((1,), "float32")
                    if head < query_heads:
                        paired[0] = T.cast(
                            query_gate[row, head * 2 * width + pair], "float32"
                        ) * T.cast(query_norm[pair], "float32")
                    else:
                        paired[0] = T.cast(
                            keys[row, (head - query_heads) * width + pair], "float32"
                        ) * T.cast(key_norm[pair], "float32")
                    paired[0] *= inverse
                    prepared[0] = (
                        prepared[0] * cosine[index]
                        + T.if_then_else(channel < half, -paired[0], paired[0]) * sine[index]
                    )
                if head < query_heads:
                    query_out[row, head, channel] = T.cast(prepared[0], dtype)
                    gate_out[row, head, channel] = query_gate[
                        row, head * 2 * width + width + channel
                    ]
                elif not publish_cache:
                    key_out[row, head - query_heads, channel] = T.cast(prepared[0], dtype)
                elif destination >= 0:
                    kv_head = head - query_heads
                    next_cache[0, destination, kv_head, channel] = T.cast(prepared[0], dtype)
                    next_cache[1, destination, kv_head, channel] = values[
                        row, kv_head * width + channel
                    ]


class _PrepareAppendEmitter:
    def __init__(self, attrs, rows, subgroup_width, subgroups, dtype):
        self.attrs, self.rows = attrs, rows
        self.subgroup_width, self.subgroups, self.dtype = subgroup_width, subgroups, dtype

    def __call__(self, operands: tuple[Any, ...]) -> None:
        # Inputs are explicitly ordered by the rule; outputs follow them.
        _prepare_append(
            *operands[:8],
            operands[8],
            operands[9],
            operands[10],
            operands[8],
            self.rows,
            self.attrs["query_heads"],
            self.attrs["kv_heads"],
            self.attrs["width"],
            self.attrs["rotary_width"],
            self.attrs["base"],
            self.attrs["sections"],
            self.attrs["epsilon"],
            self.dtype,
            self.subgroup_width,
            self.subgroups,
            True,
        )


class _PrepareEmitter:
    def __init__(self, attrs, rows, subgroup_width, subgroups, dtype):
        self.attrs, self.rows = attrs, rows
        self.subgroup_width, self.subgroups, self.dtype = subgroup_width, subgroups, dtype

    def __call__(self, operands):
        query_gate, keys, query_norm, key_norm, coordinates, query_out, key_out, gate_out = operands
        _prepare_append(
            query_gate, keys, keys, query_norm, key_norm, coordinates, key_out, coordinates,
            query_out, gate_out, key_out, key_out, self.rows,
            self.attrs["query_heads"], self.attrs["kv_heads"], self.attrs["width"],
            self.attrs["rotary_width"], self.attrs["base"], self.attrs["sections"],
            self.attrs["epsilon"], self.dtype, self.subgroup_width, self.subgroups, False,
        )


class AttentionPrepareRule:
    def build(self, graph, root, context):
        node = graph.node(root)
        attrs = node.attributes
        threads = _reduction_threads(attrs["width"], context)
        if threads is None or context.compiler_target.shared_memory_bytes <= 0:
            return ()
        source = graph.value(node.inputs[0]).spec
        subgroups = min(threads // context.compiler_target.subgroup_width,
                        attrs["query_heads"] + attrs["kv_heads"])
        return (BoundOperation(f"attention.prepare@{root}", frozenset({root}), node.inputs, node.outputs,
                               _PrepareEmitter(attrs, source.shape[0], context.compiler_target.subgroup_width,
                                               subgroups, source.dtype.value)),)


@T.macro
def _append_cache(keys, values, destinations, output, rows, heads, width, threads):
    with T.Kernel(T.ceildiv(rows * heads * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            index = block * threads + lane
            if index < rows * heads * width:
                row = index // (heads * width)
                head = index // width % heads
                channel = index % width
                destination = destinations[row]
                if destination >= 0:
                    output[0, destination, head, channel] = keys[row, head, channel]
                    output[1, destination, head, channel] = values[row, head, channel]


class _AppendEmitter:
    def __init__(self, shape, threads):
        self.shape, self.threads = shape, threads

    def __call__(self, operands):
        cache, keys, values, destinations, output = operands
        _append_cache(keys, values, destinations, output, *self.shape, self.threads)


class KVAppendRule:
    def build(self, graph, root, context):
        node = graph.node(root)
        shape = graph.value(node.inputs[1]).spec.shape
        return (BoundOperation(f"kv.append@{root}", frozenset({root}), node.inputs, node.outputs,
                               _AppendEmitter(shape, min(256, context.compiler_target.threads_per_group)),
                               aliases=((node.outputs[0], node.inputs[0]),)),)


class AttentionPrepareAppendRule:
    name = "attention-prepare-append"

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
            prepare.inputs[0],
            prepare.inputs[1],
            raw_values,
            prepare.inputs[2],
            prepare.inputs[3],
            prepare.inputs[4],
            cache,
            positions,
        )
        outputs = (prepare.outputs[0], prepare.outputs[2], append.outputs[0])
        specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
        if any(not spec.static for spec in specs):
            return ()
        rows = cast(int, specs[0].shape[0])
        width = prepare.attributes["width"]
        threads = _reduction_threads(width, context)
        if threads is None or context.compiler_target.shared_memory_bytes <= 0:
            return ()
        moved = sum(spec.storage_nbytes for spec in specs)
        subgroups = min(
            threads // context.compiler_target.subgroup_width,
            prepare.attributes["query_heads"] + prepare.attributes["kv_heads"],
        )
        return (
            BoundOperation(
                f"attention.prepare-append@{root}:{root + 2}",
                frozenset({root, root + 1, root + 2}),
                inputs,
                outputs,
                _PrepareAppendEmitter(
                    prepare.attributes,
                    rows,
                    context.compiler_target.subgroup_width,
                    subgroups,
                    specs[0].dtype.value,
                ),
                aliases=((append.outputs[0], cache),),
            ),
        )


__all__ = ["AttentionPrepareAppendRule"]
