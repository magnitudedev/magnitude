"""Channel-parallel online causal attention."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .matrix import (
    _packed_matrix,
    _packed_vector,
    _packed_vector_geometry,
    _packet_matrix_instruction,
    _packet_reduction_width,
)
from .packed import packet_format


@T.macro
def _matrix_streaming_attention(
    query,
    history,
    visible,
    gate,
    partials,
    statistics,
    tokens,
    heads,
    kv_heads,
    width,
    scale,
    partitions,
    span,
    query_tile,
    key_tile,
    threads,
    dtype,
    fuse_gate,
):
    """Stream one packed sequence through tiled QK and PV matrix products."""
    group = heads // kv_heads
    # Reuse each loaded history tile across related query heads.
    members = (
        4 if group % 4 == 0 and threads >= 512 else (2 if group % 2 == 0 and threads >= 256 else 1)
    )
    head_blocks = T.ceildiv(group, members)
    grouped_rows = query_tile * members
    log2e = 1.4426950408889634
    with T.Kernel(
        T.ceildiv(tokens, query_tile),
        kv_heads * head_blocks,
        partitions,
        threads=threads,
    ) as (
        block,
        head_block,
        partition,
    ):
        kv_head = head_block // head_blocks
        first_member = head_block % head_blocks * members
        query_fragment = T.alloc_fragment((grouped_rows, width), dtype)
        output_fragment = T.alloc_fragment((grouped_rows, width), "float32")
        scores = T.alloc_fragment((grouped_rows, key_tile), "float32")
        kv = T.alloc_shared((1, key_tile, width), "float32")
        keys = T.view(
            kv,
            shape=(1, key_tile, width * (2 if dtype in ("float16", "bfloat16") else 1)),
            dtype=dtype,
        )
        maximum = T.alloc_fragment((grouped_rows,), "float32")
        previous = T.alloc_fragment((grouped_rows,), "float32")
        denominator = T.alloc_fragment((grouped_rows,), "float32")
        local_sum = T.alloc_fragment((grouped_rows,), "float32")
        alpha = T.alloc_fragment((grouped_rows,), "float32")
        first_row = block * query_tile
        last_row = T.min(tokens - 1, first_row + query_tile - 1)
        base = T.cast(visible[first_row, 0], "int32")
        count = T.cast(visible[last_row, 1], "int32")
        first = partition * span
        partition_count = T.max(0, T.min(span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        T.clear(output_fragment)
        for grouped_row, channel in T.Parallel(grouped_rows, width):
            member = grouped_row // query_tile
            row = grouped_row % query_tile
            token = first_row + row
            head = kv_head * group + first_member + member
            query_fragment[grouped_row, channel] = T.if_then_else(
                partition_count > 0 and token < tokens and first_member + member < group,
                query[token, head, channel],
                0,
            )
        for chunk in T.serial(T.ceildiv(partition_count, key_tile)):
            aligned = first + (chunk + 1) * key_tile <= count
            if aligned:
                for item, channel in T.Parallel(key_tile, width):
                    relative = first + chunk * key_tile + item
                    keys[0, item, channel] = history[0, base + relative, kv_head, channel]
            else:
                for item, channel in T.Parallel(key_tile, width):
                    relative = first + chunk * key_tile + item
                    keys[0, item, channel] = T.if_then_else(
                        relative < count,
                        history[0, base + relative, kv_head, channel],
                        0,
                    )
            T.gemm(
                query_fragment,
                keys[0, :, :width],
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            wholly_visible = (
                aligned
                and first_row + query_tile <= tokens
                and first + (chunk + 1) * key_tile <= visible[first_row, 1]
            )
            if wholly_visible:
                for grouped_row, item in T.Parallel(grouped_rows, key_tile):
                    scores[grouped_row, item] *= scale * log2e
            else:
                for grouped_row, item in T.Parallel(grouped_rows, key_tile):
                    row = grouped_row % query_tile
                    token = first_row + row
                    relative = first + chunk * key_tile + item
                    scores[grouped_row, item] = T.if_then_else(
                        token < tokens
                        and first_member + grouped_row // query_tile < group
                        and relative < visible[token, 1],
                        scores[grouped_row, item] * scale * log2e,
                        -3.402823466e38,
                    )
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for grouped_row in T.Parallel(grouped_rows):
                alpha[grouped_row] = T.exp2(previous[grouped_row] - maximum[grouped_row])
            for grouped_row, item in T.Parallel(grouped_rows, key_tile):
                scores[grouped_row, item] = T.if_then_else(
                    scores[grouped_row, item] > -3.402823466e38,
                    T.exp2(scores[grouped_row, item] - maximum[grouped_row]),
                    0,
                )
            T.reduce_sum(scores, local_sum, dim=1)
            for grouped_row in T.Parallel(grouped_rows):
                denominator[grouped_row] = (
                    denominator[grouped_row] * alpha[grouped_row] + local_sum[grouped_row]
                )
            for grouped_row, channel in T.Parallel(grouped_rows, width):
                output_fragment[grouped_row, channel] *= alpha[grouped_row]
            for item, channel in T.Parallel(key_tile, width):
                relative = first + chunk * key_tile + item
                kv[0, item, channel] = T.if_then_else(
                    relative < count,
                    T.cast(history[1, base + relative, kv_head, channel], "float32"),
                    0,
                )
            T.gemm(scores, kv[0, :, :], output_fragment, policy=T.GemmWarpPolicy.FullRow)
        if fuse_gate and partitions == 1:
            # Prefill already has abundant row/head parallelism.  Publish the
            # normalized, gated tile directly instead of materializing a
            # capacity-sized FP32 partial tensor only to merge one useful run.
            for grouped_row, channel in T.Parallel(grouped_rows, width):
                member = grouped_row // query_tile
                row = grouped_row % query_tile
                token = first_row + row
                if token < tokens and first_member + member < group:
                    head = kv_head * group + first_member + member
                    gate_value = T.cast(gate[token, head, channel], "float32")
                    attended = T.cast(
                        output_fragment[grouped_row, channel]
                        / T.max(denominator[grouped_row], 1e-30),
                        dtype,
                    )
                    coefficient = T.cast(T.sigmoid(gate_value), dtype)
                    partials[token, head * width + channel] = T.cast(
                        T.cast(attended, "float32") * T.cast(coefficient, "float32"),
                        dtype,
                    )
        else:
            for grouped_row in T.Parallel(grouped_rows):
                member = grouped_row // query_tile
                row = grouped_row % query_tile
                token = first_row + row
                head = kv_head * group + first_member + member
                if token < tokens and first_member + member < group:
                    statistics[partition, token, head, 0] = T.if_then_else(
                        denominator[grouped_row] > 0,
                        maximum[grouped_row] / log2e,
                        -3.402823466e38,
                    )
                    statistics[partition, token, head, 1] = denominator[grouped_row]
            for grouped_row, channel in T.Parallel(grouped_rows, width):
                member = grouped_row // query_tile
                row = grouped_row % query_tile
                token = first_row + row
                if (
                    token < tokens
                    and first_member + member < group
                    and denominator[grouped_row] > 0
                ):
                    head = kv_head * group + first_member + member
                    # Publish the unnormalized numerator. Empty partitions
                    # publish only zero statistics, never a width-sized tile.
                    partials[partition, token, head, channel] = output_fragment[
                        grouped_row, channel
                    ]


@T.macro
def _register_partition_attention(
    query,
    history,
    visible,
    partials,
    statistics,
    tokens,
    heads,
    kv_heads,
    width,
    partitions,
    span,
    scale,
    subgroups,
):
    """Scan K/V once per grouped-query head set with online state in registers."""
    group = heads // kv_heads
    values_per_lane = width // 32
    subspan = T.ceildiv(span, subgroups)
    with T.Kernel(kv_heads, tokens, partitions, threads=32 * subgroups) as (
        kv_head,
        token,
        partition,
    ):
        thread = T.get_thread_binding()
        lane = thread % 32
        subgroup = thread // 32
        query_values = T.alloc_local((group, values_per_lane), "float32")
        accumulator = T.alloc_local((group, values_per_lane), "float32")
        key_values = T.alloc_local((values_per_lane,), "float32")
        value_values = T.alloc_local((values_per_lane,), "float32")
        maximum = T.alloc_local((group,), "float32")
        denominator = T.alloc_local((group,), "float32")
        scratch = T.alloc_shared((subgroups, group, width + 2), "float32")
        for member in T.unroll(group, explicit=True):
            maximum[member] = -3.402823466e38
            denominator[member] = 0.0
            for item in T.unroll(values_per_lane, explicit=True):
                channel = lane * values_per_lane + item
                query_values[member, item] = T.cast(
                    query[token, kv_head * group + member, channel], "float32"
                )
                accumulator[member, item] = 0.0
        start = T.cast(visible[token, 0], "int32")
        count = T.cast(visible[token, 1], "int32")
        partition_first = partition * span
        begin = partition_first + subgroup * subspan
        end = T.min(count, T.min(partition_first + span, begin + subspan))
        for relative in T.serial(begin, T.max(begin, end)):
            for item in T.unroll(values_per_lane, explicit=True):
                channel = lane * values_per_lane + item
                key_values[item] = T.cast(history[0, start + relative, kv_head, channel], "float32")
                value_values[item] = T.cast(
                    history[1, start + relative, kv_head, channel], "float32"
                )
            for member in T.unroll(group, explicit=True):
                dot = T.alloc_local((1,), "float32")
                dot[0] = 0.0
                for item in T.unroll(values_per_lane, explicit=True):
                    dot[0] += query_values[member, item] * key_values[item]
                score = T.warp_reduce_sum(dot[0]) * scale
                next_maximum = T.max(maximum[member], score)
                previous_weight = T.__exp(maximum[member] - next_maximum)
                current_weight = T.__exp(score - next_maximum)
                denominator[member] = denominator[member] * previous_weight + current_weight
                for item in T.unroll(values_per_lane, explicit=True):
                    accumulator[member, item] = (
                        accumulator[member, item] * previous_weight
                        + current_weight * value_values[item]
                    )
                maximum[member] = next_maximum
        for member in T.unroll(group, explicit=True):
            for item in T.unroll(values_per_lane, explicit=True):
                scratch[subgroup, member, lane * values_per_lane + item] = accumulator[member, item]
            if lane == 0:
                scratch[subgroup, member, width] = maximum[member]
                scratch[subgroup, member, width + 1] = denominator[member]
        T.sync_threads()
        if subgroup == 0:
            for member in T.unroll(group, explicit=True):
                maximum[member] = -3.402823466e38
                denominator[member] = 0.0
                for part in T.unroll(subgroups, explicit=True):
                    maximum[member] = T.max(maximum[member], scratch[part, member, width])
                for item in T.unroll(values_per_lane, explicit=True):
                    accumulator[member, item] = 0.0
                for part in T.unroll(subgroups, explicit=True):
                    weight = T.if_then_else(
                        scratch[part, member, width + 1] > 0,
                        T.__exp(scratch[part, member, width] - maximum[member]),
                        0.0,
                    )
                    denominator[member] += weight * scratch[part, member, width + 1]
                    for item in T.unroll(values_per_lane, explicit=True):
                        accumulator[member, item] += (
                            weight * scratch[part, member, lane * values_per_lane + item]
                        )
                head = kv_head * group + member
                for item in T.unroll(values_per_lane, explicit=True):
                    channel = lane * values_per_lane + item
                    if denominator[member] > 0:
                        partials[partition, token, head, channel] = accumulator[member, item]
                if lane == 0:
                    statistics[partition, token, head, 0] = maximum[member]
                    statistics[partition, token, head, 1] = denominator[member]


@T.macro
def _merge_attention(
    partials,
    statistics,
    output,
    tokens,
    heads,
    width,
    partitions,
    threads,
    dtype,
):
    with T.Kernel(heads, tokens, threads=threads) as (head, token):
        lane = T.get_thread_binding(0)
        maximum = T.alloc_local((1,), "float32")
        denominator = T.alloc_local((1,), "float32")
        answer = T.alloc_local((1,), "float32")
        maximum[0] = -3.402823466e38
        denominator[0] = 0.0
        answer[0] = 0.0
        for partition in T.serial(partitions):
            if statistics[partition, token, head, 1] > 0:
                maximum[0] = T.max(maximum[0], statistics[partition, token, head, 0])
        if lane < width:
            for partition in T.serial(partitions):
                if statistics[partition, token, head, 1] > 0:
                    weight = T.__exp(statistics[partition, token, head, 0] - maximum[0])
                    denominator[0] += weight * statistics[partition, token, head, 1]
                    answer[0] += weight * partials[partition, token, head, lane]
            output[token, head, lane] = T.cast(answer[0] / T.max(denominator[0], 1e-30), dtype)


class _MatrixAttentionEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        scale: float,
        partitions: int,
        span: int,
        tile,
        merge_threads: int,
    ):
        self.specs, self.scale, self.tile = specs, scale, tile
        self.partitions, self.span = partitions, span
        self.merge_threads = merge_threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        query, history, visible, output, partials, statistics = operands
        query_tile, key_tile, threads = self.tile
        _matrix_streaming_attention(
            query,
            history,
            visible,
            query,
            partials,
            statistics,
            tokens,
            heads,
            kv_heads,
            width,
            self.scale,
            self.partitions,
            self.span,
            query_tile,
            key_tile,
            threads,
            self.specs[0].dtype.value,
            False,
        )
        _merge_attention(
            partials,
            statistics,
            output,
            tokens,
            heads,
            width,
            self.partitions,
            self.merge_threads,
            self.specs[0].dtype.value,
        )


class _PartitionedAttentionEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        scale: float,
        partitions: int,
        span: int,
        subgroups: int,
        merge_threads: int,
    ):
        self.specs = specs
        self.scale = scale
        self.partitions = partitions
        self.span = span
        self.subgroups = subgroups
        self.merge_threads = merge_threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, output, partials, statistics = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _register_partition_attention(
            query,
            history,
            visible,
            partials,
            statistics,
            tokens,
            heads,
            kv_heads,
            width,
            self.partitions,
            self.span,
            self.scale,
            self.subgroups,
        )
        _merge_attention(
            partials,
            statistics,
            output,
            tokens,
            heads,
            width,
            self.partitions,
            self.merge_threads,
            self.specs[0].dtype.value,
        )


@T.macro
def _reference_attention(
    query,
    history,
    visible,
    output,
    tokens,
    heads,
    kv_heads,
    width,
    capacity,
    scale,
    visibility_rank,
    dtype,
):
    """Deliberately slow TileLang oracle, selectable only by reference builds."""
    group = heads // kv_heads
    with T.Kernel(width, heads, tokens, threads=1) as (output_channel, head, token):
        start = T.alloc_local((1,), "int32")
        count = T.alloc_local((1,), "int32")
        if visibility_rank == 2:
            start[0] = T.cast(visible[token, 0], "int32")
            count[0] = T.cast(visible[token, 1], "int32")
        else:
            start[0] = 0
            count[0] = T.cast(visible[token], "int32")
        maximum = T.alloc_local((1,), "float32")
        denominator = T.alloc_local((1,), "float32")
        mixed = T.alloc_local((1,), "float32")
        maximum[0] = -3.402823466e38
        denominator[0] = 0.0
        mixed[0] = 0.0
        kv_head = head // group
        for relative in T.serial(capacity):
            if relative < count[0]:
                position = start[0] + relative
                score = T.alloc_local((1,), "float32")
                score[0] = 0.0
                for channel in T.serial(width):
                    score[0] += T.cast(query[token, head, channel], "float32") * T.cast(
                        history[0, position, kv_head, channel], "float32"
                    )
                score[0] *= scale
                next_maximum = T.max(maximum[0], score[0])
                old_scale = T.__exp(maximum[0] - next_maximum)
                probability = T.__exp(score[0] - next_maximum)
                denominator[0] = denominator[0] * old_scale + probability
                mixed[0] = mixed[0] * old_scale + probability * T.cast(
                    history[1, position, kv_head, output_channel], "float32"
                )
                maximum[0] = next_maximum
        output[token, head, output_channel] = T.cast(mixed[0] / T.max(denominator[0], 1e-30), dtype)


class _ReferenceAttentionEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], scale: float):
        self.specs, self.scale = specs, scale

    def __call__(self, operands: tuple[Any, ...]) -> None:
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        _reference_attention(
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            tokens,
            heads,
            cast(int, self.specs[1].shape[2]),
            width,
            cast(int, self.specs[1].shape[1]),
            self.scale,
            self.specs[2].rank,
            self.specs[0].dtype.value,
        )


class CausalAttentionRule:
    name = "causal-attention"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if (
            node.operation != "causal_attention"
            or len(node.inputs) != 3
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        if any(not spec.static for spec in specs):
            return ()
        width = cast(int, specs[0].shape[-1])
        threads = 1 << (width - 1).bit_length()
        if threads > context.capabilities.threads_per_group:
            return ()
        rows, heads, width = cast(tuple[int, int, int], specs[0].shape)
        capacity = cast(int, specs[1].shape[1])
        operations = 4 * rows * heads * capacity * width
        candidates = []
        if (
            context.precision == "reference"
            or "reference_schedules" in context.capabilities.features
        ):
            candidates.append(
                Candidate(
                    f"causal_attention.reference@{root}",
                    frozenset({root}),
                    node.inputs,
                    node.outputs,
                    _ReferenceAttentionEmitter(specs, node.attributes["scale"]),
                    1.0 + operations / 1e9,
                    priority=-100,
                )
            )
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == specs[0].dtype
            ),
            None,
        )
        if (
            context.mode == "prefill"
            and node.attributes["sequence_count"] == 1
            and instruction is not None
            and any(
                item.input_dtype == DType.F32 and item.accumulation_dtype == DType.F32
                for item in context.capabilities.matrix_instructions
            )
            and width >= instruction.k
            and width % instruction.k == 0
        ):
            query_tile = instruction.m * 4
            key_tile = instruction.n * 4
            matrix_threads = min(
                context.capabilities.threads_per_group,
                context.capabilities.subgroup_width * 4,
            )
            shared = key_tile * width * 4
            if shared <= context.capabilities.shared_memory_bytes:
                span = 4096
                partitions = math.ceil(capacity / span)
                partials = TensorSpec((partitions, rows, heads, width), DType.F32)
                statistics = TensorSpec((partitions, rows, heads, 2), DType.F32)
                workspace = (partials, statistics)
                if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
                    return tuple(candidates)
                candidates.append(
                    Candidate(
                        f"causal_attention.matrix-streaming@{root}",
                        frozenset({root}),
                        node.inputs,
                        node.outputs,
                        _MatrixAttentionEmitter(
                            specs,
                            node.attributes["scale"],
                            partitions,
                            span,
                            (query_tile, key_tile, matrix_threads),
                            threads,
                        ),
                        5e-7 + operations / 5e12,
                        workspace=workspace,
                        kernel_count=2,
                        priority=50,
                    )
                )
        if (
            context.mode == "decode"
            and capacity > 512
            and width == 256
            and context.capabilities.subgroup_width == 32
            and context.capabilities.threads_per_group >= 64
        ):
            subgroups = 2
            stride = context.capabilities.subgroup_width * subgroups
            target_partitions = max(1, math.ceil(512 / (rows * cast(int, specs[1].shape[2]))))
            span = math.ceil(capacity / target_partitions / stride) * stride
            partitions = math.ceil(capacity / span)
            partials = TensorSpec((partitions, rows, heads, width), DType.F32)
            statistics = TensorSpec((partitions, rows, heads, 2), DType.F32)
            if partials.storage_nbytes + statistics.storage_nbytes <= context.workspace_limit:
                candidates.append(
                    Candidate(
                        f"causal_attention.register-partitioned@{root}",
                        frozenset({root}),
                        node.inputs,
                        node.outputs,
                        _PartitionedAttentionEmitter(
                            specs,
                            node.attributes["scale"],
                            partitions,
                            span,
                            subgroups,
                            threads,
                        ),
                        5e-7 + rows * heads * capacity * width / 2e12,
                        workspace=(partials, statistics),
                        kernel_count=2,
                        priority=45,
                    )
                )
        return tuple(candidates)


@T.macro
def _merge_attention_gate(
    partials,
    statistics,
    gate,
    output,
    tokens,
    heads,
    width,
    partitions,
    threads,
    dtype,
):
    with T.Kernel(heads, tokens, threads=threads) as (head, token):
        lane = T.get_thread_binding()
        maximum = T.alloc_local((1,), "float32")
        denominator = T.alloc_local((1,), "float32")
        answer = T.alloc_local((1,), "float32")
        maximum[0] = -3.402823466e38
        denominator[0] = 0.0
        answer[0] = 0.0
        for partition in T.serial(partitions):
            if statistics[partition, token, head, 1] > 0:
                maximum[0] = T.max(maximum[0], statistics[partition, token, head, 0])
        if lane < width:
            for partition in T.serial(partitions):
                if statistics[partition, token, head, 1] > 0:
                    weight = T.__exp(statistics[partition, token, head, 0] - maximum[0])
                    denominator[0] += weight * statistics[partition, token, head, 1]
                    answer[0] += weight * partials[partition, token, head, lane]
            gate_value = T.cast(gate[token, head, lane], "float32")
            attended = T.cast(answer[0] / T.max(denominator[0], 1e-30), dtype)
            coefficient = T.cast(T.sigmoid(gate_value), dtype)
            output[token, head * width + lane] = T.cast(
                T.cast(attended, "float32") * T.cast(coefficient, "float32"), dtype
            )


def _attention_output_region(graph: Graph, root: int):
    if root + 4 >= len(graph.nodes):
        return None
    attention, sigmoid, multiply, reshape, linear = graph.nodes[root : root + 5]
    if (
        attention.operation != "causal_attention"
        or sigmoid.operation != "sigmoid"
        or multiply.operation != "multiply"
        or attention.outputs[0] not in multiply.inputs
        or sigmoid.outputs[0] not in multiply.inputs
        or reshape.operation != "reshape"
        or reshape.inputs != multiply.outputs
        or linear.operation != "linear"
        or linear.inputs[0] != reshape.outputs[0]
        or len(attention.inputs) != 3
    ):
        return None
    gate = sigmoid.inputs[0]
    inputs = (*attention.inputs, gate, linear.inputs[1])
    outputs = linear.outputs
    specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
    if any(not spec.static for spec in specs) or packet_format(specs[4]) is None:
        return None
    return frozenset(range(root, root + 5)), inputs, outputs, specs, attention.attributes["scale"]


class _AttentionOutputEmitter:
    def __init__(self, specs, scale, partitions, span, subgroups, merge_threads, vector):
        self.specs, self.scale = specs, scale
        self.partitions, self.span, self.subgroups = partitions, span, subgroups
        self.merge_threads = merge_threads
        self.vector = vector

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, partials, statistics, activation = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _register_partition_attention(
            query,
            history,
            visible,
            partials,
            statistics,
            tokens,
            heads,
            kv_heads,
            width,
            self.partitions,
            self.span,
            self.scale,
            self.subgroups,
        )
        _merge_attention_gate(
            partials,
            statistics,
            gate,
            activation,
            tokens,
            heads,
            width,
            self.partitions,
            self.merge_threads,
            self.specs[0].dtype.value,
        )
        vector_threads, outputs_per_subgroup = self.vector
        _packed_vector(
            activation,
            weight,
            activation,
            output,
            self.specs[4],
            tokens,
            cast(int, self.specs[4].shape[0]),
            heads * width,
            self.specs[5].dtype.value,
            False,
            vector_threads,
            outputs_per_subgroup,
        )


class _PrefillAttentionOutputEmitter:
    def __init__(self, specs, scale, partitions, span, attention_tile, projection_tile):
        self.specs, self.scale = specs, scale
        self.partitions, self.span = partitions, span
        self.attention_tile, self.projection_tile = attention_tile, projection_tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, activation = operands[:7]
        partials, statistics = operands[7:] if self.partitions > 1 else (activation, activation)
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        query_tile, key_tile, attention_threads = self.attention_tile
        _matrix_streaming_attention(
            query,
            history,
            visible,
            gate,
            partials,
            statistics,
            tokens,
            heads,
            kv_heads,
            width,
            self.scale,
            self.partitions,
            self.span,
            query_tile,
            key_tile,
            attention_threads,
            self.specs[0].dtype.value,
            True,
        )
        if self.partitions > 1:
            _merge_attention_gate(
                partials,
                statistics,
                gate,
                activation,
                tokens,
                heads,
                width,
                self.partitions,
                max(width, 128),
                self.specs[0].dtype.value,
            )
        projection_threads, bm, bn, bk, arithmetic_dtype = self.projection_tile
        _packed_matrix(
            activation,
            weight,
            activation,
            output,
            self.specs[4],
            tokens,
            cast(int, self.specs[4].shape[0]),
            heads * width,
            arithmetic_dtype,
            self.specs[5].dtype.value,
            projection_threads,
            bm,
            bn,
            bk,
            False,
        )


class AttentionOutputRule:
    """Matrix attention, query gating, flattening, and packed output projection."""

    name = "attention-output"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        region = _attention_output_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs, scale = region
        tokens, heads, width = cast(tuple[int, int, int], specs[0].shape)
        capacity = cast(int, specs[1].shape[1])
        packet = packet_format(specs[4])
        if (
            packet is None
            or heads * width % packet.tile
            or context.capabilities.subgroup_width != 32
        ):
            return ()
        if context.mode == "prefill":
            attention = next(
                (
                    item
                    for item in context.capabilities.matrix_instructions
                    if item.input_dtype == specs[0].dtype
                ),
                None,
            )
            if (
                attention is None
                or not any(
                    item.input_dtype == DType.F32 and item.accumulation_dtype == DType.F32
                    for item in context.capabilities.matrix_instructions
                )
                or graph.nodes[root].attributes["sequence_count"] != 1
                or width < attention.k
                or width % attention.k
            ):
                return ()
            query_tile = attention.m * 4
            key_tile = attention.n * 4
            group = heads // cast(int, specs[1].shape[2])
            members = (
                4
                if group % 4 == 0 and context.capabilities.threads_per_group >= 512
                else (2 if group % 2 == 0 and context.capabilities.threads_per_group >= 256 else 1)
            )
            attention_threads = min(
                context.capabilities.threads_per_group,
                context.capabilities.subgroup_width * 4 * members,
            )
            projection_threads = min(context.capabilities.threads_per_group, 128)
            projection = _packet_matrix_instruction(context, specs[0].dtype)
            if projection is None:
                return ()
            if tokens >= 256 and min(cast(int, specs[4].shape[0]), heads * width) >= 512:
                bm, bn = 32, 64
            else:
                bm, bn = projection.m * 4, projection.n * 4
            bk = _packet_reduction_width(specs[4])
            if bk % projection.k:
                return ()
            projection_threads = min(
                projection_threads, bm // projection.m * context.capabilities.subgroup_width
            )
            attention_shared = key_tile * width * 4
            projection_shared = (bm + bn) * bk * projection.input_dtype.itemsize
            if max(attention_shared, projection_shared) > context.capabilities.shared_memory_bytes:
                return ()
            # Bound each history traversal. Whole-buffer workspace reuse keeps
            # partition storage shared across sequential layers, while short
            # histories publish the gated activation without a merge.
            partitions = math.ceil(capacity / 4096)
            span = math.ceil(capacity / (partitions * key_tile)) * key_tile
            activation = TensorSpec((tokens, heads * width), specs[0].dtype)
            workspace = (activation,)
            if partitions > 1:
                workspace += (
                    TensorSpec((partitions, tokens, heads, width), DType.F32),
                    TensorSpec((partitions, tokens, heads, 2), DType.F32),
                )
            if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
                return ()
            return (
                Candidate(
                    f"attention.matrix-streaming-gated-output@{root}:{max(nodes)}",
                    nodes,
                    inputs,
                    outputs,
                    _PrefillAttentionOutputEmitter(
                        specs,
                        scale,
                        partitions,
                        span,
                        (query_tile, key_tile, attention_threads),
                        (projection_threads, bm, bn, bk, projection.input_dtype.value),
                    ),
                    8e-7
                    + tokens * heads * capacity * width / 5e12
                    + specs[4].storage_nbytes / 5e12,
                    workspace=workspace,
                    kernel_count=2 if partitions == 1 else 3,
                    priority=110,
                ),
            )
        if context.mode != "decode" or capacity <= 512:
            return ()
        vector = _packed_vector_geometry(specs[4], context)
        if vector is None:
            return ()
        if width != 256 or context.capabilities.threads_per_group < 64:
            return ()
        subgroups = 2
        stride = context.capabilities.subgroup_width * subgroups
        target_partitions = max(1, math.ceil(512 / (tokens * cast(int, specs[1].shape[2]))))
        span = math.ceil(capacity / target_partitions / stride) * stride
        partitions = math.ceil(capacity / span)
        partials = TensorSpec((partitions, tokens, heads, width), DType.F32)
        statistics = TensorSpec((partitions, tokens, heads, 2), DType.F32)
        activation = TensorSpec((tokens, heads * width), specs[0].dtype)
        workspace = (partials, statistics, activation)
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        return (
            Candidate(
                f"attention.register-partitioned-gated-output@{root}:{max(nodes)}",
                nodes,
                inputs,
                outputs,
                _AttentionOutputEmitter(
                    specs,
                    scale,
                    partitions,
                    span,
                    subgroups,
                    1 << (width - 1).bit_length(),
                    vector,
                ),
                8e-7 + tokens * heads * capacity * width / 2e12 + specs[4].storage_nbytes / 4e12,
                workspace=workspace,
                kernel_count=3,
                priority=100,
            ),
        )


__all__ = ["AttentionOutputRule", "CausalAttentionRule"]
