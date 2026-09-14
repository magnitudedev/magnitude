"""Channel-parallel online causal attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .matrix import (
    _packed_matrix,
    _packed_vector,
    _packed_vector_geometry,
    _packet_matrix_instruction,
    _packet_reduction_width,
)
from .packed import affine_shared_bytes, packet_format


@dataclass(frozen=True, slots=True)
class _MatrixAttentionSchedule:
    tile: tuple[int, int, int]
    partitions: int
    span: int
    shared_bytes: int
    workspace: tuple[TensorSpec, TensorSpec]


def _matrix_attention_schedule(
    query: TensorSpec, history: TensorSpec, context: LoweringContext, sequence_count: int | None,
) -> _MatrixAttentionSchedule | None:
    """One streaming body geometry, whether isolated or composed with its output.

    Composition may eliminate the final merge/publication, not secretly choose
    a wider query-head register tile than the independently measured operation.
    Keep four row subgroups and a bounded FP32 PV tile in both contexts.
    """
    if context.mode != "prefill" or sequence_count != 1:
        return None
    instructions = context.capabilities.matrix_instructions
    instruction = next((item for item in instructions
                        if item.input_dtype == query.dtype and item.accumulation_dtype == DType.F32), None)
    if instruction is None or not any(item.input_dtype == item.accumulation_dtype == DType.F32
                                      for item in instructions):
        return None
    rows, heads, width = cast(tuple[int, int, int], query.shape)
    if width < instruction.k or width % instruction.k:
        return None
    query_tile, key_tile = instruction.m * 4, instruction.n * 4
    threads = context.capabilities.subgroup_width * 4
    shared_bytes = key_tile * width * 4
    if threads > context.capabilities.threads_per_group or shared_bytes > context.capabilities.shared_memory_bytes:
        return None
    capacity = cast(int, history.shape[1])
    partitions = math.ceil(capacity / 4096)
    span = math.ceil(capacity / (partitions * key_tile)) * key_tile
    return _MatrixAttentionSchedule(
        (query_tile, key_tile, threads), partitions, span, shared_bytes,
        (TensorSpec((partitions, rows, heads, width), DType.F32),
         TensorSpec((partitions, rows, heads, 2), DType.F32)),
    )


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
    log2e = 1.4426950408889634
    with T.Kernel(
        T.ceildiv(tokens, query_tile),
        heads,
        partitions,
        threads=threads,
    ) as (
        block,
        head,
        partition,
    ):
        kv_head = head // group
        query_fragment = T.alloc_fragment((query_tile, width), dtype)
        output_fragment = T.alloc_fragment((query_tile, width), "float32")
        scores = T.alloc_fragment((query_tile, key_tile), "float32")
        kv = T.alloc_shared((1, key_tile, width), "float32")
        keys = T.view(
            kv,
            shape=(1, key_tile, width * (2 if dtype in ("float16", "bfloat16") else 1)),
            dtype=dtype,
        )
        maximum = T.alloc_fragment((query_tile,), "float32")
        previous = T.alloc_fragment((query_tile,), "float32")
        denominator = T.alloc_fragment((query_tile,), "float32")
        local_sum = T.alloc_fragment((query_tile,), "float32")
        alpha = T.alloc_fragment((query_tile,), "float32")
        first_row = block * query_tile
        last_row = T.min(tokens - 1, first_row + query_tile - 1)
        base = T.cast(visible[first_row, 0], "int32")
        count = T.cast(visible[last_row, 1], "int32")
        first = partition * span
        partition_count = T.max(0, T.min(span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        T.clear(output_fragment)
        for row, channel in T.Parallel(query_tile, width):
            token = first_row + row
            query_fragment[row, channel] = T.if_then_else(
                partition_count > 0 and token < tokens,
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
                for row, item in T.Parallel(query_tile, key_tile):
                    scores[row, item] *= scale * log2e
            else:
                for row, item in T.Parallel(query_tile, key_tile):
                    token = first_row + row
                    relative = first + chunk * key_tile + item
                    scores[row, item] = T.if_then_else(
                        token < tokens
                        and relative < visible[token, 1],
                        scores[row, item] * scale * log2e,
                        -3.402823466e38,
                    )
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for row in T.Parallel(query_tile):
                alpha[row] = T.exp2(previous[row] - maximum[row])
            for row, item in T.Parallel(query_tile, key_tile):
                scores[row, item] = T.if_then_else(
                    scores[row, item] > -3.402823466e38,
                    T.exp2(scores[row, item] - maximum[row]),
                    0,
                )
            T.reduce_sum(scores, local_sum, dim=1)
            for row in T.Parallel(query_tile):
                denominator[row] = (
                    denominator[row] * alpha[row] + local_sum[row]
                )
            for row, channel in T.Parallel(query_tile, width):
                output_fragment[row, channel] *= alpha[row]
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
            for row, channel in T.Parallel(query_tile, width):
                token = first_row + row
                if token < tokens:
                    gate_value = T.cast(gate[token, head, channel], "float32")
                    attended = T.cast(
                        output_fragment[row, channel]
                        / T.max(denominator[row], 1e-30),
                        dtype,
                    )
                    coefficient = T.cast(T.sigmoid(gate_value), dtype)
                    partials[token, head * width + channel] = T.cast(
                        T.cast(attended, "float32") * T.cast(coefficient, "float32"),
                        dtype,
                    )
        else:
            for row in T.Parallel(query_tile):
                token = first_row + row
                if token < tokens:
                    statistics[partition, token, head, 0] = T.if_then_else(
                        denominator[row] > 0,
                        maximum[row] / log2e,
                        -3.402823466e38,
                    )
                    statistics[partition, token, head, 1] = denominator[row]
            for row, channel in T.Parallel(query_tile, width):
                token = first_row + row
                if (
                    token < tokens
                    and denominator[row] > 0
                ):
                    # Publish the unnormalized numerator. Empty partitions
                    # publish only zero statistics, never a width-sized tile.
                    partials[partition, token, head, channel] = output_fragment[
                        row, channel
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

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
        if (
            context.precision == "reference"
            or "reference_schedules" in context.capabilities.features
        ):
            return (
                BoundOperation(
                    f"causal_attention.reference@{root}",
                    frozenset({root}),
                    node.inputs,
                    node.outputs,
                    _ReferenceAttentionEmitter(specs, node.attributes["scale"]),
                ),
            )
        schedule = _matrix_attention_schedule(specs[0], specs[1], context, node.attributes["sequence_count"])
        if schedule is not None:
            if sum(value.storage_nbytes for value in schedule.workspace) > context.workspace_limit:
                raise ValueError("attention partition workspace exceeds available capacity")
            return (
                BoundOperation(
                    f"causal_attention.matrix-streaming@{root}",
                    frozenset({root}), node.inputs, node.outputs,
                    _MatrixAttentionEmitter(specs, node.attributes["scale"], schedule.partitions,
                                            schedule.span, schedule.tile, threads),
                    workspace=schedule.workspace, kernel_count=2,
                ),
            )
        if (
            width % 32 == 0
            and context.capabilities.subgroup_width == 32
            and context.capabilities.threads_per_group >= 64
        ):
            group = heads // cast(int, specs[1].shape[2])
            subgroups = min(2, context.capabilities.shared_memory_bytes // (group * (width + 2) * 4))
            if subgroups < 1:
                raise ValueError("one attention head group exceeds shared-memory capacity")
            stride = context.capabilities.subgroup_width * subgroups
            target_partitions = max(1, math.ceil(512 / (rows * cast(int, specs[1].shape[2]))))
            span = math.ceil(capacity / target_partitions / stride) * stride
            partitions = math.ceil(capacity / span)
            partials = TensorSpec((partitions, rows, heads, width), DType.F32)
            statistics = TensorSpec((partitions, rows, heads, 2), DType.F32)
            if partials.storage_nbytes + statistics.storage_nbytes <= context.workspace_limit:
                return (
                    BoundOperation(
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
                        workspace=(partials, statistics),
                        kernel_count=2,
                    ),
                )
            raise ValueError("attention partition workspace exceeds available capacity")
        raise ValueError("attention requires a legal tiled matrix or subgroup reduction geometry")


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

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
            schedule = _matrix_attention_schedule(
                specs[0], specs[1], context, graph.nodes[root].attributes["sequence_count"],
            )
            if schedule is None:
                return ()
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
            projection_shared = affine_shared_bytes(bm, bn, bk, projection.input_dtype)
            if max(schedule.shared_bytes, projection_shared) > context.capabilities.shared_memory_bytes:
                return ()
            # Bound each history traversal. Whole-buffer workspace reuse keeps
            # partition storage shared across sequential layers, while short
            # histories publish the gated activation without a merge.
            partitions, span = schedule.partitions, schedule.span
            activation = TensorSpec((tokens, heads * width), specs[0].dtype)
            workspace = (activation,)
            if partitions > 1:
                workspace += schedule.workspace
            if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
                return ()
            return (
                BoundOperation(
                    f"attention.matrix-streaming-gated-output@{root}:{max(nodes)}",
                    nodes,
                    inputs,
                    outputs,
                    _PrefillAttentionOutputEmitter(
                        specs,
                        scale,
                        partitions,
                        span,
                        schedule.tile,
                        (projection_threads, bm, bn, bk, projection.input_dtype.value),
                    ),
                    workspace=workspace,
                    kernel_count=2 if partitions == 1 else 3,
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
            BoundOperation(
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
                workspace=workspace,
                kernel_count=3,
            ),
        )


__all__ = ["AttentionOutputRule", "CausalAttentionRule"]
