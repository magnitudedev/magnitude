"""Channel-parallel online causal attention."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .matrix import _packed_matrix, _packed_vector
from .packed import packet_format


@T.macro
def _matrix_streaming_attention(
    query,
    history,
    visible,
    gate,
    output,
    tokens,
    heads,
    kv_heads,
    width,
    scale,
    query_tile,
    key_tile,
    threads,
    dtype,
    fuse_gate,
):
    """Stream one packed sequence through tiled QK and PV matrix products."""
    group = heads // kv_heads
    with T.Kernel(T.ceildiv(tokens, query_tile), heads, threads=threads) as (block, head):
        query_fragment = T.alloc_fragment((query_tile, width), dtype)
        output_fragment = T.alloc_fragment((query_tile, width), "float32")
        scores = T.alloc_fragment((query_tile, key_tile), "float32")
        probabilities = T.alloc_fragment((query_tile, key_tile), dtype)
        kv = T.alloc_shared((key_tile, width), dtype)
        maximum = T.alloc_fragment((query_tile,), "float32")
        previous = T.alloc_fragment((query_tile,), "float32")
        denominator = T.alloc_fragment((query_tile,), "float32")
        local_sum = T.alloc_fragment((query_tile,), "float32")
        alpha = T.alloc_fragment((query_tile,), "float32")
        first_row = block * query_tile
        last_row = T.min(tokens - 1, first_row + query_tile - 1)
        base = T.cast(visible[first_row, 0], "int32")
        count = T.cast(visible[last_row, 1], "int32")
        kv_head = head // group
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        T.clear(output_fragment)
        for row, channel in T.Parallel(query_tile, width):
            token = first_row + row
            query_fragment[row, channel] = T.if_then_else(
                token < tokens,
                query[token, head, channel],
                0,
            )
        for chunk in T.serial(T.ceildiv(count, key_tile)):
            for item, channel in T.Parallel(key_tile, width):
                relative = chunk * key_tile + item
                kv[item, channel] = T.if_then_else(
                    relative < count,
                    history[0, base + relative, kv_head, channel],
                    0,
                )
            T.gemm(
                query_fragment,
                kv,
                scores,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for row, item in T.Parallel(query_tile, key_tile):
                token = first_row + row
                relative = chunk * key_tile + item
                scores[row, item] = T.if_then_else(
                    token < tokens and relative < visible[token, 1],
                    scores[row, item] * scale,
                    -3.402823466e38,
                )
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for row in T.Parallel(query_tile):
                alpha[row] = T.exp(previous[row] - maximum[row])
            for row, item in T.Parallel(query_tile, key_tile):
                scores[row, item] = T.if_then_else(
                    scores[row, item] > -3.402823466e38,
                    T.exp(scores[row, item] - maximum[row]),
                    0,
                )
                probabilities[row, item] = T.cast(scores[row, item], dtype)
            T.reduce_sum(scores, local_sum, dim=1)
            for row in T.Parallel(query_tile):
                denominator[row] = denominator[row] * alpha[row] + local_sum[row]
            for row, channel in T.Parallel(query_tile, width):
                output_fragment[row, channel] *= alpha[row]
            for item, channel in T.Parallel(key_tile, width):
                relative = chunk * key_tile + item
                kv[item, channel] = T.if_then_else(
                    relative < count,
                    history[1, base + relative, kv_head, channel],
                    0,
                )
            T.gemm(
                probabilities,
                kv,
                output_fragment,
                policy=T.GemmWarpPolicy.FullRow,
            )
        for row, channel in T.Parallel(query_tile, width):
            token = first_row + row
            if token < tokens:
                if fuse_gate:
                    output[token, head * width + channel] = T.cast(
                        output_fragment[row, channel]
                        / T.max(denominator[row], 1e-30)
                        * T.sigmoid(T.cast(gate[token, head, channel], "float32")),
                        dtype,
                    )
                else:
                    output[token, head, channel] = T.cast(
                        output_fragment[row, channel]
                        / T.max(denominator[row], 1e-30),
                        dtype,
                    )


@T.macro
def _partition_attention(
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
    threads,
):
    group = heads // kv_heads
    with T.Kernel(kv_heads, tokens, partitions, threads=threads) as (
        kv_head,
        token,
        partition,
    ):
        lane = T.get_thread_binding(0)
        score = T.alloc_shared((group, threads), "float32")
        state = T.alloc_shared((group, 3), "float32")
        mixed = T.alloc_local((group,), "float32")
        key = T.alloc_local((1,), "float32")
        value = T.alloc_local((1,), "float32")
        T.clear(mixed)
        start = T.cast(visible[token, 0], "int32")
        count = T.cast(visible[token, 1], "int32")
        first = partition * span
        if lane == 0:
            for member in T.unroll(group):
                state[member, 0] = -3.402823466e38
                state[member, 1] = 0.0
        T.sync_threads()
        for offset in T.serial(span):
            relative = first + offset
            if relative < count:
                position = start + relative
                key[0] = T.if_then_else(
                    lane < width,
                    T.cast(history[0, position, kv_head, lane], "float32"),
                    0.0,
                )
                value[0] = T.if_then_else(
                    lane < width,
                    T.cast(history[1, position, kv_head, lane], "float32"),
                    0.0,
                )
                for member in T.unroll(group):
                    head = kv_head * group + member
                    score[member, lane] = T.if_then_else(
                        lane < width,
                        T.cast(query[token, head, lane], "float32") * key[0],
                        0.0,
                    )
                T.sync_threads()
                for reduction in T.unroll(threads.bit_length() - 1):
                    distance = threads >> (reduction + 1)
                    for member in T.unroll(group):
                        if lane < distance:
                            score[member, lane] += score[member, lane + distance]
                    T.sync_threads()
                if lane == 0:
                    for member in T.unroll(group):
                        scaled = score[member, 0] * scale
                        maximum = T.max(state[member, 0], scaled)
                        state[member, 2] = T.exp(state[member, 0] - maximum)
                        score[member, 0] = T.exp(scaled - maximum)
                        state[member, 1] = (
                            state[member, 1] * state[member, 2] + score[member, 0]
                        )
                        state[member, 0] = maximum
                T.sync_threads()
                if lane < width:
                    for member in T.unroll(group):
                        mixed[member] = (
                            mixed[member] * state[member, 2]
                            + score[member, 0] * value[0]
                        )
                T.sync_threads()
        if lane == 0:
            for member in T.unroll(group):
                head = kv_head * group + member
                statistics[partition, token, head, 0] = state[member, 0]
                statistics[partition, token, head, 1] = state[member, 1]
        if lane < width:
            for member in T.unroll(group):
                head = kv_head * group + member
                partials[partition, token, head, lane] = T.if_then_else(
                    state[member, 1] > 0,
                    mixed[member] / state[member, 1],
                    0,
                )


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
                maximum[0] = T.max(
                    maximum[0], statistics[partition, token, head, 0]
                )
        if lane < width:
            for partition in T.serial(partitions):
                if statistics[partition, token, head, 1] > 0:
                    weight = (
                        T.exp(statistics[partition, token, head, 0] - maximum[0])
                        * statistics[partition, token, head, 1]
                    )
                    denominator[0] += weight
                    answer[0] += weight * partials[partition, token, head, lane]
            output[token, head, lane] = T.cast(
                answer[0] / T.max(denominator[0], 1e-30), dtype
            )


@T.macro
def _parallel_online_attention(
    query,
    history,
    visible,
    output,
    tokens,
    heads,
    kv_heads,
    width,
    scale,
    visibility_rank,
    threads,
    context_tile,
    dtype,
):
    group = heads // kv_heads
    with T.Kernel(heads, tokens, threads=threads) as (head, token):
        lane = T.get_thread_binding(0)
        score = T.alloc_shared((threads,), "float32")
        statistics = T.alloc_shared((3,), "float32")
        mixed = T.alloc_local((1,), "float32")
        mixed[0] = 0.0
        start = T.alloc_local((1,), "int32")
        count = T.alloc_local((1,), "int32")
        if visibility_rank == 2:
            start[0] = T.cast(visible[token, 0], "int32")
            count[0] = T.cast(visible[token, 1], "int32")
        else:
            start[0] = 0
            count[0] = T.cast(visible[token], "int32")
        if lane == 0:
            statistics[0] = -3.402823466e38
            statistics[1] = 0.0
        T.sync_threads()
        kv_head = head // group
        for tile in T.serial(T.ceildiv(count[0], context_tile)):
            for offset in T.serial(context_tile):
                relative = tile * context_tile + offset
                if relative < count[0]:
                    position = start[0] + relative
                    score[lane] = T.if_then_else(
                        lane < width,
                        T.cast(query[token, head, lane], "float32")
                        * T.cast(history[0, position, kv_head, lane], "float32"),
                        0.0,
                    )
                    T.sync_threads()
                    for reduction in T.unroll(threads.bit_length() - 1):
                        distance = threads >> (reduction + 1)
                        if lane < distance:
                            score[lane] += score[lane + distance]
                        T.sync_threads()
                    if lane == 0:
                        value = score[0] * scale
                        maximum = T.max(statistics[0], value)
                        statistics[2] = T.exp(statistics[0] - maximum)
                        score[0] = T.exp(value - maximum)
                        statistics[1] = statistics[1] * statistics[2] + score[0]
                        statistics[0] = maximum
                    T.sync_threads()
                    if lane < width:
                        mixed[0] = mixed[0] * statistics[2] + score[0] * T.cast(
                            history[1, position, kv_head, lane], "float32"
                        )
                    T.sync_threads()
        if lane < width:
            output[token, head, lane] = T.cast(mixed[0] / statistics[1], dtype)


class _AttentionEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], scale: float, threads: int):
        self.specs = specs
        self.scale = scale
        self.threads = threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _parallel_online_attention(
            operands[0],
            operands[1],
            operands[2],
            operands[3],
            tokens,
            heads,
            kv_heads,
            width,
            self.scale,
            self.specs[2].rank,
            self.threads,
            32,
            self.specs[0].dtype.value,
        )


class _MatrixAttentionEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], scale: float, tile):
        self.specs, self.scale, self.tile = specs, scale, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        query_tile, key_tile, threads = self.tile
        _matrix_streaming_attention(
            operands[0],
            operands[1],
            operands[2],
            operands[0],
            operands[3],
            tokens,
            heads,
            kv_heads,
            width,
            self.scale,
            query_tile,
            key_tile,
            threads,
            self.specs[0].dtype.value,
            False,
        )


class _PartitionedAttentionEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        scale: float,
        partitions: int,
        span: int,
        threads: int,
    ):
        self.specs = specs
        self.scale = scale
        self.partitions = partitions
        self.span = span
        self.threads = threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, output, partials, statistics = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _partition_attention(
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
            self.threads,
        )
        _merge_attention(
            partials,
            statistics,
            output,
            tokens,
            heads,
            width,
            self.partitions,
            self.threads,
            self.specs[0].dtype.value,
        )


class OnlineAttentionRule:
    name = "parallel-online-attention"

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
        candidates = [
            Candidate(
                f"causal_attention.online@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _AttentionEmitter(specs, node.attributes["scale"], threads),
                5e-7 + operations / 50e9,
                priority=35,
            )
        ]
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
            and width >= instruction.k
            and width % instruction.k == 0
        ):
            query_tile = instruction.m * 4
            key_tile = instruction.n * 4
            matrix_threads = min(
                context.capabilities.threads_per_group,
                context.capabilities.subgroup_width * 4,
            )
            shared = key_tile * width * specs[0].dtype.itemsize
            if shared <= context.capabilities.shared_memory_bytes:
                candidates.append(
                    Candidate(
                        f"causal_attention.matrix-streaming@{root}",
                        frozenset({root}),
                        node.inputs,
                        node.outputs,
                        _MatrixAttentionEmitter(
                            specs,
                            node.attributes["scale"],
                            (query_tile, key_tile, matrix_threads),
                        ),
                        5e-7 + operations / 5e12,
                        priority=50,
                    )
                )
        if context.mode == "decode" and capacity > 512:
            span = 128
            partitions = math.ceil(capacity / span)
            partials = TensorSpec((partitions, rows, heads, width), DType.F32)
            statistics = TensorSpec((partitions, rows, heads, 2), DType.F32)
            if partials.storage_nbytes + statistics.storage_nbytes <= context.workspace_limit:
                candidates.append(
                    Candidate(
                        f"causal_attention.partitioned@{root}",
                        frozenset({root}),
                        node.inputs,
                        node.outputs,
                        _PartitionedAttentionEmitter(
                            specs,
                            node.attributes["scale"],
                            partitions,
                            span,
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
    partials, statistics, gate, output, tokens, heads, width, partitions, threads, dtype,
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
                    weight = T.exp(
                        statistics[partition, token, head, 0] - maximum[0]
                    ) * statistics[partition, token, head, 1]
                    denominator[0] += weight
                    answer[0] += weight * partials[partition, token, head, lane]
            gate_value = T.cast(gate[token, head, lane], "float32")
            output[token, head * width + lane] = T.cast(
                answer[0] / T.max(denominator[0], 1e-30) * T.sigmoid(gate_value), dtype
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
        or reshape.operation != "reshape" or reshape.inputs != multiply.outputs
        or linear.operation != "linear" or linear.inputs[0] != reshape.outputs[0]
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
    def __init__(self, specs, scale, partitions, span, threads):
        self.specs, self.scale = specs, scale
        self.partitions, self.span, self.threads = partitions, span, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, partials, statistics, activation = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _partition_attention(
            query, history, visible, partials, statistics, tokens, heads, kv_heads,
            width, self.partitions, self.span, self.scale, self.threads,
        )
        _merge_attention_gate(
            partials, statistics, gate, activation, tokens, heads, width,
            self.partitions, self.threads, self.specs[0].dtype.value,
        )
        _packed_vector(
            activation, weight, activation, output, self.specs[4], tokens,
            cast(int, self.specs[4].shape[0]), heads * width,
            self.specs[5].dtype.value, False,
        )


class _PrefillAttentionOutputEmitter:
    def __init__(self, specs, scale, attention_tile, projection_tile):
        self.specs, self.scale = specs, scale
        self.attention_tile, self.projection_tile = attention_tile, projection_tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, activation = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        query_tile, key_tile, attention_threads = self.attention_tile
        _matrix_streaming_attention(
            query, history, visible, gate, activation, tokens, heads, kv_heads,
            width, self.scale, query_tile, key_tile, attention_threads,
            self.specs[0].dtype.value, True,
        )
        projection_threads, bm, bn, bk = self.projection_tile
        _packed_matrix(
            activation, weight, activation, output, self.specs[4], tokens,
            cast(int, self.specs[4].shape[0]), heads * width,
            self.specs[0].dtype.value, self.specs[5].dtype.value,
            projection_threads, bm, bn, bk, False,
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
            packet is None or heads * width % packet.tile
            or context.capabilities.subgroup_width != 32
        ):
            return ()
        if context.mode == "prefill":
            attention = next(
                (
                    item for item in context.capabilities.matrix_instructions
                    if item.input_dtype == specs[0].dtype
                ),
                None,
            )
            if (
                attention is None
                or graph.nodes[root].attributes["sequence_count"] != 1
                or width < attention.k
                or width % attention.k
            ):
                return ()
            query_tile, key_tile = attention.m * 4, attention.n * 4
            attention_threads = min(
                context.capabilities.threads_per_group,
                context.capabilities.subgroup_width * 4,
            )
            projection_threads = attention_threads
            bm, bn = attention.m * 4, attention.n * 4
            bk = max(attention.k * 2, packet.packet)
            bk = math.ceil(bk / packet.packet) * packet.packet
            attention_shared = key_tile * width * specs[0].dtype.itemsize
            projection_shared = (bm + bn) * bk * specs[0].dtype.itemsize
            if max(attention_shared, projection_shared) > context.capabilities.shared_memory_bytes:
                return ()
            activation = TensorSpec((tokens, heads * width), specs[0].dtype)
            if activation.storage_nbytes > context.workspace_limit:
                return ()
            return (Candidate(
                f"attention.matrix-streaming-gated-output@{root}:{max(nodes)}",
                nodes, inputs, outputs,
                _PrefillAttentionOutputEmitter(
                    specs, scale,
                    (query_tile, key_tile, attention_threads),
                    (projection_threads, bm, bn, bk),
                ),
                8e-7 + tokens * heads * capacity * width / 5e12
                + specs[4].storage_nbytes / 5e12,
                workspace=(activation,), kernel_count=2, priority=110,
            ),)
        if context.mode != "decode" or capacity <= 512:
            return ()
        threads = 1 << (width - 1).bit_length()
        if threads > context.capabilities.threads_per_group:
            return ()
        span = 128
        partitions = math.ceil(capacity / span)
        partials = TensorSpec((partitions, tokens, heads, width), DType.F32)
        statistics = TensorSpec((partitions, tokens, heads, 2), DType.F32)
        activation = TensorSpec((tokens, heads * width), specs[0].dtype)
        workspace = (partials, statistics, activation)
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        return (Candidate(
            f"attention.partitioned-gated-output@{root}:{max(nodes)}", nodes, inputs, outputs,
            _AttentionOutputEmitter(specs, scale, partitions, span, threads),
            8e-7 + tokens * heads * capacity * width / 2e12
            + specs[4].storage_nbytes / 4e12,
            workspace=workspace, kernel_count=3, priority=100,
        ),)


__all__ = ["AttentionOutputRule", "OnlineAttentionRule"]
