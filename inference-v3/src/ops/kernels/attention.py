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
    _packet_reduction_width,
)
from .packed import affine_shared_bytes, packet_format


@dataclass(frozen=True, slots=True)
class _MatrixAttentionSchedule:
    tile: tuple[int, int, int]
    head_tile: int
    value_tile: int
    reduction_step: int
    padding: int
    partitions: int
    span: int
    shared_bytes: int
    workspace: tuple[TensorSpec, TensorSpec]


@dataclass(frozen=True, slots=True)
class _DecodeAttentionSchedule:
    heads: int
    rows: int
    keys: int
    contraction: int
    columns: int
    reduction_step: int
    threads: int
    padding: int
    partitions: int
    span: int


def _decode_attention_schedule(query, history, context):
    """Use grouped heads as matrix rows and reduce history in complete tiles.

    Only complete head cohorts use this schedule. K/V retain their storage
    precision; both contractions and all normalization state are FP32.
    """
    if context.mode != "decode":
        return None
    rows, heads, width = query.shape
    group = heads // history.shape[2]
    head_tile = 8
    matrix_rows = 16
    keys = 32
    contraction = 32
    columns = 64
    threads = context.compiler_target.subgroup_width * 4
    padding = 8
    def staging_bytes():
        return ((width + padding) * (keys + padding) * history.dtype.itemsize
                + matrix_rows * keys * DType.F32.itemsize)

    while staging_bytes() > context.compiler_target.shared_memory_bytes and keys > 8:
        keys //= 2
    # Each subgroup owns at least one complete eight-column QK strip.
    # A narrower history tile needs fewer participants, not empty column owners.
    threads = context.compiler_target.subgroup_width * min(4, keys // 8)
    if (group % head_tile or width % contraction or width % columns
            or threads > context.compiler_target.threads_per_group
            or staging_bytes() > context.compiler_target.shared_memory_bytes):
        return None
    # Each partition amortizes staging over a bounded history segment. The
    # complete grid remains independent of changing visible token counts.
    span = keys * 16
    partitions = math.ceil(history.shape[1] / span)
    return _DecodeAttentionSchedule(head_tile, matrix_rows, keys, contraction, columns,
                                    8, threads, padding, partitions, span)


def _matrix_attention_schedule(
    query: TensorSpec, history: TensorSpec, context: LoweringContext, sequence_count: int | None,
) -> _MatrixAttentionSchedule | None:
    """One streaming body geometry, whether isolated or composed with its output.

    Composition may eliminate the final merge/publication, not secretly choose
    a wider query-head register tile than the independently measured operation.
    Share compact K/V across heads; widen only immediate PV operands.
    """
    if context.mode != "prefill" or sequence_count != 1:
        return None
    rows, heads, width = cast(tuple[int, int, int], query.shape)
    if width < 8 or width % 8:
        return None
    query_tile, key_tile = 32, 32
    group = heads // cast(int, history.shape[2])
    paired_heads = group % 2 == 0 and context.compiler_target.threads_per_group >= context.compiler_target.subgroup_width * 8
    head_tile = 2 if paired_heads else 1
    threads = context.compiler_target.subgroup_width * 4 * head_tile
    padding = 8
    def staging_bytes() -> int:
        probability = query_tile * head_tile * 8 * DType.F32.itemsize
        return (width + padding) * (key_tile + padding) * query.dtype.itemsize + probability

    while staging_bytes() > context.compiler_target.shared_memory_bytes and key_tile > 8:
        key_tile //= 2
    shared_bytes = staging_bytes()
    value_tile = min(width, query_tile * 2)
    if key_tile % 8 or (query_tile * head_tile) % 8 or value_tile % 8:
        return None
    if threads > context.compiler_target.threads_per_group or shared_bytes > context.compiler_target.shared_memory_bytes:
        return None
    capacity = cast(int, history.shape[1])
    partitions = math.ceil(capacity / 4096)
    span = math.ceil(capacity / (partitions * key_tile)) * key_tile
    return _MatrixAttentionSchedule(
        (query_tile, key_tile, threads), head_tile, value_tile, 8, padding,
        partitions, span, shared_bytes,
        (TensorSpec((partitions, rows, heads, width), DType.F32),
         TensorSpec((partitions, rows, heads, 2), DType.F32)),
    )


@T.macro
def _attention_clear(output):
    T.clear(output)


def _attention_accumulators(rows, width, columns):
    """Separate full fragments keep immediate V operands bounded by columns."""
    outputs = tuple(T.alloc_fragment((rows, columns), "float32") for _ in range(math.ceil(width / columns)))
    for output in outputs:
        _attention_clear(output)
    return outputs


@T.macro
def _attention_rescale_tile(output, alpha, rows, columns):
    for row, column in T.Parallel(rows, columns):
        output[row, column] *= alpha[row]


def _attention_rescale(outputs, alpha, rows, columns):
    for output in outputs:
        _attention_rescale_tile(output, alpha, rows, columns)


@T.macro
def _attention_value_tile(probability, values, output, first_key, first_column,
                          width, columns, reduction_step):
    operand = T.alloc_fragment((reduction_step, columns), "float32")
    for item, column in T.Parallel(reduction_step, columns):
        operand[item, column] = T.if_then_else(
            first_column + column < width,
            T.cast(values[0, first_key + item, first_column + column], "float32"), 0,
        )
    # Keep fragment coordinates static without expanding the history traversal.
    with T.attr(0, "pragma_auto_unroll_max_step", 4096):
        with T.attr(0, "pragma_unroll_explicit", 1):
            T.gemm(probability, operand, output, policy=T.GemmWarpPolicy.FullRow)


def _attention_value_columns(probability, values, outputs, first_key, width, columns, reduction_step):
    for index, output in enumerate(outputs):
        _attention_value_tile(probability, values, output, first_key, index * columns,
                              width, columns, reduction_step)


@T.macro
def _attention_values(scores, values, outputs, rows, width, keys, columns, reduction_step):
    # QK accumulator ownership and PV A-operand ownership are not generally
    # compatible. Publish each narrow probability strip so TileLang can infer
    # both matrix layouts independently on every backend.
    probability = T.alloc_shared((rows, reduction_step), "float32")
    for step in T.serial(keys // reduction_step):
        for row, item in T.Parallel(rows, reduction_step):
            probability[row, item] = scores[row, step * reduction_step + item]
        T.sync_threads()
        _attention_value_columns(probability, values, outputs, step * reduction_step,
                                  width, columns, reduction_step)
        T.sync_threads()


@T.macro
def _attention_publish_tile(output, denominator, gate, partials, first_row, first_head,
                            partition, partitions, tokens, query_tile, head_tile,
                            width, columns, first_column, dtype, fuse_gate):
    for row, column in T.Parallel(query_tile * head_tile, columns):
        token = first_row + row % query_tile
        head = first_head + row // query_tile
        channel = first_column + column
        if token < tokens and channel < width:
            if fuse_gate and partitions == 1:
                attended = T.cast(output[row, column] / T.max(denominator[row], 1e-30), dtype)
                coefficient = T.cast(T.sigmoid(T.cast(gate[token, head, channel], "float32")), dtype)
                partials[token, head * width + channel] = T.cast(
                    T.cast(attended, "float32") * T.cast(coefficient, "float32"), dtype,
                )
            elif denominator[row] > 0:
                partials[partition, token, head, channel] = output[row, column]


def _attention_publish(outputs, denominator, gate, partials, first_row, first_head,
                       partition, partitions, tokens, query_tile, head_tile, width,
                       columns, dtype, fuse_gate):
    for index, output in enumerate(outputs):
        _attention_publish_tile(output, denominator, gate, partials, first_row, first_head,
                                partition, partitions, tokens, query_tile, head_tile, width,
                                columns, index * columns, dtype, fuse_gate)


@T.macro
def _matrix_visible_count(visible, first_row, query_tile, tokens, column):
    """Padding has zero visibility; the last physical row need not be valid."""
    result = T.alloc_shared((1,), "int32")
    thread = T.get_thread_binding()
    if thread < 32:
        maximum = T.alloc_local((1,), "int32")
        maximum[0] = 0
        for group in T.unroll(T.ceildiv(query_tile, 32)):
            row = first_row + group * 32 + thread
            maximum[0] = T.max(maximum[0], T.if_then_else(
                group * 32 + thread < query_tile and row < tokens,
                T.cast(visible[row, column], "int32"), 0))
        count = T.warp_reduce_max(maximum[0])
        if thread == 0:
            result[0] = count
    T.sync_threads()
    return result[0]


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
    schedule,
    dtype,
    fuse_gate,
):
    """Reuse compact K/V across query heads with immediate FP32 PV operands."""
    group = heads // kv_heads
    query_tile, key_tile, threads = schedule.tile
    partitions, span = schedule.partitions, schedule.span
    head_tile, columns = schedule.head_tile, schedule.value_tile
    query_rows = query_tile * head_tile
    padding = schedule.padding
    log2e = 1.4426950408889634
    with T.Kernel(
        T.ceildiv(tokens, query_tile),
        heads // head_tile,
        partitions,
        threads=threads,
    ) as (
        block,
        head_block,
        partition,
    ):
        first_head = head_block * head_tile
        kv_head = first_head // group
        query_fragment = T.alloc_fragment((query_rows, width), dtype)
        outputs = _attention_accumulators(query_rows, width, columns)
        scores = T.alloc_fragment((query_rows, key_tile), "float32")
        # The 3D views retain their authored physical pitch. K is oriented for
        # QK; V reuses that allocation only after QK has consumed it.
        # Both views cover the entire backing, including padding on both axes;
        # a smaller alias must not become the allocation's inferred extent.
        kv = T.alloc_shared(((width + padding) * (key_tile + padding),), dtype)
        keys = T.view(kv, shape=(1, width + padding, key_tile + padding), dtype=dtype)
        values = T.view(kv, shape=(1, key_tile + padding, width + padding), dtype=dtype)
        maximum = T.alloc_fragment((query_rows,), "float32")
        previous = T.alloc_fragment((query_rows,), "float32")
        denominator = T.alloc_fragment((query_rows,), "float32")
        local_sum = T.alloc_fragment((query_rows,), "float32")
        alpha = T.alloc_fragment((query_rows,), "float32")
        first_row = block * query_tile
        last_row = T.min(tokens - 1, first_row + query_tile - 1)
        base = T.cast(visible[first_row, 0], "int32")
        count = _matrix_visible_count(visible, first_row, query_tile, tokens, 1)
        # Visibility is a valid interval within the history allocation. The
        # single-sequence matrix schedule shares its base across this query tile.
        T.assume(base >= 0)
        T.assume(count >= 0)
        T.assume(base <= history.shape[1])
        T.assume(count <= history.shape[1] - base)
        first = partition * span
        partition_count = T.max(0, T.min(span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        for row, channel in T.Parallel(query_rows, width):
            token = first_row + row % query_tile
            head = first_head + row // query_tile
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
                    keys[0, channel, item] = history[0, base + relative, kv_head, channel]
            else:
                for item, channel in T.Parallel(key_tile, width):
                    relative = first + chunk * key_tile + item
                    keys[0, channel, item] = T.if_then_else(
                        relative < count,
                        history[0, base + relative, kv_head, channel],
                        0,
                    )
            # Expand the bounded contraction, preserving the full score reduction.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(
                        query_fragment,
                        keys[0, :width, :key_tile],
                        scores,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
            wholly_visible = (
                aligned
                and first_row + query_tile <= tokens
                and first + (chunk + 1) * key_tile <= visible[first_row, 1]
                and first + (chunk + 1) * key_tile <= visible[last_row, 1]
            )
            if wholly_visible:
                for row, item in T.Parallel(query_rows, key_tile):
                    scores[row, item] *= scale * log2e
            else:
                for row, item in T.Parallel(query_rows, key_tile):
                    token = first_row + row % query_tile
                    relative = first + chunk * key_tile + item
                    scores[row, item] = T.if_then_else(
                        token < tokens
                        and relative < visible[token, 1],
                        scores[row, item] * scale * log2e,
                        -3.402823466e38,
                    )
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for row in T.Parallel(query_rows):
                alpha[row] = T.exp2(previous[row] - maximum[row])
            for row, item in T.Parallel(query_rows, key_tile):
                scores[row, item] = T.if_then_else(
                    scores[row, item] > -3.402823466e38,
                    T.exp2(scores[row, item] - maximum[row]),
                    0,
                )
            T.reduce_sum(scores, local_sum, dim=1)
            for row in T.Parallel(query_rows):
                denominator[row] = (
                    denominator[row] * alpha[row] + local_sum[row]
                )
            _attention_rescale(outputs, alpha, query_rows, columns)
            for item, channel in T.Parallel(key_tile, width):
                relative = first + chunk * key_tile + item
                values[0, item, channel] = T.if_then_else(
                    relative < count,
                    history[1, base + relative, kv_head, channel],
                    0,
                )
            _attention_values(scores, values, outputs, query_rows, width, key_tile,
                              columns, schedule.reduction_step)
        if not fuse_gate or partitions > 1:
            for row in T.Parallel(query_rows):
                token = first_row + row % query_tile
                head = first_head + row // query_tile
                if token < tokens:
                    statistics[partition, token, head, 0] = T.if_then_else(
                        denominator[row] > 0,
                        maximum[row] / log2e,
                        -3.402823466e38,
                    )
                    statistics[partition, token, head, 1] = denominator[row]
        _attention_publish(outputs, denominator, gate, partials, first_row, first_head,
                           partition, partitions, tokens, query_tile, head_tile, width,
                           columns, dtype, fuse_gate)


@T.macro
def _decode_scores(query, keys, scores, token, first_head, heads, rows, width,
                   key_tile, contraction, key_major=False):
    """Immediate FP32 operands avoid narrowing Q/K or dynamic fragment indices."""
    q = T.alloc_fragment((rows, contraction), "float32")
    k = T.alloc_fragment((contraction, key_tile), "float32")
    T.clear(scores)
    for block in T.serial(width // contraction):
        for head, channel in T.Parallel(rows, contraction):
            q[head, channel] = T.if_then_else(
                head < heads,
                T.cast(query[token, first_head + head, block * contraction + channel], "float32"),
                0,
            )
        for channel, item in T.Parallel(contraction, key_tile):
            if key_major:
                k[channel, item] = T.cast(keys[0, item, block * contraction + channel], "float32")
            else:
                k[channel, item] = T.cast(keys[0, block * contraction + channel, item], "float32")
        T.gemm(q, k, scores, policy=T.GemmWarpPolicy.FullRow)


@T.macro
def _tiled_decode_attention(query, history, visible, partials, statistics,
                             tokens, heads, kv_heads, width, scale, schedule):
    """One K/V tile serves a complete query-head cohort and a stable softmax.

    The online dependency advances per history tile. Both QK and PV use matrix
    contraction with FP32 operands; only the original K/V staging is compact.
    """
    group = heads // kv_heads
    head_tile, matrix_rows, key_tile = schedule.heads, schedule.rows, schedule.keys
    columns, padding = schedule.columns, schedule.padding
    log2e = 1.4426950408889634
    with T.Kernel(heads // head_tile, tokens, schedule.partitions,
                  threads=schedule.threads) as (cohort, token, partition):
        first_head = cohort * head_tile
        kv_head = first_head // group
        outputs = _attention_accumulators(matrix_rows, width, columns)
        scores = T.alloc_fragment((matrix_rows, key_tile), "float32")
        # QK distributes history columns across subgroups. PV distributes
        # output channels, so every subgroup needs the probability tile.
        # Make that exchange explicit instead of copying incompatible fragments.
        probabilities = T.alloc_shared((matrix_rows, key_tile), "float32")
        staging = T.alloc_shared(((width + padding) * (key_tile + padding),), history.dtype)
        keys = T.view(staging, shape=(1, width + padding, key_tile + padding), dtype=history.dtype)
        values = T.view(staging, shape=(1, key_tile + padding, width + padding), dtype=history.dtype)
        maximum = T.alloc_fragment((matrix_rows,), "float32")
        previous = T.alloc_fragment((matrix_rows,), "float32")
        denominator = T.alloc_fragment((matrix_rows,), "float32")
        local_sum = T.alloc_fragment((matrix_rows,), "float32")
        alpha = T.alloc_fragment((matrix_rows,), "float32")
        base = T.cast(visible[token, 0], "int32")
        count = T.cast(visible[token, 1], "int32")
        # Engine visibility is a valid interval inside this physical history.
        T.assume(base >= 0)
        T.assume(count >= 0)
        T.assume(base <= history.shape[1])
        T.assume(count <= history.shape[1] - base)
        first = partition * schedule.span
        size = T.max(0, T.min(schedule.span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        for chunk in T.serial(T.ceildiv(size, key_tile)):
            chunk_first = first + chunk * key_tile
            full = chunk_first + key_tile <= count
            if full:
                for item, channel in T.Parallel(key_tile, width):
                    keys[0, channel, item] = history[0, base + chunk_first + item, kv_head, channel]
            else:
                for item, channel in T.Parallel(key_tile, width):
                    keys[0, channel, item] = T.if_then_else(
                        chunk_first + item < count,
                        history[0, base + chunk_first + item, kv_head, channel], 0)
            _decode_scores(query, keys, scores, token, first_head, head_tile, matrix_rows,
                           width, key_tile, schedule.contraction)
            for head, item in T.Parallel(matrix_rows, key_tile):
                scores[head, item] = T.if_then_else(head < head_tile and chunk_first + item < count,
                                                   scores[head, item] * scale * log2e,
                                                   -3.402823466e38)
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=False)
            for head in T.Parallel(matrix_rows):
                alpha[head] = T.exp2(previous[head] - maximum[head])
            for head, item in T.Parallel(matrix_rows, key_tile):
                scores[head, item] = T.if_then_else(head < head_tile and chunk_first + item < count,
                                                   T.exp2(scores[head, item] - maximum[head]), 0)
            T.reduce_sum(scores, local_sum, dim=1)
            for head in T.Parallel(matrix_rows):
                denominator[head] = denominator[head] * alpha[head] + local_sum[head]
            _attention_rescale(outputs, alpha, matrix_rows, columns)
            if full:
                for item, channel in T.Parallel(key_tile, width):
                    values[0, item, channel] = history[1, base + chunk_first + item, kv_head, channel]
            else:
                for item, channel in T.Parallel(key_tile, width):
                    values[0, item, channel] = T.if_then_else(
                        chunk_first + item < count,
                        history[1, base + chunk_first + item, kv_head, channel], 0)
            T.copy(scores, probabilities)
            _attention_values(probabilities, values, outputs, matrix_rows, width, key_tile,
                              columns, schedule.reduction_step)
        for head in T.Parallel(head_tile):
            statistics[partition, token, first_head + head, 0] = T.if_then_else(
                denominator[head] > 0, maximum[head] / log2e, -3.402823466e38)
            statistics[partition, token, first_head + head, 1] = denominator[head]
        _attention_publish(outputs, denominator, query, partials, token, first_head,
                           partition, schedule.partitions, tokens, 1, head_tile,
                           width, columns, query.dtype, False)


def _decode_partition_body(query, history, visible, partials, statistics, tokens,
                           heads, kv_heads, width, partitions, span, scale,
                           subgroups, schedule):
    if schedule is None:
        _register_partition_attention(query, history, visible, partials, statistics,
                                      tokens, heads, kv_heads, width, partitions,
                                      span, scale, subgroups)
    else:
        _tiled_decode_attention(query, history, visible, partials, statistics,
                                tokens, heads, kv_heads, width, scale, schedule)


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
        schedule: _MatrixAttentionSchedule,
        merge_threads: int,
    ):
        self.specs, self.scale, self.schedule = specs, scale, schedule
        self.merge_threads = merge_threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        query, history, visible, output, partials, statistics = operands
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
            self.schedule,
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
            self.schedule.partitions,
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
        schedule: _DecodeAttentionSchedule | None = None,
    ):
        self.specs = specs
        self.scale = scale
        self.partitions = partitions
        self.span = span
        self.subgroups = subgroups
        self.merge_threads = merge_threads
        self.schedule = schedule

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, output, partials, statistics = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _decode_partition_body(
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
            self.schedule,
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
            or context.compiler_target.shared_memory_bytes <= 0
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        if any(not spec.static for spec in specs):
            return ()
        width = cast(int, specs[0].shape[-1])
        threads = 1 << (width - 1).bit_length()
        if threads > context.compiler_target.threads_per_group:
            return ()
        rows, heads, width = cast(tuple[int, int, int], specs[0].shape)
        capacity = cast(int, specs[1].shape[1])
        if (
            context.precision == "reference"
            or context.compiler_target.reference_schedules
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
                    _MatrixAttentionEmitter(specs, node.attributes["scale"], schedule, threads),
                    workspace=schedule.workspace, kernel_count=2,
                ),
            )
        if (
            width % 32 == 0
            and context.compiler_target.subgroup_width == 32
            and context.compiler_target.threads_per_group >= 64
        ):
            group = heads // cast(int, specs[1].shape[2])
            subgroups = min(2, context.compiler_target.shared_memory_bytes // (group * (width + 2) * 4))
            if subgroups < 1:
                raise ValueError("one attention head group exceeds shared-memory capacity")
            stride = context.compiler_target.subgroup_width * subgroups
            target_partitions = max(1, math.ceil(512 / (rows * cast(int, specs[1].shape[2]))))
            span = math.ceil(capacity / target_partitions / stride) * stride
            partitions = math.ceil(capacity / span)
            decode_schedule = _decode_attention_schedule(specs[0], specs[1], context)
            if decode_schedule is not None:
                partitions, span = decode_schedule.partitions, decode_schedule.span
            partials = TensorSpec((partitions, rows, heads, width), DType.F32)
            statistics = TensorSpec((partitions, rows, heads, 2), DType.F32)
            if partials.storage_nbytes + statistics.storage_nbytes <= context.workspace_limit:
                return (
                    BoundOperation(
                        f"causal_attention.{'matrix-decode' if decode_schedule else 'register-partitioned'}@{root}",
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
                            decode_schedule,
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
    def __init__(self, specs, scale, partitions, span, subgroups, merge_threads, vector,
                 schedule=None):
        self.specs, self.scale = specs, scale
        self.partitions, self.span, self.subgroups = partitions, span, subgroups
        self.merge_threads = merge_threads
        self.vector = vector
        self.schedule = schedule

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, partials, statistics, activation = operands
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
        _decode_partition_body(
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
            self.schedule,
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
    def __init__(self, specs, scale, schedule, projection_tile):
        self.specs, self.scale, self.schedule = specs, scale, schedule
        self.projection_tile = projection_tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        query, history, visible, gate, weight, output, activation = operands[:7]
        partials, statistics = operands[7:] if self.schedule.partitions > 1 else (activation, activation)
        tokens, heads, width = cast(tuple[int, int, int], self.specs[0].shape)
        kv_heads = cast(int, self.specs[1].shape[2])
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
            self.schedule,
            self.specs[0].dtype.value,
            True,
        )
        if self.schedule.partitions > 1:
            _merge_attention_gate(
                partials,
                statistics,
                gate,
                activation,
                tokens,
                heads,
                width,
                self.schedule.partitions,
                max(width, 128),
                self.specs[0].dtype.value,
            )
        projection_threads, bm, bn, bk, reduction_step = self.projection_tile
        _packed_matrix(
            activation,
            weight,
            activation,
            output,
            self.specs[4],
            tokens,
            cast(int, self.specs[4].shape[0]),
            heads * width,
            reduction_step,
            self.specs[5].dtype.value,
            projection_threads,
            bm,
            bn,
            bk,
            False,
        )


def attention_projection_tile(context, activation, weight):
    """The output contraction schedule shared by dense and persistent mixers."""
    tokens, channels = activation.shape
    bm, bn = (32, 64) if tokens >= 256 and min(weight.shape[0], channels) >= 512 else (32, 32)
    bk = _packet_reduction_width(weight)
    threads = min(context.compiler_target.threads_per_group, 128,
                  bm // 8 * context.compiler_target.subgroup_width)
    if bk % 8 or affine_shared_bytes(bm, bn, bk, activation.dtype, weight) > context.compiler_target.shared_memory_bytes:
        return None
    return threads, bm, bn, bk, 8


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
            or context.compiler_target.subgroup_width != 32
        ):
            return ()
        if context.mode == "prefill":
            schedule = _matrix_attention_schedule(
                specs[0], specs[1], context, graph.nodes[root].attributes["sequence_count"],
            )
            if schedule is None:
                return ()
            activation = TensorSpec((tokens, heads * width), specs[0].dtype)
            projection_tile = attention_projection_tile(context, activation, specs[4])
            if projection_tile is None:
                return ()
            if schedule.shared_bytes > context.compiler_target.shared_memory_bytes:
                return ()
            # Bound each history traversal. Whole-buffer workspace reuse keeps
            # partition storage shared across sequential layers, while short
            # histories publish the gated activation without a merge.
            partitions = schedule.partitions
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
                        schedule,
                        projection_tile,
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
        if width != 256 or context.compiler_target.threads_per_group < 64:
            return ()
        subgroups = 2
        stride = context.compiler_target.subgroup_width * subgroups
        target_partitions = max(1, math.ceil(512 / (tokens * cast(int, specs[1].shape[2]))))
        span = math.ceil(capacity / target_partitions / stride) * stride
        partitions = math.ceil(capacity / span)
        decode_schedule = _decode_attention_schedule(specs[0], specs[1], context)
        if decode_schedule is not None:
            partitions, span = decode_schedule.partitions, decode_schedule.span
        partials = TensorSpec((partitions, tokens, heads, width), DType.F32)
        statistics = TensorSpec((partitions, tokens, heads, 2), DType.F32)
        activation = TensorSpec((tokens, heads * width), specs[0].dtype)
        workspace = (partials, statistics, activation)
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        return (
            BoundOperation(
                f"attention.{'matrix-decode' if decode_schedule else 'register-partitioned'}-gated-output@{root}:{max(nodes)}",
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
                    decode_schedule,
                ),
                workspace=workspace,
                kernel_count=3,
            ),
        )


__all__ = ["AttentionOutputRule", "CausalAttentionRule"]
