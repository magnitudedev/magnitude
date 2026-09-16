"""Stream head dimensions while preserving the full query-row reuse cohort."""
from __future__ import annotations

import math

import tilelang.language as T

from .buffers import rebase_buffer

from ..compiler.schedules import select_schedule
from ..tensor.types import DType, TensorSpec
from .attention import (
    _attention_accumulators,
    _attention_publish,
    _attention_rescale,
    _attention_value_tile,
    _matrix_visible_interval,
    _MatrixAttentionSchedule,
)
from .kv_packed import prepare_codebook, stage_current, stage_history
from .schedules import ProbabilityTransfer


def prefill_schedule(query, history, context, *, template=None, workload=()):
    """Price the streamed arena independently of a whole-head staging schedule."""
    rows, heads, width = query.shape
    group = heads // history.shape[1]
    target = context.compiler_target
    slice_width = math.gcd(64, width)
    padding = 8

    def candidate(query_tile, key_tile, head_tile, transfer):
        threads = target.subgroup_width * 4 * head_tile
        probability = (query_tile * head_tile * 8 * DType.F32.itemsize
                       if transfer == ProbabilityTransfer.SHARED else 0)
        shared = (slice_width + padding) * (key_tile + padding) * query.dtype.itemsize + probability
        if (group % head_tile or threads > target.threads_per_group
                or shared + 16 > target.shared_memory_bytes):
            return None
        partitions = math.ceil(history.shape[0] / 4096)
        span = math.ceil(history.shape[0] / (partitions * key_tile)) * key_tile
        partitions = math.ceil(history.shape[0] / span)
        workspace = (TensorSpec((partitions, rows, heads, width), DType.F32),
                     TensorSpec((partitions, rows, heads, 2), DType.F32))
        return _MatrixAttentionSchedule((query_tile, key_tile, threads), head_tile,
            slice_width, 8, padding, partitions, span, shared, workspace, transfer)

    # Row reuse and head reuse have different accumulator lifetimes. Keep both
    # decompositions available, together with a shorter history tile; none is
    # inferred to be optimal from the target's maximum resource limits.
    geometries = ((32, 64, 2), (32, 64, 1), (16, 64, 2),
                  (32, 32, 2), (64, 32, 1), (64, 64, 1), (32, 8, 1))
    candidates = tuple(
        schedule
        for geometry in geometries
        for transfer in ProbabilityTransfer
        if (schedule := candidate(*geometry, transfer)) is not None
    )
    if not candidates:
        return None
    return select_schedule(context, "attention.persistent-prefill", candidates, candidates[0],
                           template=template or dimension_tiled_prefill, workload=(query, history, workload))


@T.macro
def _resident_probability_values(scores, values, output, rows, keys, columns, reduction, transfer):
    if transfer == ProbabilityTransfer.SHARED:
        probability = T.alloc_shared((rows, reduction), 'float32')
    else:
        probability = T.alloc_fragment((rows, reduction), 'float32')
        T.sync_threads()
    # The enlarged score register file must never acquire a dynamic subscript.
    # Statically select each immediate left operand before the column GEMM.
    for step in T.unroll(keys // reduction):
        for row, key in T.Parallel(rows, reduction):
            probability[row, key] = scores[row, step * reduction + key]
        if transfer == ProbabilityTransfer.SHARED:
            T.sync_threads()
        _attention_value_tile(probability, values, output, step * reduction, 0,
                              columns, columns, reduction)
        if transfer == ProbabilityTransfer.SHARED:
            T.sync_threads()
    if transfer == ProbabilityTransfer.INFERRED:
        T.sync_threads()


@T.macro
def _stream_value_column(history, current, staging, spec, from_history, head, base,
                         first, count, keys, table, probabilities, output, rows, columns,
                         reduction, first_channel, transfer):
    if from_history:
        stage_history(history, staging, spec, "value", head, base, first, count, keys,
                      table, first_channel=first_channel, tile_width=columns)
    else:
        stage_current(current, staging, head, base, first, count, keys, columns,
                      False, first_channel)
    _resident_probability_values(probabilities, staging, output, rows, keys, columns, reduction, transfer)


def _stream_value_columns(history, current, staging, spec, from_history, head, base,
                          first, count, keys, table, probabilities, outputs, rows,
                          columns, reduction, transfer):
    for index, output in enumerate(outputs):
        _stream_value_column(history, current, staging, spec, from_history, head, base,
                             first, count, keys, table, probabilities, output, rows,
                             columns, reduction, index * columns, transfer)


@T.macro
def dimension_tiled_prefill(
    query,
    history,
    visible,
    current_keys,
    current_values,
    sums,
    partials,
    statistics,
    tokens,
    heads,
    kv_heads,
    width,
    scale,
    schedule,
    dtype,
    history_spec,
    from_history,
    partition_offset,
    visibility_column,
):
    """Stream head slices while preserving register ownership of probabilities."""
    group = heads // kv_heads
    query_tile, key_tile, threads = schedule.tile
    partitions, span = schedule.partitions, schedule.span
    head_tile, columns = schedule.head_tile, schedule.value_tile
    query_rows = query_tile * head_tile
    slice_width = math.gcd(64, width)
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
        query = rebase_buffer(query)
        current_keys = rebase_buffer(current_keys)
        current_values = rebase_buffer(current_values)
        table = prepare_codebook(history_spec, from_history)
        first_head = head_block * head_tile
        kv_head = first_head // group
        query_fragment = T.alloc_fragment((query_rows, slice_width), dtype)
        outputs = _attention_accumulators(query_rows, width, columns)
        scores = T.alloc_fragment((query_rows, key_tile), "float32")
        # Query-rich prefill amortizes decoding across its whole query tile.
        # K uses contraction-major storage; V reuses the same arena after QK.
        kv = T.alloc_shared(((slice_width + padding) * (key_tile + padding),), dtype)
        keys = T.view(kv, shape=((1, slice_width + padding, key_tile + padding) if from_history
                                else (1, key_tile + padding, slice_width + padding)), dtype=dtype)
        values = T.view(kv, shape=(1, key_tile + padding, slice_width + padding), dtype=dtype)
        maximum = T.alloc_fragment((query_rows,), "float32")
        previous = T.alloc_fragment((query_rows,), "float32")
        denominator = T.alloc_fragment((query_rows,), "float32")
        local_sum = T.alloc_fragment((query_rows,), "float32")
        alpha = T.alloc_fragment((query_rows,), "float32")
        first_row = block * query_tile
        base, count, common_first, common_last = _matrix_visible_interval(
            visible, first_row, query_tile, tokens, visibility_column + 1)
        # Visibility is a valid interval within the history allocation. The
        # staged union may include gaps; masks retain each row's own interval.
        T.assume(base >= 0)
        T.assume(count >= 0)
        T.assume(base <= (history_spec.shape[0] if from_history else current_keys.shape[0]))
        T.assume(count <= (history_spec.shape[0] if from_history else current_keys.shape[0]) - base)
        first = partition * span
        partition_count = T.max(0, T.min(span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        for chunk in T.serial(T.ceildiv(partition_count, key_tile)):
            aligned = first + (chunk + 1) * key_tile <= count
            T.clear(scores)
            for dimension in T.serial(width // slice_width):
                first_channel = dimension * slice_width
                for row, channel in T.Parallel(query_rows, slice_width):
                    token = first_row + row % query_tile
                    head = first_head + row // query_tile
                    query_fragment[row, channel] = T.if_then_else(
                        token < tokens, query[token, head, first_channel + channel], 0)
                if from_history:
                    stage_history(history, keys, history_spec, "key", kv_head, base,
                                  first + chunk * key_tile, count, key_tile, table,
                                  first_channel=first_channel, tile_width=slice_width)
                else:
                    stage_current(current_keys, keys, kv_head, base, first + chunk * key_tile,
                                  count, key_tile, slice_width, False, first_channel)
                with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                    with T.attr(0, "pragma_unroll_explicit", 1):
                        if from_history:
                            T.gemm(query_fragment, keys[0, :slice_width, :key_tile], scores,
                                   policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.gemm(query_fragment, keys[0, :key_tile, :slice_width], scores,
                                   transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            wholly_visible = (
                aligned
                and first_row + query_tile <= tokens
                and base + first + chunk * key_tile >= common_first
                and base + first + (chunk + 1) * key_tile <= common_last
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
                        and base + relative >= visible[token, visibility_column]
                        and base + relative < (visible[token, visibility_column]
                                               + visible[token, visibility_column + 1]),
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
            _stream_value_columns(history, current_values, values, history_spec, from_history,
                                   kv_head, base, first + chunk * key_tile, count, key_tile,
                                   table, scores, outputs, query_rows, columns,
                                   schedule.reduction_step, schedule.probability_transfer)
        for row in T.Parallel(query_rows):
            token = first_row + row % query_tile
            head = first_head + row // query_tile
            if token < tokens:
                statistics[partition + partition_offset, token, head, 0] = T.if_then_else(
                    denominator[row] > 0,
                    maximum[row] / log2e,
                    -3.402823466e38,
                )
                statistics[partition + partition_offset, token, head, 1] = denominator[row]
        _attention_publish(outputs, denominator, query, partials, first_row, first_head,
                           partition + partition_offset, partitions, tokens, query_tile, head_tile, width,
                           columns, dtype, False)
