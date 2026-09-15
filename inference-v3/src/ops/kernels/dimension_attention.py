"""Stream head dimensions while preserving the full query-row reuse cohort."""
from __future__ import annotations

import math

import tilelang.language as T

from ..tensor.types import TensorSpec, DType
from .attention import (_MatrixAttentionSchedule, _attention_accumulators, _attention_rescale, _attention_value_tile,
                        _attention_publish, _matrix_visible_count)
from .kv_packed import stage_history, stage_current, prepare_codebook


def prefill_schedule(query, history, context):
    """Price the streamed arena independently of a whole-head staging schedule."""
    rows, heads, width = query.shape
    group = heads // history.shape[1]
    target = context.compiler_target
    head_tile = 2 if group % 2 == 0 and target.threads_per_group >= target.subgroup_width * 8 else 1
    query_tile, key_tile = 32, 64
    threads = target.subgroup_width * 4 * head_tile
    slice_width = math.gcd(64, width)
    padding = 8
    def shared_bytes():
        return (slice_width + padding) * (key_tile + padding) * query.dtype.itemsize
    while key_tile > 8 and shared_bytes() + 4 > target.shared_memory_bytes:
        key_tile //= 2
    if threads > target.threads_per_group or shared_bytes() + 4 > target.shared_memory_bytes:
        return None
    partitions = math.ceil(history.shape[0] / 4096)
    span = math.ceil(history.shape[0] / (partitions * key_tile)) * key_tile
    partitions = math.ceil(history.shape[0] / span)
    workspace = (TensorSpec((partitions, rows, heads, width), DType.F32),
                 TensorSpec((partitions, rows, heads, 2), DType.F32))
    return _MatrixAttentionSchedule((query_tile, key_tile, threads), head_tile,
        slice_width, 8, padding, partitions, span, shared_bytes(), workspace)


@T.macro
def _resident_probability_values(scores, values, output, rows, keys, columns, reduction):
    probability = T.alloc_fragment((rows, reduction), 'float32')
    # The enlarged score register file must never acquire a dynamic subscript.
    # Statically select each immediate left operand before the column GEMM.
    for step in T.unroll(keys // reduction):
        for row, key in T.Parallel(rows, reduction):
            probability[row, key] = scores[row, step * reduction + key]
        _attention_value_tile(probability, values, output, step * reduction, 0,
                              columns, columns, reduction)


@T.macro
def _stream_value_column(history, current, staging, spec, from_history, head, base,
                         first, count, keys, table, probabilities, output, rows, columns,
                         reduction, first_channel):
    if from_history:
        stage_history(history, staging, spec, "value", head, base, first, count, keys,
                      table, first_channel=first_channel, tile_width=columns)
    else:
        stage_current(current, staging, head, base, first, count, keys, columns,
                      False, first_channel)
    _resident_probability_values(probabilities, staging, output, rows, keys, columns, reduction)


def _stream_value_columns(history, current, staging, spec, from_history, head, base,
                          first, count, keys, table, probabilities, outputs, rows,
                          columns, reduction):
    for index, output in enumerate(outputs):
        _stream_value_column(history, current, staging, spec, from_history, head, base,
                             first, count, keys, table, probabilities, output, rows,
                             columns, reduction, index * columns)


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
        last_row = T.min(tokens - 1, first_row + query_tile - 1)
        base = T.cast(visible[first_row, 0 if from_history else 2], "int32")
        count = _matrix_visible_count(visible, first_row, query_tile, tokens,
                                      1 if from_history else 3)
        # Visibility is a valid interval within the history allocation. The
        # single-sequence matrix schedule shares its base across this query tile.
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
                and first + (chunk + 1) * key_tile <= visible[first_row, 1 if from_history else 3]
                and first + (chunk + 1) * key_tile <= visible[last_row, 1 if from_history else 3]
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
                        and relative < visible[token, 1 if from_history else 3],
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
                                   schedule.reduction_step)
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
