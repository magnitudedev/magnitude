"""Direct packed-history and dense-current attention with shared FP32 softmax."""
from __future__ import annotations

import math
from dataclasses import replace

import tilelang.language as T

from ..compiler.lowering import BoundOperation
from ..compiler.schedules import schedule_boundary, select_schedule
from ..kv import AffineKVCodec, RotatedLloydMax
from ..tensor.types import DType, TensorSpec
from .attention import (
    _attention_accumulators,
    _attention_publish,
    _attention_rescale,
    _attention_shared_values,
    _decode_attention_schedule,
    _decode_scores,
    _merge_attention,
    _merge_attention_gate,
)
from .compact_attention import (
    allocate_compact,
    compact_decode_scores,
    compact_values,
    stage_compact,
)
from .dimension_attention import dimension_tiled_prefill, prefill_schedule
from .kv_contraction import (
    correct_key_scores,
    finish_bias,
    query_sums,
    update_bias,
    value_coefficients,
)
from .kv_packed import (
    prepare_codebook,
    signed_wht,
    stage_current,
    stage_history,
    validate_vector_codec,
)


@T.macro
def _rotate_history_queries(query, output, tokens, heads, width, seed, subgroup_width):
    count = width // subgroup_width
    with T.Kernel(heads, tokens, threads=subgroup_width) as (head, token):
        lane = T.get_thread_binding()
        values = T.alloc_local((count,), "float32")
        scratch = T.alloc_local((count,), "float32")
        for item in T.unroll(count):
            values[item] = T.cast(query[token, head, lane * count + item], "float32")
        signed_wht(values, scratch, width, subgroup_width, lane, seed)
        for item in T.unroll(count):
            output[token, head, lane * count + item] = values[item] * width ** -0.5


class _PersistentEmitter:
    def __init__(self, specs, scale, history_schedule, current_schedule, matrix, threads, subgroup_width,
                 fuse_gate=False):
        self.specs, self.scale = specs, scale
        self.history_schedule, self.current_schedule = history_schedule, current_schedule
        self.matrix, self.threads, self.subgroup_width = matrix, threads, subgroup_width
        self.fuse_gate = fuse_gate

    def __call__(self, operands):
        query, history, keys, values, visible = operands[:5]
        if self.fuse_gate:
            gate, output, partials, statistics, *scratch = operands[5:]
        else:
            output, partials, statistics, *scratch = operands[5:]
        query_spec, history_spec = self.specs[:2]
        tokens, heads, width = query_spec.shape
        codec = history_spec.representation.key
        history_query = query
        sums = query
        if isinstance(codec, RotatedLloydMax):
            history_query = scratch[0]
            _rotate_history_queries(query, history_query, tokens, heads, width,
                                    codec.sign_seed, self.subgroup_width)
        elif isinstance(codec, AffineKVCodec) and not self.matrix:
            sums = scratch[0]
            query_sums(query, sums, tokens, heads, width)
        for from_history, source_query, schedule, offset in (
            (True, history_query, self.history_schedule, 0),
            (False, query, self.current_schedule, self.history_schedule.partitions),
        ):
            if self.matrix:
                dimension_tiled_prefill(source_query, history, visible, keys, values, sums, partials, statistics,
                                   tokens, heads, history_spec.shape[1], width, self.scale, schedule,
                                   query_spec.dtype.value, history_spec, from_history, offset)
            else:
                _persistent_decode(source_query, history, visible, keys, values, sums, partials, statistics,
                                   tokens, heads, history_spec.shape[1], width, self.scale, schedule,
                                   history_spec, from_history, offset)
        partitions = self.history_schedule.partitions + self.current_schedule.partitions
        if self.fuse_gate:
            _merge_attention_gate(partials, statistics, gate, output, tokens, heads, width,
                                  partitions, self.threads, query_spec.dtype.value)
        else:
            _merge_attention(partials, statistics, output, tokens, heads, width,
                             partitions, self.threads, query_spec.dtype.value)


class PersistentAttentionRule:
    def build(self, graph, root, context):
        node = graph.node(root)
        specs = tuple(graph.value(value).spec for value in node.inputs)
        query, history, keys, values, visible = specs
        rows, heads, width = query.shape
        validate_vector_codec(history, context.compiler_target.subgroup_width)
        if (isinstance(history.representation.key, (AffineKVCodec, RotatedLloydMax))
                and not isinstance(history.representation.value, AffineKVCodec)):
            raise ValueError("compact history contraction requires affine value storage")
        # Select the streamed prefill or compact decode ownership by geometry.
        logical_history = TensorSpec((2, history.shape[0], history.shape[1], width), query.dtype)
        matrix = context.mode == "prefill" and node.attributes["sequence_count"] == 1
        if matrix:
            schedule = prefill_schedule(query, history, context, template=_PersistentEmitter, workload=specs)
            if schedule is None:
                raise ValueError("persistent matrix attention exceeds target resources")
            current = replace(schedule, partitions=math.ceil(keys.shape[0] / schedule.span))
        else:
            # Compact history and dense current rows have different staging
            # footprints. Select the final history realization, never overwrite
            # its geometry after the tuner has identified it.
            base = _decode_attention_schedule(
                query, logical_history, replace(context, mode="decode", schedules=None))
            if base is None:
                raise ValueError("persistent grouped decode requires a legal complete head cohort")
            compact = isinstance(history.representation.key, (AffineKVCodec, RotatedLloydMax))
            if compact:
                max_words = max(p.row_elements for p in history.representation.planes(history.shape[0] * history.shape[1])
                                if p.name.endswith(".codes"))
                key_capacity = min(64, 8192 // (max_words * 4))
                key_tile = 1 << (max(1, key_capacity).bit_length() - 1)
                default = replace(base, keys=key_tile, span=key_tile * 16,
                                  partitions=math.ceil(history.shape[0] / (key_tile * 16)))
                extra = 64 if isinstance(history.representation.key, RotatedLloydMax) else 0
                candidates = tuple(
                    replace(default, rows=physical_rows, columns=columns, keys=keys_per_tile,
                            span=span, partitions=math.ceil(history.shape[0] / span))
                    for physical_rows in (8, 16)
                    for columns in sorted({base.columns, 32, 64})
                    for keys_per_tile, span in dict.fromkeys(((key_tile, key_tile * 16), (32, 512), (64, 2048)))
                    if width % columns == 0 and keys_per_tile % 8 == 0
                    and (max_words + physical_rows) * keys_per_tile * 4 + extra <= context.compiler_target.shared_memory_bytes
                    and ((width + base.padding) * (base.keys + base.padding) * query.dtype.itemsize
                         + physical_rows * base.keys * 4 <= context.compiler_target.shared_memory_bytes)
                )
                if not candidates:
                    raise ValueError("persistent compact attention exceeds target resources")
                if default not in candidates:
                    default = candidates[0]
                schedule = select_schedule(context, "attention.decode", candidates, default,
                                           template=_PersistentEmitter, workload=(specs, "compact-history"))
                current = replace(base, rows=schedule.rows, columns=schedule.columns,
                                  partitions=math.ceil(keys.shape[0] / base.span))
            else:
                schedule = _decode_attention_schedule(
                    query, logical_history, replace(context, mode="decode"),
                    template=_PersistentEmitter, workload=specs)
                current = replace(schedule, partitions=math.ceil(keys.shape[0] / schedule.span))
        partitions = schedule.partitions + current.partitions
        if isinstance(history.representation.key, (AffineKVCodec, RotatedLloydMax)):
            max_words = max(p.row_elements for p in history.representation.planes(history.shape[0] * history.shape[1])
                            if p.name.endswith(".codes"))
            key_tile = schedule.tile[1] if matrix else schedule.keys
            # Prefill expands a bounded native tile and keeps its scores in
            # registers. Decode retains code words and a shared probability
            # bridge. Price the phase's actual storage, not a discarded body.
            shared_bytes = (schedule.shared_bytes + 16 if matrix else
                            (max_words + schedule.rows) * key_tile * 4)
            if isinstance(history.representation.key, RotatedLloydMax):
                shared_bytes += 16 * 4
            if shared_bytes > context.compiler_target.shared_memory_bytes:
                raise ValueError("compact attention exceeds target shared memory")
        workspace = (TensorSpec((partitions, rows, heads, width), DType.F32),
                     TensorSpec((partitions, rows, heads, 2), DType.F32))
        rotated = isinstance(history.representation.key, RotatedLloydMax)
        if rotated:
            workspace += (TensorSpec(query.shape, query.dtype),)
        elif isinstance(history.representation.key, AffineKVCodec) and not matrix:
            workspace += (TensorSpec(query.shape[:2], DType.F32),)
        if sum(spec.storage_nbytes for spec in workspace) > context.workspace_limit:
            raise ValueError("persistent attention workspace exceeds available capacity")
        return (BoundOperation(f"attention.persistent@{root}", frozenset({root}), node.inputs, node.outputs,
                               _PersistentEmitter(specs, node.attributes["scale"], schedule, current, matrix,
                                                  min(256, context.compiler_target.threads_per_group),
                                                  context.compiler_target.subgroup_width),
                               workspace=workspace, kernel_count=4 if rotated or (isinstance(history.representation.key, AffineKVCodec) and not matrix) else 3),)


class PersistentAttentionGateRule:
    def build(self, graph, root, context):
        if root + 3 >= len(graph.nodes):
            return ()
        attention, sigmoid, multiply, reshape = graph.nodes[root:root + 4]
        if (attention.operation != "persistent_attention" or sigmoid.operation != "sigmoid"
                or multiply.operation != "multiply" or reshape.operation != "reshape"
                or set(multiply.inputs) != {attention.outputs[0], sigmoid.outputs[0]}
                or reshape.inputs != multiply.outputs):
            return ()
        context = schedule_boundary(context, _PersistentEmitter,
                                    ("gated", graph.value(sigmoid.inputs[0]).spec, graph.value(reshape.outputs[0]).spec))
        operation = PersistentAttentionRule().build(graph, root, context)[0]
        emitter = operation.emitter
        fused = _PersistentEmitter(emitter.specs, emitter.scale, emitter.history_schedule,
                                    emitter.current_schedule, emitter.matrix, emitter.threads,
                                    emitter.subgroup_width, fuse_gate=True)
        return (replace(operation, name=f"attention.persistent-gated@{root}",
                        nodes=frozenset(range(root, root + 4)),
                        inputs=(*attention.inputs, sigmoid.inputs[0]), outputs=reshape.outputs,
                        emitter=fused),)

@T.macro
def _persistent_decode(query, history, visible, current_keys, current_values, sums, partials, statistics,
                             tokens, heads, kv_heads, width, scale, schedule, history_spec, from_history, partition_offset):
    """One K/V tile serves a complete query-head cohort and a stable softmax.

    The online dependency advances per history tile. Both QK and PV use matrix
    contraction with FP32 operands; only the original K/V staging is compact.
    """
    group = heads // kv_heads
    encoded = from_history and isinstance(history_spec.representation.key, (AffineKVCodec, RotatedLloydMax))
    head_tile, matrix_rows, key_tile = schedule.heads, schedule.rows, schedule.keys
    columns, padding = schedule.columns, schedule.padding
    log2e = 1.4426950408889634
    with T.Kernel(heads // head_tile, tokens, schedule.partitions,
                  threads=schedule.threads) as (cohort, token, partition):
        table = prepare_codebook(history_spec, from_history)
        first_head = cohort * head_tile
        kv_head = first_head // group
        outputs = _attention_accumulators(matrix_rows, width, columns)
        scores = T.alloc_fragment((matrix_rows, key_tile), "float32")
        affine_values = from_history and isinstance(history_spec.representation.value, AffineKVCodec)
        bias = T.alloc_fragment((matrix_rows,), "float32") if affine_values else None
        local_bias = T.alloc_fragment((matrix_rows,), "float32") if affine_values else None
        if affine_values:
            T.clear(bias)
        # QK distributes history columns across subgroups. PV distributes
        # output channels, so every subgroup needs the probability tile.
        # Make that exchange explicit instead of copying incompatible fragments.
        probabilities = T.alloc_shared((matrix_rows, key_tile), "float32")
        if encoded:
            packed = allocate_compact(history_spec, key_tile)
        else:
            staging = T.alloc_shared(((width + padding) * (key_tile + padding),), query.dtype)
            keys = T.view(staging, shape=(1, key_tile + padding, width + padding), dtype=query.dtype)
            values = T.view(staging, shape=(1, key_tile + padding, width + padding), dtype=query.dtype)
        maximum = T.alloc_fragment((matrix_rows,), "float32")
        previous = T.alloc_fragment((matrix_rows,), "float32")
        denominator = T.alloc_fragment((matrix_rows,), "float32")
        local_sum = T.alloc_fragment((matrix_rows,), "float32")
        alpha = T.alloc_fragment((matrix_rows,), "float32")
        base = T.cast(visible[token, 0 if from_history else 2], "int32")
        count = T.cast(visible[token, 1 if from_history else 3], "int32")
        # Engine visibility is a valid interval inside this physical history.
        T.assume(base >= 0)
        T.assume(count >= 0)
        T.assume(base <= (history_spec.shape[0] if from_history else current_keys.shape[0]))
        T.assume(count <= (history_spec.shape[0] if from_history else current_keys.shape[0]) - base)
        first = partition * schedule.span
        size = T.max(0, T.min(schedule.span, count - first))
        T.fill(maximum, -3.402823466e38)
        T.clear(denominator)
        for chunk in T.serial(T.ceildiv(size, key_tile)):
            chunk_first = first + chunk * key_tile
            full = chunk_first + key_tile <= count
            if encoded:
                stage_compact(history, packed, history_spec, "key", kv_head, base, chunk_first, count, key_tile)
                compact_decode_scores(query, packed, table, scores, token, first_head, head_tile, matrix_rows,
                                       width, key_tile, schedule.contraction, history_spec.representation.key.bits,
                                       isinstance(history_spec.representation.key, RotatedLloydMax))
            elif from_history:
                stage_history(history, keys, history_spec, "key", kv_head, base, chunk_first, count, key_tile, table, True, True)
            else:
                stage_current(current_keys, keys, kv_head, base, chunk_first, count, key_tile, width, False)
            if not encoded:
                _decode_scores(query, keys, scores, token, first_head, head_tile, matrix_rows,
                               width, key_tile, schedule.contraction, True)
            if from_history:
                correct_key_scores(scores, sums, history, history_spec, base, chunk_first, count,
                                    kv_head, matrix_rows, key_tile, 1, token, first_head, tokens, valid_rows=head_tile)
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
            if encoded:
                stage_compact(history, packed, history_spec, "value", kv_head, base, chunk_first, count, key_tile)
            elif from_history:
                stage_history(history, values, history_spec, "value", kv_head, base, chunk_first, count, key_tile, table, True, True)
            else:
                stage_current(current_values, values, kv_head, base, chunk_first, count, key_tile, width, False)
            if affine_values:
                value_coefficients(scores, probabilities, history, history_spec, base, chunk_first,
                                     count, kv_head, matrix_rows, key_tile)
                update_bias(scores, bias, local_bias, alpha, matrix_rows)
            else:
                T.copy(scores, probabilities)
            if encoded:
                compact_values(probabilities, packed, outputs, matrix_rows, key_tile, columns,
                                schedule.reduction_step, history_spec.representation.value.bits)
            else:
                _attention_shared_values(probabilities, values, outputs, matrix_rows, width, key_tile,
                                   columns, schedule.reduction_step)
        if affine_values:
            finish_bias(outputs, bias, matrix_rows, columns)
        for head in T.Parallel(matrix_rows):
            if head < head_tile:
                statistics[partition + partition_offset, token, first_head + head, 0] = T.if_then_else(
                    denominator[head] > 0, maximum[head] / log2e, -3.402823466e38)
                statistics[partition + partition_offset, token, first_head + head, 1] = denominator[head]
        _attention_publish(outputs, denominator, query, partials, token, first_head,
                           partition + partition_offset, schedule.partitions, tokens, 1, head_tile,
                           width, columns, query.dtype, False, matrix_rows)
