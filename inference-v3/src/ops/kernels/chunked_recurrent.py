"""Parallel delta preparation, two-product state scan, parallel output.

Prepare W = inverse(I+L) beta D K and U = inverse(I+L) beta V independently
for each chunk. Only R = U - W S and S_next = D_last S + K_tail^T R depend
on the preceding chunk. Save incoming states for parallel output publication.
Direct decay products preserve reset boundaries without cumulative-product ratios.
"""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec


@T.macro
def _chunk_delta_systems(
    query,
    key,
    value,
    decay,
    beta,
    offsets,
    systems,
    factors,
    weights,
    residuals,
    batch,
    chunks,
    key_heads,
    heads,
    width,
    value_width,
    chunk,
    mapping,
    threads,
    sequence_length,
):
    with T.Kernel(chunks, heads, batch, threads=threads) as (block, head, sequence):
        # Preserve the packed sequence's valid range in the compiler's bounds
        # analysis instead of treating offsets loaded from memory as unbounded.
        first = (0 if sequence_length is not None else
                 T.min(query.shape[0], T.max(0, offsets[sequence]))) + block * chunk
        end = (sequence_length if sequence_length is not None else
               T.min(query.shape[0], T.max(0, offsets[sequence + 1])))
        kh = head % key_heads if mapping == "tiled" else head // (heads // key_heads)
        q = T.alloc_fragment((chunk, width), "float32")
        k = T.alloc_shared((chunk, width), "float32")
        kk = T.alloc_fragment((chunk, chunk), "float32")
        qk = T.alloc_fragment((chunk, chunk), "float32")
        lower = T.alloc_shared((chunk, chunk), "float32")
        inverse = T.alloc_shared((chunk, chunk), "float32")
        prefix = T.alloc_shared((chunk,), "float32")
        prepared = T.alloc_fragment((chunk, chunk), "float32")
        product = T.alloc_local((1,), "float32")
        total = T.alloc_local((1,), "float32")
        if first < end:
            for i, d in T.Parallel(chunk, width):
                q[i, d] = T.if_then_else(
                    first + i < end, T.cast(query[first + i, kh, d], "float32"), 0
                )
                k[i, d] = T.if_then_else(
                    first + i < end, T.cast(key[first + i, kh, d], "float32"), 0
                )
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(k, k, kk, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.Square)
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(q, k, qk, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.Square)
            for i, j in T.Parallel(chunk, chunk):
                product[0] = 1.0
                for step in T.serial(j + 1, T.max(j + 1, i + 1)):
                    if first + step < end:
                        product[0] *= decay[first + step, head]
                lower[i, j] = T.if_then_else(
                    j < i and first + i < end,
                    kk[i, j] * product[0] * beta[first + i, head],
                    0,
                )
                systems[sequence, block, head, i, j] = T.if_then_else(
                    j <= i, qk[i, j] * product[0], 0
                )
                inverse[i, j] = T.if_then_else(i == j, 1.0, 0.0)
            T.sync_threads()
            # Independent right-hand sides; only this small chunk-local system
            # has a row dependency. All chunks prepare concurrently.
            for i in T.serial(chunk):
                for j in T.Parallel(chunk):
                    total[0] = T.if_then_else(i == j, 1.0, 0.0)
                    for prior in T.serial(i):
                        total[0] -= lower[i, prior] * inverse[prior, j]
                    inverse[i, j] = total[0]
                T.sync_threads()
            for i in T.Parallel(chunk):
                product[0] = 1.0
                for step in T.serial(i + 1):
                    if first + step < end:
                        product[0] *= decay[first + step, head]
                factors[sequence, block, head, 0, i] = product[0]
                prefix[i] = product[0]
                product[0] = 1.0
                for step in T.serial(i + 1, chunk):
                    if first + step < end:
                        product[0] *= decay[first + step, head]
                factors[sequence, block, head, 1, i] = product[0]
            T.sync_threads()
            # The lower system is dead after inversion. Its shared tile now
            # carries W/U operands, keeping the complete preparation bounded.
            for tile in T.serial(T.ceildiv(width, chunk)):
                for i, j in T.Parallel(chunk, chunk):
                    lower[i, j] = T.if_then_else(
                        first + i < end and tile * chunk + j < width,
                        beta[first + i, head] * prefix[i] * k[i, tile * chunk + j],
                        0,
                    )
                # Expand bounded matrix coordinates, leaving algorithm loops serial.
                with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                    with T.attr(0, "pragma_unroll_explicit", 1):
                        T.gemm(inverse, lower, prepared, clear_accum=True, policy=T.GemmWarpPolicy.Square)
                for i, j in T.Parallel(chunk, chunk):
                    if tile * chunk + j < width:
                        weights[sequence, block, head, i, tile * chunk + j] = prepared[i, j]
            for tile in T.serial(T.ceildiv(value_width, chunk)):
                for i, j in T.Parallel(chunk, chunk):
                    lower[i, j] = T.if_then_else(
                        first + i < end and tile * chunk + j < value_width,
                        beta[first + i, head] * T.cast(value[first + i, head, tile * chunk + j], "float32"),
                        0,
                    )
                # Expand bounded matrix coordinates, leaving algorithm loops serial.
                with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                    with T.attr(0, "pragma_unroll_explicit", 1):
                        T.gemm(inverse, lower, prepared, clear_accum=True, policy=T.GemmWarpPolicy.Square)
                for i, j in T.Parallel(chunk, chunk):
                    if tile * chunk + j < value_width:
                        residuals[sequence, block, head, i, tile * chunk + j] = prepared[i, j]


@T.macro
def _chunk_delta_scan(
    key,
    previous,
    offsets,
    factors,
    weights,
    residuals,
    boundaries,
    following,
    batch,
    key_heads,
    heads,
    width,
    value_width,
    chunk,
    columns,
    mapping,
    threads,
    sequence_length,
):
    with T.Kernel(T.ceildiv(value_width, columns), heads, batch, threads=threads) as (
        tile,
        head,
        sequence,
    ):
        kh = head % key_heads if mapping == "tiled" else head // (heads // key_heads)
        state = T.alloc_shared((columns, width), "float32")
        operand = T.alloc_shared((chunk, width), "float32")
        rhs = T.alloc_shared((chunk, columns), "float32")
        contraction = T.alloc_fragment((chunk, columns), "float32")
        update = T.alloc_fragment((columns, width), "float32")
        first = (0 if sequence_length is not None else
                 T.min(key.shape[0], T.max(0, offsets[sequence])))
        end = (sequence_length if sequence_length is not None else
               T.min(key.shape[0], T.max(0, offsets[sequence + 1])))
        count = T.max(0, end - first)
        for v, d in T.Parallel(columns, width):
            state[v, d] = T.if_then_else(
                tile * columns + v < value_width, previous[sequence, head, tile * columns + v, d], 0
            )
        for block in T.serial(
            T.min(T.ceildiv(count, chunk), T.ceildiv(key.shape[0], chunk))
        ):
            start = first + block * chunk
            for v, d in T.Parallel(columns, width):
                if tile * columns + v < value_width:
                    boundaries[sequence, block, head, tile * columns + v, d] = state[v, d]
            for i, d in T.Parallel(chunk, width):
                operand[i, d] = weights[sequence, block, head, i, d]
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(
                        operand,
                        state,
                        contraction,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.Square,
                    )
            for i, v in T.Parallel(chunk, columns):
                rhs[i, v] = T.if_then_else(
                    block * chunk + i < count and tile * columns + v < value_width,
                    residuals[sequence, block, head, i, tile * columns + v] - contraction[i, v],
                    0,
                )
            for i, v in T.Parallel(chunk, columns):
                if tile * columns + v < value_width:
                    residuals[sequence, block, head, i, tile * columns + v] = rhs[i, v]
            for i, d in T.Parallel(chunk, width):
                operand[i, d] = T.if_then_else(
                    block * chunk + i < count,
                    T.cast(key[start + i, kh, d], "float32") * factors[sequence, block, head, 1, i],
                    0,
                )
            for v, d in T.Parallel(columns, width):
                state[v, d] *= factors[sequence, block, head, 0, chunk - 1]
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(
                        rhs,
                        operand,
                        update,
                        transpose_A=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.Square,
                    )
            for v, d in T.Parallel(columns, width):
                state[v, d] += update[v, d]
        for v, d in T.Parallel(columns, width):
            if tile * columns + v < value_width:
                following[sequence, head, tile * columns + v, d] = state[v, d]


@T.macro
def _chunk_delta_output(
    query, offsets, systems, factors, residuals, boundaries, output,
    batch, chunks, key_heads, heads, width, value_width, chunk, columns,
    mapping, threads, dtype, sequence_length,
):
    with T.Kernel(T.ceildiv(value_width, columns), chunks, batch * heads,
                  threads=threads) as (tile, block, owner):
        sequence = owner // heads
        head = owner % heads
        kh = head % key_heads if mapping == "tiled" else head // (heads // key_heads)
        first = (0 if sequence_length is not None else
                 T.min(query.shape[0], T.max(0, offsets[sequence]))) + block * chunk
        end = (sequence_length if sequence_length is not None else
               T.min(query.shape[0], T.max(0, offsets[sequence + 1])))
        q = T.alloc_shared((chunk, width), "float32")
        state = T.alloc_shared((columns, width), "float32")
        rhs = T.alloc_shared((chunk, columns), "float32")
        causal = T.alloc_fragment((chunk, chunk), "float32")
        result = T.alloc_fragment((chunk, columns), "float32")
        if first < end:
            for i, d in T.Parallel(chunk, width):
                q[i, d] = T.if_then_else(first + i < end,
                                         T.cast(query[first + i, kh, d], "float32"), 0)
            for v, d in T.Parallel(columns, width):
                state[v, d] = T.if_then_else(tile * columns + v < value_width,
                    boundaries[sequence, block, head, tile * columns + v, d], 0)
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(q, state, result, transpose_B=True, clear_accum=True,
                           policy=T.GemmWarpPolicy.Square)
            for i, v in T.Parallel(chunk, columns):
                result[i, v] *= factors[sequence, block, head, 0, i]
                rhs[i, v] = T.if_then_else(tile * columns + v < value_width,
                    residuals[sequence, block, head, i, tile * columns + v], 0)
            for i, j in T.Parallel(chunk, chunk):
                causal[i, j] = systems[sequence, block, head, i, j]
            # Expand bounded matrix coordinates, leaving algorithm loops serial.
            with T.attr(0, "pragma_auto_unroll_max_step", 4096):
                with T.attr(0, "pragma_unroll_explicit", 1):
                    T.gemm(causal, rhs, result, policy=T.GemmWarpPolicy.Square)
            for i, v in T.Parallel(chunk, columns):
                if first + i < end and tile * columns + v < value_width:
                    output[first + i, head, tile * columns + v] = T.cast(result[i, v], dtype)


class _ChunkedDeltaEmitter:
    def __init__(self, specs, mapping, chunk, columns, threads, sequence_length):
        self.specs, self.mapping = specs, mapping
        self.chunk, self.columns, self.threads = chunk, columns, threads
        self.sequence_length = sequence_length

    def __call__(self, operands: tuple[Any, ...]) -> None:
        (q, k, v, decay, beta, state, offsets, output, following,
         systems, factors, weights, residuals, boundaries) = operands
        batch, heads, value_width, width = cast(tuple[int, int, int, int], self.specs[5].shape)
        rows, key_heads, _ = cast(tuple[int, int, int], self.specs[0].shape)
        _chunk_delta_systems(
            q,
            k,
            v,
            decay,
            beta,
            offsets,
            systems,
            factors,
            weights,
            residuals,
            batch,
            math.ceil(rows / self.chunk),
            key_heads,
            heads,
            width,
            value_width,
            self.chunk,
            self.mapping,
            self.threads,
            self.sequence_length,
        )
        _chunk_delta_scan(
            k,
            state,
            offsets,
            factors,
            weights,
            residuals,
            boundaries,
            following,
            batch,
            key_heads,
            heads,
            width,
            value_width,
            self.chunk,
            self.columns,
            self.mapping,
            self.threads,
            self.sequence_length,
        )
        _chunk_delta_output(
            q, offsets, systems, factors, residuals, boundaries, output,
            batch, math.ceil(rows / self.chunk), key_heads, heads, width,
            value_width, self.chunk, self.columns, self.mapping, self.threads,
            self.specs[7].dtype.value,
            self.sequence_length,
        )


class ChunkedDeltaRule:
    name = "chunked-gated-delta"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "gated_delta_recurrence" or context.mode != "prefill":
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        rows = cast(int, specs[0].shape[0])
        batch, heads, value_width, width = cast(tuple[int, int, int, int], specs[5].shape)
        chunk, columns = 32, 16
        if (
            context.compiler_target.shared_memory_bytes <= 0
            or rows < 64 * batch
            or width % 8
        ):
            return ()
        threads = (chunk // 8) * context.compiler_target.subgroup_width
        # Include conservative affine-row padding for each shared matrix.
        prepare_shared = (chunk * (width + 4) + 2 * chunk * (chunk + 4) + chunk) * 4
        scan_shared = (
            (columns + chunk) * (width + 4) + chunk * (columns + 4)
        ) * 4
        if (
            threads > context.compiler_target.threads_per_group
            or max(prepare_shared, scan_shared) > context.compiler_target.shared_memory_bytes
        ):
            return ()
        chunks = math.ceil(rows / chunk)
        workspace = (
            TensorSpec((batch, chunks, heads, chunk, chunk), DType.F32),
            TensorSpec((batch, chunks, heads, 2, chunk), DType.F32),
            TensorSpec((batch, chunks, heads, chunk, width), DType.F32),
            TensorSpec((batch, chunks, heads, chunk, value_width), DType.F32),
            TensorSpec((batch, chunks, heads, value_width, width), DType.F32),
        )
        if sum(spec.storage_nbytes for spec in workspace) > context.workspace_limit:
            return ()
        return (
            BoundOperation(
                f"gated_delta.chunked-matrix@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _ChunkedDeltaEmitter(
                    specs, node.attributes["mapping"], chunk, columns, threads,
                    node.attributes.get("sequence_length"),
                ),
                workspace=workspace,
                kernel_count=3,
            ),
        )
