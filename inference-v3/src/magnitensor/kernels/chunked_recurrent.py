"""Chunk-local delta systems followed by a matrix-based state scan.

For each chunk, solve (I + L) R = beta * (V - D K S), where
L[i,j] = beta[i] <K[i], K[j]> product(decay[j+1:i+1]) for j < i.
Outputs and the next state then follow from matrix products with R. Products,
not ratios of cumulative decays, preserve zero-decay reset boundaries.
"""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec


@T.macro
def _chunk_delta_systems(
    query,
    key,
    decay,
    beta,
    offsets,
    systems,
    factors,
    batch,
    chunks,
    key_heads,
    heads,
    width,
    chunk,
    mapping,
    threads,
):
    with T.Kernel(chunks, heads, batch, threads=threads) as (block, head, sequence):
        first = offsets[sequence] + block * chunk
        end = offsets[sequence + 1]
        kh = head % key_heads if mapping == "tiled" else head // (heads // key_heads)
        q = T.alloc_fragment((chunk, width), "float32")
        k = T.alloc_shared((chunk, width), "float32")
        kk = T.alloc_fragment((chunk, chunk), "float32")
        qk = T.alloc_fragment((chunk, chunk), "float32")
        lower = T.alloc_shared((chunk, chunk), "float32")
        inverse = T.alloc_shared((chunk, chunk), "float32")
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
            T.gemm(k, k, kk, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            T.gemm(q, k, qk, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
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
                systems[sequence, block, head, 1, i, j] = T.if_then_else(
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
            for i, j in T.Parallel(chunk, chunk):
                systems[sequence, block, head, 0, i, j] = inverse[i, j]
            for i in T.Parallel(chunk):
                product[0] = 1.0
                for step in T.serial(i + 1):
                    if first + step < end:
                        product[0] *= decay[first + step, head]
                factors[sequence, block, head, 0, i] = product[0]
                product[0] = 1.0
                for step in T.serial(i + 1, chunk):
                    if first + step < end:
                        product[0] *= decay[first + step, head]
                factors[sequence, block, head, 1, i] = product[0]


@T.macro
def _chunk_delta_scan(
    query,
    key,
    value,
    beta,
    previous,
    offsets,
    systems,
    factors,
    output,
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
    dtype,
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
        coefficients = T.alloc_shared((chunk, chunk), "float32")
        contraction = T.alloc_fragment((chunk, columns), "float32")
        solved = T.alloc_fragment((chunk, columns), "float32")
        first = offsets[sequence]
        count = offsets[sequence + 1] - first
        for v, d in T.Parallel(columns, width):
            state[v, d] = T.if_then_else(
                tile * columns + v < value_width, previous[sequence, head, tile * columns + v, d], 0
            )
        for block in T.serial(T.ceildiv(count, chunk)):
            start = first + block * chunk
            for i, d in T.Parallel(chunk, width):
                operand[i, d] = T.if_then_else(
                    block * chunk + i < count, T.cast(key[start + i, kh, d], "float32"), 0
                )
            T.gemm(
                operand,
                state,
                contraction,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, v in T.Parallel(chunk, columns):
                rhs[i, v] = T.if_then_else(
                    block * chunk + i < count and tile * columns + v < value_width,
                    beta[start + i, head]
                    * (
                        T.cast(value[start + i, head, tile * columns + v], "float32")
                        - factors[sequence, block, head, 0, i] * contraction[i, v]
                    ),
                    0,
                )
            for i, j in T.Parallel(chunk, chunk):
                coefficients[i, j] = systems[sequence, block, head, 0, i, j]
            T.gemm(coefficients, rhs, solved, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
            # The right-hand side is dead after the solve. Reuse its tile for
            # residuals, leaving space for a longer chunk and fewer state steps.
            T.copy(solved, rhs)
            for i, d in T.Parallel(chunk, width):
                operand[i, d] = T.if_then_else(
                    block * chunk + i < count, T.cast(query[start + i, kh, d], "float32"), 0
                )
            T.gemm(
                operand,
                state,
                contraction,
                transpose_B=True,
                clear_accum=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, v in T.Parallel(chunk, columns):
                contraction[i, v] *= factors[sequence, block, head, 0, i]
            for i, j in T.Parallel(chunk, chunk):
                coefficients[i, j] = systems[sequence, block, head, 1, i, j]
            T.gemm(coefficients, rhs, contraction, policy=T.GemmWarpPolicy.FullRow)
            for i, v in T.Parallel(chunk, columns):
                if block * chunk + i < count and tile * columns + v < value_width:
                    output[start + i, head, tile * columns + v] = T.cast(contraction[i, v], dtype)
            for i, d in T.Parallel(chunk, width):
                operand[i, d] = T.if_then_else(
                    block * chunk + i < count,
                    T.cast(key[start + i, kh, d], "float32") * factors[sequence, block, head, 1, i],
                    0,
                )
            for v, d in T.Parallel(columns, width):
                state[v, d] *= factors[sequence, block, head, 0, chunk - 1]
            T.gemm(rhs, operand, state, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
        for v, d in T.Parallel(columns, width):
            if tile * columns + v < value_width:
                following[sequence, head, tile * columns + v, d] = state[v, d]


class _ChunkedDeltaEmitter:
    def __init__(self, specs, mapping, chunk, columns, threads):
        self.specs, self.mapping = specs, mapping
        self.chunk, self.columns, self.threads = chunk, columns, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        q, k, v, decay, beta, state, offsets, output, following, systems, factors = operands
        batch, heads, value_width, width = cast(tuple[int, int, int, int], self.specs[5].shape)
        rows, key_heads, _ = cast(tuple[int, int, int], self.specs[0].shape)
        _chunk_delta_systems(
            q,
            k,
            decay,
            beta,
            offsets,
            systems,
            factors,
            batch,
            math.ceil(rows / self.chunk),
            key_heads,
            heads,
            width,
            self.chunk,
            self.mapping,
            self.threads,
        )
        _chunk_delta_scan(
            q,
            k,
            v,
            beta,
            state,
            offsets,
            systems,
            factors,
            output,
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
            self.specs[7].dtype.value,
        )


class ChunkedDeltaRule:
    name = "chunked-gated-delta"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "gated_delta_recurrence" or context.mode != "prefill":
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        rows = cast(int, specs[0].shape[0])
        batch, heads, value_width, width = cast(tuple[int, int, int, int], specs[5].shape)
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == DType.F32 and item.accumulation_dtype == DType.F32
            ),
            None,
        )
        chunk, columns = 32, 16
        if (
            instruction is None
            or "shared" not in context.capabilities.memory_scopes
            or rows < 64 * batch
            or width % instruction.k
            or chunk % instruction.m
            or chunk % instruction.n
            or chunk % instruction.k
            or columns % instruction.m
            or columns % instruction.n
        ):
            return ()
        threads = (chunk // instruction.m) * context.capabilities.subgroup_width
        # Include conservative affine-row padding for each shared matrix.
        prepare_shared = (chunk * (width + 4) + 2 * chunk * (chunk + 4)) * 4
        scan_shared = (
            (columns + chunk) * (width + 4) + chunk * (columns + 4) + chunk * (chunk + 4)
        ) * 4
        if (
            threads > context.capabilities.threads_per_group
            or max(prepare_shared, scan_shared) > context.capabilities.shared_memory_bytes
        ):
            return ()
        chunks = math.ceil(rows / chunk)
        workspace = (
            TensorSpec((batch, chunks, heads, 2, chunk, chunk), DType.F32),
            TensorSpec((batch, chunks, heads, 2, chunk), DType.F32),
        )
        if sum(spec.storage_nbytes for spec in workspace) > context.workspace_limit:
            return ()
        return (
            Candidate(
                f"gated_delta.chunked-matrix@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _ChunkedDeltaEmitter(specs, node.attributes["mapping"], chunk, columns, threads),
                1e-6 + rows * heads * width * value_width / 2e12,
                workspace=workspace,
                aliases=((node.outputs[1], node.inputs[5]),),
                kernel_count=2,
                priority=70,
            ),
        )
