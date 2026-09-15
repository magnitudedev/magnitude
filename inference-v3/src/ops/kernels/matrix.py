"""Production dense and packed matrix schedules.

Small-row projections are packet GEMVs. Prefill contractions stage exact codes
and reconstruct immediate FP32 operands for one continuous contraction.
No schedule in this module performs scalar logical dequantization.
"""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..representations import Affine, Dense
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .packed import (affine_gemm, affine_shared_bytes, affine_storage,
                     load_matrix_tile, packet_dot, packet_format, prepare_packet_activation)
from .publication import publish


def _dense(spec: TensorSpec) -> bool:
    return spec.representation is None or spec.representation == Dense(spec.dtype)


def _packet_reduction_width(*specs: TensorSpec) -> int:
    representations = tuple(spec.representation for spec in specs)
    assert all(isinstance(value, Affine) for value in representations)
    # Staging spans multiple groups when necessary; group boundaries no longer
    # terminate the accumulator or force an output correction.
    return min(64, max(32, math.gcd(*(cast(Affine, value).group for value in representations))))


def _packed_vector_geometry(spec: TensorSpec, context: LoweringContext) -> tuple[int, int] | None:
    packet = packet_format(spec)
    if packet is None or context.compiler_target.subgroup_width != 32:
        return None
    outputs_per_subgroup = 4 if packet.name == "mlx-q4-group64" else 2
    if packet.name == "mlx-q4-group64":
        outputs = cast(int, spec.shape[0])
        groups = min(2, max(1, (outputs + outputs_per_subgroup - 1) // outputs_per_subgroup))
        threads = min(groups * 32, context.compiler_target.threads_per_group // 32 * 32)
        return (threads, outputs_per_subgroup) if threads >= 32 else None
    outputs = cast(int, spec.shape[0])
    required_subgroups = max(
        1, min(4, (outputs + outputs_per_subgroup - 1) // outputs_per_subgroup)
    )
    threads = min(context.compiler_target.threads_per_group, required_subgroups * 32)
    threads = threads // 32 * 32
    if threads < 32:
        return None
    return threads, outputs_per_subgroup


@T.macro
def _packed_vector(
    source,
    weight,
    bias,
    result,
    spec,
    m,
    n,
    k_size,
    output_dtype,
    has_bias,
    threads,
    outputs_per_subgroup,
    extent=None,
    residual=None,
):
    packet = packet_format(spec)
    assert packet is not None
    subgroups = threads // 32
    outputs_per_group = subgroups * outputs_per_subgroup
    with T.Kernel(T.ceildiv(n, outputs_per_group), m, threads=threads) as (block, row):
        offset = extent[0] if extent is not None else 0
        thread = T.get_thread_binding()
        lane = thread % 32
        subgroup = thread // 32
        first_output = block * outputs_per_group + subgroup * outputs_per_subgroup
        values = T.alloc_local((packet.dot_packet,), "float32")
        partial = T.alloc_local((outputs_per_subgroup,), "float32")
        T.clear(partial)
        for chunk in T.serial(k_size // packet.tile):
            for item in T.unroll(packet.dot_packet, explicit=True):
                values[item] = T.cast(
                    source[row, chunk * packet.tile + lane * packet.dot_packet + item], "float32"
                )
            activation_sum, masked = prepare_packet_activation(values, source.dtype, packet)
            for owned in T.unroll(outputs_per_subgroup, explicit=True):
                output = first_output + owned
                if output < n:
                    partial[owned] += packet_dot(values, weight, spec, output, chunk, lane, activation_sum, masked)
        for owned in T.unroll(outputs_per_subgroup, explicit=True):
            output = first_output + owned
            total = T.warp_reduce_sum(partial[owned])
            if lane == 0 and output < n:
                value = total
                if has_bias:
                    value += T.cast(bias[output + offset], "float32")
                publish(result, residual, row, output + offset, value, output_dtype)


class _PackedVectorEmitter:
    def __init__(
        self,
        spec: TensorSpec,
        m: int,
        n: int,
        k: int,
        output: str,
        bias: bool,
        threads: int,
        outputs_per_subgroup: int,
    ):
        self.spec, self.m, self.n, self.k = spec, m, n, k
        self.output, self.bias = output, bias
        self.threads = threads
        self.outputs_per_subgroup = outputs_per_subgroup

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _packed_vector(
            source,
            weight,
            bias,
            result,
            self.spec,
            self.m,
            self.n,
            self.k,
            self.output,
            self.bias,
            self.threads,
            self.outputs_per_subgroup,
        )


@T.macro
def _packed_matrix(
    source,
    weight,
    bias,
    result,
    spec,
    m,
    n,
    k_size,
    reduction_step,
    output_dtype,
    threads,
    bm,
    bn,
    bk,
    has_bias,
    extent=None,
    residual=None,
):
    packet = packet_format(spec)
    assert packet is not None
    full = (
        m % bm == 0
        and n % bn == 0
        and k_size % bk == 0
        and (bn * bk // packet.matrix_packet) % threads == 0
    )
    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
        offset = extent[0] if extent is not None else 0
        storage = affine_storage(bm, bn, bk, source.dtype, reduction_step, (spec,))
        left, right, coefficients, accum, b = storage
        T.clear(accum)
        for block in T.serial(T.ceildiv(k_size, bk)):
            for i, k in T.Parallel(bm, bk):
                if full:
                    left[i, k] = source[by * bm + i, block * bk + k]
                else:
                    left[i, k] = T.if_then_else(
                        by * bm + i < m and block * bk + k < k_size,
                        source[by * bm + i, block * bk + k],
                        0,
                    )
            load_matrix_tile(right, coefficients, weight, spec, bx * bn, block * bk, n, k_size, bn, bk, threads)
            affine_gemm(storage, bm, bn, bk, bm)
        for i, j in T.Parallel(bm, bn):
            if full:
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[bx * bn + j + offset], "float32")
                publish(result, residual, by * bm + i, bx * bn + j + offset, value, output_dtype)
            elif by * bm + i < m and bx * bn + j < n:
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[bx * bn + j + offset], "float32")
                publish(result, residual, by * bm + i, bx * bn + j + offset, value, output_dtype)


class _PackedMatrixEmitter:
    def __init__(self, spec, m, n, k, reduction_step, output, threads, tile, bias):
        self.spec, self.m, self.n, self.k = spec, m, n, k
        self.reduction_step, self.output = reduction_step, output
        self.threads, self.tile, self.bias = threads, tile, bias

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _packed_matrix(
            source,
            weight,
            bias,
            result,
            self.spec,
            self.m,
            self.n,
            self.k,
            self.reduction_step,
            self.output,
            self.threads,
            *self.tile,
            self.bias,
        )


@T.macro
def _dense_vector(source, weight, bias, result, m, n, k_size, output_dtype, has_bias,
                  extent=None, residual=None):
    with T.Kernel(T.ceildiv(n, 8), m, threads=128) as (block, row):
        offset = extent[0] if extent is not None else 0
        thread = T.get_thread_binding()
        lane = thread % 32
        first_output = block * 8 + (thread // 32) * 2
        partial = T.alloc_local((2,), "float32")
        T.clear(partial)
        for chunk in T.serial(T.ceildiv(k_size, 32)):
            reduction = chunk * 32 + lane
            if reduction < k_size:
                value = T.cast(source[row, reduction], "float32")
                for owned in T.unroll(2, explicit=True):
                    output = first_output + owned
                    if output < n:
                        partial[owned] += value * T.cast(weight[output, reduction], "float32")
        for owned in T.unroll(2, explicit=True):
            total = T.warp_reduce_sum(partial[owned])
            output = first_output + owned
            if lane == 0 and output < n:
                value = total
                if has_bias:
                    value += T.cast(bias[output + offset], "float32")
                publish(result, residual, row, output + offset, value, output_dtype)


class _DenseVectorEmitter:
    def __init__(self, m, n, k, output, bias):
        self.m, self.n, self.k, self.output, self.bias = m, n, k, output, bias

    def __call__(self, operands):
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _dense_vector(source, weight, bias, result, self.m, self.n, self.k, self.output, self.bias)


@T.macro
def _dense_matrix(
    source,
    weight,
    bias,
    result,
    operation,
    m,
    n,
    k_size,
    dtype,
    output_dtype,
    threads,
    bm,
    bn,
    bk,
    has_bias,
    extent=None,
    residual=None,
):
    full = m % bm == 0 and n % bn == 0 and k_size % bk == 0
    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
        offset = extent[0] if extent is not None else 0
        left = T.alloc_shared((bm, bk), dtype)
        right = T.alloc_shared((bn, bk), dtype)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        for block in T.serial(T.ceildiv(k_size, bk)):
            for i, k in T.Parallel(bm, bk):
                if full:
                    left[i, k] = source[by * bm + i, block * bk + k]
                else:
                    left[i, k] = T.if_then_else(
                        by * bm + i < m and block * bk + k < k_size,
                        source[by * bm + i, block * bk + k],
                        0,
                    )
            for j, k in T.Parallel(bn, bk):
                row, reduction = bx * bn + j, block * bk + k
                if full:
                    right[j, k] = (
                        weight[row, reduction] if operation == "linear" else weight[reduction, row]
                    )
                else:
                    right[j, k] = T.if_then_else(
                        row < n and reduction < k_size,
                        weight[row, reduction] if operation == "linear" else weight[reduction, row],
                        0,
                    )
            T.sync_threads()
            T.gemm(left, right, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            if full or (by * bm + i < m and bx * bn + j < n):
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[bx * bn + j + offset], "float32")
                publish(result, residual, by * bm + i, bx * bn + j + offset, value, output_dtype)


class _DenseMatrixEmitter:
    def __init__(self, operation, m, n, k, dtype, output, threads, tile, bias):
        self.operation, self.m, self.n, self.k = operation, m, n, k
        self.dtype, self.output, self.threads, self.tile, self.bias = (
            dtype,
            output,
            threads,
            tile,
            bias,
        )

    def __call__(self, operands):
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _dense_matrix(
            source,
            weight,
            bias,
            result,
            self.operation,
            self.m,
            self.n,
            self.k,
            self.dtype,
            self.output,
            self.threads,
            *self.tile,
            self.bias,
        )


def _four(values):
    if len(values) == 3:
        return values[0], values[1], values[2], values[2]
    if len(values) == 4:
        return tuple(values)
    raise ValueError("parallel packet projection requires three or four branches")


@T.macro
def _packet_projection_partial(source, weight, spec, row, output, width, lane,
                               shared_values=None, shared_chunk=None, prepared_sum=None, prepared_mask=None):
    packet = packet_format(spec)
    assert packet is not None
    if shared_chunk is not None:
        return packet_dot(shared_values, weight, spec, output, shared_chunk, lane, prepared_sum, prepared_mask)
    else:
        values = T.alloc_local((packet.dot_packet,), "float32")
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for chunk in T.serial(width // packet.tile):
            for item in T.unroll(packet.dot_packet, explicit=True):
                values[item] = T.cast(
                    source[row, chunk * packet.tile + lane * packet.dot_packet + item], "float32"
                )
            partial[0] += packet_dot(values, weight, spec, output, chunk, lane)
        return partial[0]


@T.macro
def _parallel_packed(
    source,
    weight0,
    weight1,
    weight2,
    weight3,
    output0,
    output1,
    output2,
    output3,
    specs,
    output_specs,
    rows,
    width,
    sizes,
    count,
    threads,
    outputs_per_subgroup,
):
    total_outputs = sum(sizes[:count])
    packets = tuple(packet_format(spec) for spec in specs[:count])
    packet = packets[0]
    shared_geometry = all((item.tile, item.dot_packet) == (packet.tile, packet.dot_packet) for item in packets)
    subgroups = threads // 32
    outputs_per_group = subgroups * outputs_per_subgroup
    with T.Kernel(T.ceildiv(total_outputs, outputs_per_group), rows, threads=threads) as (
        block,
        row,
    ):
        thread = T.get_thread_binding()
        lane = thread % 32
        subgroup = thread // 32
        first = block * outputs_per_group + subgroup * outputs_per_subgroup
        partial = T.alloc_local((outputs_per_subgroup,), "float32")
        values = T.alloc_local((packet.dot_packet if shared_geometry else 1,), "float32")
        T.clear(partial)
        for chunk in T.serial(width // packet.tile if shared_geometry else 1):
            if shared_geometry:
                for item in T.unroll(packet.dot_packet, explicit=True):
                    values[item] = T.cast(source[row, chunk * packet.tile + lane * packet.dot_packet + item], "float32")
                activation_sum, masked = prepare_packet_activation(values, source.dtype, packet)
            else:
                activation_sum, masked = None, None
            for owned in T.unroll(outputs_per_subgroup, explicit=True):
                combined = first + owned
                if combined < total_outputs:
                    if combined < sizes[0]:
                        partial[owned] += _packet_projection_partial(
                            source, weight0, specs[0], row, combined, width, lane,
                            values, chunk if shared_geometry else None, activation_sum, masked)
                    elif combined < sizes[0] + sizes[1]:
                        partial[owned] += _packet_projection_partial(
                            source, weight1, specs[1], row, combined - sizes[0], width, lane,
                            values, chunk if shared_geometry else None, activation_sum, masked)
                    elif combined < sizes[0] + sizes[1] + sizes[2]:
                        partial[owned] += _packet_projection_partial(
                            source, weight2, specs[2], row, combined - sizes[0] - sizes[1], width, lane,
                            values, chunk if shared_geometry else None, activation_sum, masked)
                    else:
                        partial[owned] += _packet_projection_partial(
                            source, weight3, specs[3], row, combined - sizes[0] - sizes[1] - sizes[2], width, lane,
                            values, chunk if shared_geometry else None, activation_sum, masked)
        for owned in T.unroll(outputs_per_subgroup, explicit=True):
            combined = first + owned
            projected = T.warp_reduce_sum(partial[owned])
            if lane == 0 and combined < total_outputs:
                if combined < sizes[0]:
                    output0[row, combined] = T.cast(projected, output_specs[0].dtype.value)
                elif combined < sizes[0] + sizes[1]:
                    output1[row, combined - sizes[0]] = T.cast(
                        projected, output_specs[1].dtype.value
                    )
                elif combined < sizes[0] + sizes[1] + sizes[2]:
                    output2[row, combined - sizes[0] - sizes[1]] = T.cast(
                        projected, output_specs[2].dtype.value
                    )
                else:
                    output3[row, combined - sizes[0] - sizes[1] - sizes[2]] = T.cast(
                        projected, output_specs[3].dtype.value
                    )


@T.macro
def _parallel_packed_matrix(
    source,
    weight0,
    weight1,
    weight2,
    weight3,
    output0,
    output1,
    output2,
    output3,
    specs,
    output_specs,
    rows,
    width,
    sizes,
    blocks,
    count,
    reduction_step,
    threads,
    bm,
    bn,
    bk,
):
    total_blocks = sum(blocks[:count])
    full_rows = rows % bm == 0 and width % bk == 0
    with T.Kernel(total_blocks, T.ceildiv(rows, bm), threads=threads) as (branch_block, by):
        storage = affine_storage(bm, bn, bk, source.dtype, reduction_step, specs[:count])
        left, right, coefficients, accum, b = storage
        T.clear(accum)
        for reduction_block in T.serial(T.ceildiv(width, bk)):
            for i, k in T.Parallel(bm, bk):
                row = by * bm + i
                reduction = reduction_block * bk + k
                left[i, k] = T.if_then_else(
                    (full_rows or row < rows) and reduction < width,
                    source[row, reduction],
                    0,
                )
            if branch_block < blocks[0]:
                load_matrix_tile(
                    right,
                    coefficients,
                    weight0,
                    specs[0],
                    (branch_block) * bn,
                    reduction_block * bk,
                    sizes[0],
                    width,
                    bn,
                    bk,
                    threads,
                )
            elif branch_block < blocks[0] + blocks[1]:
                load_matrix_tile(
                    right,
                    coefficients,
                    weight1,
                    specs[1],
                    (branch_block - blocks[0]) * bn,
                    reduction_block * bk,
                    sizes[1],
                    width,
                    bn,
                    bk,
                    threads,
                )
            elif branch_block < blocks[0] + blocks[1] + blocks[2]:
                load_matrix_tile(
                    right,
                    coefficients,
                    weight2,
                    specs[2],
                    (branch_block - blocks[0] - blocks[1]) * bn,
                    reduction_block * bk,
                    sizes[2],
                    width,
                    bn,
                    bk,
                    threads,
                )
            else:
                load_matrix_tile(
                    right,
                    coefficients,
                    weight3,
                    specs[3],
                    (branch_block - blocks[0] - blocks[1] - blocks[2]) * bn,
                    reduction_block * bk,
                    sizes[3],
                    width,
                    bn,
                    bk,
                    threads,
                )
            affine_gemm(storage, bm, bn, bk, bm)
        for i, j in T.Parallel(bm, bn):
            row = by * bm + i
            if branch_block < blocks[0]:
                column = branch_block * bn + j
                if row < rows and column < sizes[0]:
                    output0[row, column] = T.cast(accum[i, j], output_specs[0].dtype.value)
            elif branch_block < blocks[0] + blocks[1]:
                column = (branch_block - blocks[0]) * bn + j
                if row < rows and column < sizes[1]:
                    output1[row, column] = T.cast(accum[i, j], output_specs[1].dtype.value)
            elif branch_block < blocks[0] + blocks[1] + blocks[2]:
                column = (branch_block - blocks[0] - blocks[1]) * bn + j
                if row < rows and column < sizes[2]:
                    output2[row, column] = T.cast(accum[i, j], output_specs[2].dtype.value)
            else:
                column = (branch_block - blocks[0] - blocks[1] - blocks[2]) * bn + j
                if row < rows and column < sizes[3]:
                    output3[row, column] = T.cast(accum[i, j], output_specs[3].dtype.value)


class _ParallelPackedEmitter:
    def __init__(
        self,
        weight_specs,
        output_specs,
        rows,
        width,
        mode,
        reduction_step,
        threads,
        outputs_per_subgroup,
        tile,
    ):
        self.weight_specs, self.output_specs = weight_specs, output_specs
        self.rows, self.width = rows, width
        self.mode, self.reduction_step, self.threads = mode, reduction_step, threads
        self.outputs_per_subgroup, self.tile = outputs_per_subgroup, tile

    def __call__(self, operands):
        count = len(self.weight_specs)
        weights = _four(operands[1 : count + 1])
        outputs = _four(operands[count + 1 : 2 * count + 1])
        specs = _four(self.weight_specs)
        output_specs = _four(self.output_specs)
        sizes = tuple(cast(int, spec.shape[0]) for spec in specs)
        if self.mode == "decode":
            _parallel_packed(
                operands[0],
                *weights,
                *outputs,
                specs,
                output_specs,
                self.rows,
                self.width,
                sizes,
                count,
                self.threads,
                self.outputs_per_subgroup,
            )
        else:
            bm, bn, bk = self.tile
            blocks = tuple((size + bn - 1) // bn for size in sizes)
            _parallel_packed_matrix(
                operands[0],
                *weights,
                *outputs,
                specs,
                output_specs,
                self.rows,
                self.width,
                sizes,
                blocks,
                count,
                self.reduction_step,
                self.threads,
                bm,
                bn,
                bk,
            )


class ParallelPackedMatrixRule:
    """Fuse three or four adjacent projections sharing an activation."""

    name = "parallel-packed-matrix"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        if context.mode not in {"decode", "prefill"} or context.compiler_target.subgroup_width != 32:
            return ()
        first = graph.nodes[root]
        if first.operation != "linear" or len(first.inputs) != 2:
            return ()
        source = first.inputs[0]
        nodes = []
        for index in range(root, min(root + 4, len(graph.nodes))):
            node = graph.nodes[index]
            if node.operation != "linear" or len(node.inputs) != 2 or node.inputs[0] != source:
                break
            nodes.append(node)
        if len(nodes) < 3:
            return ()
        source_spec = graph.values[source].spec
        if not source_spec.static or source_spec.rank != 2:
            return ()
        rows, width = cast(tuple[int, int], source_spec.shape)
        weight_specs = tuple(graph.values[node.inputs[1]].spec for node in nodes)
        formats = tuple(packet_format(spec) for spec in weight_specs)
        if any(
            packet is None or spec.shape[1] != width or width % packet.tile
            for spec, packet in zip(weight_specs, formats, strict=True)
        ):
            return ()
        output_specs = tuple(graph.values[node.outputs[0]].spec for node in nodes)
        mlx_affine = any(
            packet.name == "mlx-q4-group64" for packet in formats if packet is not None
        )
        threads = min(
            context.compiler_target.threads_per_group,
            64 if context.mode == "decode" and mlx_affine else 128,
        )
        threads = (
            threads // context.compiler_target.subgroup_width * context.compiler_target.subgroup_width
        )
        if threads < context.compiler_target.subgroup_width:
            return ()
        tile = None
        reduction_step = None
        outputs_per_subgroup = 1
        if context.mode == "decode":
            if rows > 8:
                return ()
            outputs_per_subgroup = (
                4
                if any(packet.name == "mlx-q4-group64" for packet in formats if packet is not None)
                else 2
            )
        else:
            reduction_step = 16
            if rows < 8:
                return ()
            bm, bn, bk = (
                (32, 64, 32)
                if rows >= 64
                else (16, 16, 8)
            )
            bk = _packet_reduction_width(*weight_specs)
            if bk % 8:
                return ()
            threads = min(threads, bm // 8 * context.compiler_target.subgroup_width)
            if affine_shared_bytes(bm, bn, bk, source_spec.dtype, *weight_specs) > context.compiler_target.shared_memory_bytes:
                return ()
            tile = (bm, bn, bk)
        moved = sum(spec.storage_nbytes for spec in weight_specs)
        return (
            BoundOperation(
                f"linear.parallel-packet-{context.mode}@{root}:{nodes[-1].id}",
                frozenset(node.id for node in nodes),
                (source, *(node.inputs[1] for node in nodes)),
                tuple(node.outputs[0] for node in nodes),
                _ParallelPackedEmitter(
                    weight_specs,
                    output_specs,
                    rows,
                    width,
                    context.mode,
                    reduction_step,
                    threads,
                    outputs_per_subgroup,
                    tile,
                ),
            ),
        )


class PackedMatrixRule:
    name = "packed-matrix"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "linear":
            return ()
        left, right = (graph.values[value].spec for value in node.inputs[:2])
        packet = packet_format(right)
        if (
            packet is None
            or not left.static
            or not right.static
            or left.rank != 2
            or right.rank != 2
            or context.compiler_target.subgroup_width != 32
        ):
            return ()
        m, k = cast(tuple[int, int], left.shape)
        n = cast(int, right.shape[0])
        if k % packet.tile:
            return ()
        output = graph.values[node.outputs[0]].spec
        vector = _packed_vector_geometry(right, context)
        reduction_step = 16
        if vector is not None and m < 8:
            threads, outputs_per_subgroup = vector
            emitter = _PackedVectorEmitter(
                right,
                m,
                n,
                k,
                output.dtype.value,
                len(node.inputs) == 3,
                threads,
                outputs_per_subgroup,
            )
            return (BoundOperation(f"linear.packet-vector@{root}", frozenset({root}),
                                   node.inputs, node.outputs, emitter),)
        bk = _packet_reduction_width(right)
        if bk % 8 == 0:
            bm, bn, threads = matrix_geometry(context, left.dtype, m, n, bk,
                                              packed_specs=(right,), storage_dtype=left.dtype)
            emitter = _PackedMatrixEmitter(right, m, n, k, reduction_step,
                                           output.dtype.value, threads, (bm, bn, bk), len(node.inputs) == 3)
            return (BoundOperation(f"linear.packet-gemm@{root}", frozenset({root}),
                                   node.inputs, node.outputs, emitter),)
        raise ValueError("packed projection cannot express the declared contraction on this device")


class DenseMatrixRule:
    name = "dense-matrix"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation not in {"linear", "matmul"}:
            return ()
        left, right = (graph.values[value].spec for value in node.inputs[:2])
        if (
            not left.static
            or not right.static
            or not _dense(right)
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        output = graph.values[node.outputs[0]].spec
        m = left.elements // cast(int, left.shape[-1])
        k = cast(int, left.shape[-1])
        n = cast(int, output.shape[-1])
        if (node.operation == "linear" and context.compiler_target.subgroup_width == 32 and
                context.compiler_target.threads_per_group >= 128 and
                m < 8):
            emitter = _DenseVectorEmitter(m, n, k, output.dtype.value, len(node.inputs) == 3)
            return (BoundOperation(f"linear.dense-vector@{root}", frozenset({root}),
                                   node.inputs, node.outputs, emitter),)
        bk = 16
        bm, bn, threads = matrix_geometry(context, left.dtype, m, n, bk)
        emitter = _DenseMatrixEmitter(node.operation, m, n, k, left.dtype.value,
                                      output.dtype.value, threads, (bm, bn, bk), len(node.inputs) == 3)
        return (BoundOperation(f"{node.operation}.dense-gemm@{root}", frozenset({root}),
                               node.inputs, node.outputs, emitter),)


def matrix_geometry(context, dtype, rows, columns, reduction, *, packed_specs=(), storage_dtype=None):
    """Choose work tiles in eight-element strips within the shared-memory budget.

    These are schedule parameters. TileLang selects instructions and infers the
    layout of the concrete GEMMs; no instruction dimensions are queried.
    """
    # A sixteen-row physical tile is the smallest shape shared by the native
    # matrix paths.  Logical tails remain masked by the surrounding loads and
    # stores; keeping the physical contraction legal lets TileLang select the
    # efficient instruction for each target without backend branches here.
    m_tiles = min(4, max(2, math.ceil(rows / 8)))
    n_tiles = min(4, max(1, math.ceil(columns / 8)))
    def shared():
        if packed_specs:
            return affine_shared_bytes(8 * m_tiles, 8 * n_tiles,
                                       reduction, storage_dtype, *packed_specs)
        return (8 * m_tiles + 8 * n_tiles) * reduction * dtype.itemsize

    while shared() > context.compiler_target.shared_memory_bytes:
        if n_tiles > 1:
            n_tiles //= 2
        elif m_tiles > 1:
            m_tiles //= 2
        else:
            raise ValueError("one matrix tile exceeds device shared-memory capacity")
    threads = min(context.compiler_target.threads_per_group,
                  context.compiler_target.subgroup_width * min(4, m_tiles))
    return 8 * m_tiles, 8 * n_tiles, threads


__all__ = [
    "DenseMatrixRule",
    "PackedMatrixRule",
    "ParallelPackedMatrixRule",
]
