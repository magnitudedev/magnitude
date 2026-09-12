"""Capability-selected dense matrix schedules authored directly in TileLang."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..compiler.tuning import TuningKey
from ..representations import (
    Affine,
    CodeInterpretation,
    Dense,
    DirectCoefficients,
)
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .quantization import represented_load


def _direct_weight(storage, spec: TensorSpec, output, reduction):
    _, columns = cast(tuple[int, int], spec.shape)
    if spec.representation is None or isinstance(spec.representation, Dense):
        return storage[output, reduction]
    return represented_load(storage, spec, output * columns + reduction)


@T.macro
def _encoded_float(storage, offset, dtype):
    if dtype == "float16":
        bits = T.cast(storage[offset], "uint16")
        bits |= T.cast(storage[offset + 1], "uint16") << 8
        return T.cast(T.reinterpret(bits, "float16"), "float32")
    if dtype == "bfloat16":
        bits = T.cast(storage[offset], "uint32")
        bits |= T.cast(storage[offset + 1], "uint32") << 8
        return T.reinterpret(bits << 16, "float32")
    bits = T.cast(storage[offset], "uint32")
    bits |= T.cast(storage[offset + 1], "uint32") << 8
    bits |= T.cast(storage[offset + 2], "uint32") << 16
    bits |= T.cast(storage[offset + 3], "uint32") << 24
    return T.reinterpret(bits, "float32")


@T.macro
def _packet_affine_linear(
    source,
    weight,
    bias,
    result,
    m,
    n,
    k_size,
    group,
    coefficient_dtype,
    output_dtype,
    has_bias,
):
    code_bytes = n * k_size // 2
    groups = n * k_size // group
    coefficient_bytes = 2 if coefficient_dtype in ("float16", "bfloat16") else 4
    bias_base = code_bytes + groups * coefficient_bytes
    pack = 16
    with T.Kernel(n, m, threads=32) as (column, row):
        lane = T.get_thread_binding(0)
        partial = T.alloc_local((1,), "float32")
        dot = T.alloc_local((1,), "float32")
        source_sum = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        for chunk in T.serial(T.ceildiv(k_size, 32 * pack)):
            base = (chunk * 32 + lane) * pack
            dot[0] = 0.0
            source_sum[0] = 0.0
            for item in T.unroll(pack // 4):
                reduction = base + item * 4
                if reduction < k_size:
                    element = column * k_size + reduction
                    packed = T.cast(weight[element // 2], "uint16")
                    packed |= T.cast(weight[element // 2 + 1], "uint16") << 8
                    for offset in T.unroll(4):
                        value = T.cast(source[row, reduction + offset], "float32")
                        code = T.cast(
                            (packed >> (offset * 4)) & T.cast(15, "uint16"),
                            "float32",
                        )
                        dot[0] += value * code
                        source_sum[0] += value
            if base < k_size:
                coefficient = (column * (k_size // group) + base // group) * coefficient_bytes
                scale = _encoded_float(weight, code_bytes + coefficient, coefficient_dtype)
                zero = _encoded_float(weight, bias_base + coefficient, coefficient_dtype)
                partial[0] += dot[0] * scale + source_sum[0] * zero
        total = T.warp_reduce_sum(partial[0])
        if lane == 0:
            value = total
            if has_bias:
                value += T.cast(bias[column], "float32")
            result[row, column] = T.cast(value, output_dtype)


class _PacketAffineEmitter:
    def __init__(
        self,
        weight_spec: TensorSpec,
        m: int,
        n: int,
        k: int,
        output_dtype: str,
        bias: bool,
    ):
        representation = weight_spec.representation
        assert isinstance(representation, Affine)
        coefficients = representation.coefficients
        assert isinstance(coefficients, DirectCoefficients)
        self.group = representation.group
        self.coefficient_dtype = coefficients.scale_dtype.value
        self.m, self.n, self.k = m, n, k
        self.output_dtype = output_dtype
        self.bias = bias

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _packet_affine_linear(
            source,
            weight,
            bias,
            result,
            self.m,
            self.n,
            self.k,
            self.group,
            self.coefficient_dtype,
            self.output_dtype,
            self.bias,
        )


class PacketAffineMatrixRule:
    """Subgroup packet GEMV over canonical direct-affine four-bit storage."""

    name = "packet-affine-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "linear":
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        representation = right.representation
        if (
            not left.static
            or not right.static
            or left.rank != 2
            or right.rank != 2
            or not isinstance(representation, Affine)
            or representation.code.bits != 4
            or representation.code.interpretation != CodeInterpretation.UNSIGNED
            or not isinstance(representation.coefficients, DirectCoefficients)
            or representation.coefficients.bias_dtype
            != representation.coefficients.scale_dtype
            or context.capabilities.subgroup_width != 32
        ):
            return ()
        m, k = cast(tuple[int, int], left.shape)
        n = cast(int, right.shape[0])
        if (
            m > 4
            or k % representation.group
            or representation.group % 16
            or right.elements % 2
        ):
            return ()
        output = graph.values[node.outputs[0]].spec
        return (
            Candidate(
                f"linear.packet-affine@{root}",
                frozenset({root}),
                tuple(node.inputs),
                tuple(node.outputs),
                _PacketAffineEmitter(
                    right,
                    m,
                    n,
                    k,
                    output.dtype.value,
                    len(node.inputs) == 3,
                ),
                2e-7 + right.storage_nbytes / 10e12,
                priority=60,
            ),
        )


@T.macro
def _fragment_matrix(
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
):
    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
        left = T.alloc_shared((bm, bk), dtype)
        right = T.alloc_shared((bn, bk), dtype)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        for block in T.serial(T.ceildiv(k_size, bk)):
            for i, k in T.Parallel(bm, bk):
                left[i, k] = T.if_then_else(
                    by * bm + i < m and block * bk + k < k_size,
                    source[by * bm + i, block * bk + k],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                if operation == "linear":
                    value = weight[bx * bn + j, block * bk + k]
                else:
                    value = weight[block * bk + k, bx * bn + j]
                right[j, k] = T.if_then_else(bx * bn + j < n and block * bk + k < k_size, value, 0)
            T.sync_threads()
            T.gemm(left, right, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            if by * bm + i < m and bx * bn + j < n:
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[bx * bn + j], "float32")
                result[by * bm + i, bx * bn + j] = T.cast(value, output_dtype)


class _TiledMatrixEmitter:
    def __init__(
        self,
        operation: str,
        m: int,
        n: int,
        k: int,
        dtype: str,
        output: str,
        threads: int,
        tile: tuple[int, int, int],
        bias: bool,
    ):
        self.operation = operation
        self.m, self.n, self.k = m, n, k
        self.dtype, self.output = dtype, output
        self.threads, self.tile, self.bias = threads, tile, bias

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _fragment_matrix(
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


@T.macro
def _fragment_encoded_linear(
    source,
    weight,
    bias,
    result,
    weight_spec,
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
):
    with T.Kernel(T.ceildiv(n, bn), T.ceildiv(m, bm), threads=threads) as (bx, by):
        left = T.alloc_shared((bm, bk), dtype)
        right = T.alloc_shared((bn, bk), dtype)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        for block in T.serial(T.ceildiv(k_size, bk)):
            for i, k in T.Parallel(bm, bk):
                left[i, k] = T.if_then_else(
                    by * bm + i < m and block * bk + k < k_size,
                    source[by * bm + i, block * bk + k],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                row = bx * bn + j
                column = block * bk + k
                if row < n and column < k_size:
                    right[j, k] = T.cast(
                        represented_load(weight, weight_spec, row * k_size + column),
                        dtype,
                    )
                else:
                    right[j, k] = 0
            T.sync_threads()
            T.gemm(left, right, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            row = by * bm + i
            column = bx * bn + j
            if row < m and column < n:
                value = accum[i, j]
                if has_bias:
                    value += T.cast(bias[column], "float32")
                result[row, column] = T.cast(value, output_dtype)


class _EncodedMatrixEmitter:
    def __init__(
        self,
        weight_spec: TensorSpec,
        m: int,
        n: int,
        k: int,
        dtype: str,
        output: str,
        threads: int,
        tile: tuple[int, int, int],
        bias: bool,
    ):
        self.weight_spec = weight_spec
        self.m, self.n, self.k = m, n, k
        self.dtype, self.output = dtype, output
        self.threads, self.tile, self.bias = threads, tile, bias

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _fragment_encoded_linear(
            source,
            weight,
            bias,
            result,
            self.weight_spec,
            self.m,
            self.n,
            self.k,
            self.dtype,
            self.output,
            self.threads,
            *self.tile,
            self.bias,
        )


@T.macro
def _direct_encoded_linear(
    source,
    weight,
    bias,
    result,
    weight_spec,
    m,
    n,
    k_size,
    output_tile,
    reduction_lanes,
    output_dtype,
    has_bias,
    encoded,
):
    with T.Kernel(
        T.ceildiv(n, output_tile),
        m,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared = T.alloc_shared((output_tile, reduction_lanes), "float32")
        T.clear(partial)
        for chunk in T.serial(T.ceildiv(k_size, reduction_lanes)):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                column = block * output_tile + slot
                reduction = chunk * reduction_lanes + lane
                if column < n and reduction < k_size:
                    if encoded:
                        coefficient = represented_load(
                            weight, weight_spec, column * k_size + reduction
                        )
                    else:
                        coefficient = weight[column, reduction]
                    partial[slot, lane] += T.cast(
                        source[row, reduction], "float32"
                    ) * T.cast(coefficient, "float32")
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared[slot, lane] = partial[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared[slot, lane] += shared[
                        slot, lane + (reduction_lanes >> (step + 1))
                    ]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            column = block * output_tile + slot
            if column < n:
                value = shared[slot, 0]
                if has_bias:
                    value += T.cast(bias[column], "float32")
                result[row, column] = T.cast(value, output_dtype)


class _DirectEncodedEmitter:
    def __init__(
        self,
        weight_spec: TensorSpec,
        m: int,
        n: int,
        k: int,
        output_tile: int,
        reduction_lanes: int,
        output_dtype: str,
        bias: bool,
        encoded: bool = True,
    ):
        self.weight_spec = weight_spec
        self.m, self.n, self.k = m, n, k
        self.output_tile = output_tile
        self.reduction_lanes = reduction_lanes
        self.output_dtype = output_dtype
        self.bias = bias
        self.encoded = encoded

    def __call__(self, operands: tuple[Any, ...]) -> None:
        source, weight = operands[:2]
        bias = operands[2] if self.bias else source
        result = operands[3] if self.bias else operands[2]
        _direct_encoded_linear(
            source,
            weight,
            bias,
            result,
            self.weight_spec,
            self.m,
            self.n,
            self.k,
            self.output_tile,
            self.reduction_lanes,
            self.output_dtype,
            self.bias,
            self.encoded,
        )


@T.macro
def _parallel_direct_linear(
    source,
    weight0,
    weight1,
    weight2,
    weight3,
    output0,
    output1,
    output2,
    output3,
    weight_specs,
    output_specs,
    rows,
    width,
    sizes,
    count,
    output_tile,
    reduction_lanes,
):
    total = sum(sizes[:count])
    with T.Kernel(
        T.ceildiv(total, output_tile),
        rows,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared = T.alloc_shared((output_tile, reduction_lanes), "float32")
        coefficient = T.alloc_local((1,), "float32")
        T.clear(partial)
        for chunk in T.serial(T.ceildiv(width, reduction_lanes)):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                combined = block * output_tile + slot
                reduction = chunk * reduction_lanes + lane
                if combined < total and reduction < width:
                    if combined < sizes[0]:
                        coefficient[0] = T.cast(
                            _direct_weight(weight0, weight_specs[0], combined, reduction),
                            "float32",
                        )
                    elif combined < sizes[0] + sizes[1]:
                        coefficient[0] = T.cast(
                            _direct_weight(
                                weight1,
                                weight_specs[1],
                                combined - sizes[0],
                                reduction,
                            ),
                            "float32",
                        )
                    elif combined < sizes[0] + sizes[1] + sizes[2]:
                        coefficient[0] = T.cast(
                            _direct_weight(
                                weight2,
                                weight_specs[2],
                                combined - sizes[0] - sizes[1],
                                reduction,
                            ),
                            "float32",
                        )
                    else:
                        coefficient[0] = T.cast(
                            _direct_weight(
                                weight3,
                                weight_specs[3],
                                combined - sizes[0] - sizes[1] - sizes[2],
                                reduction,
                            ),
                            "float32",
                        )
                    partial[slot, lane] += T.cast(
                        source[row, reduction], "float32"
                    ) * coefficient[0]
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared[slot, lane] = partial[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared[slot, lane] += shared[
                        slot, lane + (reduction_lanes >> (step + 1))
                    ]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            combined = block * output_tile + slot
            if combined < sizes[0]:
                output0[row, combined] = T.cast(
                    shared[slot, 0], output_specs[0].dtype.value
                )
            elif combined < sizes[0] + sizes[1]:
                output1[row, combined - sizes[0]] = T.cast(
                    shared[slot, 0], output_specs[1].dtype.value
                )
            elif combined < sizes[0] + sizes[1] + sizes[2]:
                output2[row, combined - sizes[0] - sizes[1]] = T.cast(
                    shared[slot, 0], output_specs[2].dtype.value
                )
            elif combined < total:
                output3[row, combined - sizes[0] - sizes[1] - sizes[2]] = T.cast(
                    shared[slot, 0], output_specs[3].dtype.value
                )


class _ParallelDirectEmitter:
    def __init__(
        self,
        weight_specs: tuple[TensorSpec, ...],
        output_specs: tuple[TensorSpec, ...],
        rows: int,
        width: int,
        output_tile: int,
        reduction_lanes: int,
    ):
        self.weight_specs = weight_specs
        self.output_specs = output_specs
        self.rows = rows
        self.width = width
        self.output_tile = output_tile
        self.reduction_lanes = reduction_lanes

    def __call__(self, operands: tuple[Any, ...]) -> None:
        count = len(self.weight_specs)
        source = operands[0]
        weights = _four_slots(operands[1 : count + 1])
        outputs = _four_slots(operands[count + 1 : 2 * count + 1])
        weight_specs = _four_slots(self.weight_specs)
        output_specs = _four_slots(self.output_specs)
        sizes = tuple(cast(int, spec.shape[0]) for spec in weight_specs)
        _parallel_direct_linear(
            source,
            *weights,
            *outputs,
            weight_specs,
            output_specs,
            self.rows,
            self.width,
            sizes,
            count,
            self.output_tile,
            self.reduction_lanes,
        )


def _four_slots(values: tuple[Any, ...]) -> tuple[Any, Any, Any, Any]:
    if len(values) == 3:
        return values[0], values[1], values[2], values[2]
    if len(values) == 4:
        return values[0], values[1], values[2], values[3]
    raise ValueError("parallel projection regions require three or four values")


class ParallelDirectMatrixRule:
    """Fuse adjacent small-row projections that read the same activation."""

    name = "parallel-direct-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
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
        if rows > 4:
            return ()
        weight_specs = tuple(graph.values[node.inputs[1]].spec for node in nodes)
        output_specs = tuple(graph.values[node.outputs[0]].spec for node in nodes)
        if any(
            not spec.static or spec.rank != 2 or spec.shape[1] != width
            for spec in weight_specs
        ):
            return ()
        lanes = min(32, context.capabilities.subgroup_width)
        output_tile = min(4, context.capabilities.threads_per_group // lanes)
        if (
            lanes < 2
            or lanes & (lanes - 1)
            or output_tile < 1
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        inputs = (source, *(node.inputs[1] for node in nodes))
        outputs = tuple(node.outputs[0] for node in nodes)
        moved = sum(spec.storage_nbytes for spec in weight_specs)
        return (
            Candidate(
                f"linear.parallel-direct@{root}:{nodes[-1].id}",
                frozenset(node.id for node in nodes),
                inputs,
                outputs,
                _ParallelDirectEmitter(
                    weight_specs,
                    output_specs,
                    rows,
                    width,
                    output_tile,
                    lanes,
                ),
                2e-7 + moved / 5e12,
                priority=50,
            ),
        )


class DirectDenseMatrixRule:
    """Exact-row dense projection when a matrix tile would be mostly masked."""

    name = "direct-dense-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "linear":
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        if (
            not left.static
            or not right.static
            or right.representation not in (None, Dense(right.dtype))
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        m, k = cast(tuple[int, int], left.shape)
        if m > 4:
            return ()
        lanes = min(32, context.capabilities.subgroup_width)
        output_tile = min(4, context.capabilities.threads_per_group // lanes)
        if (
            lanes < 2
            or lanes & (lanes - 1)
            or output_tile < 1
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        output = graph.values[node.outputs[0]].spec
        n = cast(int, right.shape[0])
        return (
            Candidate(
                f"linear.direct-dense@{root}",
                frozenset({root}),
                tuple(node.inputs),
                tuple(node.outputs),
                _DirectEncodedEmitter(
                    right,
                    m,
                    n,
                    k,
                    output_tile,
                    lanes,
                    output.dtype.value,
                    len(node.inputs) == 3,
                    encoded=False,
                ),
                3e-7 + right.storage_nbytes / 800e9,
                priority=30,
            ),
        )


class DirectEncodedMatrixRule:
    """Exact-row packed projection for decode and other very small batches."""

    name = "direct-encoded-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "linear":
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        if (
            not left.static
            or not right.static
            or right.representation is None
            or isinstance(right.representation, Dense)
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        m, k = cast(tuple[int, int], left.shape)
        if m > 4:
            return ()
        lanes = min(32, context.capabilities.subgroup_width)
        output_tile = min(4, context.capabilities.threads_per_group // lanes)
        if (
            lanes < 2
            or lanes & (lanes - 1)
            or output_tile < 1
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        output = graph.values[node.outputs[0]].spec
        n = cast(int, right.shape[0])
        packed_bytes = right.storage_nbytes
        return (
            Candidate(
                f"linear.direct-encoded@{root}",
                frozenset({root}),
                tuple(node.inputs),
                tuple(node.outputs),
                _DirectEncodedEmitter(
                    right,
                    m,
                    n,
                    k,
                    output_tile,
                    lanes,
                    output.dtype.value,
                    len(node.inputs) == 3,
                ),
                3e-7 + packed_bytes / 4e12,
                priority=30,
            ),
        )


class DenseMatrixRule:
    name = "fragment-tiled-dense-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation not in {"matmul", "linear"}:
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        if (
            not left.static
            or not right.static
            or right.representation not in (None, Dense(right.dtype))
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == left.dtype
            ),
            None,
        )
        if instruction is None:
            return ()
        output = graph.values[node.outputs[0]].spec
        n = cast(int, output.shape[-1])
        k = cast(int, left.shape[-1])
        m = left.elements // k
        if m < instruction.m or n < instruction.n or k < instruction.k:
            return ()
        if m <= instruction.m:
            bm, bn, bk = instruction.m, instruction.n * 8, instruction.k * 2
        else:
            bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
        shared = (bm * bk + bn * bk) * left.dtype.itemsize
        if shared > context.capabilities.shared_memory_bytes:
            bm, bn, bk = instruction.m * 2, instruction.n * 2, instruction.k
        threads = min(
            context.capabilities.threads_per_group, context.capabilities.subgroup_width * 4
        )
        inputs = tuple(node.inputs)
        outputs = tuple(node.outputs)
        geometry = (m, n, k, bm, bn, bk, threads)
        key = TuningKey(
            node.operation,
            geometry,
            ("dense",),
            context.precision,
            context.capabilities.fingerprint,
            context.compiler_identity,
        )
        return (
            Candidate(
                f"{node.operation}.fragment-tiled@{root}",
                frozenset({root}),
                inputs,
                outputs,
                _TiledMatrixEmitter(
                    node.operation,
                    m,
                    n,
                    k,
                    left.dtype.value,
                    output.dtype.value,
                    threads,
                    (bm, bn, bk),
                    len(inputs) == 3,
                ),
                5e-7 + (2 * m * n * k) / 5e12,
                tuning_key=key,
                priority=10,
            ),
        )


class EncodedMatrixRule:
    """Matrix-instruction projection that reads canonical packed weights directly."""

    name = "fragment-tiled-encoded-matrix"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "linear":
            return ()
        specs = tuple(graph.values[value].spec for value in node.inputs)
        left, right = specs[:2]
        if (
            not left.static
            or not right.static
            or right.representation is None
            or isinstance(right.representation, Dense)
            or left.rank != 2
            or right.rank != 2
        ):
            return ()
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == left.dtype
            ),
            None,
        )
        if instruction is None:
            return ()
        output = graph.values[node.outputs[0]].spec
        m, k = cast(tuple[int, int], left.shape)
        n = cast(int, right.shape[0])
        if m < instruction.m or n < instruction.n or k < instruction.k:
            return ()
        if m <= instruction.m:
            # A full 32-row tile makes decode execute mostly masked matrix work.
            # Keep the minimum instruction-height while widening the output tile
            # enough to retain useful weight-load and matrix-instruction density.
            bm, bn, bk = instruction.m, instruction.n * 8, instruction.k * 2
        else:
            bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
        shared = (bm * bk + bn * bk) * left.dtype.itemsize
        if shared > context.capabilities.shared_memory_bytes:
            bm, bn, bk = instruction.m * 2, instruction.n * 2, instruction.k
        threads = min(
            context.capabilities.threads_per_group, context.capabilities.subgroup_width * 4
        )
        operations = 2 * m * n * k
        return (
            Candidate(
                f"linear.fragment-encoded@{root}",
                frozenset({root}),
                tuple(node.inputs),
                tuple(node.outputs),
                _EncodedMatrixEmitter(
                    right,
                    m,
                    n,
                    k,
                    left.dtype.value,
                    output.dtype.value,
                    threads,
                    (bm, bn, bk),
                    len(node.inputs) == 3,
                ),
                5e-7 + operations / 4e12,
                priority=20,
            ),
        )
