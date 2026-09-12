"""Direct feed-forward schedules for decode and small routed batches."""

from __future__ import annotations

import math
from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..representations import Dense
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .quantization import represented_load


def _weight(storage, spec: TensorSpec, expert, output, reduction):
    experts, rows, columns = cast(tuple[int, int, int], spec.shape)
    del experts
    if spec.representation is None or isinstance(spec.representation, Dense):
        return T.cast(storage[expert, output, reduction], "float32")
    return represented_load(storage, spec, (expert * rows + output) * columns + reduction)


def _matrix_weight(storage, spec: TensorSpec, output, reduction):
    _, columns = cast(tuple[int, int], spec.shape)
    if spec.representation is None or isinstance(spec.representation, Dense):
        return T.cast(storage[output, reduction], "float32")
    return represented_load(storage, spec, output * columns + reduction)


@T.macro
def _dense_activate(
    hidden,
    gate,
    up,
    activation,
    gate_spec,
    up_spec,
    rows,
    width,
    intermediate,
    output_tile,
    reduction_lanes,
    output_dtype,
):
    with T.Kernel(
        T.ceildiv(intermediate, output_tile),
        rows,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial_gate = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        partial_up = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared_gate = T.alloc_shared((output_tile, reduction_lanes), "float32")
        shared_up = T.alloc_shared((output_tile, reduction_lanes), "float32")
        T.clear(partial_gate)
        T.clear(partial_up)
        for chunk in T.serial(T.ceildiv(width, reduction_lanes)):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                channel = block * output_tile + slot
                reduction = chunk * reduction_lanes + lane
                if channel < intermediate and reduction < width:
                    value = T.cast(hidden[row, reduction], "float32")
                    partial_gate[slot, lane] += value * _matrix_weight(
                        gate, gate_spec, channel, reduction
                    )
                    partial_up[slot, lane] += value * _matrix_weight(
                        up, up_spec, channel, reduction
                    )
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared_gate[slot, lane] = partial_gate[slot, lane]
            shared_up[slot, lane] = partial_up[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared_gate[slot, lane] += shared_gate[
                        slot, lane + (reduction_lanes >> (step + 1))
                    ]
                    shared_up[slot, lane] += shared_up[slot, lane + (reduction_lanes >> (step + 1))]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            channel = block * output_tile + slot
            if channel < intermediate:
                gate_value = shared_gate[slot, 0]
                activation[row, channel] = T.cast(
                    gate_value * T.sigmoid(gate_value) * shared_up[slot, 0],
                    output_dtype,
                )


@T.macro
def _dense_down(
    activation,
    down,
    output,
    down_spec,
    rows,
    width,
    intermediate,
    output_tile,
    reduction_lanes,
    output_dtype,
):
    with T.Kernel(
        T.ceildiv(width, output_tile),
        rows,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared = T.alloc_shared((output_tile, reduction_lanes), "float32")
        T.clear(partial)
        for chunk in T.serial(T.ceildiv(intermediate, reduction_lanes)):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                output_channel = block * output_tile + slot
                reduction = chunk * reduction_lanes + lane
                if output_channel < width and reduction < intermediate:
                    partial[slot, lane] += T.cast(
                        activation[row, reduction], "float32"
                    ) * _matrix_weight(down, down_spec, output_channel, reduction)
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared[slot, lane] = partial[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared[slot, lane] += shared[slot, lane + (reduction_lanes >> (step + 1))]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            output_channel = block * output_tile + slot
            if output_channel < width:
                output[row, output_channel] = T.cast(shared[slot, 0], output_dtype)


@T.macro
def _matrix_dense_activate(
    hidden,
    gate,
    up,
    activation,
    gate_spec,
    up_spec,
    rows,
    width,
    intermediate,
    bm,
    bn,
    bk,
    threads,
    dtype,
):
    with T.Kernel(T.ceildiv(intermediate, bn), T.ceildiv(rows, bm), threads=threads) as (
        bx,
        by,
    ):
        x = T.alloc_shared((bm, bk), dtype)
        w = T.alloc_shared((bn, bk), dtype)
        gate_accum = T.alloc_fragment((bm, bn), "float32")
        up_accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(gate_accum)
        T.clear(up_accum)
        for block in T.serial(T.ceildiv(width, bk)):
            for i, k in T.Parallel(bm, bk):
                row = by * bm + i
                reduction = block * bk + k
                x[i, k] = T.if_then_else(
                    row < rows and reduction < width,
                    hidden[row, reduction],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                channel = bx * bn + j
                reduction = block * bk + k
                if channel < intermediate and reduction < width:
                    w[j, k] = T.cast(_matrix_weight(gate, gate_spec, channel, reduction), dtype)
                else:
                    w[j, k] = 0
            T.sync_threads()
            T.gemm(x, w, gate_accum, transpose_B=True)
            T.sync_threads()
            for j, k in T.Parallel(bn, bk):
                channel = bx * bn + j
                reduction = block * bk + k
                if channel < intermediate and reduction < width:
                    w[j, k] = T.cast(_matrix_weight(up, up_spec, channel, reduction), dtype)
                else:
                    w[j, k] = 0
            T.sync_threads()
            T.gemm(x, w, up_accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            row = by * bm + i
            channel = bx * bn + j
            if row < rows and channel < intermediate:
                value = gate_accum[i, j]
                activation[row, channel] = T.cast(value * T.sigmoid(value) * up_accum[i, j], dtype)


@T.macro
def _matrix_dense_down(
    activation,
    down,
    output,
    down_spec,
    rows,
    width,
    intermediate,
    bm,
    bn,
    bk,
    threads,
    dtype,
    output_dtype,
):
    with T.Kernel(T.ceildiv(width, bn), T.ceildiv(rows, bm), threads=threads) as (bx, by):
        x = T.alloc_shared((bm, bk), dtype)
        w = T.alloc_shared((bn, bk), dtype)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        for block in T.serial(T.ceildiv(intermediate, bk)):
            for i, k in T.Parallel(bm, bk):
                row = by * bm + i
                reduction = block * bk + k
                x[i, k] = T.if_then_else(
                    row < rows and reduction < intermediate,
                    activation[row, reduction],
                    0,
                )
            for j, k in T.Parallel(bn, bk):
                channel = bx * bn + j
                reduction = block * bk + k
                if channel < width and reduction < intermediate:
                    w[j, k] = T.cast(_matrix_weight(down, down_spec, channel, reduction), dtype)
                else:
                    w[j, k] = 0
            T.sync_threads()
            T.gemm(x, w, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            row = by * bm + i
            channel = bx * bn + j
            if row < rows and channel < width:
                output[row, channel] = T.cast(accum[i, j], output_dtype)


@T.macro
def _activate(
    hidden,
    routes,
    gate,
    up,
    activation,
    gate_spec,
    up_spec,
    rows,
    selected,
    width,
    intermediate,
    output_tile,
    reduction_lanes,
    output_dtype,
):
    with T.Kernel(
        T.ceildiv(selected * intermediate, output_tile),
        rows,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial_gate = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        partial_up = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared_gate = T.alloc_shared((output_tile, reduction_lanes), "float32")
        shared_up = T.alloc_shared((output_tile, reduction_lanes), "float32")
        T.clear(partial_gate)
        T.clear(partial_up)
        for chunk in T.serial(T.ceildiv(width, reduction_lanes)):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                combined = block * output_tile + slot
                rank = combined // intermediate
                channel = combined % intermediate
                reduction = chunk * reduction_lanes + lane
                if rank < selected and reduction < width:
                    expert = routes[row, rank]
                    if 0 <= expert and expert < gate_spec.shape[0]:
                        value = T.cast(hidden[row, reduction], "float32")
                        partial_gate[slot, lane] += value * _weight(
                            gate, gate_spec, expert, channel, reduction
                        )
                        partial_up[slot, lane] += value * _weight(
                            up, up_spec, expert, channel, reduction
                        )
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared_gate[slot, lane] = partial_gate[slot, lane]
            shared_up[slot, lane] = partial_up[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared_gate[slot, lane] += shared_gate[
                        slot, lane + (reduction_lanes >> (step + 1))
                    ]
                    shared_up[slot, lane] += shared_up[slot, lane + (reduction_lanes >> (step + 1))]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            combined = block * output_tile + slot
            rank = combined // intermediate
            channel = combined % intermediate
            if rank < selected:
                gate_value = shared_gate[slot, 0]
                activation[row, rank, channel] = T.cast(
                    gate_value * T.sigmoid(gate_value) * shared_up[slot, 0],
                    output_dtype,
                )


@T.macro
def _down_reduce(
    activation,
    routes,
    scores,
    down,
    output,
    down_spec,
    rows,
    selected,
    width,
    intermediate,
    output_tile,
    reduction_lanes,
    output_dtype,
):
    with T.Kernel(
        T.ceildiv(width, output_tile),
        rows,
        threads=output_tile * reduction_lanes,
    ) as (block, row):
        partial = T.alloc_fragment((output_tile, reduction_lanes), "float32")
        shared = T.alloc_shared((output_tile, reduction_lanes), "float32")
        T.clear(partial)
        for rank in T.serial(selected):
            expert = routes[row, rank]
            score = T.cast(scores[row, rank], "float32")
            for chunk in T.serial(T.ceildiv(intermediate, reduction_lanes)):
                for slot, lane in T.Parallel(output_tile, reduction_lanes):
                    output_channel = block * output_tile + slot
                    reduction = chunk * reduction_lanes + lane
                    if output_channel < width and reduction < intermediate:
                        if 0 <= expert and expert < down_spec.shape[0]:
                            partial[slot, lane] += (
                                T.cast(activation[row, rank, reduction], "float32")
                                * _weight(
                                    down,
                                    down_spec,
                                    expert,
                                    output_channel,
                                    reduction,
                                )
                                * score
                            )
        for slot, lane in T.Parallel(output_tile, reduction_lanes):
            shared[slot, lane] = partial[slot, lane]
        T.sync_threads()
        for step in T.unroll(int(math.log2(reduction_lanes))):
            for slot, lane in T.Parallel(output_tile, reduction_lanes):
                if lane < (reduction_lanes >> (step + 1)):
                    shared[slot, lane] += shared[slot, lane + (reduction_lanes >> (step + 1))]
            T.sync_threads()
        for slot in T.Parallel(output_tile):
            output_channel = block * output_tile + slot
            if output_channel < width:
                output[row, output_channel] = T.cast(shared[slot, 0], output_dtype)


class _DirectExpertsEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        output_tile: int,
        reduction_lanes: int,
    ):
        self.specs = specs
        self.output_tile = output_tile
        self.reduction_lanes = reduction_lanes

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, routes, scores, gate, up, down, output, activation = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        intermediate = cast(int, self.specs[3].shape[1])
        _activate(
            hidden,
            routes,
            gate,
            up,
            activation,
            self.specs[3],
            self.specs[4],
            rows,
            selected,
            width,
            intermediate,
            self.output_tile,
            self.reduction_lanes,
            self.specs[0].dtype.value,
        )
        _down_reduce(
            activation,
            routes,
            scores,
            down,
            output,
            self.specs[5],
            rows,
            selected,
            width,
            intermediate,
            self.output_tile,
            self.reduction_lanes,
            self.specs[6].dtype.value,
        )


class _DenseSwiGLUEmitter:
    def __init__(
        self,
        specs: tuple[TensorSpec, ...],
        output_tile: int,
        reduction_lanes: int,
    ):
        self.specs = specs
        self.output_tile = output_tile
        self.reduction_lanes = reduction_lanes

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, gate, up, down, output, activation = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        intermediate = cast(int, self.specs[1].shape[0])
        _dense_activate(
            hidden,
            gate,
            up,
            activation,
            self.specs[1],
            self.specs[2],
            rows,
            width,
            intermediate,
            self.output_tile,
            self.reduction_lanes,
            self.specs[0].dtype.value,
        )
        _dense_down(
            activation,
            down,
            output,
            self.specs[3],
            rows,
            width,
            intermediate,
            self.output_tile,
            self.reduction_lanes,
            self.specs[4].dtype.value,
        )


class _MatrixDenseSwiGLUEmitter:
    def __init__(self, specs: tuple[TensorSpec, ...], tile):
        self.specs, self.tile = specs, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, gate, up, down, output, activation = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        intermediate = cast(int, self.specs[1].shape[0])
        bm, bn, bk, threads = self.tile
        _matrix_dense_activate(
            hidden,
            gate,
            up,
            activation,
            self.specs[1],
            self.specs[2],
            rows,
            width,
            intermediate,
            bm,
            bn,
            bk,
            threads,
            self.specs[0].dtype.value,
        )
        _matrix_dense_down(
            activation,
            down,
            output,
            self.specs[3],
            rows,
            width,
            intermediate,
            bm,
            bn,
            bk,
            threads,
            self.specs[0].dtype.value,
            self.specs[4].dtype.value,
        )


def _direct_geometry(context: LoweringContext) -> tuple[int, int] | None:
    lanes = min(32, context.capabilities.subgroup_width)
    output_tile = min(8, context.capabilities.threads_per_group // lanes)
    if (
        lanes < 2
        or lanes & (lanes - 1)
        or output_tile < 1
        or "shared" not in context.capabilities.memory_scopes
        or 4 * output_tile * lanes * 2 > context.capabilities.shared_memory_bytes
    ):
        return None
    return output_tile, lanes


def _dense_swiglu_region(graph: Graph, root: int):
    """Return a complete gate/up/activation/down region independent of schedule."""
    if not 0 <= root < len(graph.nodes):
        return None
    gate_node = graph.nodes[root]
    if gate_node.operation != "linear" or len(gate_node.inputs) != 2:
        return None
    gate_value = gate_node.outputs[0]
    gate_users = graph.users[gate_value]
    if len(gate_users) != 1:
        return None
    silu_node = graph.nodes[gate_users[0]]
    if silu_node.operation != "silu":
        return None
    silu_value = silu_node.outputs[0]
    multiply_users = graph.users[silu_value]
    if len(multiply_users) != 1:
        return None
    multiply_node = graph.nodes[multiply_users[0]]
    if multiply_node.operation != "multiply":
        return None
    up_value = next((value for value in multiply_node.inputs if value != silu_value), None)
    if up_value is None:
        return None
    up_producer = graph.values[up_value].producer
    if up_producer is None:
        return None
    up_node = graph.nodes[up_producer]
    if (
        up_node.operation != "linear"
        or len(up_node.inputs) != 2
        or up_node.inputs[0] != gate_node.inputs[0]
    ):
        return None
    multiply_value = multiply_node.outputs[0]
    down_users = graph.users[multiply_value]
    if len(down_users) != 1:
        return None
    down_node = graph.nodes[down_users[0]]
    if down_node.operation != "linear" or len(down_node.inputs) != 2:
        return None
    nodes = frozenset({gate_node.id, silu_node.id, up_node.id, multiply_node.id, down_node.id})
    inputs = (
        gate_node.inputs[0],
        gate_node.inputs[1],
        up_node.inputs[1],
        down_node.inputs[1],
    )
    outputs = tuple(down_node.outputs)
    specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
    if any(not spec.static for spec in specs):
        return None
    hidden, gate, up, down, output = specs
    if (
        hidden.rank != 2
        or gate.rank != 2
        or up.shape != gate.shape
        or down.shape != (hidden.shape[1], gate.shape[0])
        or output.shape != hidden.shape
    ):
        return None
    return nodes, inputs, outputs, specs


class DenseSwiGLURule:
    name = "direct-dense-swiglu"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        region = _dense_swiglu_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs = region
        hidden, gate, _, _, _ = specs
        rows = cast(int, hidden.shape[0])
        if context.mode != "decode" and rows > 4:
            return ()
        geometry = _direct_geometry(context)
        if geometry is None:
            return ()
        activation = TensorSpec((rows, cast(int, gate.shape[0])), hidden.dtype)
        moved = sum(spec.storage_nbytes for spec in specs) + activation.storage_nbytes * 2
        return (
            Candidate(
                f"dense_swiglu.direct@{root}:{max(nodes)}",
                nodes,
                inputs,
                outputs,
                _DenseSwiGLUEmitter(specs, *geometry),
                # Decode is weight-bandwidth bound. The fused region reads the
                # same encoded weights as three standalone projections while
                # avoiding three extra launches and two materialized pointwise
                # passes, so compare it on the target's effective packed-weight
                # bandwidth instead of the scalar fallback bandwidth.
                4e-7 + moved / 4e12,
                workspace=(activation,),
                kernel_count=2,
                priority=25,
            ),
        )


class DirectExpertsRule:
    name = "direct-selected-experts"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "routed_experts" or node.attributes["activation"] != "silu":
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        hidden, routes, _, gate, _, _, _ = specs
        rows, width = cast(tuple[int, int], hidden.shape)
        selected = cast(int, routes.shape[1])
        experts = cast(int, gate.shape[0])
        intermediate = cast(int, gate.shape[1])
        geometry = _direct_geometry(context)
        if geometry is None or (context.mode != "decode" and rows * selected >= 2 * experts):
            return ()
        output_tile, lanes = geometry
        activation = TensorSpec((rows, selected, intermediate), hidden.dtype)
        moved = sum(spec.storage_nbytes for spec in specs) + activation.storage_nbytes * 2
        return (
            Candidate(
                f"routed_experts.direct@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _DirectExpertsEmitter(specs, output_tile, lanes),
                4e-7 + moved / 100e9,
                workspace=(activation,),
                kernel_count=2,
                priority=20,
            ),
        )


class MatrixDenseSwiGLURule:
    name = "matrix-dense-swiglu"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        if context.mode != "prefill":
            return ()
        region = _dense_swiglu_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs = region
        hidden, gate, up, down, output = specs
        rows = cast(int, hidden.shape[0])
        if rows <= 4:
            return ()
        instruction = next(
            (
                item
                for item in context.capabilities.matrix_instructions
                if item.input_dtype == hidden.dtype
            ),
            None,
        )
        if instruction is None:
            return ()
        bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
        threads = min(
            context.capabilities.threads_per_group,
            context.capabilities.subgroup_width * 4,
        )
        if (bm * bk + bn * bk) * hidden.dtype.itemsize > context.capabilities.shared_memory_bytes:
            return ()
        activation = TensorSpec((rows, cast(int, gate.shape[0])), hidden.dtype)
        operations = 6 * rows * cast(int, gate.shape[0]) * cast(int, hidden.shape[1])
        return (
            Candidate(
                f"dense_swiglu.matrix@{root}:{max(nodes)}",
                nodes,
                inputs,
                outputs,
                _MatrixDenseSwiGLUEmitter(specs, (bm, bn, bk, threads)),
                5e-7 + operations / 5e12,
                workspace=(activation,),
                kernel_count=2,
                priority=40,
            ),
        )
