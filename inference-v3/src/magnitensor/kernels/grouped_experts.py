"""Grouped expert prefill pipeline selected from explicit target capabilities."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec
from .packed import decode_packet, packet_format


def _aligned_capacity(rows: int, selected: int, experts: int, tile: int) -> int:
    routes = rows * selected
    return ((routes + experts * (tile - 1) + tile - 1) // tile) * tile


@T.macro
def _group_routes(
    routes, order, inverse, block_experts, counts, cursors,
    rows, selected, experts, capacity, tile,
):
    """Build the complete expert permutation in one workgroup launch."""
    count = rows * selected
    blocks = capacity // tile
    with T.Kernel(1, threads=256):
        lane = T.get_thread_binding(0)
        if lane < experts:
            counts[lane] = 0
        for chunk in T.serial(T.ceildiv(capacity, 256)):
            position = chunk * 256 + lane
            if position < capacity:
                order[position] = -1
        for chunk in T.serial(T.ceildiv(count, 256)):
            position = chunk * 256 + lane
            if position < count:
                inverse[position] = -1
        for chunk in T.serial(T.ceildiv(blocks, 256)):
            block = chunk * 256 + lane
            if block < blocks:
                block_experts[block] = -1
        T.sync_threads()
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    T.atomic_add(counts[expert], 1)
        T.sync_threads()
        if lane == 0:
            cursor = T.alloc_local((1,), "int32")
            cursor[0] = 0
            for expert in T.serial(experts):
                cursors[expert] = cursor[0]
                for block in T.serial(T.ceildiv(counts[expert], tile)):
                    block_experts[cursor[0] // tile + block] = expert
                cursor[0] += T.ceildiv(counts[expert], tile) * tile
        T.sync_threads()
        for chunk in T.serial(T.ceildiv(count, 256)):
            route = chunk * 256 + lane
            if route < count:
                expert = routes[route // selected, route % selected]
                if 0 <= expert and expert < experts:
                    position = T.atomic_add(cursors[expert], 1, return_prev=True)
                    order[position] = route
                    inverse[route] = position


@T.macro
def _grouped_gated_projection(
    source,
    order,
    block_experts,
    gate,
    up,
    output,
    gate_spec,
    up_spec,
    blocks,
    source_width,
    output_width,
    selected,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
):
    packet = packet_format(gate_spec)
    assert packet is not None and packet_format(up_spec) == packet
    with T.Kernel(T.ceildiv(output_width, bn), blocks, threads=threads) as (bx, by):
        x = T.alloc_shared((bm, bk), gate_spec.dtype.value)
        gate_tile = T.alloc_shared((bn, bk), gate_spec.dtype.value)
        up_tile = T.alloc_shared((bn, bk), gate_spec.dtype.value)
        gate_accum = T.alloc_fragment((bm, bn), "float32")
        up_accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(gate_accum)
        T.clear(up_accum)
        expert = block_experts[by]
        for reduction_block in T.serial(T.ceildiv(source_width, bk)):
            for i, reduction_lane in T.Parallel(bm, bk):
                route = order[by * bm + i]
                reduction = reduction_block * bk + reduction_lane
                x[i, reduction_lane] = T.if_then_else(
                    route >= 0 and reduction < source_width,
                    source[
                        T.if_then_else(source_grouped, by * bm + i, route // selected),
                        reduction,
                    ],
                    0,
                )
            packets = bn * bk // packet.packet
            for iteration in T.serial(T.ceildiv(packets, threads)):
                linear = iteration * threads + T.get_thread_binding()
                j = linear // (bk // packet.packet)
                packet_column = linear % (bk // packet.packet) * packet.packet
                channel = bx * bn + j
                reduction = reduction_block * bk + packet_column
                if j < bn:
                    if expert >= 0 and channel < output_width and reduction < source_width:
                        flattened_row = expert * output_width + channel
                        decode_packet(
                            gate_tile, j, packet_column, gate, gate_spec,
                            flattened_row, reduction,
                        )
                        decode_packet(
                            up_tile, j, packet_column, up, up_spec,
                            flattened_row, reduction,
                        )
                    else:
                        for item in T.unroll(packet.packet, explicit=True):
                            gate_tile[j, packet_column + item] = 0
                            up_tile[j, packet_column + item] = 0
            T.sync_threads()
            T.gemm(x, gate_tile, gate_accum, transpose_B=True)
            T.gemm(x, up_tile, up_accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            route = order[by * bm + i]
            channel = bx * bn + j
            if route >= 0 and channel < output_width:
                gate_value = T.cast(
                    T.cast(gate_accum[i, j], gate_spec.dtype.value), "float32"
                )
                output[by * bm + i, channel] = T.cast(
                    gate_value
                    * T.sigmoid(gate_value)
                    * T.cast(T.cast(up_accum[i, j], up_spec.dtype.value), "float32"),
                    gate_spec.dtype.value,
                )


@T.macro
def _grouped_projection(
    source,
    order,
    block_experts,
    weight,
    output,
    weight_spec,
    blocks,
    source_width,
    output_width,
    selected,
    source_grouped,
    bm,
    bn,
    bk,
    threads,
):
    packet = packet_format(weight_spec)
    assert packet is not None
    with T.Kernel(T.ceildiv(output_width, bn), blocks, threads=threads) as (bx, by):
        x = T.alloc_shared((bm, bk), weight_spec.dtype.value)
        w = T.alloc_shared((bn, bk), weight_spec.dtype.value)
        accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(accum)
        expert = block_experts[by]
        for reduction_block in T.serial(T.ceildiv(source_width, bk)):
            for i, reduction_lane in T.Parallel(bm, bk):
                route = order[by * bm + i]
                reduction = reduction_block * bk + reduction_lane
                x[i, reduction_lane] = T.if_then_else(
                    route >= 0 and reduction < source_width,
                    source[
                        T.if_then_else(source_grouped, by * bm + i, route // selected),
                        reduction,
                    ],
                    0,
                )
            packets = bn * bk // packet.packet
            for iteration in T.serial(T.ceildiv(packets, threads)):
                linear = iteration * threads + T.get_thread_binding()
                j = linear // (bk // packet.packet)
                packet_column = linear % (bk // packet.packet) * packet.packet
                channel = bx * bn + j
                reduction = reduction_block * bk + packet_column
                if j < bn:
                    if expert >= 0 and channel < output_width and reduction < source_width:
                        decode_packet(
                            w,
                            j,
                            packet_column,
                            weight,
                            weight_spec,
                            expert * output_width + channel,
                            reduction,
                        )
                    else:
                        for item in T.unroll(packet.packet, explicit=True):
                            w[j, packet_column + item] = 0
            T.sync_threads()
            T.gemm(x, w, accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            route = order[by * bm + i]
            channel = bx * bn + j
            if route >= 0 and channel < output_width:
                output[by * bm + i, channel] = T.cast(
                    accum[i, j], weight_spec.dtype.value
                )


@T.macro
def _unpermute(projected, inverse, scores, output, rows, selected, width, dtype, threads):
    with T.Kernel(T.ceildiv(rows * width, threads), threads=threads) as block:
        for lane in T.Parallel(threads):
            flat = block * threads + lane
            if flat < rows * width:
                row, column = flat // width, flat % width
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for rank in T.serial(selected):
                    source = inverse[row * selected + rank]
                    if source >= 0:
                        total[0] += T.cast(projected[source, column], "float32") * T.cast(
                            scores[row, rank], "float32"
                        )
                output[row, column] = T.cast(total[0], dtype)


class _GroupedExpertsEmitter:
    def __init__(self, specs, capacity, blocks, tile):
        self.specs = specs
        self.capacity, self.blocks, self.tile = capacity, blocks, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, routes, scores, gate, up, down, output = operands[:7]
        (
            order,
            inverse,
            block_experts,
            counts,
            cursors,
            activation,
            projected,
        ) = operands[7:]
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        experts, intermediate, _ = cast(tuple[int, int, int], self.specs[3].shape)
        instruction = self.tile
        bm, bn, bk, threads = instruction
        _group_routes(
            routes,
            order,
            inverse,
            block_experts,
            counts,
            cursors,
            rows,
            selected,
            experts,
            self.capacity,
            bm,
        )
        _grouped_gated_projection(
            hidden,
            order,
            block_experts,
            gate,
            up,
            activation,
            self.specs[3],
            self.specs[4],
            self.blocks,
            width,
            intermediate,
            selected,
            False,
            bm,
            bn,
            bk,
            threads,
        )
        _grouped_projection(
            activation,
            order,
            block_experts,
            down,
            projected,
            self.specs[5],
            self.blocks,
            intermediate,
            width,
            selected,
            True,
            bm,
            bn,
            bk,
            threads,
        )
        _unpermute(
            projected,
            inverse,
            scores,
            output,
            rows,
            selected,
            width,
            self.specs[6].dtype.value,
            threads,
        )


class GroupedExpertsRule:
    name = "grouped-prefill-experts"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if (
            node.operation != "routed_experts"
            or node.attributes["activation"] != "silu"
            or context.mode == "decode"
            or DType.I32 not in context.capabilities.atomics
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        if any(not spec.static for spec in specs):
            return ()
        hidden, routes, scores, gate, up, down, output = specs
        rows, width = cast(tuple[int, int], hidden.shape)
        selected = cast(int, routes.shape[1])
        experts, intermediate, input_width = cast(tuple[int, int, int], gate.shape)
        gate_packet, up_packet, down_packet = (
            packet_format(spec) for spec in (gate, up, down)
        )
        instruction = next(
            (
                value
                for value in context.capabilities.matrix_instructions
                if value.input_dtype == hidden.dtype
            ),
            None,
        )
        if (
            instruction is None
            or rows * selected < experts
            or input_width != width
            or up.shape != gate.shape
            or down.shape != (experts, width, intermediate)
            or scores.shape != routes.shape
            or output.shape != hidden.shape
            or experts > 256
            or gate_packet is None
            or up_packet is None
            or down_packet is None
            or gate_packet != up_packet
            or width % gate_packet.tile
            or intermediate % down_packet.tile
        ):
            return ()
        bm = instruction.m * 2
        bn = instruction.n * 2
        packet_width = max(
            gate_packet.packet, up_packet.packet, down_packet.packet
        )
        bk = max(instruction.k, packet_width)
        bk = ((bk + packet_width - 1) // packet_width) * packet_width
        threads = min(
            context.capabilities.threads_per_group,
            context.capabilities.subgroup_width * 4,
        )
        shared = (bm * bk + 2 * bn * bk) * hidden.dtype.itemsize
        if shared > context.capabilities.shared_memory_bytes:
            return ()
        capacity = _aligned_capacity(rows, selected, experts, bm)
        blocks = capacity // bm
        workspace = (
            TensorSpec((capacity,), DType.I32),
            TensorSpec((rows * selected,), DType.I32),
            TensorSpec((blocks,), DType.I32),
            TensorSpec((experts,), DType.I32),
            TensorSpec((experts,), DType.I32),
            TensorSpec((capacity, intermediate), hidden.dtype),
            TensorSpec((capacity, width), hidden.dtype),
        )
        if sum(value.storage_nbytes for value in workspace) > context.workspace_limit:
            return ()
        operations = 6 * rows * selected * width * intermediate
        return (
            Candidate(
                f"routed_experts.grouped@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _GroupedExpertsEmitter(
                    specs, capacity, blocks, (bm, bn, bk, threads)
                ),
                1e-7 + operations / 5e12,
                workspace=workspace,
                kernel_count=4,
                priority=40,
            ),
        )
