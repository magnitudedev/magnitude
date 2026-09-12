"""Packet-native SwiGLU and selected-expert schedules."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import TensorSpec
from .matrix import _matrix_instruction, _packed_matrix
from .packed import decode_packet, packet_dot, packet_format


def _dense_swiglu_region(graph: Graph, root: int):
    if not 0 <= root < len(graph.nodes):
        return None
    gate = graph.nodes[root]
    if gate.operation != "linear" or len(gate.inputs) != 2:
        return None
    gate_users = graph.users[gate.outputs[0]]
    if len(gate_users) != 1 or graph.nodes[gate_users[0]].operation != "silu":
        return None
    silu = graph.nodes[gate_users[0]]
    users = graph.users[silu.outputs[0]]
    if len(users) != 1 or graph.nodes[users[0]].operation != "multiply":
        return None
    multiply = graph.nodes[users[0]]
    up_value = next((value for value in multiply.inputs if value != silu.outputs[0]), None)
    if up_value is None or graph.values[up_value].producer is None:
        return None
    up = graph.nodes[graph.values[up_value].producer]
    if up.operation != "linear" or up.inputs[0] != gate.inputs[0] or len(up.inputs) != 2:
        return None
    down_users = graph.users[multiply.outputs[0]]
    if len(down_users) != 1 or graph.nodes[down_users[0]].operation != "linear":
        return None
    down = graph.nodes[down_users[0]]
    nodes = frozenset({gate.id, up.id, silu.id, multiply.id, down.id})
    if nodes != frozenset(range(min(nodes), max(nodes) + 1)):
        return None
    inputs = (gate.inputs[0], gate.inputs[1], up.inputs[1], down.inputs[1])
    outputs = down.outputs
    specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
    hidden, gate_spec, up_spec, down_spec, output = specs
    if (
        any(not spec.static for spec in specs)
        or hidden.rank != 2 or gate_spec.rank != 2 or up_spec.shape != gate_spec.shape
        or down_spec.shape != (hidden.shape[1], gate_spec.shape[0])
        or output.shape != hidden.shape
        or any(packet_format(spec) is None for spec in (gate_spec, up_spec, down_spec))
    ):
        return None
    return nodes, inputs, outputs, specs


@T.macro
def _gated_packet_vector(hidden, gate, up, activation, gate_spec, up_spec, rows, width, intermediate):
    gate_packet = packet_format(gate_spec)
    up_packet = packet_format(up_spec)
    assert gate_packet is not None and up_packet is not None and gate_packet == up_packet
    packet = gate_packet
    with T.Kernel(intermediate, rows, threads=32) as (channel, row):
        lane = T.get_thread_binding()
        values = T.alloc_local((packet.packet,), "float32")
        partial = T.alloc_local((2,), "float32")
        T.clear(partial)
        for chunk in T.serial(width // packet.tile):
            for item in T.unroll(packet.packet, explicit=True):
                values[item] = T.cast(
                    hidden[row, chunk * packet.tile + lane * packet.packet + item], "float32"
                )
            partial[0] += packet_dot(values, gate, gate_spec, channel, chunk, lane)
            partial[1] += packet_dot(values, up, up_spec, channel, chunk, lane)
        gate_value = T.cast(T.warp_reduce_sum(partial[0]), hidden.dtype)
        up_value = T.cast(T.warp_reduce_sum(partial[1]), hidden.dtype)
        if lane == 0:
            rounded_gate = T.cast(gate_value, "float32")
            activation[row, channel] = T.cast(
                rounded_gate * T.sigmoid(rounded_gate) * T.cast(up_value, "float32"), hidden.dtype
            )


@T.macro
def _gated_packet_matrix(
    hidden, gate, up, activation, gate_spec, up_spec, rows, width, intermediate,
    dtype, threads, bm, bn, bk,
):
    packet = packet_format(gate_spec)
    assert packet is not None and packet_format(up_spec) == packet
    full = rows % bm == 0 and intermediate % bn == 0 and width % bk == 0
    with T.Kernel(T.ceildiv(intermediate, bn), T.ceildiv(rows, bm), threads=threads) as (bx, by):
        x = T.alloc_shared((bm, bk), dtype)
        gate_tile = T.alloc_shared((bn, bk), dtype)
        up_tile = T.alloc_shared((bn, bk), dtype)
        gate_accum = T.alloc_fragment((bm, bn), "float32")
        up_accum = T.alloc_fragment((bm, bn), "float32")
        T.clear(gate_accum)
        T.clear(up_accum)
        for block in T.serial(T.ceildiv(width, bk)):
            for i, k in T.Parallel(bm, bk):
                if full:
                    x[i, k] = hidden[by * bm + i, block * bk + k]
                else:
                    x[i, k] = T.if_then_else(
                        by * bm + i < rows and block * bk + k < width,
                        hidden[by * bm + i, block * bk + k], 0,
                    )
            packets = bn * bk // packet.packet
            for iteration in T.serial(T.ceildiv(packets, threads)):
                linear = iteration * threads + T.get_thread_binding()
                tile_row = linear // (bk // packet.packet)
                tile_column = linear % (bk // packet.packet) * packet.packet
                channel = bx * bn + tile_row
                reduction = block * bk + tile_column
                if full:
                    decode_packet(gate_tile, tile_row, tile_column, gate, gate_spec, channel, reduction)
                    decode_packet(up_tile, tile_row, tile_column, up, up_spec, channel, reduction)
                elif tile_row < bn:
                    if channel < intermediate and reduction < width:
                        decode_packet(gate_tile, tile_row, tile_column, gate, gate_spec, channel, reduction)
                        decode_packet(up_tile, tile_row, tile_column, up, up_spec, channel, reduction)
                    else:
                        for item in T.unroll(packet.packet, explicit=True):
                            gate_tile[tile_row, tile_column + item] = 0
                            up_tile[tile_row, tile_column + item] = 0
            T.sync_threads()
            T.gemm(x, gate_tile, gate_accum, transpose_B=True)
            T.gemm(x, up_tile, up_accum, transpose_B=True)
            T.sync_threads()
        for i, j in T.Parallel(bm, bn):
            if full or (by * bm + i < rows and bx * bn + j < intermediate):
                gate_value = T.cast(T.cast(gate_accum[i, j], dtype), "float32")
                up_value = T.cast(T.cast(up_accum[i, j], dtype), "float32")
                activation[by * bm + i, bx * bn + j] = T.cast(
                    gate_value * T.sigmoid(gate_value) * up_value, dtype
                )


class _DenseSwiGLUEmitter:
    def __init__(self, specs, mode, tile=None):
        self.specs, self.mode, self.tile = specs, mode, tile

    def __call__(self, operands: tuple[Any, ...]) -> None:
        hidden, gate, up, down, output, activation = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        intermediate = cast(int, self.specs[1].shape[0])
        if self.mode == "decode":
            _gated_packet_vector(
                hidden, gate, up, activation, self.specs[1], self.specs[2],
                rows, width, intermediate,
            )
        else:
            assert self.tile is not None
            _gated_packet_matrix(
                hidden, gate, up, activation, self.specs[1], self.specs[2],
                rows, width, intermediate, self.specs[0].dtype.value, *self.tile,
            )
        packet = packet_format(self.specs[3])
        assert packet is not None
        if self.mode == "decode":
            from .matrix import _packed_vector
            _packed_vector(
                activation, down, activation, output, self.specs[3], rows, width,
                intermediate, self.specs[4].dtype.value, False,
            )
        else:
            assert self.tile is not None
            threads, bm, bn, bk = self.tile
            _packed_matrix(
                activation, down, activation, output, self.specs[3], rows, width,
                intermediate, self.specs[0].dtype.value, self.specs[4].dtype.value,
                threads, bm, bn, bk, False,
            )


@T.macro
def _selected_activate(
    hidden, routes, gate, up, activation, gate_spec, up_spec,
    rows, selected, width, intermediate,
):
    packet = packet_format(gate_spec)
    assert packet is not None and packet_format(up_spec) == packet
    with T.Kernel(selected * intermediate, rows, threads=32) as (combined, row):
        lane = T.get_thread_binding()
        rank, channel = combined // intermediate, combined % intermediate
        expert = routes[row, rank]
        values = T.alloc_local((packet.packet,), "float32")
        partial = T.alloc_local((2,), "float32")
        T.clear(partial)
        if 0 <= expert and expert < gate_spec.shape[0]:
            weight_row = expert * intermediate + channel
            for chunk in T.serial(width // packet.tile):
                for item in T.unroll(packet.packet, explicit=True):
                    values[item] = T.cast(
                        hidden[row, chunk * packet.tile + lane * packet.packet + item], "float32"
                    )
                partial[0] += packet_dot(values, gate, gate_spec, weight_row, chunk, lane)
                partial[1] += packet_dot(values, up, up_spec, weight_row, chunk, lane)
        gate_value = T.cast(T.warp_reduce_sum(partial[0]), hidden.dtype)
        up_value = T.cast(T.warp_reduce_sum(partial[1]), hidden.dtype)
        if lane == 0:
            rounded_gate = T.cast(gate_value, "float32")
            activation[row, rank, channel] = T.cast(
                rounded_gate * T.sigmoid(rounded_gate) * T.cast(up_value, "float32"), hidden.dtype
            )


@T.macro
def _selected_down(
    activation, routes, scores, down, output, down_spec,
    rows, selected, width, intermediate, output_dtype,
):
    packet = packet_format(down_spec)
    assert packet is not None
    with T.Kernel(T.ceildiv(width, 8), rows, threads=128) as (block, row):
        thread = T.get_thread_binding()
        lane = thread % 32
        first_output = block * 8 + (thread // 32) * 2
        values = T.alloc_local((packet.packet,), "float32")
        partial = T.alloc_local((2,), "float32")
        T.clear(partial)
        for rank in T.serial(selected):
            expert = routes[row, rank]
            if 0 <= expert and expert < down_spec.shape[0]:
                for owned in T.unroll(2, explicit=True):
                    channel = first_output + owned
                    if channel < width:
                        weight_row = expert * width + channel
                        expert_sum = T.alloc_local((1,), "float32")
                        expert_sum[0] = 0
                        for chunk in T.serial(intermediate // packet.tile):
                            for item in T.unroll(packet.packet, explicit=True):
                                values[item] = T.cast(
                                    activation[
                                        row, rank,
                                        chunk * packet.tile + lane * packet.packet + item,
                                    ], "float32"
                                )
                            expert_sum[0] += packet_dot(
                                values, down, down_spec, weight_row, chunk, lane
                            )
                        partial[owned] += T.cast(scores[row, rank], "float32") * expert_sum[0]
        for owned in T.unroll(2, explicit=True):
            projected = T.warp_reduce_sum(partial[owned])
            channel = first_output + owned
            if lane == 0 and channel < width:
                output[row, channel] = T.cast(projected, output_dtype)


class _SelectedExpertsEmitter:
    def __init__(self, specs):
        self.specs = specs

    def __call__(self, operands):
        hidden, routes, scores, gate, up, down, output, activation = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        intermediate = cast(int, self.specs[3].shape[1])
        _selected_activate(
            hidden, routes, gate, up, activation, self.specs[3], self.specs[4],
            rows, selected, width, intermediate,
        )
        _selected_down(
            activation, routes, scores, down, output, self.specs[5],
            rows, selected, width, intermediate, self.specs[6].dtype.value,
        )


class DenseSwiGLURule:
    name = "packet-dense-swiglu"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        region = _dense_swiglu_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs = region
        rows, width = cast(tuple[int, int], specs[0].shape)
        intermediate = cast(int, specs[1].shape[0])
        gate_packet, up_packet, down_packet = (
            packet_format(spec) for spec in specs[1:4]
        )
        if (
            gate_packet is None
            or up_packet is None
            or down_packet is None
            or gate_packet != up_packet
            or width % gate_packet.tile
            or intermediate % down_packet.tile
        ):
            return ()
        if context.mode == "decode":
            if context.capabilities.subgroup_width != 32:
                return ()
            tile = None
        else:
            instruction = _matrix_instruction(context, specs[0].dtype)
            if instruction is None:
                return ()
            bm, bn, bk = instruction.m * 4, instruction.n * 4, instruction.k * 2
            packet_width = max(
                gate_packet.packet, up_packet.packet, down_packet.packet
            )
            bk = ((bk + packet_width - 1) // packet_width) * packet_width
            threads = min(context.capabilities.threads_per_group, context.capabilities.subgroup_width * 4)
            shared = (bm * bk + 2 * bn * bk) * specs[0].dtype.itemsize
            if shared > context.capabilities.shared_memory_bytes:
                return ()
            tile = (threads, bm, bn, bk)
        activation = TensorSpec((rows, intermediate), specs[0].dtype)
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"dense_swiglu.packet-{context.mode}@{root}:{max(nodes)}", nodes, inputs, outputs,
            _DenseSwiGLUEmitter(specs, context.mode, tile),
            4e-7 + moved / 4e12, workspace=(activation,), kernel_count=2, priority=90,
        ),)


class SelectedExpertsRule:
    name = "packet-selected-experts"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "routed_experts" or context.mode != "decode":
            return ()
        specs = tuple(graph.values[value].spec for value in (*node.inputs, *node.outputs))
        hidden, routes, scores, gate, up, down, output = specs
        if (
            any(not spec.static for spec in specs)
            or context.capabilities.subgroup_width != 32
            or node.attributes["activation"] != "silu"
            or any(packet_format(spec) is None for spec in (gate, up, down))
        ):
            return ()
        rows, width = cast(tuple[int, int], hidden.shape)
        selected = cast(int, routes.shape[1])
        experts, intermediate, input_width = cast(tuple[int, int, int], gate.shape)
        gate_packet, up_packet, down_packet = (
            packet_format(spec) for spec in (gate, up, down)
        )
        if (
            input_width != width or up.shape != gate.shape
            or down.shape != (experts, width, intermediate)
            or scores.shape != routes.shape or output.shape != hidden.shape
            or gate_packet is None
            or up_packet is None
            or down_packet is None
            or gate_packet != up_packet
            or width % gate_packet.tile
            or intermediate % down_packet.tile
        ):
            return ()
        activation = TensorSpec((rows, selected, intermediate), hidden.dtype)
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"routed_experts.packet-selected@{root}", frozenset({root}), node.inputs, node.outputs,
            _SelectedExpertsEmitter(specs), 4e-7 + moved / 4e12,
            workspace=(activation,), kernel_count=2, priority=90,
        ),)


@T.macro
def _routed_shared_activate(
    hidden, routes, expert_gate, expert_up, shared_gate, shared_up, shared_router,
    selected_activation, shared_activation, coefficient,
    expert_gate_spec, expert_up_spec, shared_gate_spec, shared_up_spec,
    rows, selected, width, expert_intermediate, shared_intermediate,
):
    expert_packet = packet_format(expert_gate_spec)
    shared_packet = packet_format(shared_gate_spec)
    assert (
        expert_packet is not None
        and shared_packet is not None
        and packet_format(expert_up_spec) == expert_packet
        and packet_format(shared_up_spec) == shared_packet
    )
    total = selected * expert_intermediate + shared_intermediate
    with T.Kernel(total, rows, threads=32) as (combined, row):
        lane = T.get_thread_binding()
        values = T.alloc_local(
            (max(expert_packet.packet, shared_packet.packet),), "float32"
        )
        partial = T.alloc_local((3,), "float32")
        T.clear(partial)
        if combined < selected * expert_intermediate:
            rank = combined // expert_intermediate
            channel = combined % expert_intermediate
            expert = routes[row, rank]
            if 0 <= expert and expert < expert_gate_spec.shape[0]:
                weight_row = expert * expert_intermediate + channel
                for chunk in T.serial(width // expert_packet.tile):
                    for item in T.unroll(expert_packet.packet, explicit=True):
                        values[item] = T.cast(
                            hidden[
                                row,
                                chunk * expert_packet.tile
                                + lane * expert_packet.packet
                                + item,
                            ],
                            "float32",
                        )
                    partial[0] += packet_dot(
                        values, expert_gate, expert_gate_spec, weight_row, chunk, lane
                    )
                    partial[1] += packet_dot(
                        values, expert_up, expert_up_spec, weight_row, chunk, lane
                    )
            gate_value = T.cast(T.warp_reduce_sum(partial[0]), hidden.dtype)
            up_value = T.cast(T.warp_reduce_sum(partial[1]), hidden.dtype)
            if lane == 0:
                rounded_gate = T.cast(gate_value, "float32")
                selected_activation[row, rank, channel] = T.cast(
                    rounded_gate
                    * T.sigmoid(rounded_gate)
                    * T.cast(up_value, "float32"),
                    hidden.dtype,
                )
        else:
            channel = combined - selected * expert_intermediate
            for chunk in T.serial(width // shared_packet.tile):
                for item in T.unroll(shared_packet.packet, explicit=True):
                    reduction = (
                        chunk * shared_packet.tile
                        + lane * shared_packet.packet
                        + item
                    )
                    values[item] = T.cast(hidden[row, reduction], "float32")
                    if channel == 0:
                        partial[2] += values[item] * T.cast(
                            shared_router[reduction], "float32"
                        )
                partial[0] += packet_dot(
                    values, shared_gate, shared_gate_spec, channel, chunk, lane
                )
                partial[1] += packet_dot(
                    values, shared_up, shared_up_spec, channel, chunk, lane
                )
            gate_value = T.cast(T.warp_reduce_sum(partial[0]), hidden.dtype)
            up_value = T.cast(T.warp_reduce_sum(partial[1]), hidden.dtype)
            router_value = T.warp_reduce_sum(partial[2])
            if lane == 0:
                rounded_gate = T.cast(gate_value, "float32")
                shared_activation[row, channel] = T.cast(
                    rounded_gate
                    * T.sigmoid(rounded_gate)
                    * T.cast(up_value, "float32"),
                    hidden.dtype,
                )
                if channel == 0:
                    coefficient[row, 0] = T.sigmoid(router_value)


@T.macro
def _routed_shared_down(
    selected_activation, shared_activation, routes, scores,
    expert_weight, shared_weight, coefficient, output,
    expert_weight_spec, shared_weight_spec,
    rows, selected, outputs, expert_inputs, shared_inputs, output_dtype,
):
    expert_packet = packet_format(expert_weight_spec)
    shared_packet = packet_format(shared_weight_spec)
    assert expert_packet is not None and shared_packet is not None
    with T.Kernel(T.ceildiv(outputs, 8), rows, threads=128) as (block, row):
        thread = T.get_thread_binding()
        lane = thread % 32
        first_output = block * 8 + (thread // 32) * 2
        values = T.alloc_local(
            (max(expert_packet.packet, shared_packet.packet),), "float32"
        )
        partial = T.alloc_local((2,), "float32")
        T.clear(partial)
        for rank in T.serial(selected):
            expert = routes[row, rank]
            if 0 <= expert and expert < expert_weight_spec.shape[0]:
                for chunk in T.serial(expert_inputs // expert_packet.tile):
                    for item in T.unroll(expert_packet.packet, explicit=True):
                        values[item] = T.cast(
                            selected_activation[
                                row,
                                rank,
                                chunk * expert_packet.tile
                                + lane * expert_packet.packet
                                + item,
                            ],
                            "float32",
                        )
                    for owned in T.unroll(2, explicit=True):
                        channel = first_output + owned
                        if channel < outputs:
                            partial[owned] += T.cast(scores[row, rank], "float32") * packet_dot(
                                values,
                                expert_weight,
                                expert_weight_spec,
                                expert * outputs + channel,
                                chunk,
                                lane,
                            )
        for chunk in T.serial(shared_inputs // shared_packet.tile):
            for item in T.unroll(shared_packet.packet, explicit=True):
                values[item] = T.cast(
                    shared_activation[
                        row,
                        chunk * shared_packet.tile
                        + lane * shared_packet.packet
                        + item,
                    ],
                    "float32",
                )
            for owned in T.unroll(2, explicit=True):
                channel = first_output + owned
                if channel < outputs:
                    partial[owned] += T.cast(coefficient[row, 0], "float32") * packet_dot(
                        values, shared_weight, shared_weight_spec, channel, chunk, lane
                    )
        for owned in T.unroll(2, explicit=True):
            projected = T.warp_reduce_sum(partial[owned])
            channel = first_output + owned
            if lane == 0 and channel < outputs:
                output[row, channel] = T.cast(projected, output_dtype)


def _routed_shared_region(graph: Graph, root: int):
    selected = graph.nodes[root]
    if selected.operation != "routed_experts":
        return None
    hidden = selected.inputs[0]
    dense = None
    for candidate in range(root + 1, min(root + 8, len(graph.nodes))):
        dense = _dense_swiglu_region(graph, candidate)
        if dense is not None and dense[1][0] == hidden:
            break
        dense = None
    if dense is None:
        return None
    dense_nodes, dense_inputs, _, _ = dense
    shared_output = graph.nodes[max(dense_nodes)].outputs[0]
    row_dot = next(
        (
            node for node in graph.nodes[max(dense_nodes) + 1 : min(len(graph.nodes), max(dense_nodes) + 8)]
            if node.operation == "row_dot" and node.inputs[0] == hidden
        ),
        None,
    )
    if row_dot is None or len(graph.users[row_dot.outputs[0]]) != 1:
        return None
    coefficient_node = graph.nodes[graph.users[row_dot.outputs[0]][0]]
    if coefficient_node.operation != "sigmoid" or len(graph.users[coefficient_node.outputs[0]]) != 1:
        return None
    coefficient_value = coefficient_node.outputs[0]
    coefficient_nodes = {row_dot.id, coefficient_node.id}
    coefficient_user = graph.nodes[graph.users[coefficient_value][0]]
    if coefficient_user.operation == "cast":
        coefficient_nodes.add(coefficient_user.id)
        coefficient_value = coefficient_user.outputs[0]
    multiply = next(
        (
            graph.nodes[user] for user in graph.users[shared_output]
            if graph.nodes[user].operation == "multiply" and coefficient_value in graph.nodes[user].inputs
        ),
        None,
    )
    if multiply is None or len(graph.users[multiply.outputs[0]]) != 1:
        return None
    add = graph.nodes[graph.users[multiply.outputs[0]][0]]
    if add.operation != "add" or selected.outputs[0] not in add.inputs:
        return None
    nodes = frozenset(
        {selected.id, *dense_nodes, *coefficient_nodes, multiply.id, add.id}
    )
    if nodes != frozenset(range(min(nodes), max(nodes) + 1)):
        return None
    inputs = (
        hidden, selected.inputs[1], selected.inputs[2], selected.inputs[3],
        selected.inputs[4], selected.inputs[5], dense_inputs[1], dense_inputs[2],
        dense_inputs[3], row_dot.inputs[1],
    )
    outputs = add.outputs
    specs = tuple(graph.values[value].spec for value in (*inputs, *outputs))
    if any(not spec.static for spec in specs):
        return None
    return nodes, inputs, outputs, specs


class _RoutedSharedEmitter:
    def __init__(self, specs):
        self.specs = specs

    def __call__(self, operands: tuple[Any, ...]) -> None:
        (
            hidden, routes, scores, expert_gate, expert_up, expert_down,
            shared_gate, shared_up, shared_down, shared_router, output,
            selected_activation, shared_activation, coefficient,
        ) = operands
        rows, width = cast(tuple[int, int], self.specs[0].shape)
        selected = cast(int, self.specs[1].shape[1])
        intermediate = cast(int, self.specs[3].shape[1])
        shared_intermediate = cast(int, self.specs[6].shape[0])
        _routed_shared_activate(
            hidden,
            routes,
            expert_gate,
            expert_up,
            shared_gate,
            shared_up,
            shared_router,
            selected_activation,
            shared_activation,
            coefficient,
            self.specs[3],
            self.specs[4],
            self.specs[6],
            self.specs[7],
            rows,
            selected,
            width,
            intermediate,
            shared_intermediate,
        )
        _routed_shared_down(
            selected_activation,
            shared_activation,
            routes,
            scores,
            expert_down,
            shared_down,
            coefficient,
            output,
            self.specs[5],
            self.specs[8],
            rows,
            selected,
            width,
            intermediate,
            shared_intermediate,
            self.specs[10].dtype.value,
        )


class RoutedSharedExpertsRule:
    """One decode pipeline for routed and gated shared experts."""

    name = "routed-shared-experts"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        if context.mode != "decode" or context.capabilities.subgroup_width != 32:
            return ()
        region = _routed_shared_region(graph, root)
        if region is None:
            return ()
        nodes, inputs, outputs, specs = region
        (
            expert_gate_packet,
            expert_up_packet,
            expert_down_packet,
            shared_gate_packet,
            shared_up_packet,
            shared_down_packet,
        ) = tuple(packet_format(spec) for spec in specs[3:9])
        rows, width = cast(tuple[int, int], specs[0].shape)
        selected = cast(int, specs[1].shape[1])
        intermediate = cast(int, specs[3].shape[1])
        shared_intermediate = cast(int, specs[6].shape[0])
        if (
            expert_gate_packet is None
            or expert_up_packet is None
            or expert_down_packet is None
            or shared_gate_packet is None
            or shared_up_packet is None
            or shared_down_packet is None
            or expert_gate_packet != expert_up_packet
            or width % expert_gate_packet.tile
            or width % shared_gate_packet.tile
            or intermediate % expert_down_packet.tile
            or shared_intermediate % shared_down_packet.tile
        ):
            return ()
        workspace = (
            TensorSpec((rows, selected, intermediate), specs[0].dtype),
            TensorSpec((rows, shared_intermediate), specs[0].dtype),
            TensorSpec((rows, 1), specs[0].dtype),
        )
        moved = sum(spec.storage_nbytes for spec in specs)
        return (Candidate(
            f"routed_experts.packet-shared@{root}:{max(nodes)}", nodes, inputs, outputs,
            _RoutedSharedEmitter(specs), 8e-7 + moved / 4e12,
            workspace=workspace, kernel_count=2, priority=120,
        ),)


__all__ = [
    "DenseSwiGLURule", "SelectedExpertsRule", "RoutedSharedExpertsRule",
]
