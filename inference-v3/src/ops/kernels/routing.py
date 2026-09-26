"""Subgroup-parallel router selection."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from ..tensor.types import DType, TensorSpec


@T.macro
def _finish_topk(
    raw,
    indices,
    weights,
    row,
    experts,
    selected,
    scoring,
    normalize,
    threads,
):
    thread = T.get_thread_binding()
    lane = thread % 32
    subgroup = thread // 32
    subgroups = threads // 32
    subgroup_scores = T.alloc_shared((subgroups,), "float32")
    subgroup_indices = T.alloc_shared((subgroups,), "int32")
    winners = T.alloc_shared((selected,), "float32")
    winner_indices = T.alloc_shared((selected,), "int32")
    peak = T.alloc_shared((1,), "float32")
    full_denominator = T.alloc_shared((1,), "float32")
    selected_denominator = T.alloc_shared((1,), "float32")
    value = T.alloc_local((1,), "float32")
    value[0] = T.if_then_else(thread < experts, raw, -3.402823466e38)

    if scoring == "softmax":
        local_peak = T.warp_reduce_max(value[0])
        if lane == 0:
            subgroup_scores[subgroup] = local_peak
        T.sync_threads()
        if subgroup == 0:
            candidate = T.if_then_else(lane < subgroups, subgroup_scores[lane], -3.402823466e38)
            global_peak = T.warp_reduce_max(candidate)
            if lane == 0:
                peak[0] = global_peak
        T.sync_threads()
        value[0] = T.if_then_else(thread < experts, T.exp(value[0] - peak[0]), 0.0)
        local_sum = T.warp_reduce_sum(value[0])
        if lane == 0:
            subgroup_scores[subgroup] = local_sum
        T.sync_threads()
        if subgroup == 0:
            candidate = T.if_then_else(lane < subgroups, subgroup_scores[lane], 0.0)
            total = T.warp_reduce_sum(candidate)
            if lane == 0:
                full_denominator[0] = total
        T.sync_threads()
    else:
        value[0] = T.if_then_else(thread < experts, T.sigmoid(value[0]), -1.0)

    for rank in T.serial(selected):
        available = T.alloc_local((1,), "int32")
        available[0] = T.if_then_else(thread < experts, 1, 0)
        for prior in T.serial(rank):
            if winner_indices[prior] == thread:
                available[0] = 0
        candidate = T.if_then_else(available[0] != 0, value[0], -3.402823466e38)
        local_maximum = T.warp_reduce_max(candidate)
        local_index = T.warp_reduce_max(T.if_then_else(candidate == local_maximum, thread, -1))
        if lane == 0:
            subgroup_scores[subgroup] = local_maximum
            subgroup_indices[subgroup] = local_index
        T.sync_threads()
        if subgroup == 0:
            cross_score = T.if_then_else(lane < subgroups, subgroup_scores[lane], -3.402823466e38)
            cross_maximum = T.warp_reduce_max(cross_score)
            cross_index = T.warp_reduce_max(
                T.if_then_else(
                    lane < subgroups and cross_score == cross_maximum,
                    subgroup_indices[lane],
                    -1,
                )
            )
            if lane == 0:
                winners[rank] = cross_maximum
                winner_indices[rank] = cross_index
        T.sync_threads()

    if normalize:
        if thread == 0:
            total_selected = T.alloc_local((1,), "float32")
            total_selected[0] = 0.0
            for rank in T.serial(selected):
                total_selected[0] += winners[rank]
            selected_denominator[0] = total_selected[0]
        T.sync_threads()
    if thread < selected:
        slot = selected - 1 - thread
        indices[row, slot] = winner_indices[thread]
        if normalize:
            weights[row, slot] = winners[thread] / selected_denominator[0]
        elif scoring == "softmax":
            weights[row, slot] = winners[thread] / full_denominator[0]
        else:
            weights[row, slot] = winners[thread]


@T.macro
def _subgroup_topk(source, indices, weights, rows, experts, selected, scoring, normalize, threads):
    with T.Kernel(rows, threads=threads) as row:
        thread = T.get_thread_binding()
        raw = T.if_then_else(
            thread < experts,
            T.cast(source[row, thread], "float32"),
            -3.402823466e38,
        )
        _finish_topk(raw, indices, weights, row, experts, selected, scoring, normalize, threads)


@T.macro
def _tiled_topk(source, indices, weights, rows, experts, selected, scoring, normalize, threads):
    extent = 1 << (experts - 1).bit_length()
    with T.Kernel(rows, threads=threads) as row:
        scores = T.alloc_fragment((extent,), "float32")
        candidates = T.alloc_fragment((extent,), "int32")
        layout = T.Fragment((extent,), forward_thread_fn=lambda i: i % threads,
                            forward_index_fn=lambda i: i // threads)
        T.annotate_layout({scores: layout, candidates: layout})
        peak = T.alloc_fragment((1,), "float32")
        total = T.alloc_fragment((1,), "float32")
        winner = T.alloc_fragment((1,), "int32")
        shared_peak = T.alloc_shared((1,), "float32")
        shared_total = T.alloc_shared((1,), "float32")
        shared_winner = T.alloc_shared((1,), "int32")
        winners = T.alloc_shared((selected,), "float32")
        for expert in T.Parallel(extent):
            scores[expert] = T.if_then_else(expert < experts, T.cast(source[row, expert], "float32"), -float("inf"))
        if scoring == "softmax":
            T.reduce_max(scores, peak, dim=0)
            T.copy(peak, shared_peak)
            T.sync_threads()
            for expert in T.Parallel(extent):
                scores[expert] = T.if_then_else(expert < experts, T.exp(scores[expert] - shared_peak[0]), 0.0)
            T.reduce_sum(scores, total, dim=0)
            T.copy(total, shared_total)
            T.sync_threads()
        else:
            for expert in T.Parallel(extent):
                scores[expert] = T.if_then_else(expert < experts, T.sigmoid(scores[expert]), 0.0)
        for expert in T.Parallel(extent):
            scores[expert] = T.if_then_else(expert < experts, scores[expert], -float("inf"))
        for rank in T.serial(selected):
            T.reduce_max(scores, peak, dim=0)
            T.copy(peak, shared_peak)
            T.sync_threads()
            for expert in T.Parallel(extent):
                candidates[expert] = T.if_then_else(expert < experts and scores[expert] == shared_peak[0], expert, -1)
            T.reduce_max(candidates, winner, dim=0)
            T.copy(winner, shared_winner)
            T.sync_threads()
            if T.get_thread_binding() == 0:
                indices[row, selected - 1 - rank] = shared_winner[0]
                winners[rank] = shared_peak[0] / shared_total[0] if scoring == "softmax" and not normalize else shared_peak[0]
            for expert in T.Parallel(extent):
                scores[expert] = T.if_then_else(expert == shared_winner[0], -float("inf"), scores[expert])
        T.sync_threads()
        denominator = T.alloc_local((1,), "float32")
        denominator[0] = 1.0
        if normalize:
            denominator[0] = 0.0
            for rank in T.serial(selected):
                denominator[0] += winners[rank]
        for rank in T.Parallel(selected):
            weights[row, selected - 1 - rank] = winners[rank] / denominator[0]


class _RoutingEmitter:
    def __init__(self, rows, experts, selected, scoring, normalize, threads):
        self.args = rows, experts, selected, scoring, normalize, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        body = _tiled_topk if self.args[1] > self.args[-1] else _subgroup_topk
        body(operands[0], operands[1], operands[2], *self.args)


class RoutingRule:
    name = "subgroup-routing"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "route_topk" or context.compiler_target.shared_memory_bytes <= 0:
            return ()
        source = graph.values[node.inputs[0]].spec
        if not source.static or source.rank != 2 or context.compiler_target.subgroup_width != 32:
            return ()
        rows, experts = cast(tuple[int, int], source.shape)
        threads = min(max(32, 1 << (experts - 1).bit_length()),
                      1 << (context.compiler_target.threads_per_group.bit_length() - 1))
        selected = node.attributes["k"]
        shared_bytes = (selected * 4 + 12 if experts > threads
                        else selected * 8 + (threads // 32) * 8 + 12)
        if threads < 32 or shared_bytes > context.compiler_target.shared_memory_bytes:
            return ()
        return (
            BoundOperation(
                f"route_topk.subgroup@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _RoutingEmitter(
                    rows,
                    experts,
                    selected,
                    node.attributes["scoring"],
                    node.attributes["normalize"],
                    threads,
                ),
            ),
        )


@T.macro
def _router_projection(hidden, router, logits, rows, width, experts, threads):
    """Distribute independent expert dots across the device before selection."""
    subgroups = threads // 32
    with T.Kernel(T.ceildiv(experts, subgroups), rows, threads=threads) as (block, row):
        thread = T.get_thread_binding()
        lane = thread % 32
        expert = block * subgroups + thread // 32
        partial = T.alloc_local((1,), "float32")
        partial[0] = 0.0
        if expert < experts:
            for channel_block in T.serial(T.ceildiv(width, 32)):
                channel = channel_block * 32 + lane
                if channel < width:
                    partial[0] += T.cast(hidden[row, channel], "float32") * T.cast(
                        router[expert, channel], "float32")
        projected = T.warp_reduce_sum(partial[0])
        if lane == 0 and expert < experts:
            logits[row, expert] = projected


class _RouterTopKEmitter:
    def __init__(self, rows, width, experts, selected, scoring, normalize, threads):
        self.args = rows, width, experts, selected, scoring, normalize, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        rows, width, experts, selected, scoring, normalize, threads = self.args
        _router_projection(operands[0], operands[1], operands[4], rows, width, experts, min(128, threads))
        _subgroup_topk(operands[4], operands[2], operands[3],
                       rows, experts, selected, scoring, normalize, threads)


class RouterTopKRule:
    name = "fused-router-topk"

    def build(self, graph: Graph, root: int, context: LoweringContext):
        linear = graph.nodes[root]
        if linear.operation != "linear" or len(linear.inputs) != 2:
            return ()
        users = graph.users[linear.outputs[0]]
        if len(users) != 1:
            return ()
        route = graph.nodes[users[0]]
        if route.operation != "route_topk" or route.id != root + 1:
            return ()
        hidden = graph.values[linear.inputs[0]].spec
        router = graph.values[linear.inputs[1]].spec
        if (
            not hidden.static
            or not router.static
            or hidden.rank != 2
            or router.rank != 2
            or router.representation is not None
            or context.compiler_target.shared_memory_bytes <= 0
            or context.compiler_target.subgroup_width != 32
        ):
            return ()
        rows, width = cast(tuple[int, int], hidden.shape)
        if context.mode != "decode" and rows > 1:
            # Prefill reuses the router matrix across tokens. Keep the ordinary
            # batched contraction and subsequent top-k in the same submission;
            # a fused sequence of per-row dots discards that matrix reuse.
            return ()
        experts, router_width = cast(tuple[int, int], router.shape)
        threads = max(32, 1 << (experts - 1).bit_length())
        if router_width != width or threads > context.compiler_target.threads_per_group:
            return ()
        logits = TensorSpec((rows, experts), DType.F32)
        if logits.storage_nbytes > context.workspace_limit:
            return ()
        return (
            BoundOperation(
                f"route_topk.parallel-router@{root}:{route.id}",
                frozenset({root, route.id}),
                linear.inputs,
                route.outputs,
                _RouterTopKEmitter(
                    rows,
                    width,
                    experts,
                    route.attributes["k"],
                    route.attributes["scoring"],
                    route.attributes["normalize"],
                    threads,
                ),
                workspace=(logits,),
                kernel_count=2,
            ),
        )


__all__ = ["RouterTopKRule", "RoutingRule"]
