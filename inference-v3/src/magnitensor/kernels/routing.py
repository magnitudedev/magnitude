"""Subgroup-parallel router selection."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph


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


class _RoutingEmitter:
    def __init__(self, rows, experts, selected, scoring, normalize, threads):
        self.args = rows, experts, selected, scoring, normalize, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        _subgroup_topk(operands[0], operands[1], operands[2], *self.args)


class RoutingRule:
    name = "subgroup-routing"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "route_topk" or "shared" not in context.capabilities.memory_scopes:
            return ()
        source = graph.values[node.inputs[0]].spec
        if not source.static or source.rank != 2 or context.capabilities.subgroup_width != 32:
            return ()
        rows, experts = cast(tuple[int, int], source.shape)
        threads = 1 << (experts - 1).bit_length()
        if threads > context.capabilities.threads_per_group:
            return ()
        selected = node.attributes["k"]
        return (
            Candidate(
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
                3e-7 + rows * experts / 1e11,
                priority=60,
            ),
        )


@T.macro
def _router_topk(
    hidden,
    router,
    indices,
    weights,
    rows,
    width,
    experts,
    selected,
    scoring,
    normalize,
    threads,
):
    with T.Kernel(rows, threads=threads) as row:
        thread = T.get_thread_binding()
        lane = thread % 32
        subgroup = thread // 32
        subgroups = threads // 32
        logits = T.alloc_shared((experts,), "float32")
        partial = T.alloc_local((1,), "float32")
        for expert_block in T.serial(T.ceildiv(experts, subgroups)):
            expert = expert_block * subgroups + subgroup
            partial[0] = 0.0
            if expert < experts:
                for channel_block in T.serial(T.ceildiv(width, 32)):
                    channel = channel_block * 32 + lane
                    if channel < width:
                        partial[0] += T.cast(hidden[row, channel], "float32") * T.cast(
                            router[expert, channel], "float32"
                        )
            projected = T.warp_reduce_sum(partial[0])
            if lane == 0 and expert < experts:
                logits[expert] = projected
        T.sync_threads()
        score = T.if_then_else(thread < experts, logits[thread], -3.402823466e38)
        _finish_topk(score, indices, weights, row, experts, selected, scoring, normalize, threads)


class _RouterTopKEmitter:
    def __init__(self, rows, width, experts, selected, scoring, normalize, threads):
        self.args = rows, width, experts, selected, scoring, normalize, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        _router_topk(operands[0], operands[1], operands[2], operands[3], *self.args)


class RouterTopKRule:
    name = "fused-router-topk"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
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
            or "shared" not in context.capabilities.memory_scopes
            or context.capabilities.subgroup_width != 32
        ):
            return ()
        rows, width = cast(tuple[int, int], hidden.shape)
        experts, router_width = cast(tuple[int, int], router.shape)
        threads = 1 << (experts - 1).bit_length()
        if router_width != width or threads > context.capabilities.threads_per_group:
            return ()
        return (
            Candidate(
                f"route_topk.fused-router@{root}:{route.id}",
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
                4e-7 + 2 * rows * width * experts / 2e12,
                priority=100,
            ),
        )


__all__ = ["RouterTopKRule", "RoutingRule"]
