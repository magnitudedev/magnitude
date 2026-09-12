"""Subgroup-parallel router selection."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph


@T.macro
def _subgroup_topk(source, indices, weights, rows, experts, selected, scoring, normalize, threads):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        scores = T.alloc_shared((threads,), "float32")
        chosen = T.alloc_shared((threads,), "int32")
        winners = T.alloc_shared((selected,), "float32")
        winner_indices = T.alloc_shared((selected,), "int32")
        denominator = T.alloc_shared((1,), "float32")
        softmax_denominator = T.alloc_shared((1,), "float32")
        peak = T.alloc_shared((1,), "float32")
        value = T.alloc_local((1,), "float32")
        value[0] = (
            T.cast(source[row, lane], "float32")
            if lane < experts else -3.402823466e38
        )
        if scoring == "softmax":
            scores[lane] = value[0]
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    scores[lane] = T.max(scores[lane], scores[lane + distance])
                T.sync_threads()
            peak[0] = scores[0]
            T.sync_threads()
            value[0] = T.if_then_else(
                lane < experts, T.exp(value[0] - peak[0]), 0.0
            )
            scores[lane] = value[0]
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    scores[lane] += scores[lane + distance]
                T.sync_threads()
            softmax_denominator[0] = scores[0]
            T.sync_threads()
        else:
            value[0] = T.if_then_else(
                lane < experts, T.sigmoid(value[0]), -1.0
            )
        for rank in T.serial(selected):
            available = T.alloc_local((1,), "int32")
            available[0] = 1
            for prior in T.serial(rank):
                if winner_indices[prior] == lane:
                    available[0] = 0
            scores[lane] = T.if_then_else(
                available[0] != 0, value[0], -3.402823466e38
            )
            chosen[lane] = lane
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    other = scores[lane + distance]
                    other_index = chosen[lane + distance]
                    if other > scores[lane] or (other == scores[lane] and other_index > chosen[lane]):
                        scores[lane] = other
                        chosen[lane] = other_index
                T.sync_threads()
            if lane == 0:
                winners[rank] = scores[0]
                winner_indices[rank] = chosen[0]
            T.sync_threads()
        if normalize:
            if lane == 0:
                denominator[0] = 0.0
                for rank in T.serial(selected):
                    denominator[0] += winners[rank]
            T.sync_threads()
        if lane < selected:
            slot = selected - 1 - lane
            indices[row, slot] = winner_indices[lane]
            if normalize:
                weights[row, slot] = winners[lane] / denominator[0]
            elif scoring == "softmax":
                weights[row, slot] = winners[lane] / softmax_denominator[0]
            else:
                weights[row, slot] = winners[lane]


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
        if not source.static or source.rank != 2:
            return ()
        rows, experts = cast(tuple[int, int], source.shape)
        threads = 1 << (experts - 1).bit_length()
        if threads > context.capabilities.threads_per_group:
            return ()
        selected = node.attributes["k"]
        return (Candidate(
            f"route_topk.subgroup@{root}", frozenset({root}), node.inputs, node.outputs,
            _RoutingEmitter(
                rows, experts, selected, node.attributes["scoring"],
                node.attributes["normalize"], threads,
            ),
            3e-7 + rows * experts / 1e11, priority=60,
        ),)


@T.macro
def _router_topk(
    hidden, router, indices, weights, rows, width, experts, selected,
    scoring, normalize, threads,
):
    with T.Kernel(rows, threads=threads) as row:
        lane = T.get_thread_binding()
        candidates = T.alloc_shared((threads,), "float32")
        candidate_indices = T.alloc_shared((threads,), "int32")
        winners = T.alloc_shared((selected,), "float32")
        winner_indices = T.alloc_shared((selected,), "int32")
        denominator = T.alloc_shared((1,), "float32")
        peak = T.alloc_shared((1,), "float32")
        score = T.alloc_local((1,), "float32")
        score[0] = 0.0
        if lane < experts:
            for channel in T.serial(width):
                score[0] += T.cast(hidden[row, channel], "float32") * T.cast(
                    router[lane, channel], "float32"
                )
        else:
            score[0] = -3.402823466e38
        if scoring == "softmax":
            candidates[lane] = score[0]
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    candidates[lane] = T.max(candidates[lane], candidates[lane + distance])
                T.sync_threads()
            peak[0] = candidates[0]
            T.sync_threads()
            score[0] = T.if_then_else(
                lane < experts, T.exp(score[0] - peak[0]), 0.0
            )
            candidates[lane] = score[0]
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    candidates[lane] += candidates[lane + distance]
                T.sync_threads()
            denominator[0] = candidates[0]
            T.sync_threads()
        else:
            score[0] = T.if_then_else(lane < experts, T.sigmoid(score[0]), -1.0)
        for rank in T.serial(selected):
            available = T.alloc_local((1,), "int32")
            available[0] = 1
            for prior in T.serial(rank):
                if winner_indices[prior] == lane:
                    available[0] = 0
            candidates[lane] = T.if_then_else(
                available[0] != 0, score[0], -3.402823466e38
            )
            candidate_indices[lane] = lane
            T.sync_threads()
            for step in T.unroll(threads.bit_length() - 1):
                distance = threads >> (step + 1)
                if lane < distance:
                    other = candidates[lane + distance]
                    other_index = candidate_indices[lane + distance]
                    if other > candidates[lane] or (
                        other == candidates[lane] and other_index > candidate_indices[lane]
                    ):
                        candidates[lane] = other
                        candidate_indices[lane] = other_index
                T.sync_threads()
            if lane == 0:
                winners[rank] = candidates[0]
                winner_indices[rank] = candidate_indices[0]
            T.sync_threads()
        if normalize:
            if lane == 0:
                denominator[0] = 0.0
                for rank in T.serial(selected):
                    denominator[0] += winners[rank]
            T.sync_threads()
        if lane < selected:
            slot = selected - 1 - lane
            indices[row, slot] = winner_indices[lane]
            if normalize:
                weights[row, slot] = winners[lane] / denominator[0]
            elif scoring == "softmax":
                weights[row, slot] = winners[lane] / denominator[0]
            else:
                weights[row, slot] = winners[lane]


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
            not hidden.static or not router.static or hidden.rank != 2 or router.rank != 2
            or router.representation is not None
            or "shared" not in context.capabilities.memory_scopes
        ):
            return ()
        rows, width = cast(tuple[int, int], hidden.shape)
        experts, router_width = cast(tuple[int, int], router.shape)
        threads = 1 << (experts - 1).bit_length()
        if router_width != width or threads > context.capabilities.threads_per_group:
            return ()
        return (Candidate(
            f"route_topk.fused-router@{root}:{route.id}",
            frozenset({root, route.id}), linear.inputs, route.outputs,
            _RouterTopKEmitter(
                rows, width, experts, route.attributes["k"], route.attributes["scoring"],
                route.attributes["normalize"], threads,
            ),
            4e-7 + 2 * rows * width * experts / 2e12, priority=100,
        ),)


__all__ = ["RouterTopKRule", "RoutingRule"]
