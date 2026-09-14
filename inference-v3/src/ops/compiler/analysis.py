"""Whole-graph facts used to validate bounded lowering regions."""

from __future__ import annotations

from dataclasses import dataclass

from ..tensor.graph import Graph


@dataclass(frozen=True, slots=True)
class GraphAnalysis:
    users: tuple[tuple[int, ...], ...]
    predecessors: tuple[frozenset[int], ...]
    successors: tuple[frozenset[int], ...]
    dominators: tuple[frozenset[int], ...]
    postdominators: tuple[frozenset[int], ...]


def analyze(graph: Graph) -> GraphAnalysis:
    predecessors = []
    successors = [set() for _ in graph.nodes]
    for node in graph.nodes:
        before = {
            graph.values[value].producer
            for value in node.inputs
            if graph.values[value].producer is not None
        }
        predecessors.append(frozenset(before))
        for producer in before:
            assert producer is not None
            successors[producer].add(node.id)
    frozen_successors = tuple(frozenset(items) for items in successors)
    return GraphAnalysis(
        graph.users,
        tuple(predecessors),
        frozen_successors,
        _fixed_dominators(tuple(predecessors)),
        _fixed_postdominators(frozen_successors),
    )


def _fixed_dominators(predecessors: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    all_nodes = frozenset(range(len(predecessors)))
    values = [
        ({index} if not before else set(all_nodes)) for index, before in enumerate(predecessors)
    ]
    changed = True
    while changed:
        changed = False
        for node, before in enumerate(predecessors):
            updated = {node}
            if before:
                updated |= set.intersection(*(values[item] for item in before))
            if updated != values[node]:
                values[node] = updated
                changed = True
    return tuple(frozenset(value) for value in values)


def _fixed_postdominators(successors: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    all_nodes = frozenset(range(len(successors)))
    values = [({index} if not after else set(all_nodes)) for index, after in enumerate(successors)]
    changed = True
    while changed:
        changed = False
        for node in range(len(successors) - 1, -1, -1):
            after = successors[node]
            updated = {node}
            if after:
                updated |= set.intersection(*(values[item] for item in after))
            if updated != values[node]:
                values[node] = updated
                changed = True
    return tuple(frozenset(value) for value in values)
