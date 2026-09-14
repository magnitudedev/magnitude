"""Representation-aware indexing schedules."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import Candidate, LoweringContext
from ..tensor.graph import Graph
from .packed import decode_packet, packet_format


@T.macro
def _packed_embedding(indices, table, output, table_spec, tokens, width, threads):
    packet = packet_format(table_spec)
    assert packet is not None
    with T.Kernel(tokens, threads=threads) as token:
        row = indices[token]
        for iteration in T.serial(T.ceildiv(width // packet.matrix_packet, threads)):
            packet_index = iteration * threads + T.get_thread_binding()
            if packet_index < width // packet.matrix_packet:
                first = packet_index * packet.matrix_packet
                decode_packet(output, token, first, table, table_spec, row, first)


class _PackedEmbeddingEmitter:
    def __init__(self, table_spec, tokens, width, threads):
        self.table_spec, self.tokens, self.width, self.threads = table_spec, tokens, width, threads

    def __call__(self, operands: tuple[Any, ...]) -> None:
        _packed_embedding(
            operands[0],
            operands[1],
            operands[2],
            self.table_spec,
            self.tokens,
            self.width,
            self.threads,
        )


class PackedEmbeddingRule:
    name = "packed-embedding"

    def enumerate(self, graph: Graph, root: int, context: LoweringContext):
        node = graph.nodes[root]
        if node.operation != "embedding":
            return ()
        indices, table = (graph.values[value].spec for value in node.inputs)
        packet = packet_format(table)
        if packet is None or not indices.static or not table.static or table.rank != 2:
            return ()
        tokens = indices.elements
        width = cast(int, table.shape[1])
        if width % packet.matrix_packet:
            return ()
        threads = min(128, context.capabilities.threads_per_group)
        return (
            Candidate(
                f"embedding.packet@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _PackedEmbeddingEmitter(table, tokens, width, threads),
                2e-7 + tokens * width * table.storage_nbytes / table.elements / 4e12,
                priority=60,
            ),
        )


__all__ = ["PackedEmbeddingRule"]
