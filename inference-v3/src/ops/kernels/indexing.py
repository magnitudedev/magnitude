"""Representation-aware indexing schedules."""

from __future__ import annotations

from typing import Any, cast

import tilelang.language as T

from ..compiler.lowering import BoundOperation, LoweringContext
from ..tensor.graph import Graph
from .packed import decode_packet, packet_format


@T.macro
def _streamed_embedding(table, destinations, output, table_spec, step, threads):
    packet = packet_format(table_spec)
    width = table_spec.shape[1]
    with T.Kernel(step, threads=threads) as token:
        source_row = destinations[token, 0]
        output_row = destinations[token, 1]
        if output_row >= 0:
            if packet is not None:
                for iteration in T.serial(T.ceildiv(width // packet.matrix_packet, threads)):
                    index = iteration * threads + T.get_thread_binding()
                    if index < width // packet.matrix_packet:
                        first = index * packet.matrix_packet
                        decode_packet(output, output_row, first, table, table_spec, source_row, first)
            else:
                for column in T.Parallel(width):
                    output[output_row, column] = table[source_row, column]


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

    def build(self, graph: Graph, root: int, context: LoweringContext):
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
        threads = min(128, context.compiler_target.threads_per_group)
        return (
            BoundOperation(
                f"embedding.packet@{root}",
                frozenset({root}),
                node.inputs,
                node.outputs,
                _PackedEmbeddingEmitter(table, tokens, width, threads),
            ),
        )


__all__ = ["PackedEmbeddingRule"]
