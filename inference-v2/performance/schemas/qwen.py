"""Typed field schemas over the architecture's execution bindings."""

from magnitude_engine.models.architectures.qwen35.attention.operation import GatedAttention
from magnitude_engine.models.architectures.qwen35.decode import ResidentDecode
from magnitude_engine.models.architectures.qwen35.feedforward.operation import (
    DenseFeedForward,
    RoutedFeedForward,
)
from magnitude_engine.models.architectures.qwen35.mtp.program import MTPProgram
from magnitude_engine.models.architectures.qwen35.program import Qwen35Program
from magnitude_engine.models.architectures.qwen35.program import readout as qwen_readout
from magnitude_engine.models.architectures.qwen35.recurrence.operation import RecurrentMixer
from performance.bindings import Fields, Use, foreign, neural, schema
from performance.facts import AttentionGeometry, NeuralParameters, RecurrentGeometry
from performance.parameters import projection_shape


def qwen35_program(a: Qwen35Program | ResidentDecode, context: None) -> Fields[NeuralParameters]:
    children = {"embedding": Use(a.embedding)}
    operands = {"norm": a.norm}
    for i, block in enumerate(a.blocks):
        children[f"layers.{i}.mixer"] = Use(block.mixer)
        children[f"layers.{i}.feedforward"] = Use(block.feedforward)
        operands[f"layers.{i}.input_norm"] = block.mixer_norm
        operands[f"layers.{i}.post_norm"] = block.feedforward_norm
    children["readout"] = foreign(
        a.output,
        qwen_readout,
        neural(operands={"output": a.output}),
    )
    if isinstance(a, Qwen35Program) and a.decode is not None:
        children["decode"] = Use(a.decode)
    return neural(operands=operands, children=children)


@schema(RecurrentMixer)
def recurrent_mixer(a: RecurrentMixer, context: None) -> Fields[NeuralParameters]:
    graph = a.operation.graph
    _, size = projection_shape(graph.qkv)
    geometry = RecurrentGeometry(
        key_heads=graph.key_heads,
        value_heads=graph.value_heads,
        key_width=graph.key_width,
        value_width=graph.value_width,
        element_bytes=size,
    )
    return neural(
        operands={
            "qkv": graph.qkv,
            "gate": graph.output_gate,
            "beta": graph.beta,
            "decay": graph.decay,
            "conv": graph.convolution,
            "log_rates": graph.log_rates,
            "time_bias": graph.time_bias,
            "norm": graph.normalize_output,
            "output": graph.output,
        },
        children={"update": Use(graph.recurrence, geometry)},
        settings={"window": graph.window},
        sources=(graph, a.operation),
    )


@schema(GatedAttention)
def gated_attention(a: GatedAttention, context: None) -> Fields[NeuralParameters]:
    _, size = projection_shape(a.queries_and_gate)
    geometry = AttentionGeometry(
        query_heads=a.query_heads,
        kv_heads=a.kv_heads,
        key_width=a.head_width,
        value_width=a.head_width,
        element_bytes=size,
    )
    return neural(
        operands={
            "q_gate": a.queries_and_gate,
            "k": a.keys,
            "v": a.values,
            "output": a.output,
            "q_norm": a.query_norm,
            "k_norm": a.key_norm,
        },
        children={"attention": Use(a.attention, geometry)},
        sources=(a.positions,),
    )


@schema(DenseFeedForward)
def dense_feed_forward(a: DenseFeedForward, context: None) -> Fields[NeuralParameters]:
    return neural(operands={"mlp": a.call})


@schema(RoutedFeedForward)
def routed_feed_forward(a: RoutedFeedForward, context: None) -> Fields[NeuralParameters]:
    return neural(
        operands={"router": a.router, "shared": a.shared, "shared_gate": a.shared_gate},
        children={"experts": Use(a.experts, a.top_k)},
        settings={"normalize": a.normalize, "top_k": a.top_k},
    )


@schema(MTPProgram)
def mtp_program(a: MTPProgram, context: None) -> Fields[NeuralParameters]:
    return neural(
        operands={
            "embedding_norm": a.normalize_embedding,
            "conditioning_norm": a.normalize_conditioning,
            "combine": a.combine,
            "layers": a.layers,
            "output_norm": a.normalize_output,
            "output": a.project,
        },
        children={"embedding": Use(a.embedding)},
    )


schema(Qwen35Program)(qwen35_program)
schema(ResidentDecode)(qwen35_program)
