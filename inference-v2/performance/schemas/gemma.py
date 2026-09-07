"""Typed field schemas over the architecture's execution bindings."""

from dataclasses import dataclass

from magnitude_engine.models.architectures.gemma4.program import (
    ExpertBranch,
    GeGLU,
    Gemma4Program,
    GemmaAttention,
    GemmaFeedForward,
    KVProducer,
    LayerInput,
    PerLayerInputs,
)
from magnitude_engine.models.architectures.gemma4.program import readout as gemma_readout
from performance.bindings import Fields, Use, foreign, neural, schema
from performance.facts import AttentionGeometry, NeuralParameters
from performance.parameters import projection_output_width, projection_shape


@dataclass(frozen=True)
class GemmaAttentionUse:
    producer: KVProducer
    shared: Use


@schema(KVProducer)
def kv_producer(a: KVProducer, context: None) -> Fields[NeuralParameters]:
    return neural(
        operands={
            "keys": a.keys,
            "values": a.values,
            "key_norm": a.key_norm,
            "value_norm": a.value_norm,
        },
        settings={"heads": a.heads},
        sources=(a.positions,),
    )


@schema(GemmaAttention, context=GemmaAttentionUse)
def gemma_attention(a: GemmaAttention, context: GemmaAttentionUse) -> Fields[NeuralParameters]:
    producer, shared = (context.producer, context.shared)
    _, size = projection_shape(producer.keys)
    width = projection_output_width(producer.keys) // producer.heads
    geometry = AttentionGeometry(
        query_heads=a.heads,
        kv_heads=producer.heads,
        key_width=width,
        value_width=width,
        element_bytes=size,
        window=a.window,
        kv_source=a.source,
    )
    children: dict[str, Use] = {"attention": Use(a.operation, geometry)}
    dependencies = {}
    if a.producer is not None:
        children["producer"] = shared
    else:
        dependencies["kv_producer"] = shared
    return neural(
        operands={"queries": a.queries, "query_norm": a.query_norm, "output": a.output},
        children=children,
        dependencies=dependencies,
        settings={"window": a.window, "source": a.source},
        sources=(a.positions,),
    )


@schema(GeGLU)
def geglu(a: GeGLU, context: None) -> Fields[NeuralParameters]:
    return neural(operands={"gate": a.gate, "up": a.up, "down": a.down})


@schema(ExpertBranch)
def expert_branch(a: ExpertBranch, context: None) -> Fields[NeuralParameters]:
    return neural(
        operands={
            "router": a.router.projection,
            "scale": a.router.scale,
            "expert_scale": a.router.expert_scale,
            "input_norm": a.input_norm,
            "output_norm": a.output_norm,
        },
        children={"experts": Use(a.operation, a.router.top_k)},
        settings={"epsilon": a.router.epsilon, "top_k": a.router.top_k},
        sources=(a.router,),
    )


@schema(GemmaFeedForward)
def gemma_feed_forward(a: GemmaFeedForward, context: None) -> Fields[NeuralParameters]:
    children = {"dense": Use(a.dense)}
    if a.experts is not None:
        children["experts"] = Use(a.experts)
    return neural(
        operands={
            "input_norm": a.input_norm,
            "output_norm": a.output_norm,
            "dense_norm": a.dense_norm,
        },
        children=children,
    )


@schema(PerLayerInputs)
def per_layer_inputs(a: PerLayerInputs, context: None) -> Fields[NeuralParameters]:
    return neural(
        operands={"projection": a.projection, "norm": a.norm},
        children={"embedding": Use(a.embedding)},
        settings={
            "layers": a.layers,
            "width": a.width,
            "embedding_scale": a.embedding_scale,
            "projection_scale": a.projection_scale,
            "combination_scale": a.combination_scale,
        },
    )


@schema(LayerInput, context=Use)
def layer_input(a: LayerInput, context: Use) -> Fields[NeuralParameters]:
    prepared = context
    return neural(
        operands={"gate": a.gate, "projection": a.projection, "norm": a.norm},
        dependencies={"prepared": prepared},
    )


@schema(Gemma4Program)
def gemma4_program(a: Gemma4Program, context: None) -> Fields[NeuralParameters]:
    children = {"embedding": Use(a.embedding)}
    if a.per_layer is not None:
        children["inputs"] = Use(a.per_layer)
    producers = {}
    operands: dict[str, object] = {"norm": a.norm}
    for i, block in enumerate(a.blocks):
        mixer = block.attention
        if mixer.producer is not None:
            producers[mixer.source] = (mixer.producer, Use(mixer.producer))
        producer, shared = producers[mixer.source]
        children[f"layers.{i}.mixer"] = Use(mixer, GemmaAttentionUse(producer, shared))
        children[f"layers.{i}.feedforward"] = Use(block.feedforward)
        if block.layer_input is not None:
            children[f"layers.{i}.inputs"] = Use(block.layer_input, children["inputs"])
        operands[f"layers.{i}.input_norm"] = block.input_norm
        operands[f"layers.{i}.attention_norm"] = block.attention_norm
        operands[f"layers.{i}.scalar"] = block.scalar
    children["readout"] = foreign(
        a.output,
        gemma_readout,
        neural(operands={"output": a.output}, settings={"softcap": a.softcap}),
    )
    return neural(
        operands=operands, children=children, settings={"embedding_scale": a.embedding_scale}
    )
