"""Bind Gemma parameter containers to explicit owned execution dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.experts.contracts import ExpertOperator
from magnitude_engine.models.state.arena import LayerGeometry

from .program import (
    ExpertBranch,
    GeGLU,
    Gemma4Program,
    GemmaAttention,
    GemmaBlock,
    GemmaFeedForward,
    GemmaRouter,
    KVProducer,
    LayerInput,
    PerLayerInputs,
)


@dataclass(frozen=True)
class Gemma4Binding:
    program: Gemma4Program
    attention: tuple[LayerGeometry, ...]


def bind_gemma4(
    model: Any,
    *,
    embedding: EmbeddingLookup,
    per_layer_embedding: EmbeddingLookup | None,
    experts: Mapping[int, ExpertOperator],
    attention: PagedAttention,
) -> Gemma4Binding:
    inner = model.model
    layers = inner.layers
    routed = {i for i, layer in enumerate(layers) if layer.enable_moe}
    if set(experts) != routed:
        raise ValueError("construction must supply exactly one expert operation per routed layer")
    width = inner.hidden_size_per_layer_input
    if (width > 0) != (per_layer_embedding is not None):
        raise ValueError("construction must supply the declared per-layer embedding operation")
    sources = inner.previous_kvs
    if len(sources) != len(layers):
        raise ValueError("Gemma KV source layout must cover every neural layer")
    physical: dict[int, int] = {}
    geometries = []
    blocks = []
    for index, layer in enumerate(layers):
        a = layer.self_attn
        source = sources[index]
        geometry = LayerGeometry(a.n_kv_heads, a.head_dim, a.head_dim)
        producer = None
        if source == index:
            if not a.has_kv:
                raise ValueError("Gemma KV producer is missing its projection parameters")
            physical[index] = len(geometries)
            geometries.append(geometry)
            producer = KVProducer(
                a.k_proj,
                None if a.use_k_eq_v else a.v_proj,
                a.k_norm,
                a.v_norm,
                a.rope,
                a.n_kv_heads,
            )
        elif source not in physical or a.has_kv:
            raise ValueError(
                "Gemma KV sharing requires an earlier producer and no duplicate writer"
            )
        elif geometries[physical[source]] != geometry:
            raise ValueError("Gemma shared KV geometry differs from its producer")
        elif layers[source].layer_type != layer.layer_type:
            raise ValueError("Gemma shared KV requires matching source attention semantics")
        m = layer.mlp
        expert_branch = None
        if index in routed:
            r = layer.router
            expert_branch = ExpertBranch(
                GemmaRouter(r.proj, r.scale, r.per_expert_scale, r.eps, r.config.top_k_experts),
                experts[index],
                layer.pre_feedforward_layernorm_2,
                layer.post_feedforward_layernorm_2,
            )
        blocks.append(
            GemmaBlock(
                layer.input_layernorm,
                GemmaAttention(
                    physical[source],
                    producer,
                    a.q_proj,
                    a.q_norm,
                    a.rope,
                    a.o_proj,
                    a.n_heads,
                    inner.window_size if a.is_sliding else None,
                    attention,
                ),
                layer.post_attention_layernorm,
                GemmaFeedForward(
                    layer.pre_feedforward_layernorm,
                    GeGLU(m.gate_proj, m.up_proj, m.down_proj),
                    layer.post_feedforward_layernorm,
                    layer.post_feedforward_layernorm_1 if expert_branch is not None else None,
                    expert_branch,
                ),
                LayerInput(
                    layer.per_layer_input_gate,
                    layer.per_layer_projection,
                    layer.post_per_layer_input_norm,
                )
                if width
                else None,
                layer.layer_scalar,
            )
        )
    per_layer = (
        PerLayerInputs(
            per_layer_embedding,
            inner.per_layer_model_projection,
            inner.per_layer_projection_norm,
            len(layers),
            width,
            inner.embed_tokens_per_layer_scale,
            inner.per_layer_projection_scale,
            inner.per_layer_input_scale,
        )
        if per_layer_embedding is not None
        else None
    )
    output = inner.embed_tokens.as_linear if model.tie_word_embeddings else model.lm_head
    return Gemma4Binding(
        Gemma4Program(
            embedding,
            inner.embed_scale,
            tuple(blocks),
            inner.norm,
            output,
            per_layer,
            model.final_logit_softcapping,
        ),
        tuple(geometries),
    )
