"""Bind the pinned MLX-LM Qwen text parameter layout to owned execution.

Library modules are borrowed weight/operation containers. This binding neither
patches their execution nor loads the tensors assigned to a streaming owner.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.nn as nn

from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.experts.contracts import ExpertOperator
from magnitude_engine.models.projections import ParallelProjections, bind_readout
from magnitude_engine.models.state.arena import LayerGeometry
from magnitude_engine.models.state.recurrent import RecurrentLayout

from .contracts import (
    AttentionFactory,
    FeedForwardFactory,
    RecurrentFactory,
)
from .program import HybridBlock, Qwen35Program


@dataclass(frozen=True)
class Qwen35Binding:
    program: Qwen35Program
    attention: tuple[LayerGeometry, ...]
    recurrence: tuple[RecurrentLayout, ...]


def bind_qwen35(
    model: Any,
    *,
    embedding: EmbeddingLookup,
    experts: Mapping[int, ExpertOperator],
    attention: AttentionFactory,
    recurrence: RecurrentFactory,
    feedforward: FeedForwardFactory,
    state_dtype: Any,
    projections: Mapping[tuple[str, ...], nn.Module] | None = None,
) -> Qwen35Binding:
    projections = projections or {}
    blocks = []
    geometries = []
    layouts = []
    routed = {i for i, layer in enumerate(model.layers) if hasattr(layer.mlp, "switch_mlp")}
    if set(experts) != routed:
        raise ValueError("construction must supply exactly one expert operation per routed layer")
    for index, layer in enumerate(model.layers):
        if layer.is_linear:
            g = layer.linear_attn
            names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
            packed = projections.get(
                tuple(f"model.layers.{index}.linear_attn.{name}" for name in names)
            )
            inputs = ParallelProjections(tuple(getattr(g, name) for name in names), packed)
            mixer = recurrence.bind(g, len(layouts), inputs)
            layouts.append(mixer.operation.layout(state_dtype))
        else:
            a = layer.self_attn
            names = ("q_proj", "k_proj", "v_proj")
            packed = projections.get(
                tuple(f"model.layers.{index}.self_attn.{name}" for name in names)
            )
            inputs = ParallelProjections(tuple(getattr(a, name) for name in names), packed)
            mixer = attention.bind(a, len(geometries), inputs)
            geometries.append(LayerGeometry(a.num_key_value_heads, a.head_dim, a.head_dim))
        m = layer.mlp
        routing = None
        if index in experts:
            names = (
                f"model.layers.{index}.mlp.gate",
                f"model.layers.{index}.mlp.shared_expert_gate",
            )
            routing = ParallelProjections((m.gate, m.shared_expert_gate), projections.get(names))
        block_feedforward = feedforward.bind(m, experts.get(index), routing)
        blocks.append(
            HybridBlock(
                layer.input_layernorm, mixer, layer.post_attention_layernorm, block_feedforward
            )
        )
    output = bind_readout(
        model.model.embed_tokens if model.args.tie_word_embeddings else model.lm_head
    )
    return Qwen35Binding(
        Qwen35Program(embedding, tuple(blocks), model.model.norm, output),
        tuple(geometries),
        tuple(layouts),
    )
