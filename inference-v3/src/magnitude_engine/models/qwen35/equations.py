"""Qwen tensor equations with no physical execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import magnitensor as mt


@dataclass(frozen=True, slots=True)
class DenseFeedForwardTensors:
    gate: mt.Tensor
    up: mt.Tensor
    down: mt.Tensor


@dataclass(frozen=True, slots=True)
class RoutedFeedForwardTensors:
    router: mt.Tensor
    shared_router: mt.Tensor
    expert_gate: mt.Tensor
    expert_up: mt.Tensor
    expert_down: mt.Tensor
    shared: DenseFeedForwardTensors
    selected: int
    normalize_selected: bool


@dataclass(frozen=True, slots=True)
class AttentionTensors:
    query_gate: mt.Tensor
    key: mt.Tensor
    value: mt.Tensor
    query_norm: mt.Tensor
    key_norm: mt.Tensor
    output: mt.Tensor
    query_heads: int
    kv_heads: int
    width: int
    rotary_width: int
    rotary_base: float
    rotary_sections: tuple[int, int, int, int]
    epsilon: float


@dataclass(frozen=True, slots=True)
class RecurrentTensors:
    query_key_value: mt.Tensor
    gate: mt.Tensor
    beta: mt.Tensor
    alpha: mt.Tensor
    convolution: mt.Tensor
    decay: mt.Tensor
    time_bias: mt.Tensor
    norm: mt.Tensor
    output: mt.Tensor
    key_heads: int
    value_heads: int
    width: int
    convolution_width: int
    epsilon: float
    head_mapping: Literal["tiled", "grouped"] = "tiled"


@dataclass(frozen=True, slots=True)
class BlockTensors:
    input_norm: mt.Tensor
    mixer: AttentionTensors | RecurrentTensors
    feedforward_norm: mt.Tensor
    feedforward: DenseFeedForwardTensors | RoutedFeedForwardTensors
    epsilon: float


@dataclass(frozen=True, slots=True)
class DecoderTensors:
    embedding: mt.Tensor
    blocks: tuple[BlockTensors, ...]
    output_norm: mt.Tensor
    readout: mt.Tensor
    epsilon: float


def dense_feedforward(hidden: mt.Tensor, weights: DenseFeedForwardTensors) -> mt.Tensor:
    """SwiGLU followed by the down projection."""

    gate = mt.linear(hidden, weights.gate)
    up = mt.linear(hidden, weights.up)
    return mt.linear(mt.silu(gate) * up, weights.down)


def routed_feedforward(hidden: mt.Tensor, weights: RoutedFeedForwardTensors) -> mt.Tensor:
    """Selected experts plus the independently gated shared expert."""

    logits = mt.linear(hidden, weights.router, output_dtype=mt.DType.F32)
    routes, scores = mt.route_topk(
        logits,
        weights.selected,
        scoring="softmax",
        normalize=weights.normalize_selected,
    )
    selected = mt.routed_experts(
        hidden,
        routes,
        scores,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
    )
    shared = dense_feedforward(hidden, weights.shared)
    coefficient = mt.cast(
        mt.sigmoid(mt.row_dot(hidden, weights.shared_router, output_dtype=mt.DType.F32)),
        shared.dtype,
    )
    return selected + shared * coefficient


def attention_mixer(
    hidden: mt.Tensor,
    coordinates: mt.Tensor,
    history: mt.Tensor,
    destinations: mt.Tensor,
    visible: mt.Tensor,
    weights: AttentionTensors,
    sequence_count: int,
) -> tuple[mt.Tensor, mt.Tensor]:
    query_gate = mt.linear(hidden, weights.query_gate)
    raw_keys = mt.linear(hidden, weights.key)
    raw_values = mt.linear(hidden, weights.value)
    queries, keys, gate = mt.attention_prepare(
        query_gate,
        raw_keys,
        weights.query_norm,
        weights.key_norm,
        coordinates,
        query_heads=weights.query_heads,
        kv_heads=weights.kv_heads,
        width=weights.width,
        rotary_width=weights.rotary_width,
        base=weights.rotary_base,
        sections=weights.rotary_sections,
        epsilon=weights.epsilon,
    )
    values = mt.reshape(raw_values, keys.shape)
    history = mt.kv_append(history, keys, values, destinations)
    attended = mt.causal_attention(
        queries,
        history,
        visible,
        sequence_count=sequence_count,
    )
    rows = cast(int, hidden.shape[0])
    mixed = mt.reshape(attended * mt.sigmoid(gate), (rows, weights.query_heads * weights.width))
    return mt.linear(mixed, weights.output), history


def recurrent_mixer(
    hidden: mt.Tensor,
    convolution_state: mt.Tensor,
    delta_state: mt.Tensor,
    row_offsets: mt.Tensor,
    weights: RecurrentTensors,
) -> tuple[mt.Tensor, mt.Tensor, mt.Tensor]:
    projected = mt.linear(hidden, weights.query_key_value)
    gate = mt.linear(hidden, weights.gate)
    beta_input = mt.linear(hidden, weights.beta)
    alpha = mt.linear(hidden, weights.alpha)
    queries, keys, values, beta, decay, convolution_state = mt.recurrent_prepare(
        projected,
        weights.convolution,
        convolution_state,
        alpha,
        beta_input,
        weights.decay,
        weights.time_bias,
        row_offsets,
        key_heads=weights.key_heads,
        value_heads=weights.value_heads,
        width=weights.width,
        convolution_width=weights.convolution_width,
        epsilon=weights.epsilon,
    )
    mixed, delta_state = mt.gated_delta_recurrence(
        queries,
        keys,
        values,
        decay,
        beta,
        delta_state,
        row_offsets,
        mapping=weights.head_mapping,
    )
    rows = cast(int, hidden.shape[0])
    normalized = mt.rms_norm(mixed, weights.norm, epsilon=weights.epsilon)
    flattened = mt.reshape(normalized, (rows, weights.value_heads * weights.width))
    gated = flattened * mt.silu(gate)
    return mt.linear(gated, weights.output), convolution_state, delta_state


def block(
    hidden: mt.Tensor,
    weights: BlockTensors,
    *,
    coordinates: mt.Tensor | None = None,
    history: mt.Tensor | None = None,
    destinations: mt.Tensor | None = None,
    visible: mt.Tensor | None = None,
    convolution_state: mt.Tensor | None = None,
    delta_state: mt.Tensor | None = None,
    recurrent_offsets: mt.Tensor | None = None,
    sequence_count: int = 1,
):
    normalized = mt.rms_norm(hidden, weights.input_norm, epsilon=weights.epsilon)
    if isinstance(weights.mixer, AttentionTensors):
        if coordinates is None or history is None or destinations is None or visible is None:
            raise ValueError("attention block requires rotary and KV operands")
        mixer, history = attention_mixer(
            normalized,
            coordinates,
            history,
            destinations,
            visible,
            weights.mixer,
            sequence_count,
        )
        state = (history,)
    else:
        if convolution_state is None or delta_state is None or recurrent_offsets is None:
            raise ValueError("recurrent block requires recurrent state operands")
        mixer, convolution_state, delta_state = recurrent_mixer(
            normalized, convolution_state, delta_state, recurrent_offsets, weights.mixer
        )
        state = convolution_state, delta_state
    residual = hidden + mixer
    normalized = mt.rms_norm(residual, weights.feedforward_norm, epsilon=weights.epsilon)
    feedforward = (
        dense_feedforward(normalized, weights.feedforward)
        if isinstance(weights.feedforward, DenseFeedForwardTensors)
        else routed_feedforward(normalized, weights.feedforward)
    )
    return (residual + feedforward, *state)


def decoder(
    tokens: mt.Tensor,
    coordinates: mt.Tensor,
    destinations: tuple[mt.Tensor, ...],
    visible: tuple[mt.Tensor, ...],
    attention_state: tuple[mt.Tensor, ...],
    convolution_state: tuple[mt.Tensor, ...],
    delta_state: tuple[mt.Tensor, ...],
    recurrent_offsets: mt.Tensor | None,
    weights: DecoderTensors,
    *,
    sequence_count: int = 1,
    output_rows: mt.Tensor | None = None,
    feature_values: mt.Tensor | None = None,
    feature_rows: mt.Tensor | None = None,
):
    """One complete decoder specialization with explicit logical state boundaries."""

    hidden = mt.embedding(tokens, weights.embedding)
    if feature_values is not None or feature_rows is not None:
        if feature_values is None or feature_rows is None:
            raise ValueError("conditioned rows require both values and destinations")
        hidden = mt.overlay_rows(hidden, mt.cast(feature_values, hidden.dtype), feature_rows)
    attention_index = 0
    recurrent_index = 0
    next_attention = []
    next_convolution = []
    next_delta = []
    for layer in weights.blocks:
        if isinstance(layer.mixer, AttentionTensors):
            result = block(
                hidden,
                layer,
                coordinates=coordinates,
                history=attention_state[attention_index],
                destinations=destinations[attention_index],
                visible=visible[attention_index],
                sequence_count=sequence_count,
            )
            hidden = result[0]
            next_attention.append(result[1])
            attention_index += 1
        else:
            result = block(
                hidden,
                layer,
                convolution_state=convolution_state[recurrent_index],
                delta_state=delta_state[recurrent_index],
                recurrent_offsets=recurrent_offsets,
                sequence_count=sequence_count,
            )
            hidden = result[0]
            next_convolution.append(result[1])
            next_delta.append(result[2])
            recurrent_index += 1
    states = tuple(next_attention), tuple(next_convolution), tuple(next_delta)
    if output_rows is None:
        return states
    selected = mt.take_rows(hidden, output_rows)
    selected = mt.rms_norm(selected, weights.output_norm, epsilon=weights.epsilon)
    logits = mt.linear(selected, weights.readout, output_dtype=mt.DType.F32)
    return logits, *states
