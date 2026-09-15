"""Qwen tensor equations with no physical execution policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import ops


@dataclass(frozen=True, slots=True)
class DenseFeedForwardTensors:
    gate: ops.Tensor
    up: ops.Tensor
    down: ops.Tensor


@dataclass(frozen=True, slots=True)
class RoutedFeedForwardTensors:
    router: ops.Tensor
    shared_router: ops.Tensor
    expert_gate: ops.Tensor
    expert_up: ops.Tensor
    expert_down: ops.Tensor
    shared: DenseFeedForwardTensors
    selected: int
    normalize_selected: bool


@dataclass(frozen=True, slots=True)
class AttentionTensors:
    query_gate: ops.Tensor
    key: ops.Tensor
    value: ops.Tensor
    query_norm: ops.Tensor
    key_norm: ops.Tensor
    output: ops.Tensor
    query_heads: int
    kv_heads: int
    width: int
    rotary_width: int
    rotary_base: float
    rotary_sections: tuple[int, int, int, int]
    epsilon: float


@dataclass(frozen=True, slots=True)
class RecurrentTensors:
    query_key_value: ops.Tensor
    gate: ops.Tensor
    beta: ops.Tensor
    alpha: ops.Tensor
    convolution: ops.Tensor
    decay: ops.Tensor
    time_bias: ops.Tensor
    norm: ops.Tensor
    output: ops.Tensor
    key_heads: int
    value_heads: int
    width: int
    convolution_width: int
    epsilon: float
    head_mapping: Literal["tiled", "grouped"] = "tiled"


@dataclass(frozen=True, slots=True)
class BlockTensors:
    input_norm: ops.Tensor
    mixer: AttentionTensors | RecurrentTensors
    feedforward_norm: ops.Tensor
    feedforward: DenseFeedForwardTensors | RoutedFeedForwardTensors
    epsilon: float


@dataclass(frozen=True, slots=True)
class DecoderTensors:
    embedding: ops.Tensor
    blocks: tuple[BlockTensors, ...]
    output_norm: ops.Tensor
    readout: ops.Tensor
    epsilon: float


@ops.formula(id="qwen35.dense_feedforward", version=1)
def dense_feedforward(hidden: ops.Tensor, weights: DenseFeedForwardTensors) -> ops.Tensor:
    """SwiGLU followed by the down projection."""

    ops.quantity("tokens", hidden.shape[0], unit=ops.units.token)
    gate = ops.linear(hidden, weights.gate)
    up = ops.linear(hidden, weights.up)
    return ops.linear(ops.silu(gate) * up, weights.down)


@ops.formula(id="qwen35.routed_feedforward", version=1)
def routed_feedforward(hidden: ops.Tensor, weights: RoutedFeedForwardTensors) -> ops.Tensor:
    """Selected experts plus the independently gated shared expert."""

    logits = ops.linear(hidden, weights.router, output_dtype=ops.DType.F32)
    routes, scores = ops.route_topk(
        logits,
        weights.selected,
        scoring="softmax",
        normalize=weights.normalize_selected,
    )
    selected = ops.routed_experts(
        hidden,
        routes,
        scores,
        weights.expert_gate,
        weights.expert_up,
        weights.expert_down,
    )
    shared = dense_feedforward(hidden, weights.shared)
    coefficient = ops.cast(
        ops.sigmoid(ops.row_dot(hidden, weights.shared_router, output_dtype=ops.DType.F32)),
        shared.dtype,
    )
    return selected + shared * coefficient


@ops.formula(id="qwen35.attention_mixer", version=1)
def attention_mixer(
    hidden: ops.Tensor,
    coordinates: ops.Tensor,
    history: ops.Tensor,
    destinations: ops.Tensor,
    visible: ops.Tensor,
    weights: AttentionTensors,
    sequence_count: int,
) -> tuple[ops.Tensor, ops.Tensor]:
    attended, gate, history = _attention_state(
        hidden,
        coordinates,
        history,
        destinations,
        visible,
        weights,
        sequence_count,
        attend=True,
    )
    assert attended is not None
    rows = cast(int, hidden.shape[0])
    mixed = ops.reshape(attended * ops.sigmoid(gate), (rows, weights.query_heads * weights.width))
    return ops.linear(mixed, weights.output), history


@ops.formula(id="qwen35.attention_state", version=1)
def _attention_state(
    hidden: ops.Tensor,
    coordinates: ops.Tensor,
    history: ops.Tensor,
    destinations: ops.Tensor,
    visible: ops.Tensor,
    weights: AttentionTensors,
    sequence_count: int,
    *,
    attend: bool,
) -> tuple[ops.Tensor | None, ops.Tensor, ops.Tensor]:
    """Produce the KV transition and, when needed, the stateless mixer value."""

    query_gate = ops.linear(hidden, weights.query_gate)
    raw_keys = ops.linear(hidden, weights.key)
    raw_values = ops.linear(hidden, weights.value)
    queries, keys, gate = ops.attention_prepare(
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
    values = ops.reshape(raw_values, keys.shape)
    # History consumption and persistence share the prepared inputs. Attention
    # reads only the pre-advance visible interval; fresh rows use dense K/V.
    attended = (
        ops.persistent_attention(
            queries,
            history,
            keys,
            values,
            visible,
            sequence_count=sequence_count,
        )
        if attend
        else None
    )
    next_history = ops.kv_append(history, keys, values, destinations, reserved=True)
    return attended, gate, next_history


@ops.formula(id="qwen35.recurrent_mixer", version=1)
def recurrent_mixer(
    hidden: ops.Tensor,
    convolution_state: ops.Tensor,
    delta_state: ops.Tensor,
    row_offsets: ops.Tensor,
    weights: RecurrentTensors,
    *,
    sequence_length: int | None = None,
) -> tuple[ops.Tensor, ops.Tensor, ops.Tensor]:
    mixed, gate, convolution_state, delta_state = _recurrent_state(
        hidden,
        convolution_state,
        delta_state,
        row_offsets,
        weights,
        sequence_length=sequence_length,
    )
    rows = cast(int, hidden.shape[0])
    normalized = ops.rms_norm(mixed, weights.norm, epsilon=weights.epsilon)
    flattened = ops.reshape(normalized, (rows, weights.value_heads * weights.width))
    gated = flattened * ops.silu(gate)
    return ops.linear(gated, weights.output), convolution_state, delta_state


@ops.formula(id="qwen35.recurrent_state", version=1)
def _recurrent_state(
    hidden: ops.Tensor,
    convolution_state: ops.Tensor,
    delta_state: ops.Tensor,
    row_offsets: ops.Tensor,
    weights: RecurrentTensors,
    *,
    sequence_length: int | None = None,
) -> tuple[ops.Tensor, ops.Tensor, ops.Tensor, ops.Tensor]:
    """Produce recurrent state transitions before the stateless output suffix."""

    projected = ops.linear(hidden, weights.query_key_value)
    gate = ops.linear(hidden, weights.gate)
    beta_input = ops.linear(hidden, weights.beta)
    alpha = ops.linear(hidden, weights.alpha)
    queries, keys, values, beta, decay, convolution_state = ops.recurrent_prepare(
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
        # Preparation uses a sum-of-squares L2 denominator. Qwen specifies
        # RMS normalization, so convert its mean-domain epsilon explicitly.
        epsilon=weights.epsilon * weights.width,
    )
    mixed, delta_state = ops.gated_delta_recurrence(
        queries,
        keys,
        values,
        decay,
        beta,
        delta_state,
        row_offsets,
        mapping=weights.head_mapping,
        sequence_length=sequence_length,
    )
    return mixed, gate, convolution_state, delta_state


@ops.formula(id="qwen35.block", version=1)
def block(
    hidden: ops.Tensor,
    weights: BlockTensors,
    *,
    coordinates: ops.Tensor | None = None,
    history: ops.Tensor | None = None,
    destinations: ops.Tensor | None = None,
    visible: ops.Tensor | None = None,
    convolution_state: ops.Tensor | None = None,
    delta_state: ops.Tensor | None = None,
    recurrent_offsets: ops.Tensor | None = None,
    sequence_count: int = 1,
    recurrent_sequence_length: int | None = None,
):
    normalized = ops.rms_norm(
        hidden, weights.input_norm, epsilon=weights.epsilon, output_dtype=weights.mixer.output.dtype
    )
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
            normalized, convolution_state, delta_state, recurrent_offsets, weights.mixer,
            sequence_length=recurrent_sequence_length,
        )
        state = convolution_state, delta_state
    residual = hidden + ops.cast(mixer, hidden.dtype)
    normalized = ops.rms_norm(
        residual,
        weights.feedforward_norm,
        epsilon=weights.epsilon,
        output_dtype=weights.mixer.output.dtype,
    )
    feedforward = (
        dense_feedforward(normalized, weights.feedforward)
        if isinstance(weights.feedforward, DenseFeedForwardTensors)
        else routed_feedforward(normalized, weights.feedforward)
    )
    return (residual + ops.cast(feedforward, residual.dtype), *state)


@ops.formula(id="qwen35.decoder", version=1)
def decoder(
    tokens: ops.Tensor,
    coordinates: ops.Tensor,
    destinations: tuple[ops.Tensor, ...],
    visible: tuple[ops.Tensor, ...],
    attention_state: tuple[ops.Tensor, ...],
    convolution_state: tuple[ops.Tensor, ...],
    delta_state: tuple[ops.Tensor, ...],
    recurrent_offsets: ops.Tensor | None,
    weights: DecoderTensors,
    *,
    sequence_count: int = 1,
    recurrent_sequence_length: int | None = None,
    output_rows: ops.Tensor | None = None,
    feature_values: ops.Tensor | None = None,
    feature_rows: ops.Tensor | None = None,
):
    """One complete decoder specialization with explicit logical state boundaries."""

    # Retain the residual stream in FP32; only normalized operator inputs are
    # published in the compact activation dtype used by projections and KV.
    hidden = ops.cast(ops.embedding(tokens, weights.embedding), ops.DType.F32)
    if feature_values is not None or feature_rows is not None:
        if feature_values is None or feature_rows is None:
            raise ValueError("conditioned rows require both values and destinations")
        hidden = ops.overlay_rows(hidden, ops.cast(feature_values, hidden.dtype), feature_rows)
    attention_index = 0
    recurrent_index = 0
    next_attention = []
    next_convolution = []
    next_delta = []
    for layer_index, layer in enumerate(weights.blocks):
        state_only = output_rows is None and layer_index + 1 == len(weights.blocks)
        if state_only:
            normalized = ops.rms_norm(
                hidden,
                layer.input_norm,
                epsilon=layer.epsilon,
                output_dtype=layer.mixer.output.dtype,
            )
            if isinstance(layer.mixer, AttentionTensors):
                _, _, history = _attention_state(
                    normalized,
                    coordinates,
                    attention_state[attention_index],
                    destinations[attention_index],
                    visible[attention_index],
                    layer.mixer,
                    sequence_count,
                    attend=False,
                )
                next_attention.append(history)
                attention_index += 1
            else:
                _, _, next_conv, next_recurrent = _recurrent_state(
                    normalized,
                    convolution_state[recurrent_index],
                    delta_state[recurrent_index],
                    cast(ops.Tensor, recurrent_offsets),
                    layer.mixer,
                    sequence_length=recurrent_sequence_length,
                )
                next_convolution.append(next_conv)
                next_delta.append(next_recurrent)
                recurrent_index += 1
            break
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
                recurrent_sequence_length=recurrent_sequence_length,
            )
            hidden = result[0]
            next_convolution.append(result[1])
            next_delta.append(result[2])
            recurrent_index += 1
    states = tuple(next_attention), tuple(next_convolution), tuple(next_delta)
    if output_rows is None:
        return states
    selected = ops.take_rows(hidden, output_rows)
    selected = ops.rms_norm(selected, weights.output_norm, epsilon=weights.epsilon)
    logits = ops.linear(selected, weights.readout, output_dtype=ops.DType.F32)
    return logits, *states
