"""Upstream execution over borrowed owned weights, without a second model allocation."""

from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated, Qwen3NextSparseMoeBlock
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from magnitude_engine.components import component


def separate_projections(group):
    """Reconstruct upstream operations from encoded views, independently of grouped execution."""
    if not group.packed:
        return tuple(group.operations)
    packed = group.operations[0]
    result, offset = [], 0
    for width in group.sizes:
        module = (
            nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            if isinstance(packed, nn.QuantizedLinear)
            else nn.Linear.__new__(nn.Linear)
        )
        nn.Module.__init__(module)
        for name, value in packed.parameters().items():
            module[name] = value[offset : offset + width]
        if isinstance(packed, nn.QuantizedLinear):
            module.bits, module.group_size, module.mode = (
                packed.bits,
                packed.group_size,
                packed.mode,
            )
        module.eval()
        result.append(module)
        offset += width
    return tuple(result)


def switch_linear(projection):
    # Module initialization supplies bookkeeping only; avoid allocating random weights.
    module = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(module)
    module.weight = projection.weight
    module.scales = projection.scales
    module.biases = projection.biases
    module.group_size = projection.encoding.group_size
    module.bits = projection.encoding.bits
    module.mode = "affine"
    module.eval()
    return module


def switch_experts(operation):
    module = SimpleNamespace(
        up_proj=switch_linear(operation.weights.up),
        gate_proj=switch_linear(operation.weights.gate),
        down_proj=switch_linear(operation.weights.down),
        activation=operation.math.activation,
        training=False,
    )
    return partial(SwitchGLU.__call__, cast(Any, module))


@component("MODEL:EXPERTS:MAG:UPSTREAM_ADAPTER")
def experts(operation):
    switch = switch_experts(operation)

    def weighted(hidden, assignments, scores):
        return (switch(hidden, assignments) * scores[..., None]).sum(axis=-2)

    return weighted


@component("MODEL:QWEN35.FEEDFORWARD:MAG:UPSTREAM_ADAPTER")
def feedforward(operation):
    router, shared_gate = separate_projections(operation.routing)
    module = SimpleNamespace(
        gate=router,
        top_k=operation.top_k,
        norm_topk_prob=operation.normalize,
        switch_mlp=switch_experts(operation.experts),
        shared_expert=operation.shared,
        shared_expert_gate=shared_gate,
        sharding_group=None,
    )
    return partial(Qwen3NextSparseMoeBlock.__call__, cast(Any, module))


@component("MODEL:QWEN35.RECURRENCE:MAG:UPSTREAM_ADAPTER")
def recurrence(operation):
    g = operation.graph
    norm = Qwen3NextRMSNormGated(g.value_width, eps=g.normalize_output.eps)
    norm.weight = g.normalize_output.weight
    qkv, gate, beta, decay = separate_projections(g.projections)
    module = SimpleNamespace(
        in_proj_qkv=qkv,
        in_proj_z=gate,
        in_proj_b=beta,
        in_proj_a=decay,
        num_v_heads=g.value_heads,
        num_k_heads=g.key_heads,
        head_v_dim=g.value_width,
        head_k_dim=g.key_width,
        conv_kernel_size=g.window + 1,
        conv_dim=2 * g.key_heads * g.key_width + g.value_heads * g.value_width,
        key_dim=g.key_heads * g.key_width,
        conv1d=g.convolution,
        A_log=g.log_rates,
        dt_bias=g.time_bias,
        norm=norm,
        out_proj=g.output,
        sharding_group=None,
        training=False,
    )
    return partial(GatedDeltaNet.__call__, cast(Any, module))


@component("MODEL:EMBEDDING:MAG:UPSTREAM_ADAPTER")
def embedding(operation):
    if not hasattr(operation, "encoding"):
        module = nn.Embedding.__new__(nn.Embedding)
        nn.Module.__init__(module)
        module.weight = operation.weight
        module.eval()
        return module
    module = nn.QuantizedEmbedding.__new__(nn.QuantizedEmbedding)
    nn.Module.__init__(module)
    module.weight = operation.weight
    module.scales = operation.scales
    module.biases = operation.biases
    module.group_size = operation.encoding.group_size
    module.bits = operation.encoding.bits
    module.mode = "affine"
    module.eval()
    return module


@component("MODEL:QWEN35.ATTENTION:MAG:UPSTREAM_ADAPTER")
def attention(operation):
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention, Qwen3_5RotaryEmbedding

    if operation.positions.rotation is None:
        raise ValueError("attention reference requires canonical text rotary binding")
    module = Qwen3_5Attention.__new__(Qwen3_5Attention)
    nn.Module.__init__(module)
    module.num_key_value_heads = operation.kv_heads
    module.num_attention_heads = operation.query_heads
    module.head_dim = operation.head_width
    module.scale = operation.head_width**-0.5
    module.q_proj, module.k_proj, module.v_proj = separate_projections(operation.inputs)
    module.o_proj = operation.output
    module.q_norm, module.k_norm = operation.query_norm, operation.key_norm
    dims = operation.positions.rotation.dim
    module.rotary_emb = Qwen3_5RotaryEmbedding(
        dims,
        base=operation.positions.operation.base,
        mrope_section=[dims // 2, 0, 0],
    )
    module.eval()
    return module


def attention_projections(operation, hidden, offset, projections):
    q_proj, k_proj, v_proj = projections
    batch, count, _ = hidden.shape
    hq, hk, d = operation.query_heads, operation.kv_heads, operation.head_width
    queries, gate = mx.split(
        q_proj(hidden).reshape(batch, count, hq, 2 * d),
        2,
        axis=-1,
    )
    queries = operation.query_norm(queries).transpose(0, 2, 1, 3)
    keys = operation.key_norm(k_proj(hidden).reshape(batch, count, hk, d))
    queries, keys = operation.positions(queries, keys.transpose(0, 2, 1, 3), offset=offset)
    values = v_proj(hidden).reshape(batch, count, hk, d).transpose(0, 2, 1, 3)
    return queries, keys, values, gate.reshape(batch, count, hq * d)


def attention_core_equation(queries, keys, values):
    """Broadcast KV groups without physically repeating their history tensors."""
    batch, hq, count, d = queries.shape
    hk = keys.shape[1]
    q = queries.astype(mx.float32).reshape(batch, hk, hq // hk, count, d)
    scores = (q @ keys.astype(mx.float32)[:, :, None].swapaxes(-1, -2)) * d**-0.5
    positions = (
        mx.arange(keys.shape[2])[None, :] <= (keys.shape[2] - count + mx.arange(count))[:, None]
    )
    scores = mx.where(positions, scores, -mx.inf)
    attended = mx.softmax(scores, axis=-1, precise=True) @ values.astype(mx.float32)[:, :, None]
    return attended.reshape(batch, hq, count, d).astype(queries.dtype)


@component("MODEL:QWEN35.ATTENTION:MAG:FP32_EQUATION")
def attention_equation(operation):
    """Gated attention with the explicit FP32 softmax/reduction contract.

    Shared upstream projections/norms stay fixed while the core equation is
    independent of both paged Metal and native BF16 SDPA implementations.
    """

    projections = separate_projections(operation.inputs)

    def apply(hidden, *, cache, mask=None):
        q, k, v, gate = attention_projections(operation, hidden, cache.offset, projections)
        k, v = cache.update_and_fetch(k, v)
        attended = attention_core_equation(q, k, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(hidden.shape[0], hidden.shape[1], -1)
        return operation.output(attended * mx.sigmoid(gate))

    return apply


@component("MODEL:QWEN35.READOUT:MAG:UPSTREAM_ADAPTER")
def readout(operation):
    return operation


@component("MODEL:FORWARD:LM:STANDARD")
class LMForward:
    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
