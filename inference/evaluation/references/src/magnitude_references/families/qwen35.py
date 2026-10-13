"""Independent float32 reference for `qwen35` and `qwen35moe` GGUFs (Qwen3.5 / 3.6 / 3.8 text model:
Qwen3.5-4B/9B, Qwen3.8-27B dense; Qwen3.6-35B-A3B routed).

Model definition: transformers 5.17.0 `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`
(`Qwen3_5TextModel`, `Qwen3_5MoeTextModel`):
- layer l is full attention when (l + 1) % `full_attention_interval` == 0, otherwise a gated delta
  net (linear attention); pre-norm residual blocks, every RMSNorm zero-centered: x * (1 + w);
- gated delta net: in_proj_qkv -> depthwise causal conv (width K, no bias, from a zero window) + SiLU
  -> q, k (num_k_heads x d_k), v (num_v_heads x d_v); q, k L2-normalized (eps 1e-6), q / sqrt(d_k);
  q, k repeated per v head in grouped order (v head h reads k head h // (n_v / n_k));
  beta = sigmoid(in_proj_b x), g = -exp(A_log) * softplus(in_proj_a x + dt_bias); per v head from a
  zero state S [d_k, d_v]: S <- exp(g) S; S <- S + k (beta (v - S^T k))^T; o = S^T q (float32);
  gated RMSNorm per v head (plain weight, eps rms_norm_eps): norm(o) * w * silu(in_proj_z x); out_proj;
- attention: q_proj gives per head [query | gate]; q/k RMSNorm per head; rotate-half rope over the
  first `rope.dimension_count` dims (partial 0.25) with rope_theta; GQA, scale head_dim^-0.5; output
  * sigmoid(gate), o_proj. The interleaved multimodal rope (sections [11, 11, 10]) reduces to this for
  text, where the temporal, height and width positions are all the token position;
- FFN: SwiGLU; routed (`qwen35moe`): softmax router over all experts, top-k, renormalized; experts
  SwiGLU; plus a SwiGLU shared expert scaled by sigmoid(shared_expert_gate x);
- final RMSNorm, LM head (tied to the embedding when the GGUF has no `output`). The trailing
  `nextn_predict_layers` blocks are the MTP head and are not part of the forward.

Converter transforms undone (llama.cpp `Qwen3NextModel`, `_LinearAttentionVReorderBase`): the stored
layer, q/k and final norms are 1 + w (applied as stored); `ssm_a` is -exp(A_log); `ssm_conv1d` is the
squeezed [C, K] kernel; and with n_v > n_k every v-head-indexed tensor (v rows of `attn_qkv` and of
the conv kernel, `attn_gate` (z) rows, `ssm_beta`/`ssm_alpha` rows, `ssm_a`, `ssm_dt`, `ssm_out`
columns) is stored in tiled order (k-head fastest); the reference restores the grouped order.
"""
from __future__ import annotations

import torch

from magnitude_references.gguf import Package
from magnitude_references.model import (Reference, Rotary, Weights, attention, default_inv_freq, heads, linear,
                                        mixture, rms_norm, route_top_k, silu, softplus, visibility)


def grouped(tensor: torch.Tensor, dim: int, k_heads: int, per_k: int, width: int) -> torch.Tensor:
    """Inverse of the converter's `_reorder_v_heads`: v heads stored tiled [per_k, k_heads, width] along
    `dim` back to grouped [k_heads, per_k, width]."""
    dim %= tensor.dim()
    shape = list(tensor.shape)
    split = tensor.reshape(*shape[:dim], per_k, k_heads, width, *shape[dim + 1:])
    return split.transpose(dim, dim + 1).reshape(shape)


def causal_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv from a zero window, no bias: out[t, c] = sum_k kernel[c, k] u[t-(K-1)+k, c]."""
    taps = kernel.shape[1]
    padded = torch.nn.functional.pad(u, (0, 0, taps - 1, 0))
    return sum(padded[:, k:k + u.shape[1]] * kernel[:, k] for k in range(taps))


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


CHUNK = 64  # rows per chunk of the gated delta rule


def gated_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The gated delta rule from a zero state, in float32: per head, S <- exp(g_t) S;
    S <- S + k_t (beta_t (v_t - S^T k_t))^T; o_t = S^T q_t.

    Evaluated chunk by chunk as transformers' `torch_chunk_gated_delta_rule` does (the same recurrence
    rearranged exactly: within a chunk the delta-rule updates are one unit-lower-triangular solve,
    between chunks the state carries over), which turns T sequential steps into T / CHUNK.
    q, k [B, T, H, Dk] (already per v head), v [B, T, H, Dv], g, beta [B, T, H] -> o [B, T, H, Dv].
    """
    batch, steps, head_count, _ = k.shape
    pad = (CHUNK - steps % CHUNK) % CHUNK
    # [B, H, chunks, CHUNK, D]; padded rows have k = v = 0, beta = 0, g = 0, so they change nothing.
    q, k, v = (torch.nn.functional.pad(x.transpose(1, 2), (0, 0, 0, pad)).unflatten(2, (-1, CHUNK)) for x in (q, k, v))
    beta, g = (torch.nn.functional.pad(x.transpose(1, 2), (0, pad)).unflatten(2, (-1, CHUNK)) for x in (beta, g))
    upper = torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=q.device).triu(1)
    decay = g.cumsum(-1)  # log of the decay from the chunk start through each row
    pairwise = (decay[..., :, None] - decay[..., None, :]).masked_fill(upper, float("-inf")).exp()
    k_beta, v_beta = k * beta[..., None], v * beta[..., None]
    # Lower triangular; solved as unit triangular, so only its strictly lower part enters.
    system = (k_beta @ k.transpose(-1, -2)) * pairwise
    values = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
    decayed_keys = torch.linalg.solve_triangular(system, k_beta * decay.exp()[..., None], upper=False, unitriangular=True)
    within = (q @ k.transpose(-1, -2)) * pairwise
    q = q * decay.exp()[..., None]
    k = k * (decay[..., -1:] - decay).exp()[..., None]
    chunk_decay = decay[..., -1].exp()[..., None, None]
    state = torch.zeros(batch, head_count, k.shape[-1], v.shape[-1], dtype=v.dtype, device=v.device)
    out = []
    for c in range(q.shape[2]):
        fresh = values[:, :, c] - decayed_keys[:, :, c] @ state
        out.append(q[:, :, c] @ state + within[:, :, c] @ fresh)
        state = state * chunk_decay[:, :, c] + k[:, :, c].transpose(-1, -2) @ fresh
    return torch.stack(out, dim=2).flatten(2, 3)[:, :, :steps].transpose(1, 2)


class Qwen35Reference(Reference):
    """`qwen35` (dense SwiGLU) and `qwen35moe` (routed experts with a gated shared expert)."""

    # architecture -> routed
    ARCHITECTURES = {"qwen35": False, "qwen35moe": True}

    @staticmethod
    def output_gate(z: torch.Tensor) -> torch.Tensor:
        """The gated delta net's output gate on z."""
        return silu(z)

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        self.architecture = package.architecture
        if self.architecture not in self.ARCHITECTURES:
            raise ValueError(f"not a {type(self).__name__} package: {self.architecture}")
        self.mtp_layers = package.key("nextn_predict_layers", 0)
        main = package.key("block_count") - self.mtp_layers
        if layers is not None and not 0 < layers <= main:
            raise ValueError(f"--layers {layers} outside 1..{main} (the rest are MTP)")
        self.block_count = main if layers is None else layers
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = package.key("attention.head_count_kv")
        self.head_dim = package.key("attention.key_length")
        if package.key("attention.value_length") != self.head_dim:
            raise ValueError("value_length differs from key_length")
        self.rope_dim = package.key("rope.dimension_count")
        self.rope_base = package.key("rope.freq_base")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.interval = package.key("full_attention_interval")
        self.conv_width = package.key("ssm.conv_kernel")
        self.k_width = package.key("ssm.state_size")
        self.k_heads = package.key("ssm.group_count")
        self.v_heads = package.key("ssm.time_step_rank")
        self.v_width = package.key("ssm.inner_size") // self.v_heads
        if self.v_heads % self.k_heads:
            raise ValueError(f"{self.v_heads} value heads do not group over {self.k_heads} key heads")
        self.per_k = self.v_heads // self.k_heads
        self.routed = self.ARCHITECTURES[self.architecture]
        if self.routed:
            self.experts = package.key("expert_count")
            self.experts_used = package.key("expert_used_count")
            self.expert_size = package.key("expert_feed_forward_length")
            self.shared_size = package.key("expert_shared_feed_forward_length")
        else:
            self.intermediate_size = package.key("feed_forward_length")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")

    def is_attention(self, layer: int) -> bool:
        return (layer + 1) % self.interval == 0

    # linear attention weights in the released (grouped) order ----------------------------------

    def v_rows(self, tensor: torch.Tensor, width: int) -> torch.Tensor:
        return grouped(tensor, 0, self.k_heads, self.per_k, width)

    def qkv_weight(self, layer: int, name: str) -> torch.Tensor:
        """`attn_qkv` rows (or the conv kernel channels): q and k as stored, v restored to grouped."""
        stored = self.weights(f"blk.{layer}.{name}")
        qk = 2 * self.k_heads * self.k_width
        return torch.cat((stored[:qk], self.v_rows(stored[qk:], self.v_width)))

    def z_weight(self, layer: int) -> torch.Tensor:
        return self.v_rows(self.weights(f"blk.{layer}.attn_gate.weight"), self.v_width)

    def head_rows(self, layer: int, name: str) -> torch.Tensor:
        """Per-v-head rows (`ssm_beta`, `ssm_alpha`) or vectors (`ssm_a`, `ssm_dt.bias`), grouped."""
        stored = self.weights(f"blk.{layer}.{name}")
        return self.v_rows(stored.reshape(self.v_heads, -1), 1).reshape(stored.shape)

    def out_weight(self, layer: int) -> torch.Tensor:
        return grouped(self.weights(f"blk.{layer}.ssm_out.weight"), 1, self.k_heads, self.per_k, self.v_width)

    def delta_net(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = x.shape
        mixed = silu(causal_conv(linear(x, self.qkv_weight(layer, "attn_qkv.weight")),
                                 self.qkv_weight(layer, "ssm_conv1d.weight")))
        key_size = self.k_heads * self.k_width
        q, k, v = torch.split(mixed, [key_size, key_size, self.v_heads * self.v_width], dim=-1)
        q = l2_normalize(q.view(batch, steps, self.k_heads, self.k_width)) * self.k_width ** -0.5
        k = l2_normalize(k.view(batch, steps, self.k_heads, self.k_width))
        v = v.view(batch, steps, self.v_heads, self.v_width)
        beta = torch.sigmoid(linear(x, self.head_rows(layer, "ssm_beta.weight")))
        a = -self.head_rows(layer, "ssm_a")  # exp(A_log)
        g = -a * softplus(linear(x, self.head_rows(layer, "ssm_alpha.weight")) + self.head_rows(layer, "ssm_dt.bias"))
        q = q.repeat_interleave(self.per_k, dim=2)
        k = k.repeat_interleave(self.per_k, dim=2)
        o = gated_delta(q, k, v, g, beta)
        z = linear(x, self.z_weight(layer)).view(batch, steps, self.v_heads, self.v_width)
        o = rms_norm(o, w(p + "ssm_norm.weight"), self.eps) * self.output_gate(z)
        return linear(o.reshape(batch, steps, -1), self.out_weight(layer))

    def self_attention(self, layer: int, x: torch.Tensor, rotary: Rotary, visible: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = x.shape
        query, gate = linear(x, w(p + "attn_q.weight")).view(batch, steps, self.head_count, 2 * self.head_dim).chunk(2, -1)
        q = rms_norm(query, w(p + "attn_q_norm.weight"), self.eps).transpose(1, 2)
        k = rms_norm(heads(linear(x, w(p + "attn_k.weight")), self.kv_heads), w(p + "attn_k_norm.weight"), self.eps)
        v = heads(linear(x, w(p + "attn_v.weight")), self.kv_heads)
        mixed = attention(rotary.apply_partial(q), rotary.apply_partial(k), v, visible, self.head_dim ** -0.5)
        return linear(mixed * torch.sigmoid(gate.reshape(batch, steps, -1)), w(p + "attn_output.weight"))

    def feed_forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        if not self.routed:
            return linear(silu(linear(x, w(p + "ffn_gate.weight"))) * linear(x, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
        rows = x.reshape(-1, self.hidden_size)
        probabilities = torch.softmax(linear(rows, w(p + "ffn_gate_inp.weight")), dim=-1)
        selected = route_top_k(probabilities, self.experts_used)
        weights = torch.gather(probabilities, 1, selected)
        weights = weights / weights.sum(-1, keepdim=True)

        def expert(index: int, h: torch.Tensor) -> torch.Tensor:
            gate = linear(h, w.expert(p + "ffn_gate_exps.weight", index))
            up = linear(h, w.expert(p + "ffn_up_exps.weight", index))
            return linear(silu(gate) * up, w.expert(p + "ffn_down_exps.weight", index))

        routed = mixture(rows, selected, weights, expert)
        shared = linear(silu(linear(rows, w(p + "ffn_gate_shexp.weight"))) * linear(rows, w(p + "ffn_up_shexp.weight")),
                        w(p + "ffn_down_shexp.weight"))
        shared = shared * torch.sigmoid(rows @ w(p + "ffn_gate_inp_shexp.weight"))[:, None]
        return (routed + shared).reshape(x.shape)

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = Rotary.build(positions, default_inv_freq(self.rope_base, self.rope_dim))
        visible = visibility(positions, positions)
        x = w.rows("token_embd.weight", tokens)
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            p = f"blk.{layer}."
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            x = x + (self.self_attention(layer, h, rotary, visible) if self.is_attention(layer) else self.delta_net(layer, h))
            x = x + self.feed_forward(layer, rms_norm(x, w(p + "post_attention_norm.weight"), self.eps))
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        config = {
            "hidden_size": self.hidden_size, "num_hidden_layers": self.block_count,
            "num_attention_heads": self.head_count, "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim,
            "rms_norm_eps": self.eps, "vocab_size": self.vocab_size, "max_position_embeddings": self.context_length,
            "tie_word_embeddings": self.tied, "hidden_act": "silu", "attention_bias": False,
            "layer_types": ["full_attention" if self.is_attention(l) else "linear_attention"
                            for l in range(self.block_count)],
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_base,
                                "partial_rotary_factor": self.rope_dim / self.head_dim,
                                "mrope_section": [11, 11, 10], "mrope_interleaved": True},
            "linear_conv_kernel_dim": self.conv_width, "linear_key_head_dim": self.k_width,
            "linear_value_head_dim": self.v_width, "linear_num_key_heads": self.k_heads,
            "linear_num_value_heads": self.v_heads,
        }
        if self.routed:
            return config | {"model_type": "qwen3_5_moe_text", "num_experts": self.experts,
                             "num_experts_per_tok": self.experts_used, "moe_intermediate_size": self.expert_size,
                             "shared_expert_intermediate_size": self.shared_size}
        return config | {"model_type": "qwen3_5_text", "intermediate_size": self.intermediate_size}

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.embed_tokens.weight": w("token_embd.weight"), "model.norm.weight": w("output_norm.weight") - 1}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            parameters[h + "input_layernorm.weight"] = w(p + "attn_norm.weight") - 1
            parameters[h + "post_attention_layernorm.weight"] = w(p + "post_attention_norm.weight") - 1
            if self.is_attention(layer):
                for ours, theirs in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")]:
                    parameters[h + f"self_attn.{theirs}.weight"] = w(p + ours + ".weight")
                parameters[h + "self_attn.q_norm.weight"] = w(p + "attn_q_norm.weight") - 1
                parameters[h + "self_attn.k_norm.weight"] = w(p + "attn_k_norm.weight") - 1
            else:
                a = h + "linear_attn."
                parameters |= {
                    a + "in_proj_qkv.weight": self.qkv_weight(layer, "attn_qkv.weight"),
                    a + "in_proj_z.weight": self.z_weight(layer),
                    a + "in_proj_b.weight": self.head_rows(layer, "ssm_beta.weight"),
                    a + "in_proj_a.weight": self.head_rows(layer, "ssm_alpha.weight"),
                    a + "conv1d.weight": self.qkv_weight(layer, "ssm_conv1d.weight")[:, None, :],
                    a + "A_log": torch.log(-self.head_rows(layer, "ssm_a")),
                    a + "dt_bias": self.head_rows(layer, "ssm_dt.bias"),
                    a + "norm.weight": w(p + "ssm_norm.weight"),
                    a + "out_proj.weight": self.out_weight(layer),
                }
            m = h + "mlp."
            if not self.routed:
                parameters |= {m + "gate_proj.weight": w(p + "ffn_gate.weight"), m + "up_proj.weight": w(p + "ffn_up.weight"),
                               m + "down_proj.weight": w(p + "ffn_down.weight")}
                continue
            parameters |= {
                m + "gate.weight": w(p + "ffn_gate_inp.weight"),
                m + "experts.gate_up_proj": torch.stack([
                    torch.cat((w.expert(p + "ffn_gate_exps.weight", e), w.expert(p + "ffn_up_exps.weight", e)))
                    for e in range(self.experts)]),
                m + "experts.down_proj": torch.stack([w.expert(p + "ffn_down_exps.weight", e) for e in range(self.experts)]),
                m + "shared_expert.gate_proj.weight": w(p + "ffn_gate_shexp.weight"),
                m + "shared_expert.up_proj.weight": w(p + "ffn_up_shexp.weight"),
                m + "shared_expert.down_proj.weight": w(p + "ffn_down_shexp.weight"),
                m + "shared_expert_gate.weight": w(p + "ffn_gate_inp_shexp.weight")[None, :],
            }
        return parameters
