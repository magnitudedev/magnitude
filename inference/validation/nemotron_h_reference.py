#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Independent float32 reference for `nemotron_h_moe` GGUFs (Nemotron 3 Super / Ultra, Nemotron 3.5 Lightning).

Model definition: transformers 5.17.0 `NemotronHForCausalLM` with the config of
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (and the Super/Ultra configs):
- every layer is one pre-norm residual sublayer: Mamba-2, attention, or MoE (from the per-layer
  `head_count_kv` / `feed_forward_length` arrays; trailing `nextn_predict_layers` blocks are the MTP
  head and are not part of the forward);
- Mamba-2: in_proj -> z | xBC | dt; depthwise causal conv (with bias) + SiLU over xBC; x | B | C;
  delta = softplus(dt + dt_bias); per head h (group g = h / (NH/G)):
  S <- exp(delta A) S + delta x B^T, y = S C + D x, recurrent state in float32; y * SiLU(z), then
  RMSNorm over each group of D_i/G channels (gate before norm); out_proj;
- attention: no rope, no q/k norm, GQA, scale head_dim^-0.5;
- MoE: float32 router, sigmoid, top-k on score + selection bias, weights = unbiased scores /
  (sum + 1e-20) * routed scale; experts relu(up x)^2 -> down in the latent width when
  `ffn_latent_down/up` exist (Super/Ultra), plus a relu^2 shared expert on the full width;
- final RMSNorm, untied head. NVFP4 tensors include their `.scale` (see reference_model.Weights).

Transformers is internally inconsistent about the time step: its chunked (prefill) scan clamps
delta to >= time_step_min (1e-3) while its single-token update does not; the released
`time_step_limit` is (0, inf) and llama.cpp never clamps. The reference does not clamp (see
references.md); the transformers cross-check sets the mixers' `time_step_limit` to (0, inf).
"""
from __future__ import annotations

import torch

from reference_model import (Reference, Weights, attention, heads, layer_count, linear, mixture, per_layer,
                             relu_squared, rms_norm, route_top_k, silu, softplus, visibility)
from reference_gguf import Package


def causal_conv(u: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv from a zero window: out[t, c] = bias[c] + sum_k kernel[c, k] u[t - (K-1) + k, c]."""
    taps = kernel.shape[1]
    padded = torch.nn.functional.pad(u, (0, 0, taps - 1, 0))
    return bias + sum(padded[:, k:k + u.shape[1]] * kernel[:, k] for k in range(taps))


def selective_scan(x: torch.Tensor, delta: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                   d: torch.Tensor) -> torch.Tensor:
    """Mamba-2 recurrence from a zero state, row by row in float32.

    x [B, T, NH, P], delta [B, T, NH], a/d [NH], b/c [B, T, G, N] -> y [B, T, NH, P].
    """
    batch, steps, head_count, width = x.shape
    groups = b.shape[2]
    b = b.repeat_interleave(head_count // groups, dim=2)
    c = c.repeat_interleave(head_count // groups, dim=2)
    state = torch.zeros(batch, head_count, width, b.shape[-1], dtype=x.dtype, device=x.device)
    out = []
    for t in range(steps):
        decay = torch.exp(delta[:, t] * a)[..., None, None]
        state = state * decay + (delta[:, t, :, None] * x[:, t])[..., None] * b[:, t, :, None, :]
        out.append(torch.einsum("bhpn,bhn->bhp", state, c[:, t]) + d[:, None] * x[:, t])
    return torch.stack(out, dim=1)


class NemotronHReference(Reference):
    architecture = "nemotron_h_moe"

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        self.mtp_layers = package.key("nextn_predict_layers", 0)
        main = package.key("block_count") - self.mtp_layers
        self.block_count = main if layers is None else layer_count(package, layers)
        if self.block_count > main:
            raise ValueError(f"only {main} main layers (the rest are MTP)")
        self.hidden_size = package.key("embedding_length")
        kv = per_layer(package.key("attention.head_count_kv"), main)
        ffn = per_layer(package.key("feed_forward_length"), main)
        self.kinds = ["mamba" if kv[l] == 0 and ffn[l] == 0 else "attention" if ffn[l] == 0
                      else "moe" if weights.has(f"blk.{l}.ffn_gate_inp.weight") else "mlp" for l in range(main)]
        self.kv_heads = kv
        self.head_count = package.key("attention.head_count")
        self.head_dim = package.key("attention.key_length")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.inner = package.key("ssm.inner_size")
        self.state_size = package.key("ssm.state_size")
        self.ssm_heads = package.key("ssm.time_step_rank")
        self.groups = package.key("ssm.group_count")
        self.conv_width = package.key("ssm.conv_kernel")
        self.head_width = self.inner // self.ssm_heads
        self.experts = package.key("expert_count")
        self.experts_used = package.key("expert_used_count")
        if package.key("expert_group_count", 1) != 1:
            raise ValueError("group-limited routing is outside the in-scope Nemotron models")
        if not package.key("expert_weights_norm"):
            raise ValueError("Nemotron-H routing normalizes the selected weights")
        self.routed_scale = package.key("expert_weights_scale")
        self.expert_size = package.key("expert_feed_forward_length")
        self.shared_size = package.key("expert_shared_feed_forward_length")
        self.latent = package.key("moe_latent_size", 0)
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")

    def mamba(self, layer: int, u: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = u.shape
        channels = self.inner + 2 * self.groups * self.state_size
        z, xbc, dt = torch.split(linear(u, w(p + "ssm_in.weight")), [self.inner, channels, self.ssm_heads], dim=-1)
        xbc = silu(causal_conv(xbc, w(p + "ssm_conv1d.weight"), w(p + "ssm_conv1d.bias")))
        x, b, c = torch.split(xbc, [self.inner, self.groups * self.state_size, self.groups * self.state_size], dim=-1)
        delta = softplus(dt + w(p + "ssm_dt.bias"))
        y = selective_scan(x.view(batch, steps, self.ssm_heads, self.head_width), delta, w(p + "ssm_a").reshape(-1),
                           b.view(batch, steps, self.groups, self.state_size),
                           c.view(batch, steps, self.groups, self.state_size), w(p + "ssm_d").reshape(-1))
        gated = y.reshape(batch, steps, self.inner) * silu(z)
        normed = rms_norm(gated.view(batch, steps, self.groups, -1), None, self.eps).reshape(batch, steps, self.inner)
        return linear(normed * w(p + "ssm_norm.weight").reshape(-1), w(p + "ssm_out.weight"))

    def attention(self, layer: int, u: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        q = heads(linear(u, w(p + "attn_q.weight")), self.head_count)
        k = heads(linear(u, w(p + "attn_k.weight")), self.kv_heads[layer])
        v = heads(linear(u, w(p + "attn_v.weight")), self.kv_heads[layer])
        return linear(attention(q, k, v, visible, self.head_dim ** -0.5), w(p + "attn_output.weight"))

    def moe(self, layer: int, u: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        rows = u.reshape(-1, self.hidden_size)
        scores = torch.sigmoid(linear(rows, w(p + "ffn_gate_inp.weight")))
        selected = route_top_k(scores + w(p + "exp_probs_b.bias"), self.experts_used)
        weights = torch.gather(scores, 1, selected)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * self.routed_scale
        latent = linear(rows, w(p + "ffn_latent_down.weight")) if self.latent else rows
        routed = mixture(latent, selected, weights, lambda e, x: linear(
            relu_squared(linear(x, w.expert(p + "ffn_up_exps.weight", e))), w.expert(p + "ffn_down_exps.weight", e)))
        if self.latent:
            routed = linear(routed, w(p + "ffn_latent_up.weight"))
        shared = linear(relu_squared(linear(rows, w(p + "ffn_up_shexp.weight"))), w(p + "ffn_down_shexp.weight"))
        return (routed + shared).reshape(u.shape)

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        visible = visibility(positions, positions)
        x = w("token_embd.weight")[tokens]
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            u = rms_norm(x, w(f"blk.{layer}.attn_norm.weight"), self.eps)
            kind = self.kinds[layer]
            if kind == "mamba":
                x = x + self.mamba(layer, u)
            elif kind == "attention":
                x = x + self.attention(layer, u, visible)
            elif kind == "moe":
                x = x + self.moe(layer, u)
            else:
                x = x + linear(relu_squared(linear(u, w(f"blk.{layer}.ffn_up.weight"))), w(f"blk.{layer}.ffn_down.weight"))
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        return {  # the legacy block names ("mamba", "attention", "moe", "mlp") are remapped by the config
            "model_type": "nemotron_h", "hidden_size": self.hidden_size, "layers_block_type": self.kinds[:self.block_count],
            "num_hidden_layers": self.block_count, "num_attention_heads": self.head_count,
            "num_key_value_heads": max(self.kv_heads), "head_dim": self.head_dim, "layer_norm_epsilon": self.eps,
            "vocab_size": self.vocab_size, "max_position_embeddings": self.context_length,
            "tie_word_embeddings": self.tied, "mamba_num_heads": self.ssm_heads, "mamba_head_dim": self.head_width,
            "ssm_state_size": self.state_size, "n_groups": self.groups, "conv_kernel": self.conv_width,
            "use_conv_bias": True, "mamba_hidden_act": "silu", "mlp_hidden_act": "relu2", "use_bias": False,
            "mlp_bias": False, "n_routed_experts": self.experts, "num_experts_per_tok": self.experts_used,
            "moe_intermediate_size": self.expert_size, "moe_shared_expert_intermediate_size": self.shared_size,
            "moe_latent_size": self.latent or None, "routed_scaling_factor": self.routed_scale,
            "n_group": 1, "topk_group": 1, "norm_topk_prob": True, "num_nextn_predict_layers": 0,
            "intermediate_size": self.expert_size, "chunk_size": 128,
        }

    def hf_build(self, config):
        model = super().hf_build(config)
        for block in model.model.layers:
            if hasattr(block.mixer, "time_step_limit"):
                block.mixer.time_step_limit = (0.0, float("inf"))
        return model

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.embeddings.weight": w("token_embd.weight"), "model.norm_f.weight": w("output_norm.weight")}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            parameters[h + "norm.weight"] = w(p + "attn_norm.weight")
            m, kind = h + "mixer.", self.kinds[layer]
            if kind == "mamba":
                parameters |= {m + "in_proj.weight": w(p + "ssm_in.weight"), m + "out_proj.weight": w(p + "ssm_out.weight"),
                               m + "conv1d.weight": w(p + "ssm_conv1d.weight")[:, None, :],
                               m + "conv1d.bias": w(p + "ssm_conv1d.bias"), m + "dt_bias": w(p + "ssm_dt.bias"),
                               m + "A_log": torch.log(-w(p + "ssm_a").reshape(-1)), m + "D": w(p + "ssm_d").reshape(-1),
                               m + "norm.weight": w(p + "ssm_norm.weight").reshape(-1)}
            elif kind == "attention":
                for ours, theirs in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"), ("attn_output", "o_proj")]:
                    parameters[m + f"{theirs}.weight"] = w(p + ours + ".weight")
            elif kind == "moe":
                parameters |= {m + "gate.weight": w(p + "ffn_gate_inp.weight"),
                               m + "gate.e_score_correction_bias": w(p + "exp_probs_b.bias"),
                               m + "experts.up_proj": torch.stack([w.expert(p + "ffn_up_exps.weight", e) for e in range(self.experts)]),
                               m + "experts.down_proj": torch.stack([w.expert(p + "ffn_down_exps.weight", e) for e in range(self.experts)]),
                               m + "shared_experts.up_proj.weight": w(p + "ffn_up_shexp.weight"),
                               m + "shared_experts.down_proj.weight": w(p + "ffn_down_shexp.weight")}
                if self.latent:
                    parameters[m + "fc1_latent_proj.weight"] = w(p + "ffn_latent_down.weight")
                    parameters[m + "fc2_latent_proj.weight"] = w(p + "ffn_latent_up.weight")
            else:
                parameters[m + "up_proj.weight"] = w(p + "ffn_up.weight")
                parameters[m + "down_proj.weight"] = w(p + "ffn_down.weight")
        return parameters


if __name__ == "__main__":
    from reference_cli import main
    main(NemotronHReference)
