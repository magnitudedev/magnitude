#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Independent float32 reference for `lfm2` and `lfm2moe` GGUFs (LFM2 / LFM2.5, dense and MoE).

Model definition: transformers 5.17.0 `Lfm2ForCausalLM` / `Lfm2MoeForCausalLM` with the configs of
`LiquidAI/LFM2.5-2.6B` and `LiquidAI/LFM2.5-8B-A1B`:
- pre-norm blocks; the mixer is a short convolution (`head_count_kv[l] == 0`) or GQA attention with
  per-head q/k RMSNorm and rotate-half rope over the full head;
- short conv: in_proj -> B, C, x; u = B*x; depthwise causal conv of width L over u (the state is
  the last L-1 rows of u, float32 here); y = C * conv; out_proj;
- FFN: SwiGLU; for `lfm2moe` layers >= `leading_dense_block_count` a sigmoid router with a
  selection-only expert bias, top-k, weights = unbiased sigmoid / (sum + 1e-6) * routed scale (1.0);
- final RMSNorm (GGUF `token_embd_norm`, HF `embedding_norm`, applied at the *end*), tied head.

Differences from llama.cpp (the model definition is followed): the routing renormalization is
w / (sum + 1e-6) in transformers and w / max(sum, 6.103515625e-5) in llama.cpp.
"""
from __future__ import annotations

import torch

from reference_model import (Reference, Rotary, Weights, attention, default_inv_freq, heads, layer_count,
                             linear, mixture, per_layer, rms_norm, route_top_k, silu, visibility)
from reference_gguf import Package


def short_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv from an empty (zero) window: out[t, c] = sum_k kernel[c, k] u[t - (L-1) + k, c].

    u [B, T, C] float32, kernel [C, L]; the last tap multiplies the current row.
    """
    taps = kernel.shape[1]
    padded = torch.nn.functional.pad(u, (0, 0, taps - 1, 0))
    return sum(padded[:, k:k + u.shape[1]] * kernel[:, k] for k in range(taps))


class Lfm2Reference(Reference):
    """`lfm2` (all layers dense) and `lfm2moe` (leading dense layers, then routed)."""

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        self.architecture = package.architecture
        if self.architecture not in ("lfm2", "lfm2moe"):
            raise ValueError(f"not an LFM2 package: {self.architecture}")
        if package.key("attention.sliding_window", 0):
            raise ValueError("sliding-window LFM2 attention is outside the in-scope models")
        self.block_count = layer_count(package, layers)
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = per_layer(package.key("attention.head_count_kv"), self.block_count)
        self.head_dim = self.hidden_size // self.head_count
        self.rope_base = package.key("rope.freq_base")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.conv_width = package.key("shortconv.l_cache")
        self.intermediate_size = package.key("feed_forward_length")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")
        self.routed = self.architecture == "lfm2moe"
        if self.routed:
            if package.key("expert_gating_func") != 2:
                raise ValueError("LFM2-MoE routing is sigmoid (expert_gating_func 2)")
            self.dense_layers = package.key("leading_dense_block_count")
            self.experts = package.key("expert_count")
            self.experts_used = package.key("expert_used_count")
            self.expert_size = package.key("expert_feed_forward_length")
            self.routed_scale = 1.0  # routed_scaling_factor of the released config; not stored in the GGUF
        else:
            self.dense_layers = self.block_count

    def is_attention(self, layer: int) -> bool:
        return self.kv_heads[layer] > 0

    def mixer(self, layer: int, h: torch.Tensor, rotary: Rotary, visible: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        if self.is_attention(layer):
            q = rms_norm(heads(linear(h, w(p + "attn_q.weight")), self.head_count), w(p + "attn_q_norm.weight"), self.eps)
            k = rms_norm(heads(linear(h, w(p + "attn_k.weight")), self.kv_heads[layer]), w(p + "attn_k_norm.weight"), self.eps)
            v = heads(linear(h, w(p + "attn_v.weight")), self.kv_heads[layer])
            mixed = attention(rotary.apply(q), rotary.apply(k), v, visible, self.head_dim ** -0.5)
            return linear(mixed, w(p + "attn_output.weight"))
        b, c, x = linear(h, w(p + "shortconv.in_proj.weight")).chunk(3, dim=-1)
        convolved = short_conv(b * x, w(p + "shortconv.conv.weight"))
        return linear(c * convolved, w(p + "shortconv.out_proj.weight"))

    def feed_forward(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        if layer < self.dense_layers:
            return linear(silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
        rows = h.reshape(-1, self.hidden_size)
        probabilities = torch.sigmoid(linear(rows, w(p + "ffn_gate_inp.weight")))
        selected = route_top_k(probabilities + w(p + "exp_probs_b.bias"), self.experts_used)
        weights = torch.gather(probabilities, 1, selected)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6) * self.routed_scale

        def expert(index: int, x: torch.Tensor) -> torch.Tensor:
            gate = linear(x, w.expert(p + "ffn_gate_exps.weight", index))
            up = linear(x, w.expert(p + "ffn_up_exps.weight", index))
            return linear(silu(gate) * up, w.expert(p + "ffn_down_exps.weight", index))

        return mixture(rows, selected, weights, expert).reshape(h.shape)

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = Rotary.build(positions, default_inv_freq(self.rope_base, self.head_dim))
        visible = visibility(positions, positions)
        x = w("token_embd.weight")[tokens]
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            x = x + self.mixer(layer, rms_norm(x, w(f"blk.{layer}.attn_norm.weight"), self.eps), rotary, visible)
            x = x + self.feed_forward(layer, rms_norm(x, w(f"blk.{layer}.ffn_norm.weight"), self.eps))
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("token_embd_norm.weight"), self.eps), tapped

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        config = {
            "hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.block_count, "num_attention_heads": self.head_count,
            "num_key_value_heads": max(self.kv_heads), "norm_eps": self.eps, "vocab_size": self.vocab_size,
            "max_position_embeddings": self.context_length, "tie_word_embeddings": self.tied,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_base},
            "conv_bias": False, "conv_L_cache": self.conv_width,
            "layer_types": ["full_attention" if self.is_attention(l) else "conv" for l in range(self.block_count)],
        }
        if len({kv for kv in self.kv_heads if kv}) > 1:
            raise ValueError("transformers LFM2 takes one kv head count for every attention layer")
        if self.routed:
            return config | {"model_type": "lfm2_moe", "moe_intermediate_size": self.expert_size,
                             "num_dense_layers": self.dense_layers, "num_experts": self.experts,
                             "num_experts_per_tok": self.experts_used, "use_expert_bias": True,
                             "routed_scaling_factor": self.routed_scale, "norm_topk_prob": True}
        return config | {"model_type": "lfm2", "block_auto_adjust_ff_dim": False}

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.embed_tokens.weight": w("token_embd.weight"),
                      "model.embedding_norm.weight": w("token_embd_norm.weight")}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            parameters[h + "operator_norm.weight"] = w(p + "attn_norm.weight")
            parameters[h + "ffn_norm.weight"] = w(p + "ffn_norm.weight")
            if self.is_attention(layer):
                for ours, theirs in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"),
                                     ("attn_output", "out_proj"), ("attn_q_norm", "q_layernorm"),
                                     ("attn_k_norm", "k_layernorm")]:
                    parameters[h + f"self_attn.{theirs}.weight"] = w(p + ours + ".weight")
            else:
                parameters[h + "conv.in_proj.weight"] = w(p + "shortconv.in_proj.weight")
                parameters[h + "conv.conv.weight"] = w(p + "shortconv.conv.weight")[:, None, :]
                parameters[h + "conv.out_proj.weight"] = w(p + "shortconv.out_proj.weight")
            f = h + "feed_forward."
            if layer < self.dense_layers:
                parameters |= {f + "w1.weight": w(p + "ffn_gate.weight"), f + "w3.weight": w(p + "ffn_up.weight"),
                               f + "w2.weight": w(p + "ffn_down.weight")}
            else:
                parameters[f + "gate.weight"] = w(p + "ffn_gate_inp.weight")
                parameters[f + "expert_bias"] = w(p + "exp_probs_b.bias")
                parameters[f + "experts.gate_up_proj"] = torch.stack([
                    torch.cat((w.expert(p + "ffn_gate_exps.weight", e), w.expert(p + "ffn_up_exps.weight", e)))
                    for e in range(self.experts)])
                parameters[f + "experts.down_proj"] = torch.stack(
                    [w.expert(p + "ffn_down_exps.weight", e) for e in range(self.experts)])
        return parameters

    def hf_checkpoint(self, name: str, read) -> torch.Tensor:
        """The released LFM2-MoE checkpoint stores experts one by one (`experts.<e>.w1|w2|w3`)."""
        prefix, _, fused = name.rpartition("experts.")
        if fused == "gate_up_proj":
            return torch.stack([torch.cat((read(f"{prefix}experts.{e}.w1.weight"), read(f"{prefix}experts.{e}.w3.weight")))
                                for e in range(self.experts)])
        if fused == "down_proj":
            return torch.stack([read(f"{prefix}experts.{e}.w2.weight") for e in range(self.experts)])
        return read(name)


if __name__ == "__main__":
    from reference_cli import main
    main(Lfm2Reference)
