#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Independent float32 reference for `llama` GGUFs of the MiniCPM5 shape (plain GQA Llama).

Model definition: transformers 5.17.0 `LlamaForCausalLM` (`models/llama/modeling_llama.py`) with the
config of `openbmb/MiniCPM5-2B` (no rope scaling, no biases, untied head). Only that feature set is
admitted, as the engine's strict `llama` family admits it.

Converter transform undone: llama.cpp's `LlamaModel.permute` reorders each head's q/k rows from
rotate-half order into adjacent pairs (for its NORM rope); the reference restores the checkpoint
order and applies transformers' rotate-half rope.
"""
from __future__ import annotations

import torch

from reference_model import (Reference, Rotary, Weights, attention, default_inv_freq, heads, layer_count, linear,
                             rms_norm, silu, unpermute_rows, visibility)
from reference_gguf import Package

REJECTED_KEYS = ("rope.scaling.type", "rope.scaling.factor", "attention.sliding_window", "expert_count")


class LlamaReference(Reference):
    architecture = "llama"

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        for key in REJECTED_KEYS:
            if f"llama.{key}" in package.metadata:
                raise ValueError(f"llama.{key} is outside the admitted MiniCPM5 feature set")
        self.block_count = layer_count(package, layers)
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_head_count = package.key("attention.head_count_kv")
        self.head_dim = package.key("attention.key_length", self.hidden_size // self.head_count)
        if package.key("attention.value_length", self.head_dim) != self.head_dim:
            raise ValueError("value_length differs from key_length")
        if package.key("rope.dimension_count", self.head_dim) != self.head_dim:
            raise ValueError("partial rotary is outside the admitted feature set")
        self.rope_base = package.key("rope.freq_base")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.intermediate_size = package.key("feed_forward_length")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")

    def q_weight(self, layer: int) -> torch.Tensor:
        return unpermute_rows(self.weights(f"blk.{layer}.attn_q.weight"), self.head_count)

    def k_weight(self, layer: int) -> torch.Tensor:
        return unpermute_rows(self.weights(f"blk.{layer}.attn_k.weight"), self.kv_head_count)

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
            p = f"blk.{layer}."
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            q = rotary.apply(heads(linear(h, self.q_weight(layer)), self.head_count))
            k = rotary.apply(heads(linear(h, self.k_weight(layer)), self.kv_head_count))
            v = heads(linear(h, w(p + "attn_v.weight")), self.kv_head_count)
            x = x + linear(attention(q, k, v, visible, self.head_dim ** -0.5), w(p + "attn_output.weight"))
            h = rms_norm(x, w(p + "ffn_norm.weight"), self.eps)
            gated = silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight"))
            x = x + linear(gated, w(p + "ffn_down.weight"))
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        return {
            "model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size, "num_hidden_layers": self.block_count,
            "num_attention_heads": self.head_count, "num_key_value_heads": self.kv_head_count,
            "head_dim": self.head_dim, "rms_norm_eps": self.eps, "vocab_size": self.vocab_size,
            "max_position_embeddings": self.context_length, "tie_word_embeddings": self.tied,
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_base},
            "attention_bias": False, "mlp_bias": False, "hidden_act": "silu",
        }

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.embed_tokens.weight": w("token_embd.weight"), "model.norm.weight": w("output_norm.weight")}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            parameters |= {
                h + "input_layernorm.weight": w(p + "attn_norm.weight"),
                h + "post_attention_layernorm.weight": w(p + "ffn_norm.weight"),
                h + "self_attn.q_proj.weight": self.q_weight(layer),
                h + "self_attn.k_proj.weight": self.k_weight(layer),
                h + "self_attn.v_proj.weight": w(p + "attn_v.weight"),
                h + "self_attn.o_proj.weight": w(p + "attn_output.weight"),
                h + "mlp.gate_proj.weight": w(p + "ffn_gate.weight"),
                h + "mlp.up_proj.weight": w(p + "ffn_up.weight"),
                h + "mlp.down_proj.weight": w(p + "ffn_down.weight"),
            }
        return parameters


if __name__ == "__main__":
    from reference_cli import main
    main(LlamaReference)
