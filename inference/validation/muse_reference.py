#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Independent float32 reference for `muse-glimmer` GGUFs (text model of Muse Glimmer).

Model definition: transformers 5.17.0 `MuseGlimmerForConditionalGeneration` text path
(`MuseGlimmerTextModel` + `lm_head`) with the config of `meta-models/Muse-Glimmer-30B`:
- embedding followed by an unweighted RMSNorm (eps `rms_norm_eps`);
- sandwich blocks: x += post_attn_norm(attn(pre_norm(x))); x += post_ffn_norm(mlp(pre_ffn_norm(x))),
  every norm centered, (1 + w); post norms use `post_norm_eps` = 1e-8 (config; not stored in the GGUF);
- attention: unweighted RMSNorm on q and k, q scaled by `qk_scale_factor`, rotate-half rope over the
  whole head on sliding layers only (`layer_rope_theta` is 0 on full layers: NoPE), per-element
  sigmoid gate from `gate_proj` on the attention input, scale head_dim^-0.5;
- SwiGLU MLP; final weighted RMSNorm; untied head; logits * `output_multiplier`, then the tanh softcap.

Converter transforms undone: q/k rows are un-permuted from llama.cpp's NORM (adjacent pair) order;
the four layer norms store w + 1 (the reference applies the stored value, which is 1 + w); the q/k
norm tensors are the converter's synthesized constants (q: `qk_scale_factor`, k: ones) and are
checked to be exactly that. Layer types come from `attention.sliding_window_pattern` (a per-layer
bool array, or a period n: layer l slides iff l % n < n - 1); full layers are NoPE.
"""
from __future__ import annotations

import torch

from reference_model import (Reference, Rotary, Weights, attention, default_inv_freq, heads, layer_count, linear,
                             per_layer, rms_norm, silu, unpermute_rows, visibility)
from reference_gguf import Package

POST_NORM_EPS = 1e-8  # MuseGlimmerTextConfig.post_norm_eps of the released config; the GGUF does not carry it


def sliding_layers(pattern, count: int) -> list[bool]:
    """llama.cpp `set_swa_pattern` semantics for a period, or the per-layer array."""
    if isinstance(pattern, list):
        return [bool(p) for p in per_layer(pattern, count)]
    return [pattern == 0 or layer % pattern < pattern - 1 for layer in range(count)]


class MuseReference(Reference):
    architecture = "muse-glimmer"

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        self.block_count = layer_count(package, layers)
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = package.key("attention.head_count_kv")
        self.head_dim = package.key("attention.key_length")
        if package.key("attention.value_length") != self.head_dim:
            raise ValueError("value_length differs from key_length")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.window = package.key("attention.sliding_window")
        self.sliding = sliding_layers(package.key("attention.sliding_window_pattern"), package.key("block_count"))[:self.block_count]
        self.rope_base = package.key("rope.freq_base")
        if package.key("rope.dimension_count", self.head_dim) != self.head_dim:
            raise ValueError("Muse rope rotates the whole head")
        self.intermediate_size = package.key("feed_forward_length")
        self.softcap = package.key("final_logit_softcapping")
        self.logit_scale = package.key("logit_scale")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")
        self.qk_scale = self.synthesized_qk_scale()

    def synthesized_qk_scale(self) -> float:
        """The converter writes q_norm = qk_scale_factor and k_norm = 1 for every layer; anything else
        is not a converted Muse checkpoint."""
        w = self.weights
        scale = float(w("blk.0.attn_q_norm.weight")[0])
        for layer in range(self.block_count):
            q, k = w(f"blk.{layer}.attn_q_norm.weight"), w(f"blk.{layer}.attn_k_norm.weight")
            if not (torch.all(q == scale) and torch.all(k == 1)):
                raise ValueError(f"blk.{layer} q/k norms are not the converter's qk_scale_factor/ones constants")
        return scale

    def attention(self, layer: int, h: torch.Tensor, rotary: Rotary, visible: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        q = rms_norm(heads(linear(h, unpermute_rows(w(p + "attn_q.weight"), self.head_count)), self.head_count), None, self.eps) * self.qk_scale
        k = rms_norm(heads(linear(h, unpermute_rows(w(p + "attn_k.weight"), self.kv_heads)), self.kv_heads), None, self.eps)
        v = heads(linear(h, w(p + "attn_v.weight")), self.kv_heads)
        if self.sliding[layer]:
            q, k = rotary.apply(q), rotary.apply(k)
        mixed = attention(q, k, v, visible, self.head_dim ** -0.5)
        return linear(mixed * torch.sigmoid(linear(h, w(p + "attn_gate.weight"))), w(p + "attn_output.weight"))

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = Rotary.build(positions, default_inv_freq(self.rope_base, self.head_dim))
        visible_full = visibility(positions, positions)
        visible_sliding = visibility(positions, positions, window=self.window)
        x = rms_norm(w("token_embd.weight")[tokens], None, self.eps)
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            p = f"blk.{layer}."
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            mixed = self.attention(layer, h, rotary, visible_sliding if self.sliding[layer] else visible_full)
            x = x + rms_norm(mixed, w(p + "post_attention_norm.weight"), POST_NORM_EPS)
            h = rms_norm(x, w(p + "ffn_norm.weight"), self.eps)
            f = linear(silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
            x = x + rms_norm(f, w(p + "post_ffw_norm.weight"), POST_NORM_EPS)
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self.head(hidden) * self.logit_scale
        return torch.tanh(z / self.softcap) * self.softcap

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        """The multimodal config with a one-layer vision tower; the text-only forward never runs it."""
        return {"model_type": "muse_glimmer", "text_config": self.hf_text_config(), "tie_word_embeddings": self.tied,
                "out_hidden_size": 64,
                "projector_hidden_size": 64,
                "vision_config": {"num_hidden_layers": 1, "hidden_size": 32, "intermediate_size": 64,
                                  "num_attention_heads": 2}}

    def hf_build(self, config):
        from transformers import MuseGlimmerForConditionalGeneration
        return MuseGlimmerForConditionalGeneration._from_config(config, dtype=torch.float32, attn_implementation="eager")

    def hf_unused_prefixes(self) -> tuple[str, ...]:
        return ("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")

    def hf_text_config(self) -> dict:
        return {
            "model_type": "muse_glimmer_text", "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size, "num_hidden_layers": self.block_count,
            "num_attention_heads": self.head_count, "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim,
            "rms_norm_eps": self.eps, "post_norm_eps": POST_NORM_EPS, "vocab_size": self.vocab_size,
            "max_position_embeddings": self.context_length, "tie_word_embeddings": self.tied,
            "sliding_window": self.window, "hidden_activation": "silu", "attention_bias": False,
            "layer_types": ["sliding_attention" if s else "full_attention" for s in self.sliding],
            "layer_rope_theta": [self.rope_base if s else 0 for s in self.sliding],
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_base},
            "qk_scale_factor": self.qk_scale, "output_multiplier": self.logit_scale,
            "final_logit_softcapping": self.softcap,
        }

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.language_model.embed_tokens.weight": w("token_embd.weight"),
                      "model.language_model.norm.weight": w("output_norm.weight")}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.language_model.layers.{layer}."
            parameters |= {
                h + "input_layernorm.weight": w(p + "attn_norm.weight") - 1,
                h + "post_attention_layernorm.weight": w(p + "post_attention_norm.weight") - 1,
                h + "pre_feedforward_layernorm.weight": w(p + "ffn_norm.weight") - 1,
                h + "post_feedforward_layernorm.weight": w(p + "post_ffw_norm.weight") - 1,
                h + "self_attn.q_proj.weight": unpermute_rows(w(p + "attn_q.weight"), self.head_count),
                h + "self_attn.k_proj.weight": unpermute_rows(w(p + "attn_k.weight"), self.kv_heads),
                h + "self_attn.v_proj.weight": w(p + "attn_v.weight"),
                h + "self_attn.o_proj.weight": w(p + "attn_output.weight"),
                h + "self_attn.gate_proj.weight": w(p + "attn_gate.weight"),
                h + "mlp.gate_proj.weight": w(p + "ffn_gate.weight"),
                h + "mlp.up_proj.weight": w(p + "ffn_up.weight"),
                h + "mlp.down_proj.weight": w(p + "ffn_down.weight"),
            }
        return parameters


if __name__ == "__main__":
    from reference_cli import main
    main(MuseReference)
