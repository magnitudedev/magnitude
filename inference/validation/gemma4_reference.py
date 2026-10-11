#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Independent float32 reference for `gemma4` GGUFs (Gemma 4 text model: E2B, E4B, 12B, 26B-A4B, 31B).

Model definition: transformers 5.17.0 `Gemma4ForCausalLM` (`Gemma4TextModel`) with the configs of
`google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it`:
- embedding * sqrt(hidden); per-layer inputs (PLE, when `embedding_length_per_layer_input` > 0):
  (RMSNorm_chunk(W_proj e * hidden^-0.5) + table[token] * sqrt(P)) * 2^-0.5, one P-slice per layer;
- sandwich blocks with plain-weight RMSNorms; attention scale 1.0; weighted q/k RMSNorm, unweighted
  v RMSNorm; V = K projection (before the k norm) where a full layer has no `attn_v`
  (`attention_k_eq_v`); KV-shared layers (the last `shared_kv_layers`) attend with the K/V of the
  last non-shared layer of the same type;
- sliding layers: window W (q - k < W), default rope over head_dim_swa; full layers: proportional rope
  over head_dim with the first `partial` of the pairs rotating (the partial is read from the
  converter's `rope_freqs` tensor: 1 for rotated pairs, 1e30 for the rest);
- FFN: GELU-tanh GLU (per-layer widths); 26B: dense branch + routed branch (softmax router on
  RMS(h) * scale * hidden^-0.5, top-k, renormalized, * per-expert scale; experts on their own
  pre-norm) each post-normed, summed, post-normed;
- PLE tail: h += RMSNorm(W_up(gelu(W_gate h) * ple[l])); then h *= layer_scalar;
- final RMSNorm, tied head, softcap `c * tanh(z / c)`.
"""
from __future__ import annotations

import torch

from reference_model import (Reference, Rotary, Weights, attention, default_inv_freq, gelu_tanh, heads, layer_count,
                             linear, mixture, per_layer, proportional_inv_freq, rms_norm, route_top_k, visibility)
from reference_gguf import Package


class Gemma4Reference(Reference):
    architecture = "gemma4"

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        total = package.key("block_count")
        self.block_count = layer_count(package, layers)
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = per_layer(package.key("attention.head_count_kv"), total)
        self.sliding = [bool(s) for s in per_layer(package.key("attention.sliding_window_pattern"), total)]
        self.window = package.key("attention.sliding_window")
        self.full_dim = package.key("attention.key_length")
        self.sliding_dim = package.key("attention.key_length_swa")
        if (package.key("attention.value_length"), package.key("attention.value_length_swa")) != (self.full_dim, self.sliding_dim):
            raise ValueError("value lengths differ from key lengths")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.full_base = package.key("rope.freq_base")
        self.sliding_base = package.key("rope.freq_base_swa")
        if (package.key("rope.dimension_count"), package.key("rope.dimension_count_swa")) != (self.full_dim, self.sliding_dim):
            raise ValueError("Gemma 4 ropes span the whole head (proportional on full layers)")
        self.full_partial = self.proportional_partial()
        self.shared = package.key("attention.shared_kv_layers", 0)
        self.first_shared = total - self.shared
        self.sources = {}
        for layer in range(self.first_shared, total):
            owners = [l for l in range(self.first_shared) if self.sliding[l] == self.sliding[layer]]
            self.sources[layer] = owners[-1]
        self.widths = per_layer(package.key("feed_forward_length"), total)
        self.ple = package.key("embedding_length_per_layer_input", 0)
        self.total_layers = total
        self.routed = package.key("expert_count", 0) > 0
        if self.routed:
            self.experts = package.key("expert_count")
            self.experts_used = package.key("expert_used_count")
            self.expert_size = package.key("expert_feed_forward_length")
        self.softcap = package.key("final_logit_softcapping")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")

    def proportional_partial(self) -> float:
        factors = self.weights("rope_freqs.weight")
        rotated = int((factors == 1).sum())
        if factors.shape[0] != self.full_dim // 2 or not torch.all(factors[:rotated] == 1) or not torch.all(factors[rotated:] == 1e30):
            raise ValueError("rope_freqs is not the converter's proportional-rope table (ones, then 1e30)")
        return 2 * rotated / self.full_dim

    def k_eq_v(self, layer: int) -> bool:
        return not self.weights.has(f"blk.{layer}.attn_v.weight")

    def per_layer_inputs(self, tokens: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        w, count, width = self.weights, self.total_layers, self.ple
        token_part = w.rows("per_layer_token_embd.weight", tokens) * width ** 0.5
        projected = linear(embedded, w("per_layer_model_proj.weight")) * self.hidden_size ** -0.5
        projected = rms_norm(projected.view(*tokens.shape, count, width), w("per_layer_proj_norm.weight"), self.eps)
        return (projected + token_part.view(*tokens.shape, count, width)) * 2.0 ** -0.5

    def feed_forward(self, layer: int, h1: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        x = rms_norm(h1, w(p + "ffn_norm.weight"), self.eps)
        dense = linear(gelu_tanh(linear(x, w(p + "ffn_gate.weight"))) * linear(x, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))
        if not self.routed:
            return dense
        rows = h1.reshape(-1, self.hidden_size)
        routed_input = rms_norm(rows, None, self.eps) * w(p + "ffn_gate_inp.scale") * self.hidden_size ** -0.5
        probabilities = torch.softmax(linear(routed_input, w(p + "ffn_gate_inp.weight")), dim=-1)
        selected = route_top_k(probabilities, self.experts_used)
        weights = torch.gather(probabilities, 1, selected)
        weights = weights / weights.sum(-1, keepdim=True) * w(p + "ffn_down_exps.scale")[selected]
        size = self.expert_size

        def expert(index: int, x: torch.Tensor) -> torch.Tensor:
            gate_up = linear(x, w.expert(p + "ffn_gate_up_exps.weight", index))
            return linear(gelu_tanh(gate_up[:, :size]) * gate_up[:, size:], w.expert(p + "ffn_down_exps.weight", index))

        routed = mixture(rms_norm(rows, w(p + "pre_ffw_norm_2.weight"), self.eps), selected, weights, expert)
        return (rms_norm(dense, w(p + "post_ffw_norm_1.weight"), self.eps)
                + rms_norm(routed.view(h1.shape), w(p + "post_ffw_norm_2.weight"), self.eps))

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = {
            False: Rotary.build(positions, proportional_inv_freq(self.full_base, self.full_dim, self.full_partial)),
            True: Rotary.build(positions, default_inv_freq(self.sliding_base, self.sliding_dim)),
        }
        visible = {False: visibility(positions, positions), True: visibility(positions, positions, window=self.window)}
        x = w("token_embd.weight")[tokens] * self.hidden_size ** 0.5
        ple = self.per_layer_inputs(tokens, x) if self.ple else None
        cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            p, sliding = f"blk.{layer}.", self.sliding[layer]
            h = rms_norm(x, w(p + "attn_norm.weight"), self.eps)
            q = rotary[sliding].apply(rms_norm(heads(linear(h, w(p + "attn_q.weight")), self.head_count), w(p + "attn_q_norm.weight"), self.eps))
            if layer in self.sources:
                k, v = cache[self.sources[layer]]
            else:
                raw_k = heads(linear(h, w(p + "attn_k.weight")), self.kv_heads[layer])
                raw_v = raw_k if self.k_eq_v(layer) else heads(linear(h, w(p + "attn_v.weight")), self.kv_heads[layer])
                k = rotary[sliding].apply(rms_norm(raw_k, w(p + "attn_k_norm.weight"), self.eps))
                v = rms_norm(raw_v, None, self.eps)
                cache[layer] = (k, v)
            mixed = linear(attention(q, k, v, visible[sliding], 1.0), w(p + "attn_output.weight"))
            x = x + rms_norm(mixed, w(p + "post_attention_norm.weight"), self.eps)
            x = x + rms_norm(self.feed_forward(layer, x), w(p + "post_ffw_norm.weight"), self.eps)
            if self.ple:
                gated = gelu_tanh(linear(x, w(p + "inp_gate.weight"))) * ple[:, :, layer]
                x = x + rms_norm(linear(gated, w(p + "proj.weight")), w(p + "post_norm.weight"), self.eps)
            x = x * w(p + "layer_output_scale.weight")
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self.head(hidden)
        return torch.tanh(z / self.softcap) * self.softcap

    # transformers cross-check -----------------------------------------------------------------

    def hf_config(self) -> dict:
        count = self.block_count
        swa_kv = {self.kv_heads[l] for l in range(count) if self.sliding[l]}
        full_kv = {self.kv_heads[l] for l in range(count) if not self.sliding[l]}
        base_width = self.widths[0]
        double = any(self.widths[l] == 2 * base_width for l in range(self.first_shared, count))
        if len(swa_kv) > 1 or len(full_kv) > 1:
            raise ValueError("transformers Gemma 4 takes one kv head count per layer type")
        config = {
            "model_type": "gemma4_text", "hidden_size": self.hidden_size, "intermediate_size": base_width,
            "num_hidden_layers": count, "num_attention_heads": self.head_count,
            "num_key_value_heads": swa_kv.pop(), "head_dim": self.sliding_dim, "global_head_dim": self.full_dim,
            "layer_types": ["sliding_attention" if s else "full_attention" for s in self.sliding[:count]],
            "sliding_window": self.window, "rms_norm_eps": self.eps, "vocab_size": self.vocab_size,
            "max_position_embeddings": self.context_length, "tie_word_embeddings": self.tied,
            "final_logit_softcapping": self.softcap, "hidden_activation": "gelu_pytorch_tanh",
            "attention_k_eq_v": any(self.k_eq_v(l) for l in range(min(count, self.first_shared)) if not self.sliding[l]),
            "num_kv_shared_layers": max(0, count - self.first_shared), "use_double_wide_mlp": double,
            "hidden_size_per_layer_input": self.ple, "vocab_size_per_layer_input": self.vocab_size,
            "enable_moe_block": self.routed, "attention_bias": False,
            "rope_parameters": {
                "full_attention": {"rope_type": "proportional", "rope_theta": self.full_base,
                                   "partial_rotary_factor": self.full_partial},
                "sliding_attention": {"rope_type": "default", "rope_theta": self.sliding_base}},
        }
        if full_kv:
            config["num_global_key_value_heads"] = full_kv.pop()
        if self.routed:
            config |= {"num_experts": self.experts, "top_k_experts": self.experts_used,
                       "moe_intermediate_size": self.expert_size}
        return config

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w = self.weights
        parameters = {"model.embed_tokens.weight": w("token_embd.weight"), "model.norm.weight": w("output_norm.weight")}
        if not self.tied:
            parameters["lm_head.weight"] = w("output.weight")
        if self.ple:
            # A truncated reference (--layers) keeps the first layers' P-slices of the per-layer tables.
            width = self.block_count * self.ple
            parameters |= {"model.embed_tokens_per_layer.weight": w("per_layer_token_embd.weight")[:, :width],
                           "model.per_layer_model_projection.weight": w("per_layer_model_proj.weight")[:width],
                           "model.per_layer_projection_norm.weight": w("per_layer_proj_norm.weight")}
        names = [("attn_norm", "input_layernorm"), ("post_attention_norm", "post_attention_layernorm"),
                 ("ffn_norm", "pre_feedforward_layernorm"), ("post_ffw_norm", "post_feedforward_layernorm"),
                 ("layer_output_scale", "layer_scalar"), ("attn_q", "self_attn.q_proj"), ("attn_q_norm", "self_attn.q_norm"),
                 ("attn_output", "self_attn.o_proj"), ("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"),
                 ("ffn_down", "mlp.down_proj"), ("inp_gate", "per_layer_input_gate"), ("proj", "per_layer_projection"),
                 ("post_norm", "post_per_layer_input_norm"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
                 ("attn_k_norm", "self_attn.k_norm"), ("ffn_gate_inp", "router.proj"), ("pre_ffw_norm_2", "pre_feedforward_layernorm_2"),
                 ("post_ffw_norm_1", "post_feedforward_layernorm_1"), ("post_ffw_norm_2", "post_feedforward_layernorm_2")]
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            for ours, theirs in names:
                if w.has(p + ours + ".weight"):
                    parameters[h + theirs + ("" if theirs == "layer_scalar" else ".weight")] = w(p + ours + ".weight")
            if self.routed:
                parameters[h + "router.scale"] = w(p + "ffn_gate_inp.scale")
                parameters[h + "router.per_expert_scale"] = w(p + "ffn_down_exps.scale")
                parameters[h + "experts.gate_up_proj"] = torch.stack(
                    [w.expert(p + "ffn_gate_up_exps.weight", e) for e in range(self.experts)])
                parameters[h + "experts.down_proj"] = torch.stack(
                    [w.expert(p + "ffn_down_exps.weight", e) for e in range(self.experts)])
        return parameters


if __name__ == "__main__":
    from reference_cli import main
    main(Gemma4Reference)
