"""Independent float32 reference for `lfm3` GGUFs (LFM3 1.7B and the LFM3-VL 2B decoder).

Model definition: `Lfm3ForCausalLM` (`modeling_lfm3.py` of `LiquidAI/LFM3-1.7B`, transformers 5.12.1)
with the released config:
- sandwich-norm blocks: x += post_norm(mixer(norm(x))), then x += post_norm(ffn(norm(x)));
- the mixer is a short convolution (`head_count_kv[l] == 0`) or GQA attention with per-head q/k
  RMSNorm, rotate-half rope, and a sigmoid gate per head on the attention output before `o_proj`
  (`gate_proj`, GGUF `attn_gate`);
- window layers (`sliding_window_pattern`) see the last `sliding_window` keys (q - k < W) and rotate
  whole heads at `rope.freq_base_swa`; full layers see every earlier key and rotate the first half of
  the pairs at frequencies over the whole head width (`partial_rotary_factor = 0.5`, stated in the
  GGUF only by `rope_freqs`: ones, then 1e30);
- short conv: in_proj -> B, C, x; u = B*x; depthwise causal conv of width L over u; y = C * conv;
  out_proj;
- FFN: SwiGLU; final RMSNorm (`output_norm`); tied head.
"""
from __future__ import annotations

import torch

from magnitude_references.families.lfm2 import short_conv
from magnitude_references.gguf import Package
from magnitude_references.model import (Reference, Rotary, Weights, attention, default_inv_freq, heads, layer_count,
                                        linear, per_layer, proportional_inv_freq, rms_norm, silu, visibility)

# `partial_rotary_factor` of the full layers in the released config.
FULL_PARTIAL = 0.5


class Lfm3Reference(Reference):
    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        self.architecture = package.architecture
        if self.architecture != "lfm3":
            raise ValueError(f"not an LFM3 package: {self.architecture}")
        self.block_count = layer_count(package, layers)
        total = package.key("block_count")
        self.hidden_size = package.key("embedding_length")
        self.head_count = package.key("attention.head_count")
        self.kv_heads = per_layer(package.key("attention.head_count_kv"), total)
        self.sliding = [bool(s) for s in per_layer(package.key("attention.sliding_window_pattern"), total)]
        self.window = package.key("attention.sliding_window")
        self.head_dim = package.key("attention.key_length")
        if (package.key("attention.value_length"), package.key("rope.dimension_count")) != (self.head_dim, self.head_dim):
            raise ValueError("LFM3 heads are rotated and valued over their whole width")
        self.rope_base = package.key("rope.freq_base")
        self.rope_base_swa = package.key("rope.freq_base_swa")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.conv_width = package.key("shortconv.l_cache")
        self.intermediate_size = package.key("feed_forward_length")
        self.context_length = package.key("context_length")
        self.vocab_size = weights.shape("token_embd.weight")[0]
        self.tied = not weights.has("output.weight")
        factors = weights("rope_freqs.weight")
        rotated = int(FULL_PARTIAL * self.head_dim) // 2
        if (factors.shape[0] != self.head_dim // 2 or not torch.all(factors[:rotated] == 1)
                or not torch.all(factors[rotated:] == 1e30)):
            raise ValueError("rope_freqs disagrees with the released partial_rotary_factor")

    def is_attention(self, layer: int) -> bool:
        return self.kv_heads[layer] > 0

    def mixer(self, layer: int, h: torch.Tensor, rotary: dict[bool, Rotary], visible: dict[bool, torch.Tensor]) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        if self.is_attention(layer):
            sliding = self.sliding[layer]
            q = rms_norm(heads(linear(h, w(p + "attn_q.weight")), self.head_count), w(p + "attn_q_norm.weight"), self.eps)
            k = rms_norm(heads(linear(h, w(p + "attn_k.weight")), self.kv_heads[layer]), w(p + "attn_k_norm.weight"), self.eps)
            v = heads(linear(h, w(p + "attn_v.weight")), self.kv_heads[layer])
            mixed = attention(rotary[sliding].apply(q), rotary[sliding].apply(k), v, visible[sliding], self.head_dim ** -0.5)
            gate = torch.sigmoid(linear(h, w(p + "attn_gate.weight")))
            batch, rows, _ = mixed.shape
            gated = (mixed.view(batch, rows, self.head_count, self.head_dim) * gate[..., None]).reshape(batch, rows, -1)
            return linear(gated, w(p + "attn_output.weight"))
        if self.sliding[layer]:
            raise ValueError(f"layer {layer} convolves but is flagged a window layer")
        b, c, x = linear(h, w(p + "shortconv.in_proj.weight")).chunk(3, dim=-1)
        convolved = short_conv(b * x, w(p + "shortconv.conv.weight"))
        return linear(c * convolved, w(p + "shortconv.out_proj.weight"))

    def feed_forward(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        return linear(silu(linear(h, w(p + "ffn_gate.weight"))) * linear(h, w(p + "ffn_up.weight")), w(p + "ffn_down.weight"))

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = {
            False: Rotary.build(positions, proportional_inv_freq(self.rope_base, self.head_dim, FULL_PARTIAL)),
            True: Rotary.build(positions, default_inv_freq(self.rope_base_swa, self.head_dim)),
        }
        visible = {False: visibility(positions, positions), True: visibility(positions, positions, window=self.window)}
        x = w("token_embd.weight")[tokens]
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(x)
            p = f"blk.{layer}."
            mixed = self.mixer(layer, rms_norm(x, w(p + "attn_norm.weight"), self.eps), rotary, visible)
            x = x + rms_norm(mixed, w(p + "post_attention_norm.weight"), self.eps)
            fed = self.feed_forward(layer, rms_norm(x, w(p + "ffn_norm.weight"), self.eps))
            x = x + rms_norm(fed, w(p + "post_ffw_norm.weight"), self.eps)
        if self.block_count in taps:
            tapped.append(x)
        return rms_norm(x, w("output_norm.weight"), self.eps), tapped
