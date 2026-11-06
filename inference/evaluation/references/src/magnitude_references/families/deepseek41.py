"""Independent float32 reference for `deepseek41` GGUFs (DeepSeek V4.1 Flash text model).

Model definition: DeepSeek's `inference/{model.py, kernel.py, engram.py}` of
deepseek-ai/DeepSeek-V4.1-Flash-1017 at `CODE[1]` (`deepseek41_released.py` runs that code as the
check). The forward, with S hyper-connection streams X [S, H] per row:

- the embedding enters as S copies; Engram layers add to X before the block (below);
- every sublayer s reads its coefficients from the stream it is given: with
  m = (flatten(X) fn^T) / rms(flatten(X)), pre = sigmoid(m[:S] a0 + b) + eps, post = 2 sigmoid(m[S:2S] a1 + b),
  comb = Sinkhorn(softmax(m[2S:] a2 + b) + eps) over 20 normalisations. The sublayer's input is
  RMSNorm(sum_c pre'[c] X[c]) where pre' is the *previous* sublayer's pre (copy 0 before layer 0),
  and X'[k] = post[k] y + sum_j comb[j, k] X[j]. The exit collapses with the last FFN's pre, then
  RMSNorm and the head;
- attention: qr = RMSNorm(wq_a x), q = wq_b qr per head, one shared key = value kv = RMSNorm(wkv x);
  adjacent-pair rope on the last `rope_dim` dims (theta `rope.freq_base`, no YaRN, at ratio 0;
  otherwise `compress_rope_freq_base` with YaRN and no amplitude). Keys: the window (t - k < W) plus,
  at ratio r > 0, the compressed rows the row selects; softmax with a per-head sink in the
  denominator; the rope is undone on the output (conjugate rotation), then the grouped wo_a and wo_b;
- compressed rows (kv source layers; later layers read the latest source): group j of r rows is
  RMSNorm(sum_i kv_i softmax_i(gate)) per channel (r = 1: RMSNorm(wkv x)), roped at position j r,
  visible to row t once (t + 1) // r > j. Index keys RMSNorm(wk latent) roped at j r;
- selection (index source layers; later layers reuse the latest): score(t, j) =
  sum_h relu(q_h . k_j) w_h with q from qr and w = proj(x) / sqrt(heads width). The candidate source
  keeps the best `candidate_top_k_blocks` blocks of `candidate_block_size` rows (block score = best
  row, the block of the newest visible row pinned) and later index layers score only inside them.
  Row t attends its min(top_k, visible) best visible rows; ties go to the lower row (DeepSeek's
  top-k leaves them unspecified);
- MoE: s = sqrt(softplus(router x)), experts = top-k of s + bias, weights s / (sum s + 1e-20) x scale;
  SwiGLU with up clamped to +-limit and gate to <= limit, routed experts plus the shared one;
- Engram: per row, `cols` table rows hashed from the compressed ids of the n-gram ending at the row
  (a slot before the sequence reads the pad id); key_c, value = wkv(rows); X[c] += value
  sigmoid(signed sqrt(<X[c], gate_c key_c> / (rms X[c] rms key_c sqrt H))).

Every row is a function of the tokens at or before it, so a prompt computed whole, in chunks, or a
verify block row by row gives the same rows: there is no batch-level selection anywhere.

The roundings the model was trained with (`Roundings`), each off by default: the window KV to E4M3
with a UE8M0 scale per 32, compressed rows to E2M1 with an E4M3 scale per 16, indexer queries and
keys to E2M1 with a UE8M0 scale per 32 (`kernel.act_quant`, `kernel.fp4_act_quant`). The E4M3
activation rounding before the GEMMs (`kernel.act_quant` into `fp8_gemm` / `fp4_gemm`, and before
wo_a) is not model numerics and is never applied.

The file (`inference/conversion` `deepseek41.py`): llama.cpp `deepseek41` names and metadata,
checkpoint layout (no row permutations); FP8 weights are E4M3 codes with UE8M0 block exponents,
dequantized by `Weights`; the Engram gate is the stored product q_weight k_weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from magnitude_references.gguf import Package
from magnitude_references.model import (Reference, Rotary, Weights, attention, default_inv_freq, layer_count, linear,
                                        lower_index_top, mixture, per_layer, rms_norm, route_top_k, silu, softplus,
                                        visibility, yarn_inv_freq)

CODE = ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e1088993b9b82de08f59de5be5a34e42170")
SQRT_SOFTPLUS = 4  # llama.cpp `expert_gating_func`
ROUTE_EPS = 1e-20  # the routing weights' normaliser (DeepSeek's Gate: not norm_eps)
ENGRAM_CLAMP = 1e-6


# ---------------------------------------------------------------------------------------------
# Trained roundings (kernel.py), exact in float32


@dataclass(frozen=True)
class Codes:
    """A small float code: `mantissa` bits, smallest normal exponent, largest finite value."""
    mantissa: int
    min_exponent: int
    max: float


E4M3 = Codes(3, -6, 448.0)
E2M1 = Codes(1, 0, 6.0)


def round_codes(x: torch.Tensor, codes: Codes) -> torch.Tensor:
    """x (within +-codes.max) to the nearest code, ties to even."""
    _, exponent = torch.frexp(x)  # |x| in [2^(e-1), 2^e)
    step = torch.ldexp(torch.ones_like(x), torch.clamp(exponent - 1, min=codes.min_exponent) - codes.mantissa)
    return torch.round(x / step) * step


def power_of_two_ceil(x: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2 x) of positive normal x (`fast_round_scale`'s exponent arithmetic)."""
    mantissa, exponent = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), exponent - (mantissa == 0.5).to(exponent.dtype))


def block_round(x: torch.Tensor, block: int, codes: Codes, scale_of) -> torch.Tensor:
    """Quantize-dequantize the last axis of x in blocks: code(clamp(x / s)) s, s = scale_of(amax)."""
    blocks = x.unflatten(-1, (-1, block))
    scale = scale_of(blocks.abs().amax(-1, keepdim=True))
    return (round_codes(torch.clamp(blocks / scale, -codes.max, codes.max), codes) * scale).flatten(-2)


def round_window(x: torch.Tensor) -> torch.Tensor:
    """`act_quant(x, 32, "ue8m0", inplace=True)`: E4M3 codes, scale 2^ceil(log2(max(amax, 1e-4) * f32(1/448)))."""
    return block_round(x, 32, E4M3, lambda amax: power_of_two_ceil(
        torch.maximum(amax, amax.new_tensor(1e-4)) * amax.new_tensor(1 / E4M3.max)))


def round_index(x: torch.Tensor) -> torch.Tensor:
    """`fp4_act_quant(x, 32, inplace=True)`: E2M1 codes, scale 2^ceil(log2(max(amax, 6 2^-126) * f32(1/6)))."""
    return block_round(x, 32, E2M1, lambda amax: power_of_two_ceil(
        torch.maximum(amax, amax.new_tensor(6 * 2.0 ** -126)) * amax.new_tensor(1 / E2M1.max)))


def round_compressed(x: torch.Tensor) -> torch.Tensor:
    """`fp4_act_quant(x, 16, inplace=True, scale_dtype=e4m3)`: E2M1 codes, scale E4M3(max(amax, 6 2^-9) / 6)."""
    return block_round(x, 16, E2M1, lambda amax: round_codes(
        torch.maximum(amax, amax.new_tensor(6 * 2.0 ** -9)) / E2M1.max, E4M3))


@dataclass(frozen=True)
class Roundings:
    """The roundings the model was trained with, each off by default."""
    window: bool = False
    compressed: bool = False
    index: bool = False

    @staticmethod
    def trained() -> "Roundings":
        return Roundings(True, True, True)


@dataclass
class Shared:
    """What a layer hands to the later layers that read it: the latest kv source's compressed rows
    [B, N, D] and index keys [B, N, W], the latest index source's selection [B, T, N], and the
    candidate source's blocks [B, T, N]."""
    rows: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    selected: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


class DeepSeek41Reference(Reference):
    """`deepseek41`: hyper-connections, compressed and selected attention, Engram, sqrt-softplus MoE."""

    tied = False

    def __init__(self, package: Package, weights: Weights, layers: int | None = None,
                 roundings: Roundings = Roundings()):
        super().__init__(package, weights, layers)
        key = package.key
        self.roundings = roundings
        self.architecture = package.architecture
        if self.architecture != "deepseek41":
            raise ValueError(f"not a deepseek41 package: {self.architecture}")
        self.vocab_size, self.hidden_size = weights.shape("token_embd.weight")
        self.block_count = count = layer_count(package, layers)
        self.head_count = key("attention.head_count")
        self.head_dim = key("attention.key_length")
        self.rope_dim = key("rope.dimension_count")
        self.eps = key("attention.layer_norm_rms_epsilon")
        self.window = key("attention.sliding_window")
        self.groups = key("attention.output_group_count")
        self.output_rank = key("attention.output_lora_rank")
        self.ratios = per_layer(key("attention.compress_ratios"), count)
        self.rope_base = key("rope.freq_base")
        self.compress_base = key("attention.compress_rope_freq_base")
        self.yarn = (key("rope.scaling.factor"), key("rope.scaling.original_context_length"),
                     key("rope.scaling.yarn_beta_fast"), key("rope.scaling.yarn_beta_slow"))
        self.streams = key("hyper_connection.count")
        self.sinkhorn = key("hyper_connection.sinkhorn_iterations")
        self.hc_eps = key("hyper_connection.epsilon")
        self.experts = key("expert_count")
        self.experts_used = key("expert_used_count")
        self.route_scale = key("expert_weights_scale")
        self.route_norm = bool(key("expert_weights_norm"))
        if key("expert_gating_func") != SQRT_SOFTPLUS:
            raise ValueError(f"deepseek41 routes by sqrt-softplus, the header says {key('expert_gating_func')}")
        self.clamp_routed = per_layer(key("swiglu_clamp_exp"), count)
        self.clamp_shared = per_layer(key("swiglu_clamp_shexp"), count)
        self.index_heads = key("attention.indexer.head_count")
        self.index_width = key("attention.indexer.key_length")
        self.index_top = key("attention.indexer.top_k")
        # GGUF holds no empty arrays: a slice without such layers omits the key
        self.kv_sources = [int(layer) for layer in key("attention.kv_source_layers", [])]
        self.index_sources = [int(layer) for layer in key("attention.indexer.source_layers", [])]
        self.candidate = key("attention.indexer.candidate_source_layer", -1)
        if self.candidate >= 0:
            self.candidate_blocks = key("attention.indexer.candidate_top_k_blocks")
            self.candidate_block = key("attention.indexer.candidate_block_size")
        for layer in range(count):
            if self.ratios[layer] and not (self.source(self.kv_sources, layer) is not None
                                           and self.source(self.index_sources, layer) is not None):
                raise ValueError(f"layer {layer} compresses (ratio {self.ratios[layer]}) without a kv and index source")
        self.engram_layers = [int(layer) for layer in key("ple.layers", [])]
        if self.engram_layers:
            self.ngram = key("ple.ngram_size")
            self.ngram_heads = key("ple.heads_per_ngram")
            self.multipliers = torch.tensor([int(m) for m in key("ple.layer_multipliers")]).view(len(self.engram_layers), -1)
            self.bucket_sizes = torch.tensor([int(s) for s in key("ple.head_vocab_sizes")]).view(len(self.engram_layers), -1)
            self.bucket_offsets = torch.tensor([int(o) for o in key("ple.head_offsets")]).view(len(self.engram_layers), -1)
            self.token_map = torch.tensor([int(t) for t in key("engram.token_map")])
            self.pad = int(self.token_map[key("engram.pad_token_id")])

    @staticmethod
    def source(sources: list[int], layer: int) -> int | None:
        """The latest of `sources` at or before `layer`."""
        return max((s for s in sources if s <= layer), default=None)

    # hyper-connections -------------------------------------------------------------------------

    def mixes(self, streams: torch.Tensor, prefix: str):
        """(pre [.., S], post [.., S], comb [.., S, S]) of a sublayer from its streams [.., S, H]."""
        w, s, eps = self.weights, self.streams, self.hc_eps
        flat = streams.flatten(-2)
        mixed = linear(flat, w(prefix + "_fn.weight")) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.eps)
        scale, base = w(prefix + "_scale.weight"), w(prefix + "_base.weight")
        pre = torch.sigmoid(mixed[..., :s] * scale[0] + base[:s]) + eps
        post = 2 * torch.sigmoid(mixed[..., s:2 * s] * scale[1] + base[s:2 * s])
        comb = torch.softmax((mixed[..., 2 * s:] * scale[2] + base[2 * s:]).unflatten(-1, (s, s)), dim=-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(self.sinkhorn - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def collapse(streams: torch.Tensor, pre: torch.Tensor | None) -> torch.Tensor:
        return streams[..., 0, :] if pre is None else (pre[..., None] * streams).sum(-2)

    @staticmethod
    def expand(out: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        return post[..., None] * out[..., None, :] + torch.einsum("...jk,...jd->...kd", comb, streams)

    # attention ---------------------------------------------------------------------------------

    def rope(self, x: torch.Tensor, rotary: Rotary, inverse: bool = False) -> torch.Tensor:
        """Rotate the last `rope_dim` dims of x [..., T, D]."""
        r = self.rope_dim
        return torch.cat((x[..., :-r], rotary.apply_pairs(x[..., -r:], inverse)), dim=-1)

    def rotaries(self, positions: torch.Tensor) -> dict[bool, Rotary]:
        """By `ratio > 0`: plain rope at the base theta, or YaRN at the compressed theta."""
        return {False: Rotary.build(positions, default_inv_freq(self.rope_base, self.rope_dim)),
                True: Rotary.build(positions, yarn_inv_freq(self.compress_base, self.rope_dim, *self.yarn)[0])}

    def latent(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """The compressor's complete groups [B, T // ratio, D], before rope."""
        w, p, ratio = self.weights, f"blk.{layer}.", self.ratios[layer]
        kv = linear(x, w(p + "attn_compressor_kv.weight"))
        if ratio > 1:
            groups = x.shape[1] // ratio
            gate = linear(x, w(p + "attn_compressor_gate.weight"))[:, :groups * ratio].unflatten(1, (groups, ratio))
            kv = (kv[:, :groups * ratio].unflatten(1, (groups, ratio)) * torch.softmax(gate, dim=2)).sum(2)
        return rms_norm(kv, w(p + "attn_compressor_norm.weight"), self.eps)

    def compress(self, layer: int, x: torch.Tensor, rotary: Rotary, shared: Shared) -> None:
        """A kv source's compressed rows and (if it indexes) index keys, each roped at its group's first row."""
        w, p, ratio = self.weights, f"blk.{layer}.", self.ratios[layer]
        latent = self.latent(layer, x)
        starts = torch.arange(latent.shape[1], device=x.device) * ratio
        at = Rotary(rotary.cos[starts], rotary.sin[starts])
        if layer in self.index_sources:
            keys = rms_norm(linear(latent, w(p + "indexer.attn_k.weight")), w(p + "indexer.k_norm.weight"), self.eps)
            keys = self.rope(keys, at)
            shared.index_keys = round_index(keys) if self.roundings.index else keys
        rows = self.rope(latent, at)
        shared.rows = round_compressed(rows) if self.roundings.compressed else rows

    def select(self, layer: int, x: torch.Tensor, qr: torch.Tensor, rotary: Rotary, positions: torch.Tensor,
               shared: Shared) -> None:
        """An index source's selection [B, T, N] of compressed rows per query row."""
        w, p, ratio = self.weights, f"blk.{layer}.", self.ratios[layer]
        batch, steps, _ = x.shape
        q = linear(qr, w(p + "indexer.attn_q_b.weight")).view(batch, steps, self.index_heads, self.index_width)
        q = self.rope(q.transpose(1, 2), rotary)  # [B, h, T, W]
        if self.roundings.index:
            q = round_index(q)
        head_weights = linear(x, w(p + "indexer.proj.weight")) / math.sqrt(self.index_width * self.index_heads)
        keys = shared.index_keys
        scores = (torch.relu(q @ keys[:, None].transpose(-1, -2)) * head_weights.transpose(1, 2)[..., None]).sum(1)
        counts = (positions + 1) // ratio  # visible rows of each query row
        entries = keys.shape[1]
        visible = torch.arange(entries, device=x.device)[None, :] < counts[:, None]
        scores = scores.masked_fill(~visible, float("-inf"))
        if layer == self.candidate:
            shared.candidates = self.candidate_rows(scores, counts)
        elif 0 <= self.candidate < layer:
            scores = scores.masked_fill(~shared.candidates, float("-inf"))
        top = min(self.index_top, entries)
        best = lower_index_top(scores, top)
        taken = torch.arange(top, device=x.device)[None, :] < counts.clamp(max=self.index_top)[:, None]
        chosen = torch.zeros(batch, steps, entries, dtype=torch.bool, device=x.device)
        chosen.scatter_(2, best, taken.expand(batch, steps, top))
        shared.selected = chosen & visible

    def candidate_rows(self, scores: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """`select_candidate_blocks`: the rows [B, T, N] of each query's best blocks."""
        block, entries = self.candidate_block, scores.shape[-1]
        blocks = -(-entries // block)
        padded = torch.nn.functional.pad(scores, (0, blocks * block - entries), value=float("-inf"))
        best = padded.unflatten(-1, (blocks, block)).amax(-1)
        newest = torch.div(counts - 1, block, rounding_mode="floor")
        best = best.masked_fill(torch.arange(blocks, device=scores.device)[None, :] == newest[:, None], float("inf"))
        top = lower_index_top(best, min(self.candidate_blocks, blocks))
        keep = torch.zeros_like(best, dtype=torch.bool).scatter_(-1, top, best.gather(-1, top) > float("-inf"))
        return keep.repeat_interleave(block, dim=-1)[..., :entries]

    def attention(self, layer: int, x: torch.Tensor, rotaries: dict[bool, Rotary], positions: torch.Tensor,
                  shared: Shared) -> torch.Tensor:
        w, p, ratio = self.weights, f"blk.{layer}.", self.ratios[layer]
        rotary = rotaries[ratio > 0]
        batch, steps, _ = x.shape
        qr = rms_norm(linear(x, w(p + "attn_q_a.weight")), w(p + "attn_q_a_norm.weight"), self.eps)
        q = linear(qr, w(p + "attn_q_b.weight")).view(batch, steps, self.head_count, self.head_dim)
        q = self.rope(q.transpose(1, 2), rotary)  # [B, heads, T, D]
        kv = self.rope(rms_norm(linear(x, w(p + "attn_kv.weight")), w(p + "attn_kv_a_norm.weight"), self.eps), rotary)
        if self.roundings.window:
            kv = round_window(kv)
        visible = visibility(positions, positions, window=self.window).expand(batch, steps, steps)
        if ratio:
            if layer in self.kv_sources:
                self.compress(layer, x, rotary, shared)
            if layer in self.index_sources:
                self.select(layer, x, qr, rotary, positions, shared)
            kv = torch.cat((kv, shared.rows), dim=1)
            visible = torch.cat((visible, shared.selected), dim=2)
        keys = kv[:, None]
        out = attention(q, keys, keys, visible, self.head_dim ** -0.5, w(p + "attn_sinks.weight"))
        out = self.rope(out.view(batch, steps, self.head_count, self.head_dim).transpose(1, 2), rotary, inverse=True)
        out = out.transpose(1, 2).reshape(batch, steps, self.groups, -1)
        wo_a = w(p + "attn_output_a.weight").view(self.groups, self.output_rank, -1)
        return linear(torch.einsum("btgd,grd->btgr", out, wo_a).flatten(2), w(p + "attn_output_b.weight"))

    # experts -----------------------------------------------------------------------------------

    @staticmethod
    def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, limit: float) -> torch.Tensor:
        g, u = linear(x, gate), linear(x, up)
        if limit > 0:
            g, u = g.clamp(max=limit), u.clamp(-limit, limit)
        return linear(silu(g) * u, down)

    def feed_forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        flat = x.reshape(-1, self.hidden_size)
        scores = torch.sqrt(softplus(linear(flat, w(p + "ffn_gate_inp.weight"))))
        selected = route_top_k(scores + w(p + "exp_probs_b.bias"), self.experts_used)
        weights = scores.gather(1, selected)
        if self.route_norm and self.experts_used > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + ROUTE_EPS)
        weights = weights * self.route_scale
        limit = self.clamp_routed[layer]

        def expert(index: int, rows: torch.Tensor) -> torch.Tensor:
            return self.swiglu(rows, w.expert(p + "ffn_gate_exps.weight", index), w.expert(p + "ffn_up_exps.weight", index),
                               w.expert(p + "ffn_down_exps.weight", index), limit)

        routed = mixture(flat, selected, weights, expert)
        shared = self.swiglu(flat, w(p + "ffn_gate_shexp.weight"), w(p + "ffn_up_shexp.weight"),
                             w(p + "ffn_down_shexp.weight"), self.clamp_shared[layer])
        return (routed + shared).view_as(x)

    # Engram ------------------------------------------------------------------------------------

    def ngram_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        """`NgramHashState`: table rows int64 [B, T, Engram layers, cols], order-major then head."""
        compressed = self.token_map.to(tokens.device)[tokens]
        steps = tokens.shape[1]
        shifted = [torch.cat((torch.full_like(compressed[:, :shift], self.pad), compressed[:, :steps - shift]), dim=1)
                   for shift in range(self.ngram)]  # a slot before the sequence is blocked, as every longer one
        products = torch.stack(shifted, dim=-1)[:, :, None, :] * self.multipliers.to(tokens.device)  # [B, T, E, n]
        sizes = self.bucket_sizes.to(tokens.device).unflatten(-1, (self.ngram - 1, self.ngram_heads))
        rolling, rows = products[..., 0], []
        for order in range(1, self.ngram):
            rolling = torch.bitwise_xor(rolling, products[..., order])
            rows.append(rolling[..., None] % sizes[:, order - 1])
        return torch.cat(rows, dim=-1) + self.bucket_offsets.to(tokens.device)

    def engram(self, layer: int, streams: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        w, p, s, hidden = self.weights, f"blk.{layer}.", self.streams, self.hidden_size
        kv = linear(w.rows("per_layer_token_embd.weight", rows).flatten(-2), w(p + "engram_kv.weight"))
        key, value = kv[..., :s * hidden].unflatten(-1, (s, hidden)), kv[..., s * hidden:]
        scale = torch.rsqrt(streams.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (streams * w(p + "engram_gate.weight") * key).sum(-1) * scale / math.sqrt(hidden)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(ENGRAM_CLAMP).sqrt(), dot))
        return streams + gate[..., None] * value[..., None, :]

    # decoder -----------------------------------------------------------------------------------

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotaries = self.rotaries(positions)
        streams = w.rows("token_embd.weight", tokens)[..., None, :].expand(-1, -1, self.streams, -1)
        ngrams = self.ngram_rows(tokens) if self.engram_layers else None
        shared, pre, tapped = Shared(), None, []
        for layer in range(self.block_count):
            p = f"blk.{layer}."
            if layer in self.engram_layers:
                streams = self.engram(layer, streams, ngrams[:, :, self.engram_layers.index(layer)])
            if layer in taps:
                tapped.append(streams.mean(-2))
            attention_pre, post, comb = self.mixes(streams, p + "hc_attn")
            h = rms_norm(self.collapse(streams, pre), w(p + "attn_norm.weight"), self.eps)
            streams = self.expand(self.attention(layer, h, rotaries, positions, shared), streams, post, comb)
            pre, post, comb = self.mixes(streams, p + "hc_ffn")
            h = rms_norm(self.collapse(streams, attention_pre), w(p + "ffn_norm.weight"), self.eps)
            streams = self.expand(self.feed_forward(layer, h), streams, post, comb)
        if self.block_count in taps:
            tapped.append(streams.mean(-2))
        return rms_norm(self.collapse(streams, pre), w("output_norm.weight"), self.eps), tapped

    # DeepSeek's code cross-check (deepseek41_released.py) ----------------------------------------

    released = "deepseek"
