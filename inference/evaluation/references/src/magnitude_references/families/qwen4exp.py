"""Independent float32 reference for `qwen4exp` GGUFs (Qwen3.8 Flash Next text model).

Model definition: mlx_vlm `models/qwen4_exp/language.py` (the port of the released Transformers
code; transformers 5.17.0 has no qwen4_exp) and the model's config. It is the routed Qwen3.5 decoder
(`qwen35.py`) with four differences:

- hyper-connections: the residual is `hc_count` streams [S, H], entered as copies of the embedding.
  Every sublayer input is a mix site and every output a scatter (`Qwen4ExpGatedResidual`): with
  xn = per-stream RMSNorm(X) * hc_norm, g = sigmoid(up(silu(down(xn) / S))), the input is
  mean_s(g * xn) and, after the sublayer, X[s] += out * 2 sigmoid(inject(xn) / S)[s]. The exit is a
  mix site without inject (`hyper_connection_mixer`) followed directly by the LM head;
- the gated delta net's output gate is sigmoid(z) (`output_gate_type: sigmoid`);
- QSA selected attention (`Qwen4ExpQSAIndexer`): q = rope(RMSNorm(q_proj x)) per indexer head, raw
  keys k_proj x; block b's key is rope(RMSNorm(mean of its `compress_ratio` raw keys)) at the
  position of its first row; score(t, b) = sum_h relu(q_h . K_b) / sqrt(width) over the complete
  blocks of row t. A row with more than `top_k / compress_ratio` complete blocks attends only to the
  best of them plus its incomplete tail; ties go to the lower block (mlx leaves them unspecified);
- the n-gram input (`Qwen4ExpPLELayer`) at the head of the `ple.layers` blocks, before the mix site:
  per position, 16 table rows hashed from the token and its predecessors (`_shift_right_ignore_eos`:
  a predecessor at or before the latest reset token preceding the row reads as the reset token),
  gated into each stream by sign(g) sqrt(max(|g|, 1e-6)) with g = <RMSNorm(key e)[s],
  RMSNorm(X[s])> / sqrt(H), then v + silu(dilated causal depthwise conv(RMSNorm(v))).

Converter transforms undone (llama.cpp `Qwen4ExpTextModel`): every RMSNorm weight is stored 1 + w
(applied as stored; the gated delta norm is stored plain); `index_qk_proj` is split into
`indexer.{q,k}_proj`; `ple.conv1d` is the squeezed [C, K] kernel; the n-gram table's shards are one
`per_layer_token_embd` table; and the gated delta tensors are reordered as for `qwen35`.
"""
from __future__ import annotations

import math

import torch

from magnitude_references.families.qwen35 import Qwen35Reference
from magnitude_references.gguf import Package
from magnitude_references.model import (Rotary, Weights, attention, default_inv_freq, heads, linear, products,
                                        rms_norm, sigmoid, silu, unit_selection, visibility)


def ngram_rows(tokens: torch.Tensor, context: int, heads_per_order: int, multipliers: list[int], sizes: list[int],
               offsets: list[int], reset: int) -> torch.Tensor:
    """The n-gram table rows of every position, int64 [B, T, context * heads_per_order], order-major.

    tokens [B, T] start a sequence (its predecessors are reset tokens)."""
    batch, steps = tokens.shape
    history = torch.cat((torch.full((batch, context), reset, dtype=torch.int64, device=tokens.device),
                         tokens.to(torch.int64)), dim=1)
    positions = torch.arange(history.shape[1], device=tokens.device)
    resets = torch.where(history == reset, positions, -1).cummax(dim=1).values
    # The latest reset strictly before each position: its own reset does not cut its context.
    previous = torch.cat((torch.full((batch, 1), -1, dtype=torch.int64, device=tokens.device), resets[:, :-1]), dim=1)
    in_segment = positions - (previous + 1)
    shifted = []
    for shift in range(context + 1):
        source = (positions - shift).clamp(min=0)
        value = history[:, source]
        valid = (in_segment >= shift) & (positions - shift >= 0)
        shifted.append(torch.where(valid, value, reset))
    rows = []
    for order in range(2, context + 2):
        mixed = shifted[0] * multipliers[0]
        for distance in range(1, order):
            mixed = mixed ^ (shifted[distance] * multipliers[distance])
        first = (order - 2) * heads_per_order
        for head in range(first, first + heads_per_order):
            rows.append(mixed % sizes[head] + offsets[head])
    return torch.stack(rows, dim=-1)[:, context:]


def dilated_causal_conv(u: torch.Tensor, kernel: torch.Tensor, dilation: int) -> torch.Tensor:
    """Depthwise causal conv from a zero window: out[t, c] = sum_k kernel[c, k] u[t - dilation (K-1-k), c]."""
    taps = kernel.shape[1]
    reach = dilation * (taps - 1)
    padded = torch.nn.functional.pad(u, (0, 0, reach, 0))
    return sum(padded[:, k * dilation:k * dilation + u.shape[1]] * kernel[:, k] for k in range(taps))


class Qwen4ExpReference(Qwen35Reference):
    """`qwen4exp`: routed Qwen3.5 with hyper-connections, selected attention and an n-gram input."""

    ARCHITECTURES = {"qwen4exp": True}

    @staticmethod
    def output_gate(z: torch.Tensor) -> torch.Tensor:
        return sigmoid(z)

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        key = package.key
        self.streams = key("hyper_connection.count")
        self.rank = key("hyper_connection.low_rank")
        self.indexer_heads = key("attention.indexer.head_count")
        self.indexer_width = key("attention.indexer.key_length")
        self.selected_tokens = key("attention.indexer.top_k")
        self.block_sizes = list(key("attention.compress_ratios"))
        self.ngram_layers = list(key("ple.layers"))
        self.ngram_size = key("ple.ngram_size")
        self.heads_per_ngram = key("ple.heads_per_ngram")
        self.ngram_kernel = key("ple.conv_kernel")
        self.reset = key("ple.eos_token_id")
        self.multipliers = [int(m) for m in key("ple.layer_multipliers")]
        self.head_sizes = [int(s) for s in key("ple.head_vocab_sizes")]
        self.head_offsets = [int(o) for o in key("ple.head_offsets")]
        if len(self.ngram_layers) != 1:
            raise ValueError(f"one n-gram layer expected, the header names {self.ngram_layers}")

    # hyper-connections ---------------------------------------------------------------------------

    def stream_norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Per-stream RMSNorm of [..., S, H] with a stored [S * H] weight."""
        return rms_norm(x, self.weights(name).view(self.streams, self.hidden_size), self.eps)

    def mix(self, streams: torch.Tensor, prefix: str, inject: bool):
        """A mix site over streams [B, T, S, H]: the sublayer input and, with `inject`, the scatter
        coefficients [B, T, S]."""
        w = self.weights
        normed = self.stream_norm(streams, f"{prefix}_norm.weight")
        flat = normed.flatten(-2)
        gate = sigmoid(linear(silu(linear(flat, w(f"{prefix}_down.weight")) / self.streams),
                                    w(f"{prefix}_up.weight"))).view_as(normed)
        mixed = (gate * normed).mean(-2)
        if not inject:
            return mixed, None
        return mixed, 2 * sigmoid(linear(flat, w(f"{prefix}_inject.weight")) / self.streams)

    @staticmethod
    def scatter(streams: torch.Tensor, out: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        return streams + out[..., None, :] * coefficients[..., None]

    # selected attention ------------------------------------------------------------------------

    def selection(self, layer: int, x: torch.Tensor, rotary: Rotary, positions: torch.Tensor,
                  visible: torch.Tensor) -> torch.Tensor:
        """The rows [B, T, T] each query attends: every visible row until it has more than the
        selected blocks' count of complete blocks, then the best blocks and its tail."""
        w, p = self.weights, f"blk.{layer}."
        block = self.block_sizes[layer]
        chosen = self.selected_tokens // block
        batch, steps, _ = x.shape
        complete = steps // block
        if complete <= chosen:
            return visible.expand(batch, steps, steps)
        q = rms_norm(linear(x, w(p + "indexer.q_proj.weight")).view(batch, steps, self.indexer_heads, self.indexer_width),
                     w(p + "indexer.q_norm.weight"), self.eps).transpose(1, 2)
        q = rotary.apply_partial(q)  # [B, h, T, W]
        raw = linear(x, w(p + "indexer.k_proj.weight"))  # [B, T, W]
        pooled = raw[:, :complete * block].view(batch, complete, block, self.indexer_width).mean(2)
        keys = rms_norm(pooled, w(p + "indexer.k_norm.weight"), self.eps)
        starts = Rotary(rotary.cos[positions[::block][:complete]], rotary.sin[positions[::block][:complete]])
        keys = starts.apply_partial(keys)  # [B, blocks, W]
        scores = torch.relu(products(q, keys[:, None])).sum(1) / math.sqrt(self.indexer_width)
        slots = unit_selection(scores, positions + 1, chosen, block)
        # its rows as a mask over the steps, an empty slot (-1) marking a dropped column
        rows = torch.zeros(batch, steps, steps + 1, dtype=torch.bool, device=x.device)
        return rows.scatter_(2, slots.masked_fill(slots < 0, steps), True)[..., :steps]

    def selected_attention(self, layer: int, x: torch.Tensor, rotary: Rotary, positions: torch.Tensor,
                           visible: torch.Tensor) -> torch.Tensor:
        return self.self_attention(layer, x, rotary, self.selection(layer, x, rotary, positions, visible))

    # n-gram input ------------------------------------------------------------------------------

    def ngram(self, layer: int, streams: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """The n-gram input's [B, T, S, H] addition to the streams."""
        w, p = self.weights, f"blk.{layer}."
        rows = ngram_rows(tokens, self.ngram_size - 1, self.heads_per_ngram, self.multipliers, self.head_sizes,
                          self.head_offsets, self.reset)
        embedding = w.rows("per_layer_token_embd.weight", rows).flatten(-2)  # [B, T, heads * width]
        keys = self.stream_norm(linear(embedding, w(p + "ple_key.weight")).view_as(streams), p + "ple_norm_key.weight")
        queries = self.stream_norm(streams, p + "ple_norm_query.weight")
        gate = (keys * queries).sum(-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.sign() * gate.abs().clamp(min=1e-6).sqrt()
        values = sigmoid(gate) * linear(embedding, w(p + "ple_value.weight"))[..., None, :]
        normed = self.stream_norm(values, p + "ple_norm_conv.weight").flatten(-2)
        convolved = dilated_causal_conv(normed, w(p + "ple_conv1d.weight"), self.ngram_size)
        return values + silu(convolved).view_as(values)

    # decoder -----------------------------------------------------------------------------------

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        rotary = Rotary.build(positions, default_inv_freq(self.rope_base, self.rope_dim))
        visible = visibility(positions, positions)
        streams = w.rows("token_embd.weight", tokens)[..., None, :].expand(-1, -1, self.streams, -1)
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(self.mix(streams, "output_hc", False)[0])
            p = f"blk.{layer}."
            if layer in self.ngram_layers:
                streams = streams + self.ngram(layer, streams, tokens)
            h, coefficients = self.mix(streams, p + "hc_attn", True)
            out = (self.selected_attention(layer, h, rotary, positions, visible) if self.is_attention(layer)
                   else self.delta_net(layer, h))
            streams = self.scatter(streams, out, coefficients)
            h, coefficients = self.mix(streams, p + "hc_ffn", True)
            streams = self.scatter(streams, self.feed_forward(layer, h), coefficients)
        if self.block_count in taps:
            tapped.append(self.mix(streams, "output_hc", False)[0])
        return self.mix(streams, "output_hc", False)[0], tapped

    # transformers has no qwen4_exp and its released code is mlx_vlm alone, so it has no released-code check (checks.py)
    released = "mlx_vlm"
