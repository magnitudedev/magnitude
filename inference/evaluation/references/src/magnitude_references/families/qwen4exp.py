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
from magnitude_references.model import (Rotary, Weights, attention, default_inv_freq, heads, linear, rms_norm, silu,
                                        visibility)


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


def lower_index_top(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Indices of the `count` largest scores along the last axis, ties to the lower index."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :count]


class Qwen4ExpReference(Qwen35Reference):
    """`qwen4exp`: routed Qwen3.5 with hyper-connections, selected attention and an n-gram input."""

    ARCHITECTURES = {"qwen4exp": True}

    @staticmethod
    def output_gate(z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z)

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
        gate = torch.sigmoid(linear(silu(linear(flat, w(f"{prefix}_down.weight")) / self.streams),
                                    w(f"{prefix}_up.weight"))).view_as(normed)
        mixed = (gate * normed).mean(-2)
        if not inject:
            return mixed, None
        return mixed, 2 * torch.sigmoid(linear(flat, w(f"{prefix}_inject.weight")) / self.streams)

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
        scores = torch.relu(q @ keys[:, None].transpose(-1, -2)).sum(1) / math.sqrt(self.indexer_width)
        ends = positions + 1
        counts = ends // block  # complete blocks per row
        valid = torch.arange(complete, device=x.device)[None, :] < counts[:, None]
        scores = scores.masked_fill(~valid[None], float("-inf"))
        best = lower_index_top(scores, chosen)  # [B, T, chosen]
        token_blocks = torch.arange(steps, device=x.device) // block
        in_best = torch.zeros(batch, steps, complete + 1, dtype=torch.bool, device=x.device)
        in_best.scatter_(2, best, True)
        selected = in_best.gather(2, token_blocks.clamp(max=complete)[None, None].expand(batch, steps, steps))
        tail = (torch.arange(steps, device=x.device)[None, :] >= (counts * block)[:, None]) & visible
        sparse = (counts > chosen)[None, :, None]
        return torch.where(sparse, (selected & visible) | tail, visible.expand(batch, steps, steps))

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
        values = torch.sigmoid(gate) * linear(embedding, w(p + "ple_value.weight"))[..., None, :]
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

    # mlx_vlm cross-check (checks.py): transformers has no qwen4_exp -----------------------------

    released = "mlx_vlm"

    def check_prompts(self, tokens: torch.Tensor) -> torch.Tensor:
        """Random prompts with reset tokens placed so n-grams restart mid-sequence."""
        tokens = tokens.clone()
        tokens[:, 5::11] = self.reset
        return tokens

    def mlx_config(self) -> dict:
        """mlx_vlm `qwen4_exp` TextConfig fields; mlx derives the hash constants from them, and the
        check compares those with the header's."""
        return {
            "model_type": "qwen4_exp_text", "hidden_size": self.hidden_size, "num_hidden_layers": self.block_count,
            "num_attention_heads": self.head_count, "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim,
            "linear_num_value_heads": self.v_heads, "linear_num_key_heads": self.k_heads,
            "linear_key_head_dim": self.k_width, "linear_value_head_dim": self.v_width,
            "linear_conv_kernel_dim": self.conv_width, "num_experts": self.experts,
            "num_experts_per_tok": self.experts_used, "moe_intermediate_size": self.expert_size,
            "shared_expert_intermediate_size": self.shared_size, "rms_norm_eps": self.eps,
            "vocab_size": self.vocab_size, "max_position_embeddings": self.context_length,
            "full_attention_interval": self.interval, "hc_count": self.streams, "hc_lowrank": self.rank,
            "indexer_n_heads": self.indexer_heads, "indexer_head_dim": self.indexer_width,
            "indexer_budget": self.selected_tokens, "indexer_compress_ratio": self.block_sizes[self.interval - 1],
            "ple_layer_ids": [layer + 1 for layer in self.ngram_layers], "ngram_size": self.ngram_size,
            "heads_per_ngram": self.heads_per_ngram, "ple_conv_kernel_size": self.ngram_kernel,
            # mlx's first head size is the first prime at or above the base, and its table pads
            # the heads' rows up to a multiple: the stored table's rows, which hold them all.
            "ngram_vocab_size_base": self.head_sizes[0],
            "make_ngram_vocab_size_divisible_by": self.weights.shape("per_layer_token_embd.weight")[0],
            "split_ngram_parts": 1, "eos_token_id": self.reset, "output_gate_type": "sigmoid",
            "rope_parameters": {"type": "default", "mrope_section": [self.rope_dim // 2, 0, 0],
                                "rope_theta": self.rope_base, "partial_rotary_factor": self.rope_dim / self.head_dim},
        }

    def mlx_parameters(self) -> dict[str, torch.Tensor]:
        """The mlx_vlm LanguageModel parameters (checkpoint form: zero-centered norms)."""
        w = self.weights
        centered = lambda name: w(name) - 1  # noqa: E731

        def site(prefix: str, ours: str, inject: bool) -> dict[str, torch.Tensor]:
            out = {prefix + "hc_norm.weight": centered(ours + "_norm.weight"),
                   prefix + "input_mix_weight_down.weight": w(ours + "_down.weight"),
                   prefix + "input_mix_weight_up.weight": w(ours + "_up.weight")}
            if inject:
                out[prefix + "block_inject_weight.weight"] = w(ours + "_inject.weight")
            return out

        parameters = {"model.embed_tokens.weight": w("token_embd.weight"), "lm_head.weight": w("output.weight")}
        parameters |= site("model.hyper_connection_mixer.", "output_hc", False)
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"model.layers.{layer}."
            parameters |= site(h + "attn_hyper_connection.", p + "hc_attn", True)
            parameters |= site(h + "mlp_hyper_connection.", p + "hc_ffn", True)
            if self.is_attention(layer):
                a = h + "self_attn."
                parameters |= {a + "q_proj.weight": w(p + "attn_q.weight"), a + "k_proj.weight": w(p + "attn_k.weight"),
                               a + "v_proj.weight": w(p + "attn_v.weight"),
                               a + "o_proj.weight": w(p + "attn_output.weight"),
                               a + "q_norm.weight": centered(p + "attn_q_norm.weight"),
                               a + "k_norm.weight": centered(p + "attn_k_norm.weight"),
                               a + "indexer.index_qk_proj.weight": torch.cat(
                                   (w(p + "indexer.q_proj.weight"), w(p + "indexer.k_proj.weight"))),
                               a + "indexer.q_layernorm.weight": centered(p + "indexer.q_norm.weight"),
                               a + "indexer.k_layernorm.weight": centered(p + "indexer.k_norm.weight")}
            else:
                a = h + "linear_attn."
                parameters |= {
                    a + "in_proj_qkv.weight": self.qkv_weight(layer, "attn_qkv.weight"),
                    a + "in_proj_z.weight": self.z_weight(layer),
                    a + "in_proj_b.weight": self.head_rows(layer, "ssm_beta.weight"),
                    a + "in_proj_a.weight": self.head_rows(layer, "ssm_alpha.weight"),
                    a + "conv1d.weight": self.qkv_weight(layer, "ssm_conv1d.weight")[:, :, None],
                    a + "A_log": torch.log(-self.head_rows(layer, "ssm_a")),
                    a + "dt_bias": self.head_rows(layer, "ssm_dt.bias"),
                    a + "norm.weight": w(p + "ssm_norm.weight"),
                    a + "out_proj.weight": self.out_weight(layer),
                }
            if layer in self.ngram_layers:
                n = h + "ple."
                parameters |= {
                    n + "ple_embedding.ngram_embedding.shards.0.weight": w("per_layer_token_embd.weight"),
                    n + "key_proj.weight": w(p + "ple_key.weight"), n + "value_proj.weight": w(p + "ple_value.weight"),
                    n + "norm_key.weight": centered(p + "ple_norm_key.weight"),
                    n + "norm_query.weight": centered(p + "ple_norm_query.weight"),
                    n + "norm_conv.weight": centered(p + "ple_norm_conv.weight"),
                    n + "conv1d.weight": w(p + "ple_conv1d.weight")[:, :, None],
                }
            m = h + "mlp."
            parameters |= {
                m + "gate.weight": w(p + "ffn_gate_inp.weight"),
                m + "switch_mlp.gate_proj.weight": torch.stack(
                    [w.expert(p + "ffn_gate_exps.weight", e) for e in range(self.experts)]),
                m + "switch_mlp.up_proj.weight": torch.stack(
                    [w.expert(p + "ffn_up_exps.weight", e) for e in range(self.experts)]),
                m + "switch_mlp.down_proj.weight": torch.stack(
                    [w.expert(p + "ffn_down_exps.weight", e) for e in range(self.experts)]),
                m + "shared_expert.gate_proj.weight": w(p + "ffn_gate_shexp.weight"),
                m + "shared_expert.up_proj.weight": w(p + "ffn_up_shexp.weight"),
                m + "shared_expert.down_proj.weight": w(p + "ffn_down_shexp.weight"),
                m + "shared_expert_gate.weight": w(p + "ffn_gate_inp_shexp.weight")[None, :],
            }
        return parameters

    def mlx_constants(self) -> dict[str, list[int]]:
        """The hash constants mlx derives, by parameter name, which must equal the header's."""
        layer = self.ngram_layers[0]
        e = f"model.layers.{layer}.ple.ple_embedding."
        return {e + "layer_multipliers": self.multipliers, e + "ngram_heads_vocab_sizes": self.head_sizes,
                e + "ngram_heads_offsets": self.head_offsets}
