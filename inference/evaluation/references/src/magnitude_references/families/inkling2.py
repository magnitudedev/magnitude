"""Independent float32 reference for `inkling2` GGUFs (Inkling 2 Small and looped Inkling 2 Mini / Varloop).

Model definition: Thinking Machines Lab Inkling 2 (matching llama.cpp `inkling2` + PR #32695 loop extensions):
- embedding + weighted RMSNorm (`token_embd_norm.weight`, text tokens only);
- Block Attention Residuals (`B = attn_res.block_size`, `normalize_values = True`, `eps = norm_eps`):
  depth-wise softmax attention over `[*blocks, cur]` with RMS-normalized keys and values, scaled by `hidden^-0.5`;
  on `ip % B == 0`, the pre-attention `cur` is pushed to `blocks` and the attention output starts a fresh `cur`;
  otherwise `cur = cur + attn_out`. FFN always adds `cur = cur + ffn_out`;
- optional looped core (`num_loops > 1`, `[core_start, core_end)`):
  each repeat pass resets `(blocks, cur)` to the prelude snapshot, banks the previous pass's core blocks
  `blocks[len(prelude_blocks)+1:]` + `bank_last = cur`, mixes cross-pass messages with `loop_res_query` /
  `loop_kv_res_query`, and modulates Q/K/V/MLP inputs via low-rank `(down, gain, forget)` message controllers:
  `(1 - tanh(F h)) * x + tanh(G h) * msg` where `h = D [RMS(x); RMS(msg)]`;
- NoPE attention with per-head L2-clipped Q (`attn_q_clip`), K (unscaled), R (`attn_rel_clip`), unweighted V-RMSNorm,
  global `tau = 1 + alpha * log(max(1, (pos + 1) / floor))` scaling on Q and R for full-attention layers,
  relative-position bias `R @ rel_bias[q - k]` (0 for `q - k >= extent`), per-token dynamic sinks (`attn_sink_proj`),
  elementwise sigmoid output gate (`attn_gate`), and causal short convolutions (`attn_k_conv`, `attn_v_conv`,
  `attn_post_conv`, `ffn_post_conv`) with per-head input-dependent dynamic taps on full-attention K/V;
- MoE FFN: sigmoid router + optional selection bias `exp_probs_b.bias`, top-k, `w / (sum + 1e-20) * route_scale`,
  SITU activation `softcap(gate_cap, g) * sigmoid(g) * softcap(up_cap, u)`;
- exit: Block AttnRes (`output_res_query`) -> `output_norm` -> `output` -> optional softcap -> slice `:vocab_unpadded`.
"""
from __future__ import annotations

import math

import torch

from magnitude_references.gguf import Package
from magnitude_references.model import (
    F32,
    Reference,
    Weights,
    heads,
    layer_count,
    linear,
    mixture,
    per_layer,
    rms_norm,
    route_top_k,
    sigmoid,
    visibility,
)

CLIP_EPS = 1e-6
ROUTE_EPS = 1e-20


def l2_clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float) -> torch.Tensor:
    """Per-head L2 norm clipping on `[B, H, T, D]`: optional `[H, D]` scale, then clamp ||v||_2 <= threshold."""
    if scale is not None:
        x = x * scale[None, :, None, :]
    norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True)) + CLIP_EPS
    factor = torch.clamp(threshold / norm, max=1.0)
    return x * factor


def block_attn_res_mix(
    blocks: list[torch.Tensor],
    cur: torch.Tensor,
    queries: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Block AttnRes mix with `normalize_values = True`.

    `blocks`: list of `[B, T, H]` tensors; `cur`: `[B, T, H]`; `queries`: `[Q, H]` or `[H]`.
    Returns `[Q, B, T, H]` if `queries` is 2-D, or `[B, T, H]` if `queries` is 1-D.
    """
    squeeze = queries.ndim == 1
    q = queries.unsqueeze(0) if squeeze else queries  # [Q, H]
    states = torch.stack([*blocks, cur], dim=0)  # [S, B, T, H]
    normed = rms_norm(states, None, eps)  # [S, B, T, H]
    scale = cur.shape[-1] ** -0.5
    scores = torch.einsum("sbth,qh->qsbt", normed, q) * scale  # [Q, S, B, T]
    weights = torch.softmax(scores, dim=1)  # [Q, S, B, T]
    out = torch.einsum("qsbt,sbth->qbth", weights, normed)  # [Q, B, T, H]
    return out[0] if squeeze else out


def causal_short_conv(
    x: torch.Tensor,
    static_w: torch.Tensor,
    residual_scale: float,
    dynamic_taps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal short convolution with residual add, optional dynamic taps, and scaling.

    `x`: `[B, T, C]`; `static_w`: `[C, K]` (tap 0 = oldest, tap K-1 = current);
    `dynamic_taps`: optional `[B, T, H, K]` (lag 0 = current, lag K-1 = oldest).
    """
    bsz, seqlen, channels = x.shape
    k = static_w.shape[1]
    padded = torch.nn.functional.pad(x, (0, 0, k - 1, 0))  # [B, T + K - 1, C]
    out = x.clone()
    for tap in range(k):
        window = padded[:, tap:tap + seqlen, :]  # [B, T, C]
        out = out + window * static_w[:, tap]
        if dynamic_taps is not None:
            lag = k - 1 - tap
            n_heads = dynamic_taps.shape[2]
            head_dim = channels // n_heads
            tap_w = dynamic_taps[:, :, :, lag].unsqueeze(-1)  # [B, T, H, 1]
            out = out + (window.view(bsz, seqlen, n_heads, head_dim) * tap_w).reshape(bsz, seqlen, channels)
    return out * residual_scale


def situ(gate: torch.Tensor, up: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    """SITU gated activation: `softcap(gate_cap, g) * sigmoid(g) * softcap(up_cap, u)`."""
    g_cap = gate_cap * torch.tanh(gate / gate_cap)
    u_cap = up_cap * torch.tanh(up / up_cap)
    return g_cap * sigmoid(gate) * u_cap


def inkling2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    rel_bias: torch.Tensor,
    sinks: torch.Tensor,
    visible: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Float32 NoPE GQA attention with relative-position bias and per-token dynamic sinks.

    `q`: `[B, H, T, D]`, `k`/`v`: `[B, H_kv, T, D]`, `r`: `[B, H, T, d_rel]`,
    `rel_bias`: `[extent, d_rel]`, `sinks`: `[B, T, H]`, `visible`: `[T, T]` bool.
    Returns `[B, T, H * D]`.
    """
    bsz, n_head, seqlen, head_dim = q.shape
    n_kv = k.shape[1]
    group = n_head // n_kv
    k_exp = k.repeat_interleave(group, dim=1)  # [B, H, T, D]
    v_exp = v.repeat_interleave(group, dim=1)  # [B, H, T, D]

    # Raw QK scores: [B, H, T_q, T_k]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k_exp) * scale

    # Relative bias: [B, H, T_q, extent + 1] gathered by clamp(q_pos - k_pos, 0, extent)
    extent = rel_bias.shape[0]
    padded_bias = torch.cat((rel_bias, rel_bias.new_zeros((1, rel_bias.shape[1]))), dim=0)  # [extent + 1, d_rel]
    table = torch.einsum("bhqr,er->bhqe", r, padded_bias)  # [B, H, T_q, extent + 1]
    pos = torch.arange(seqlen, device=q.device)
    dist = torch.clamp(pos[:, None] - pos[None, :], min=0, max=extent)  # [T_q, T_k]
    bias = table[:, :, pos[:, None], dist]  # [B, H, T_q, T_k]
    scores = scores + bias

    scores = scores.masked_fill(~visible[None, None, :, :], float("-inf"))
    sink = sinks.transpose(1, 2).unsqueeze(-1)  # [B, H, T_q, 1]
    maximum = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
    weights = torch.exp(scores - maximum)
    denom = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
    probs = weights / denom
    out = torch.einsum("bhqk,bhkd->bqhd", probs, v_exp)
    return out.reshape(bsz, seqlen, n_head * head_dim)


class Inkling2Reference(Reference):
    architecture = "inkling2"

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        total = package.key("block_count")
        self.total_layers = total
        self.block_count = layer_count(package, layers)
        self.hidden_size = package.key("embedding_length")
        self.eps = package.key("attention.layer_norm_rms_epsilon")
        self.qk_clip = package.key("attention.qk_clip")
        self.attn_res_block_size = package.key("attn_res.block_size")
        self.conv_k = package.key("shortconv.l_cache")
        self.d_rel = package.key("attention.relative_length")
        self.rel_extent = package.key("attention.relative_extent")
        self.rel_clip = package.key("attention.relative_clip")
        self.log_scale_floor = package.key("attention.log_scale_floor")
        self.log_scale_alpha = package.key("attention.log_scale_alpha")
        self.vocab_padded = package.key("vocab_size")
        self.vocab_unpadded = package.key("vocab_size_unpadded")
        self.vocab_size = self.vocab_unpadded
        self.head_dim_full = package.key("attention.key_length")
        self.head_dim_swa = package.key("attention.key_length_swa", self.head_dim_full)
        self.sliding_window = package.key("attention.sliding_window")
        self.is_swa = [bool(s) for s in per_layer(package.key("attention.sliding_window_pattern"), total)]
        self.heads = per_layer(package.key("attention.head_count"), total)
        self.kv_heads = per_layer(package.key("attention.head_count_kv"), total)
        self.experts = package.key("expert_count")
        self.expert_used = per_layer(package.key("expert_used_count"), total)
        self.expert_ff = per_layer(package.key("expert_feed_forward_length"), total)
        self.expert_weights_scale = package.key("expert_weights_scale", 1.0)
        self.situ_beta = package.key("activation.situ_beta")
        self.situ_linear_beta = package.key("activation.situ_linear_beta")
        self.softcap = package.key("final_logit_softcapping", 0.0)
        self.context_length = package.key("context_length")

        # Optional looped core configuration (Inkling 2 Mini / Varloop).
        core_end = package.key("loop.core_end", 0)
        if core_end > 0:
            self.num_loops = package.key("num_loops", 1)
            self.core_start = package.key("loop.core_start")
            self.core_end = core_end
            self.n_core = self.core_end - self.core_start
        else:
            self.num_loops = 1
            self.core_start = 0
            self.core_end = 0
            self.n_core = 0

        self.static_conv_scale = 1.0 / math.sqrt(self.conv_k + 1)
        self.dynamic_conv_scale = 1.0 / math.sqrt(2 * self.conv_k + 1)

    def visit(self, il: int) -> tuple[int, int]:
        """Map logical visit `il` to `(physical_layer, pass_idx)`."""
        if self.num_loops < 2 or il < self.core_start:
            return il, 0
        core_visits = self.num_loops * self.n_core
        if il < self.core_start + core_visits:
            offset = il - self.core_start
            return self.core_start + offset % self.n_core, offset // self.n_core
        return il - (self.num_loops - 1) * self.n_core, 0

    def control(self, cur: torch.Tensor, message: torch.Tensor, ip: int, module: str) -> torch.Tensor:
        """Cross-pass message controller: `(1 - tanh(F h)) * cur + tanh(G h) * message`."""
        w, p = self.weights, f"blk.{ip}.loop_{module}_"
        h = torch.cat((rms_norm(cur, None, self.eps), rms_norm(message, None, self.eps)), dim=-1)
        h = linear(h, w(p + "down.weight"))
        gain = torch.tanh(linear(h, w(p + "gain.weight")))
        forget = torch.tanh(linear(h, w(p + "forget.weight")))
        return (1.0 - forget) * cur + gain * message

    def feed_forward(self, ip: int, il: int, h: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{ip}."
        rows = h.reshape(-1, self.hidden_size)
        probabilities = sigmoid(linear(rows, w(p + "ffn_gate_inp.weight")))
        bias_name = p + "exp_probs_b.bias"
        selection_scores = probabilities + w(bias_name) if w.has(bias_name) else probabilities
        selected = route_top_k(selection_scores, self.expert_used[il])
        weights = torch.gather(probabilities, 1, selected)
        weights = weights / (weights.sum(-1, keepdim=True) + ROUTE_EPS) * self.expert_weights_scale
        size = self.expert_ff[il]

        def expert(index: int, x: torch.Tensor) -> torch.Tensor:
            gate_up = linear(x, w.expert(p + "ffn_gate_up_exps.weight", index))
            activated = situ(gate_up[:, :size], gate_up[:, size:], self.situ_beta, self.situ_linear_beta)
            return linear(activated, w.expert(p + "ffn_down_exps.weight", index))

        return mixture(rows, selected, weights, expert).reshape(h.shape)

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        seqlen = tokens.shape[1]
        positions = torch.arange(seqlen, device=tokens.device)
        visible_full = visibility(positions, positions)
        visible_swa = visibility(positions, positions, window=self.sliding_window)

        if self.log_scale_alpha > 0.0:
            ratio = torch.clamp((positions.to(F32) + 1.0) / float(self.log_scale_floor), min=1.0)
            tau = (1.0 + self.log_scale_alpha * torch.log(ratio)).view(1, 1, seqlen, 1)
        else:
            tau = None

        cur = rms_norm(w.rows("token_embd.weight", tokens), w("token_embd_norm.weight"), self.eps)
        blocks: list[torch.Tensor] = []
        prelude_blocks: list[torch.Tensor] = []
        prelude_cur: torch.Tensor | None = None
        bank: list[torch.Tensor] = []
        bank_last: torch.Tensor | None = None
        tapped: list[torch.Tensor] = []

        for il in range(self.block_count):
            if il in taps:
                tapped.append(cur)
            ip, pass_idx = self.visit(il)
            p = f"blk.{ip}."
            swa = self.is_swa[il]
            n_head = self.heads[il]
            n_kv = self.kv_heads[il]
            head_dim = self.head_dim_swa if swa else self.head_dim_full

            messages: list[torch.Tensor | None] = [None, None, None, None]
            if self.num_loops > 1 and self.core_start <= ip < self.core_end:
                if ip == self.core_start:
                    if pass_idx == 0:
                        prelude_blocks = list(blocks)
                        prelude_cur = cur
                    else:
                        bank = list(blocks[len(prelude_blocks) + 1:])
                        bank_last = cur
                        blocks = list(prelude_blocks)
                        assert prelude_cur is not None
                        cur = prelude_cur
                if pass_idx > 0:
                    assert bank_last is not None
                    row = 2 * (ip - self.core_start)
                    q_res = w("loop_res_query.weight")[row:row + 2]
                    q_kv = w("loop_kv_res_query.weight")[row:row + 2]
                    m_res = block_attn_res_mix(bank, bank_last, q_res, self.eps)  # [2, B, T, H]
                    m_kv = block_attn_res_mix(bank, bank_last, q_kv, self.eps)    # [2, B, T, H]
                    messages = [m_res[0], m_kv[0], m_kv[1], m_res[1]]

            inp_sa = cur
            if ip == 0:
                inputs = [cur, cur, cur]
            else:
                q_attn = w(p + "attn_res_query.weight")
                if swa:
                    m = block_attn_res_mix(blocks, cur, q_attn, self.eps)
                    inputs = [m, m, m]
                else:
                    m = block_attn_res_mix(blocks, cur, q_attn, self.eps)  # [3, B, T, H]
                    inputs = [m[0], m[1], m[2]]

            if messages[0] is not None:
                for j, mod in enumerate(("attn", "k", "v")):
                    msg = messages[j]
                    assert msg is not None
                    inputs[j] = self.control(inputs[j], msg, ip, mod)

            if swa and messages[0] is not None:
                nw = w(p + "attn_norm.weight")
                inputs = [rms_norm(inp, nw, self.eps) for inp in inputs]
            elif swa:
                normed = rms_norm(inputs[0], w(p + "attn_norm.weight"), self.eps)
                inputs = [normed, normed, normed]
            else:
                inputs = [
                    rms_norm(inputs[0], w(p + "attn_q_inp_norm.weight"), self.eps),
                    rms_norm(inputs[1], w(p + "attn_k_inp_norm.weight"), self.eps),
                    rms_norm(inputs[2], w(p + "attn_v_inp_norm.weight"), self.eps),
                ]

            q_raw = linear(inputs[0], w(p + "attn_q.weight"))
            k_raw = linear(inputs[1], w(p + "attn_k.weight"))
            v_raw = linear(inputs[2], w(p + "attn_v.weight"))
            r_raw = linear(inputs[0], w(p + "attn_rel.weight"))

            if swa:
                k_conv = causal_short_conv(k_raw, w(p + "attn_k_conv.weight"), self.static_conv_scale)
                v_conv = causal_short_conv(v_raw, w(p + "attn_v_conv.weight"), self.static_conv_scale)
            else:
                k_dyn = linear(inputs[1], w(p + "attn_k_conv_proj.weight")).view(tokens.shape[0], seqlen, n_kv, self.conv_k)
                v_dyn = linear(inputs[2], w(p + "attn_v_conv_proj.weight")).view(tokens.shape[0], seqlen, n_kv, self.conv_k)
                k_conv = causal_short_conv(k_raw, w(p + "attn_k_conv.weight"), self.dynamic_conv_scale, k_dyn)
                v_conv = causal_short_conv(v_raw, w(p + "attn_v_conv.weight"), self.dynamic_conv_scale, v_dyn)

            q = l2_clip(heads(q_raw, n_head), w(p + "attn_q_clip.weight"), self.qk_clip)
            k = l2_clip(heads(k_conv, n_kv), None, self.qk_clip)
            v = rms_norm(heads(v_conv, n_kv), None, self.eps)
            r = l2_clip(heads(r_raw, n_head), w(p + "attn_rel_clip.weight"), self.rel_clip)

            if not swa and tau is not None:
                q = q * tau
                r = r * tau

            sinks = linear(inputs[0], w(p + "attn_sink_proj.weight"))  # [B, T, H]
            attn_out = inkling2_attention(
                q, k, v, r, w(p + "attn_rel_bias.weight"), sinks,
                visible_swa if swa else visible_full,
                head_dim ** -0.5,
            )
            gate = sigmoid(linear(inputs[0], w(p + "attn_gate.weight")))
            attn_out = linear(attn_out * gate, w(p + "attn_output.weight"))
            attn_out = causal_short_conv(attn_out, w(p + "attn_post_conv.weight"), self.static_conv_scale)

            if ip % self.attn_res_block_size == 0:
                blocks.append(inp_sa)
                cur = attn_out
            else:
                cur = inp_sa + attn_out

            ffn_inp = cur
            ffn_mixed = block_attn_res_mix(blocks, cur, w(p + "ffn_res_query.weight"), self.eps)
            if messages[3] is not None:
                ffn_mixed = self.control(ffn_mixed, messages[3], ip, "mlp")
            ffn_normed = rms_norm(ffn_mixed, w(p + "ffn_norm.weight"), self.eps)
            ffn_out = self.feed_forward(ip, il, ffn_normed)
            ffn_out = causal_short_conv(ffn_out, w(p + "ffn_post_conv.weight"), self.static_conv_scale)
            cur = ffn_inp + ffn_out

        if self.block_count in taps:
            tapped.append(cur)

        exit_mixed = block_attn_res_mix(blocks, cur, w("output_res_query.weight"), self.eps)
        return rms_norm(exit_mixed, w("output_norm.weight"), self.eps), tapped

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        z = linear(hidden, self.weights("output.weight"))
        if self.softcap != 0.0:
            z = torch.tanh(z / self.softcap) * self.softcap
        return z[..., :self.vocab_unpadded]
