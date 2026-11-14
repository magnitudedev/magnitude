"""Independent float32 reference for `glm5-next` GGUFs (GLM-5.3-Flash text model).

Model definition: transformers 5.17.0 `Glm5NextTextModel` (`models/glm5_next/modeling_glm5_next.py`, checked as
`Glm5NextForConditionalGeneration` without images), in float32 with no activation roundings. With S hyper-connection
streams X [S, H] per row:

- the embedding enters as S copies; every sublayer reads (pre, post, comb) from the streams it is given, the Sinkhorn
  mixes DeepSeek V4.1 uses (`model.sinkhorn_mixes`); its input is RMSNorm(sum_c pre[c] X[c]) with its *own* site's
  pre, and X'[k] = post[k] y + sum_j comb[j, k] X[j]. The exit is RMSNorm(mean_c X[c]), then the untied head;
- a layer with `attention.head_count_kv` 0 is KDA, 1 is MLA with its own DSA indexer;
- KDA: q, k, v projections, each a depthwise causal conv (no bias) and SiLU; q and k L2-normalized (eps 1e-6), q / sqrt(D);
  beta = sigmoid(ssm_beta x) per head; the per-channel log decay g = lower * sigmoid(exp(A_log)[h] (f_b f_a x + dt_bias));
  per head from a zero state S [D, D], token by token: S <- diag(exp(g_t)) S; S <- S + k_t (beta_t (v_t - S^T k_t))^T;
  o_t = S^T q_t. Then RMSNorm(o) w sigmoid(g_b g_a x) per head, and the output projection;
- MLA without positions: q = q_b RMSNorm(q_a x) per head, latent c = RMSNorm(kv_a x), k_h = W_UK_h c, v_h = W_UV_h c,
  softmax(q k^T / sqrt(D)) over the rows the layer's indexer selects, then the output projection. It is computed
  absorbed, as one latent row serves as key and value: head h's scores (W_UK_h^T q_h) . c over the selected rows
  (`model.gathered_attention`), its output W_UV_h of their weighted sum;
- the indexer: q = wq_b of the query latent; per-row key LayerNorm(wk x) (weight and bias); each complete pool of
  `kpool` rows has the key sum_i softmax_i(gate x_i + ape_i) k_i per channel; score(t, j) = sum_h w_h relu(q_h . K_j
  / sqrt(W)) with w = proj(x) / sqrt(heads). Row t attends the rows of its best min(top_k / kpool, visible) pools (a pool
  is visible once its last row is at or before t; ties go to the lower pool) and its tail, the rows after its last
  complete pool;
- FFN: SwiGLU with gate <= limit and up in [-limit, limit]; the first `leading_dense_block_count` layers are dense, the
  rest route: s = sigmoid(router x), experts = top-k of s + bias, weights s / (sum s + 1e-20) x scale (the expert output
  times its weight), plus the shared expert. Routed experts clamp at `swiglu_clamp_exp`, the shared expert and the dense
  FFN at `swiglu_clamp_shexp`, as llama.cpp's graph reads them.

Every row is a function of the tokens at or before it, the selection included. Its products and pointwise ops are the
shared ones (`model.linear`, `products`, `pointwise`), tiled where the platform makes a row's bits its own
(`model.tiled`), and every softmax runs along its last axis, so there a prefix's rows are the whole prompt's bit for
bit.

The file (upstream llama.cpp v0.9.0 `glm5-next`, `conversion/glm.py` `Glm5NextModel`): `kv_b` split into `attn_k_b`
[heads, latent, D] (transposed) and `attn_v_b` [heads, D, latent]; conv kernels [1, channels, 1, K]; `ssm_a` holds
-exp(A_log); `dt_bias` is `ssm_dt.bias`. FP8 weights are E4M3 codes under an F32 tile grid (`Weights`). The MTP block
is a separate file.
"""
from __future__ import annotations

from collections.abc import Callable

from gguf.constants import ExpertGatingFuncType
import torch

from magnitude_references.gguf import Package
from magnitude_references.model import (F32, Reference, Weights, causal_conv, gathered_attention, l2_normalize,
                                        layer_count, layer_norm, linear, mixture, per_layer, products, rms_norm,
                                        route_top_k, sigmoid, silu, sinkhorn_mixes, stream_collapse, stream_expand,
                                        unit_selection)

ROUTE_EPS = 1e-20  # the routing weights' normaliser (transformers' Glm5NextTextTopkRouter)
# transformers builds the vision tower with the model; a text forward never runs it, so the check builds it small.
CHECK_VISION = {"depth": 1, "hidden_size": 16, "num_heads": 2, "intermediate_size": 16, "out_hidden_size": 16,
                "projection_intermediate_size": 16}


def kimi_delta(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The Kimi delta rule from a zero state, token by token (its definition): per head, S <- diag(exp(g_t)) S;
    S <- S + k_t (beta_t (v_t - S^T k_t))^T; o_t = S^T q_t. q, k, g [B, T, H, Dk], v [B, T, H, Dv], beta [B, T, H]
    -> o [B, T, H, Dv]."""
    batch, steps, head_count, width = k.shape
    state = torch.zeros(batch, head_count, width, v.shape[-1], dtype=F32, device=v.device)
    out = []
    for t in range(steps):
        state = state * g[:, t].exp()[..., None]
        delta = (v[:, t] - (state * k[:, t, ..., None]).sum(-2)) * beta[:, t, :, None]
        state = state + k[:, t, ..., None] * delta[..., None, :]
        out.append((state * q[:, t, ..., None]).sum(-2))
    return torch.stack(out, dim=1)


class Glm5NextReference(Reference):
    """`glm5-next`: mHC streams over KDA and NoPE MLA layers with a pooled DSA indexer, sigmoid-routed MoE."""

    tied = False

    def __init__(self, package: Package, weights: Weights, layers: int | None = None):
        super().__init__(package, weights, layers)
        key = package.key
        self.architecture = package.architecture
        if self.architecture != "glm5-next":
            raise ValueError(f"not a glm5-next package: {self.architecture}")
        self.mtp_layers = key("nextn_predict_layers", 0)
        self.vocab_size, self.hidden_size = weights.shape("token_embd.weight")
        # The trailing `nextn_predict_layers` blocks are the MTP head, the drafter's, not the text model's.
        main = key("block_count") - self.mtp_layers
        self.block_count = count = main if layers is None else layer_count(package, layers)
        if count > main:
            raise ValueError(f"only {main} main layers (the rest are MTP)")
        self.context_length = key("context_length")
        self.eps = key("attention.layer_norm_rms_epsilon")
        self.head_count = key("attention.head_count")
        self.linear = [kv == 0 for kv in per_layer(key("attention.head_count_kv"), count)]
        # KDA
        self.conv_width = key("ssm.conv_kernel")
        self.linear_dim = key("kda.head_dim")
        self.lower_bound = key("kda.gate_lower_bound")
        # MLA
        self.q_rank, self.kv_rank = key("attention.q_lora_rank"), key("attention.kv_lora_rank")
        if key("rope.dimension_count") != 0:
            raise ValueError("glm5-next attention has no positions: rope.dimension_count must be 0")
        self.key_dim, self.value_dim = key("attention.key_length_mla"), key("attention.value_length_mla")
        # indexer
        self.index_norm_eps = key("attention.layer_norm_epsilon")
        self.index_heads = key("attention.indexer.head_count")
        self.index_width = key("attention.indexer.key_length")
        self.index_top = key("attention.indexer.top_k")
        self.kpool = key("attention.indexer.kpool")
        if self.kpool < 1 or self.index_top % self.kpool:
            raise ValueError(f"indexer top_k {self.index_top} is not a whole number of pools of {self.kpool}")
        if not key("attention.indexer.kpool_select_tail", True):
            raise ValueError("a row before its first complete pool attends only its tail: kpool_select_tail must hold")
        if not all(per_layer(key("attention.indexer.types", True), count)):
            raise ValueError("shared indexer layers (attention.indexer.types false) are not part of this reference")
        # hyper-connections
        self.streams = key("hyper_connection.count")
        self.sinkhorn = key("hyper_connection.sinkhorn_iterations")
        self.hc_eps = key("hyper_connection.epsilon")
        # FFN
        self.dense = key("leading_dense_block_count", 0)
        self.clamp_routed = per_layer(key("swiglu_clamp_exp"), count)
        self.clamp_shared = per_layer(key("swiglu_clamp_shexp"), count)
        if self.dense:
            self.dense_size = key("feed_forward_length")
        if count > self.dense:
            self.experts = key("expert_count")
            self.experts_used = key("expert_used_count")
            self.expert_size = key("expert_feed_forward_length")
            self.shared_experts = key("expert_shared_count")
            self.route_scale = key("expert_weights_scale")
            self.route_norm = bool(key("expert_weights_norm"))
            if key("expert_gating_func", ExpertGatingFuncType.SIGMOID) != ExpertGatingFuncType.SIGMOID:
                raise ValueError(f"glm5-next routes by sigmoid, the header says {key('expert_gating_func')}")
            if (key("expert_group_count", 1), key("expert_group_used_count", 1)) != (1, 1):
                raise ValueError("group-limited routing is not part of this reference")

    # decoder -----------------------------------------------------------------------------------

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        w = self.weights
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        streams = w.rows("token_embd.weight", tokens)[..., None, :].expand(-1, -1, self.streams, -1)
        tapped = []
        for layer in range(self.block_count):
            if layer in taps:
                tapped.append(streams.mean(-2))
            p = f"blk.{layer}."
            mixer = (lambda h: self.kda(layer, h)) if self.linear[layer] else (lambda h: self.attention(layer, h, positions))
            streams = self.sublayer(streams, p + "hc_attn", p + "attn_norm.weight", mixer)
            streams = self.sublayer(streams, p + "hc_ffn", p + "ffn_norm.weight", lambda h: self.feed_forward(layer, h))
        if self.block_count in taps:
            tapped.append(streams.mean(-2))
        return rms_norm(streams.mean(-2), w("output_norm.weight"), self.eps), tapped

    def sublayer(self, streams: torch.Tensor, site: str, norm: str,
                 body: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """`Glm5NextTextHyperConnection` around one sublayer: its input collapses with the site's own pre."""
        w = self.weights
        pre, post, comb = sinkhorn_mixes(streams, w(site + "_fn.weight"), w(site + "_scale.weight"),
                                         w(site + "_base.weight"), self.sinkhorn, self.hc_eps, self.eps)
        return stream_expand(body(rms_norm(stream_collapse(streams, pre), w(norm), self.eps)), streams, post, comb)

    # KDA ---------------------------------------------------------------------------------------

    def kda(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """`Glm5NextTextLinearAttention` with `Glm5NextTextForgetGate`'s bounded decay."""
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = x.shape
        shape = (batch, steps, self.head_count, self.linear_dim)

        def convolved(part: str) -> torch.Tensor:
            kernel = w(p + f"ssm_conv1d_{part}.weight").reshape(-1, self.conv_width)
            return silu(causal_conv(linear(x, w(p + f"attn_{part}.weight")), kernel)).view(shape)

        q = l2_normalize(convolved("q")) * self.linear_dim ** -0.5
        k = l2_normalize(convolved("k"))
        v = convolved("v")
        beta = sigmoid(linear(x, w(p + "ssm_beta.weight")))
        rate = -w(p + "ssm_a")  # exp(A_log) per head
        forget = linear(linear(x, w(p + "ssm_f_a.weight")), w(p + "ssm_f_b.weight")) + w(p + "ssm_dt.bias")
        g = self.lower_bound * sigmoid(rate[:, None] * forget.view(shape))
        o = kimi_delta(q, k, v, g, beta)
        gate = linear(linear(x, w(p + "ssm_g_a.weight")), w(p + "ssm_g_b.weight")).view(shape)
        o = rms_norm(o, w(p + "ssm_norm.weight"), self.eps) * sigmoid(gate)
        return linear(o.reshape(batch, steps, -1), w(p + "attn_output.weight"))

    # MLA and the indexer -----------------------------------------------------------------------

    def attention(self, layer: int, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`Glm5NextTextAttention`: NoPE MLA over the rows the layer's indexer selects, absorbed (module docstring)."""
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = x.shape
        qr = rms_norm(linear(x, w(p + "attn_q_a.weight")), w(p + "attn_q_a_norm.weight"), self.eps)
        q = linear(qr, w(p + "attn_q_b.weight")).view(batch * steps, self.head_count, self.key_dim)
        latent = rms_norm(linear(x, w(p + "attn_kv_a_mqa.weight")), w(p + "attn_kv_a_norm.weight"), self.eps)
        k_b, v_b = w(p + "attn_k_b.weight"), w(p + "attn_v_b.weight")  # [h, latent, D], [h, D, latent]
        q = products(q.transpose(0, 1), k_b).transpose(0, 1)  # W_UK_h^T q_h: [B·T, h, latent]
        out = gathered_attention(q.reshape(batch, steps, self.head_count, self.kv_rank), latent, None,
                                 self.selection(layer, x, qr, positions), self.key_dim ** -0.5)
        out = products(out.flatten(0, 1).transpose(0, 1), v_b).transpose(0, 1)  # W_UV_h o_h: [B·T, h, D]
        return linear(out.reshape(batch, steps, -1), w(p + "attn_output.weight"))

    def selection(self, layer: int, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`Glm5NextTextIndexer`: each row's slots [B, T, K], its best visible pools' rows and its tail
        (`unit_selection` over its pools' scores)."""
        w, p = self.weights, f"blk.{layer}."
        batch, steps, _ = x.shape
        pool = self.kpool
        complete = steps // pool
        span = complete * pool
        q = linear(qr, w(p + "indexer.attn_q_b.weight")).view(batch, steps, self.index_heads, self.index_width)
        keys = layer_norm(linear(x, w(p + "indexer.attn_k.weight")), w(p + "indexer.k_norm.weight"),
                          w(p + "indexer.k_norm.bias"), self.index_norm_eps)
        gate = linear(x, w(p + "indexer_compressor_gate.weight"))[:, :span].unflatten(1, (complete, pool))
        # each pool's softmax over its rows, taken along the last axis: torch's softmax over an inner axis rounds by
        # how many pools it is given, along the last a row's bits are its own
        gate = (gate + w(p + "indexer_compressor_ape.weight")).transpose(-1, -2)
        mix = torch.softmax(gate, dim=-1).transpose(-1, -2)
        pooled = (mix * keys[:, :span].unflatten(1, (complete, pool))).sum(2)  # [B, pools, W]
        scores = torch.relu(products(q.transpose(1, 2), pooled[:, None]) * self.index_width ** -0.5)  # [B, h, T, pools]
        head_weights = linear(x, w(p + "indexer.proj.weight")) * self.index_heads ** -0.5  # [B, T, h]
        scores = (scores * head_weights.transpose(1, 2)[..., None]).sum(1)  # [B, T, pools]
        return unit_selection(scores, positions + 1, self.index_top // pool, pool)

    # FFN ---------------------------------------------------------------------------------------

    @staticmethod
    def swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, limit: float) -> torch.Tensor:
        """SwiGLU with the gate clamped to <= limit and up to [-limit, limit]."""
        return linear(silu(linear(x, gate).clamp(max=limit)) * linear(x, up).clamp(-limit, limit), down)

    def feed_forward(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        w, p = self.weights, f"blk.{layer}."
        if layer < self.dense:
            return self.swiglu(x, *(w(p + f"ffn_{part}.weight") for part in ("gate", "up", "down")),
                               self.clamp_shared[layer])
        flat = x.reshape(-1, self.hidden_size)
        scores = sigmoid(linear(flat, w(p + "ffn_gate_inp.weight")))
        selected = route_top_k(scores + w(p + "exp_probs_b.bias"), self.experts_used)
        weights = scores.gather(1, selected)
        if self.route_norm:
            weights = weights / (weights.sum(-1, keepdim=True) + ROUTE_EPS)
        weights = weights * self.route_scale
        names = tuple(p + f"ffn_{part}_exps.weight" for part in ("gate", "up", "down"))

        def expert(index: int, rows: torch.Tensor) -> torch.Tensor:
            return self.swiglu(rows, *(w.expert(name, index) for name in names), self.clamp_routed[layer])

        out = mixture(flat, selected, weights, expert)
        if self.shared_experts:
            out = out + self.swiglu(flat, *(w(p + f"ffn_{part}_shexp.weight") for part in ("gate", "up", "down")),
                                    self.clamp_shared[layer])
        return out.view_as(x)

    # transformers cross-check ------------------------------------------------------------------

    def hf_config(self) -> dict:
        count = self.block_count
        limits = {self.clamp_routed[l] for l in range(self.dense, count)} | {self.clamp_shared[l] for l in range(count)}
        if len(limits) != 1:
            raise ValueError(f"transformers has one swiglu_limit, the file has {sorted(limits)}")
        text = {
            "vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "num_hidden_layers": count,
            "num_attention_heads": self.head_count, "num_key_value_heads": self.head_count,
            "max_position_embeddings": self.context_length, "rms_norm_eps": self.eps, "tie_word_embeddings": False,
            "pad_token_id": None, "hidden_act": "silu", "swiglu_limit": limits.pop(),
            "layer_types": ["linear_attention" if self.linear[l] else "deepseek_sparse_attention" for l in range(count)],
            "mlp_layer_types": ["dense" if l < self.dense else "sparse" for l in range(count)],
            "indexer_types": ["full"] * count,
            "linear_head_dim": self.linear_dim, "linear_num_heads": self.head_count,
            "linear_conv_kernel_dim": self.conv_width, "linear_lower_bound": self.lower_bound,
            "q_lora_rank": self.q_rank, "kv_lora_rank": self.kv_rank, "qk_rope_head_dim": 0,
            "qk_nope_head_dim": self.key_dim, "v_head_dim": self.value_dim,
            "index_topk": self.index_top, "index_head_dim": self.index_width, "index_n_heads": self.index_heads,
            "index_kpool": self.kpool, "index_kpool_always_select_tail": True,
            "hc_mult": self.streams, "hc_eps": self.hc_eps, "hc_sinkhorn_iters": self.sinkhorn,
        }
        if self.dense:
            text["intermediate_size"] = self.dense_size
        if count > self.dense:
            text |= {"n_routed_experts": self.experts, "num_experts_per_tok": self.experts_used,
                     "moe_intermediate_size": self.expert_size, "n_shared_experts": self.shared_experts,
                     "routed_scaling_factor": self.route_scale, "norm_topk_prob": self.route_norm,
                     "n_group": 1, "topk_group": 1}
        return {"model_type": "glm5_next", "text_config": text, "vision_config": CHECK_VISION}

    def hf_build(self, config):
        from transformers import Glm5NextForConditionalGeneration
        return Glm5NextForConditionalGeneration._from_config(config, dtype=F32, attn_implementation="eager")

    def hf_unused_prefixes(self) -> tuple[str, ...]:
        return ("model.visual.",)

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        w, m = self.weights, "model.language_model."
        parameters = {m + "embed_tokens.weight": w("token_embd.weight"), m + "norm.weight": w("output_norm.weight"),
                      "lm_head.weight": w("output.weight")}
        for layer in range(self.block_count):
            p, h = f"blk.{layer}.", f"{m}layers.{layer}."
            parameters[h + "input_layernorm.weight"] = w(p + "attn_norm.weight")
            parameters[h + "post_attention_layernorm.weight"] = w(p + "ffn_norm.weight")
            for site in ("attn", "ffn"):
                for part in ("fn", "base", "scale"):
                    parameters[f"{h}{site}_hc.{part}"] = w(f"{p}hc_{site}_{part}.weight")
            a = h + "self_attn."
            if self.linear[layer]:
                for theirs, ours in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
                                     ("o_proj", "attn_output"), ("b_proj", "ssm_beta"), ("g_a_proj", "ssm_g_a"),
                                     ("g_b_proj", "ssm_g_b"), ("o_norm", "ssm_norm"),
                                     ("forget_gate.f_a_proj", "ssm_f_a"), ("forget_gate.f_b_proj", "ssm_f_b")):
                    parameters[f"{a}{theirs}.weight"] = w(f"{p}{ours}.weight")
                parameters[a + "conv1d.weight"] = torch.cat(
                    [w(p + f"ssm_conv1d_{part}.weight").reshape(-1, 1, self.conv_width) for part in "qkv"])
                parameters[a + "forget_gate.dt_bias"] = w(p + "ssm_dt.bias")
                parameters[a + "forget_gate.A_log"] = torch.log(-w(p + "ssm_a"))
            else:
                for theirs, ours in (("q_a_proj", "attn_q_a"), ("q_a_layernorm", "attn_q_a_norm"),
                                     ("q_b_proj", "attn_q_b"), ("kv_a_proj_with_mqa", "attn_kv_a_mqa"),
                                     ("kv_a_layernorm", "attn_kv_a_norm"), ("o_proj", "attn_output"),
                                     ("indexer.wq_b", "indexer.attn_q_b"), ("indexer.wk", "indexer.attn_k"),
                                     ("indexer.k_norm", "indexer.k_norm"), ("indexer.weights_proj", "indexer.proj")):
                    parameters[f"{a}{theirs}.weight"] = w(f"{p}{ours}.weight")
                parameters[a + "indexer.k_norm.bias"] = w(p + "indexer.k_norm.bias")
                for part in ("ape", "gate"):
                    parameters[f"{a}indexer.index_kpool_compress_{part}"] = w(f"{p}indexer_compressor_{part}.weight")
                k_b, v_b = w(p + "attn_k_b.weight"), w(p + "attn_v_b.weight")  # [h, latent, D], [h, D, latent]
                parameters[a + "kv_b_proj.weight"] = torch.cat((k_b.transpose(1, 2), v_b), dim=1).reshape(-1, self.kv_rank)
            f = h + "mlp."
            if layer < self.dense:
                for part in ("gate", "up", "down"):
                    parameters[f"{f}{part}_proj.weight"] = w(f"{p}ffn_{part}.weight")
                continue
            parameters[f + "gate.weight"] = w(p + "ffn_gate_inp.weight")
            parameters[f + "gate.e_score_correction_bias"] = w(p + "exp_probs_b.bias")
            parameters[f + "experts.gate_up_proj"] = torch.cat((w(p + "ffn_gate_exps.weight"), w(p + "ffn_up_exps.weight")), 1)
            parameters[f + "experts.down_proj"] = w(p + "ffn_down_exps.weight")
            if self.shared_experts:
                for part in ("gate", "up", "down"):
                    parameters[f"{f}shared_experts.{part}_proj.weight"] = w(f"{p}ffn_{part}_shexp.weight")
        return parameters
