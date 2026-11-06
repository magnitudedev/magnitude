"""DeepSeek's own V4.1 inference code over a `deepseek41` reference's weights: the released-code check
of `families/deepseek41.py` (`checks.check_hf`).

`inference/model.py` and `inference/engram.py` are read at the pinned revision (`deepseek41.CODE`).
Their CUDA kernels (`kernel.py`, tilelang) are replaced by float32 torch transcriptions of the same
definitions. Every weight is loaded widened (`dtype="bf16"`, no FP4 experts, then float32), so no GEMM
runs on quantized activations; the in-place roundings run as the reference's `Roundings` say: the
3-D `act_quant` is the window KV, `fp4_act_quant` with an E4M3 scale the compressed rows and with a
UE8M0 scale the index rows. The 4-D `act_quant` (the attention output before wo_a) is an activation
rounding, not model numerics, and stays off like the reference's.

The prompt runs as DeepSeek's `generate.py` runs it: one prefill over its first half, then one decode
step per row, so the check also covers the reference's row-by-row definition against DeepSeek's
incremental state (partial compressor groups, decode-time selection).
"""
from __future__ import annotations

from functools import partial
import importlib.util
from pathlib import Path
import sys
import types
import urllib.request

import torch

from magnitude_references.families.deepseek41 import (CODE, DeepSeek41Reference, round_compressed, round_index,
                                                      round_window)
from magnitude_references.gguf import HEADER_CACHE
from magnitude_references.model import F32

MODULES = ("engram", "model")


def code() -> Path:
    """The pinned `inference/` files, read once into the cache."""
    repository, revision = CODE
    directory = HEADER_CACHE.parent / "deepseek-code" / revision
    for name in MODULES:
        path = directory / f"{name}.py"
        if not path.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{repository}/resolve/{revision}/inference/{name}.py"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
    return directory


def kernels(reference: DeepSeek41Reference) -> types.ModuleType:
    """`kernel.py` in float32 torch (K: lines of `inference/kernel.py`)."""
    roundings = reference.roundings
    module = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=F32, inplace=False):
        if not inplace:
            raise ValueError("an FP8 GEMM ran: every weight is loaded widened")
        if x.dim() == 3 and roundings.window:
            x.copy_(round_window(x))
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if not inplace:
            raise ValueError("an FP4 GEMM ran: every weight is loaded widened")
        compressed = scale_dtype == torch.float8_e4m3fn
        if roundings.compressed if compressed else roundings.index:
            x.copy_(round_compressed(x) if compressed else round_index(x))
        return x

    def gemm(*_, **__):
        raise ValueError("a quantized GEMM ran: every weight is loaded widened")

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """K:310-403: the indexed rows (-1 skipped), running max from -1e30, the sink in the denominator."""
        valid = topk_idxs >= 0
        rows = kv[torch.arange(kv.shape[0])[:, None, None], topk_idxs.clamp(min=0).long()]  # [b, m, k, d]
        scores = torch.einsum("bmhd,bmkd->bmhk", q, rows) * softmax_scale
        scores = scores.masked_fill(~valid[:, :, None], float("-inf"))
        peak = scores.amax(-1, keepdim=True).clamp(min=-1e30)
        weights = torch.exp(scores - peak)
        total = weights.sum(-1, keepdim=True) + torch.exp(attn_sink.view(1, 1, -1, 1) - peak)
        return torch.einsum("bmhk,bmkd->bmhd", weights / total, rows)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        """K:406-475."""
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    module.act_quant, module.fp4_act_quant, module.sparse_attn = act_quant, fp4_act_quant, sparse_attn
    module.hc_split_sinkhorn, module.fp8_gemm, module.fp4_gemm = hc_split_sinkhorn, gemm, gemm
    return module


def load(reference: DeepSeek41Reference) -> types.ModuleType:
    """DeepSeek's `model` module, imported over the stand-in kernels (vision is not loaded)."""
    directory = code()
    vision = types.ModuleType("vision")
    vision.ViT = vision.Aligner = None
    image_processor = types.ModuleType("image_processor")
    image_processor.IMAGE, image_processor.IMAGE_END, image_processor.IMAGE_NEW_LINE, image_processor.IMAGE_START = range(4)
    stand_ins = {"kernel": kernels(reference), "vision": vision, "image_processor": image_processor}
    saved = {name: sys.modules.get(name) for name in (*stand_ins, *MODULES)}
    try:
        sys.modules.update(stand_ins)
        for name in MODULES:  # engram first: model imports it
            spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
            sys.modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules[name])
        model = sys.modules["model"]
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return model


def arguments(reference: DeepSeek41Reference, model: types.ModuleType, tokens: torch.Tensor):
    r, w = reference, reference.weights
    if len(set(r.clamp_routed + r.clamp_shared)) != 1:
        raise ValueError("DeepSeek's ModelArgs has one swiglu_limit")
    engram = {}
    if r.engram_layers:
        engram = {"engram_layer_ids": tuple(r.engram_layers),
                  "engram_num_embeddings": tuple(int(sizes.sum()) for sizes in r.bucket_sizes),
                  "engram_max_ngram_size": r.ngram, "engram_vocab_size": int(r.bucket_sizes[0, 0]),
                  "engram_n_heads": r.ngram_heads, "engram_head_dim": w.shape("per_layer_token_embd.weight")[1],
                  "engram_pad_id": r.package.key("engram.pad_token_id"),
                  "engram_compressed_vocab_size": int(r.token_map.max()) + 1}
    candidates = ({"candidate_source_layer": r.candidate, "candidate_topk_blocks": r.candidate_blocks,
                   "candidate_block_size": r.candidate_block} if r.candidate >= 0 else {})
    factor, original, beta_fast, beta_slow = r.yarn
    return model.ModelArgs(
        max_batch_size=tokens.shape[0], max_seq_len=tokens.shape[1], temperature=0, dtype="bf16", expert_dtype=None,
        vocab_size=r.vocab_size, dim=r.hidden_size, moe_inter_dim=w.shape("blk.0.ffn_gate_shexp.weight")[0],
        n_layers=r.block_count, n_mtp_layers=0, n_heads=r.head_count, n_routed_experts=r.experts, n_shared_experts=1,
        n_activated_experts=r.experts_used, score_func="sqrtsoftplus", norm_topk_prob=r.route_norm,
        route_scale=r.route_scale, swiglu_limit=r.clamp_routed[0], q_lora_rank=w.shape("blk.0.attn_q_a.weight")[0],
        head_dim=r.head_dim, rope_head_dim=r.rope_dim, norm_eps=r.eps, o_groups=r.groups, o_lora_rank=r.output_rank,
        window_size=r.window, compress_ratios=tuple(r.ratios), kv_source_layers=tuple(r.kv_sources),
        index_source_layers=tuple(r.index_sources), compress_rope_theta=r.compress_base, original_seq_len=original,
        rope_theta=r.rope_base, rope_factor=factor, beta_fast=beta_fast, beta_slow=beta_slow,
        index_n_heads=r.index_heads, index_head_dim=r.index_width, index_topk=r.index_top, hc_mult=r.streams,
        hc_sinkhorn_iters=r.sinkhorn, hc_eps=r.hc_eps, vision_n_layers=0, **candidates, **engram)


def parameters(reference: DeepSeek41Reference) -> dict[str, torch.Tensor]:
    """The checkpoint's state dict, from the reference's weights (FP8 and FP4 widened)."""
    r, w = reference, reference.weights
    out = {"embed.weight": w("token_embd.weight"), "head.weight": w("output.weight"), "norm.weight": w("output_norm.weight")}
    for layer in range(r.block_count):
        p, d = f"blk.{layer}.", f"layers.{layer}."
        out |= {d + f"hc_{site}_{part}": w(p + f"hc_{site}_{part}.weight")
                for site in ("attn", "ffn") for part in ("fn", "base", "scale")}
        out |= {d + "attn_norm.weight": w(p + "attn_norm.weight"), d + "ffn_norm.weight": w(p + "ffn_norm.weight"),
                d + "attn.attn_sink": w(p + "attn_sinks.weight"), d + "attn.wq_a.weight": w(p + "attn_q_a.weight"),
                d + "attn.q_norm.weight": w(p + "attn_q_a_norm.weight"), d + "attn.wq_b.weight": w(p + "attn_q_b.weight"),
                d + "attn.wkv.weight": w(p + "attn_kv.weight"), d + "attn.kv_norm.weight": w(p + "attn_kv_a_norm.weight"),
                d + "attn.wo_a.weight": w(p + "attn_output_a.weight"), d + "attn.wo_b.weight": w(p + "attn_output_b.weight"),
                d + "ffn.gate.weight": w(p + "ffn_gate_inp.weight"), d + "ffn.gate.bias": w(p + "exp_probs_b.bias"),
                d + "ffn.shared_experts.w1.weight": w(p + "ffn_gate_shexp.weight"),
                d + "ffn.shared_experts.w2.weight": w(p + "ffn_down_shexp.weight"),
                d + "ffn.shared_experts.w3.weight": w(p + "ffn_up_shexp.weight")}
        for expert in range(r.experts):
            for name, stored in (("w1", "gate"), ("w2", "down"), ("w3", "up")):
                out[d + f"ffn.experts.{expert}.{name}.weight"] = w.expert(p + f"ffn_{stored}_exps.weight", expert)
        a = d + "attn."
        if layer in r.kv_sources:
            out |= {a + "compressor.wkv.weight": w(p + "attn_compressor_kv.weight"),
                    a + "compressor.norm.weight": w(p + "attn_compressor_norm.weight")}
            if r.ratios[layer] > 1:
                out[a + "compressor.wgate.weight"] = w(p + "attn_compressor_gate.weight")
        if layer in r.index_sources:
            out |= {a + "indexer.wq_b.weight": w(p + "indexer.attn_q_b.weight"),
                    a + "indexer.weights_proj.weight": w(p + "indexer.proj.weight")}
            if layer in r.kv_sources:
                out |= {a + "indexer.wk.weight": w(p + "indexer.attn_k.weight"),
                        a + "indexer.k_norm.weight": w(p + "indexer.k_norm.weight")}
        if layer in r.engram_layers:
            module = r.engram_layers.index(layer)
            first, count = int(r.bucket_offsets[module, 0]), int(r.bucket_sizes[module].sum())
            table = w.rows("per_layer_token_embd.weight", torch.arange(first, first + count))
            gate = w(p + "engram_gate.weight")
            out |= {d + "engram.embed.weight": table, d + "engram.embed.scale": torch.ones(count, table.shape[1] // 32),
                    d + "engram.wkv.weight": w(p + "engram_kv.weight"), d + "engram.q_weight": gate,
                    d + "engram.k_weight": torch.ones_like(gate)}
    return out


def logits(reference: DeepSeek41Reference, tokens: torch.Tensor) -> torch.Tensor:
    """DeepSeek's logits [B, T, V] of `tokens` (CPU): a prefill of the first half, then decode steps."""
    model = load(reference)
    args = arguments(reference, model, tokens)
    if reference.engram_layers:  # the header's token map stands in for the tokenizer it is built from
        token_map = reference.token_map.tolist()
        model.NgramHashState.__init__.__globals__["build_compressed_token_map"] = (
            lambda tokenizer: (token_map, args.engram_compressed_vocab_size))
    # The Engram lookup casts its rows to bfloat16 (exact: E4M3 times a power of two) for bfloat16 GEMMs.
    widened = model.linear
    model.linear = lambda x, weight, bias=None: widened(x.to(weight.dtype), weight, bias)
    transformer = model.Transformer(args).float()
    state = {name: value.to("cpu", F32) for name, value in parameters(reference).items()}
    transformer.load_state_dict(state, strict=True)
    transformer.head.forward = partial(type(transformer.head).forward, transformer.head, full_logits=True)
    if reference.engram_layers:
        hashing = transformer.engram_hash
        hashing.multipliers.copy_(reference.multipliers)  # a slice keeps its source layers' constants
        hashing.primes.copy_(reference.bucket_sizes.view_as(hashing.primes))
        hashing.offsets.copy_(reference.bucket_offsets - reference.bucket_offsets[:, :1])
    prefill = tokens.shape[1] // 2 + 1
    rows = [transformer(tokens[:, :prefill], 0)[1]]
    rows += [transformer(tokens[:, step:step + 1], step)[1] for step in range(prefill, tokens.shape[1])]
    return torch.cat(rows, dim=1)
