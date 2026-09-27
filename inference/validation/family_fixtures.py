#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "transformers==5.17.0", "safetensors"]
# ///
"""Synthetic variation fixtures for the in-scope model families (plan §5).

uv run inference/validation/family_fixtures.py [--output DIR] [--cases a,b] [--no-transformers] [--list]

Each case is a small synthetic GGUF of one family, built from the metadata of a locked real GGUF of that
family (every `<architecture>.*` key of the real header is written, with synthetic values, and no
other architecture key; the tokenizer keeps the real model/pre/special-token keys over a small
vocabulary), plus the independent reference's outputs for decoding schedules over it. Format:
`handoffs/26-09-27/model-families/references.md` ("Fixture format"). Templates are read once from
Hugging Face by byte range and cached under `results/headers/`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dflash_reference import DFlashReference  # noqa: E402
from model_references import open_reference  # noqa: E402
from reference_cli import check_hf  # noqa: E402
from reference_gguf import HEADER_CACHE, Package, RemoteSource, Value, Writer, quantize  # noqa: E402
from reference_model import Weights, to_numpy  # noqa: E402

OUTPUT = ROOT / "results" / "fixtures" / "families"

TEMPLATES = {  # locked catalog (or plan-candidate) GGUFs whose headers the synthetic models mirror
    "minicpm5": ("openbmb/MiniCPM5-2B-GGUF", "2079a22f3beaa4e306449978533478fe0522f4b3", "MiniCPM5-2B-Q8_0.gguf"),
    "lfm2": ("LiquidAI/LFM2.5-2.6B-GGUF", "b421ad1d549afeda6a0fb2ad3a697cb5a7879adc", "LFM2.5-2.6B-Q8_0.gguf"),
    "lfm2moe": ("LiquidAI/LFM2.5-8B-A1B-GGUF", "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0", "LFM2.5-8B-A1B-Q8_0.gguf"),
    "muse": ("unsloth/Muse-Glimmer-30B-GGUF", "1afeb8e879f60116d206cf724425dbe1e1a2f7f5", "Muse-Glimmer-30B-UD-Q8_K_XL.gguf"),
    "gemma4-e2b": ("unsloth/gemma-4-E2B-it-qat-GGUF", "66a399f68ddd113b06dff02fca9523e55465d11d",
                   "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
    "gemma4-12b": ("unsloth/gemma-4-12B-it-qat-GGUF", "980b060c40a8539ac159e0501a3e0f66a6365af3",
                   "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"),
    "gemma4-26b": ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", "7b92b5b28818151e8669af2e45e88d6086f490dd",
                   "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"),
    "nemotron-lightning": ("ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "88d7ce0b0fa385c5108866ce5d33690927531a37",
                           "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"),
    "nemotron-super": ("unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF", "036038fb30334a2d56a146c6f0d4871ab5edccbb",
                       "MXFP4_MOE/NVIDIA-Nemotron-3-Super-120B-A12B-MXFP4_MOE-00001-of-00003.gguf"),
}
TOKENIZER_ID_KEYS = ("bos_token_id", "eos_token_id", "unknown_token_id", "padding_token_id", "seperator_token_id",
                     "mask_token_id", "eot_token_id")


def template(name: str) -> dict[str, Value]:
    return RemoteSource(*TEMPLATES[name], cache=HEADER_CACHE).header.metadata


class Synthetic:
    """A synthetic GGUF: metadata mirrored from a real header, random tensors in GGUF layout."""

    def __init__(self, source: dict[str, Value], name: str, vocab: int, seed: int):
        self.source = source
        self.architecture = source["general.architecture"].value
        self.rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.metadata: dict[str, Value] = {
            "general.architecture": source["general.architecture"],
            "general.type": Value(V.STRING, "model"),
            "general.name": Value(V.STRING, f"synthetic {name}"),
        }
        self.tensors: list[tuple[str, np.ndarray, Q]] = []
        self.tokenizer()

    def tokenizer(self) -> None:
        source, count = self.source, self.vocab
        remapped = {}
        for suffix in TOKENIZER_ID_KEYS:
            key = f"tokenizer.ggml.{suffix}"
            if key in source:
                remapped.setdefault(source[key].value, len(remapped))
                self.metadata[key] = Value(source[key].kind, remapped[source[key].value])
        real_tokens = source["tokenizer.ggml.tokens"].value
        tokens = [f"<t{i}>" for i in range(count)]
        types = [1] * count
        for real, small in remapped.items():
            tokens[small] = real_tokens[real]
            types[small] = 3
        self.metadata["tokenizer.ggml.tokens"] = Value(V.ARRAY, tokens, V.STRING)
        self.metadata["tokenizer.ggml.token_type"] = Value(V.ARRAY, types, source["tokenizer.ggml.token_type"].item)
        for key, value in source.items():
            if not key.startswith("tokenizer.") or key in self.metadata:
                continue
            if key == "tokenizer.ggml.merges":
                self.metadata[key] = Value(V.ARRAY, [], V.STRING)
            elif key == "tokenizer.ggml.scores":
                self.metadata[key] = Value(V.ARRAY, [0.0] * count, V.FLOAT32)
            elif key == "tokenizer.ggml.suppress_tokens":
                self.metadata[key] = Value(V.ARRAY, [count - 1, count - 2], value.item)
            else:
                self.metadata[key] = value

    def has(self, suffix: str) -> bool:
        return f"{self.architecture}.{suffix}" in self.source

    def key(self, suffix: str, value) -> None:
        """Set `<architecture>.<suffix>`, a key of the real header, with the real header's GGUF type."""
        key = f"{self.architecture}.{suffix}"
        real = self.source[key]
        if (real.kind == V.ARRAY) != isinstance(value, list):
            raise ValueError(f"{key}: the real header stores {'an array' if real.kind == V.ARRAY else 'a scalar'}")
        self.metadata[key] = Value(real.kind, value, real.item)

    def layered(self, suffix: str, values: list) -> None:
        """A per-layer value, as an array or as one scalar, the way the real header stores it."""
        if self.source[f"{self.architecture}.{suffix}"].kind == V.ARRAY:
            self.key(suffix, values)
        elif len(set(values)) == 1:
            self.key(suffix, values[0])
        else:
            raise ValueError(f"{suffix} varies per layer but the real header stores one scalar")

    def tensor(self, name: str, values: np.ndarray, kind: Q = Q.F16) -> None:
        self.tensors.append((name, np.asarray(values, dtype=np.float32), kind))

    def normal(self, *shape: int, scale: float = 1.0) -> np.ndarray:
        return (self.rng.standard_normal(shape) * scale).astype(np.float32)

    def matrix(self, name: str, rows: int, cols: int, gain: float = 1.0, kind: Q = Q.F16) -> None:
        self.tensor(name, self.normal(rows, cols, scale=gain / math.sqrt(cols)), kind)

    def experts(self, name: str, count: int, rows: int, cols: int, gain: float = 1.0, kind: Q = Q.F16) -> None:
        self.tensor(name, self.normal(count, rows, cols, scale=gain / math.sqrt(cols)), kind)

    def norm(self, name: str, width: int, center: float = 1.0, spread: float = 0.1) -> None:
        self.tensor(name, center + self.normal(width, scale=spread), Q.F32)

    def check_keys(self) -> None:
        """Every architecture key of the real header is written."""
        prefix = f"{self.architecture}."
        missing = {k for k in self.source if k.startswith(prefix)} - set(self.metadata)
        if missing:
            raise ValueError(f"{self.architecture} keys of the real header not written: {sorted(missing)}")

    def write(self, path: Path) -> None:
        writer = Writer(path, self.metadata)
        encoded = [(name, quantize(values, kind), values.shape, kind) for name, values, kind in self.tensors]
        for name, _, shape, kind in encoded:
            writer.declare(name, tuple(reversed(shape)), kind)
        writer.begin()
        for name, data, _, _ in encoded:
            writer.write(name, data)
        writer.close()


# ---------------------------------------------------------------------------------------------
# Family builders. Shapes below are NumPy order ([out, in] for matrices; GGUF order is reversed).


def common_attention(s: Synthetic, p: str, hidden: int, heads: int, kv: int, dim: int, norms: bool) -> None:
    s.norm(p + "attn_norm.weight", hidden)
    s.matrix(p + "attn_q.weight", heads * dim, hidden)
    s.matrix(p + "attn_k.weight", kv * dim, hidden)
    s.matrix(p + "attn_v.weight", kv * dim, hidden)
    s.matrix(p + "attn_output.weight", hidden, heads * dim)
    if norms:
        s.norm(p + "attn_q_norm.weight", dim)
        s.norm(p + "attn_k_norm.weight", dim)


def swiglu(s: Synthetic, p: str, hidden: int, width: int, suffix: str = "") -> None:
    s.matrix(f"{p}ffn_gate{suffix}.weight", width, hidden)
    s.matrix(f"{p}ffn_up{suffix}.weight", width, hidden)
    s.matrix(f"{p}ffn_down{suffix}.weight", hidden, width)


def head(s: Synthetic, hidden: int, tied: bool, gain: float = 3.0) -> None:
    s.tensor("token_embd.weight", s.normal(s.vocab, hidden))
    s.norm("output_norm.weight", hidden)
    if not tied:
        s.matrix("output.weight", s.vocab, hidden, gain=gain)


def llama(case: "Case") -> Synthetic:
    a = case.axes
    s = Synthetic(template("minicpm5"), case.name, case.vocab, case.seed)
    hidden, heads, kv, dim, layers, ffn = a["hidden"], a["heads"], a["kv_heads"], a["head_dim"], a["layers"], a["ffn"]
    for key, value in [("block_count", layers), ("context_length", 4096), ("embedding_length", hidden),
                       ("feed_forward_length", ffn), ("attention.head_count", heads), ("attention.head_count_kv", kv),
                       ("rope.freq_base", 5e6), ("attention.layer_norm_rms_epsilon", 1e-6), ("attention.key_length", dim),
                       ("attention.value_length", dim), ("vocab_size", case.vocab), ("rope.dimension_count", dim)]:
        s.key(key, value)
    head(s, hidden, tied=False)
    for layer in range(layers):
        p = f"blk.{layer}."
        common_attention(s, p, hidden, heads, kv, dim, norms=False)
        s.norm(p + "ffn_norm.weight", hidden)
        swiglu(s, p, hidden, ffn)
    s.check_keys()
    return s


def lfm2(case: "Case") -> Synthetic:
    a = case.axes
    routed = a.get("experts", 0) > 0
    s = Synthetic(template("lfm2moe" if routed else "lfm2"), case.name, case.vocab, case.seed)
    hidden, heads, kv, layers, ffn = a["hidden"], a["heads"], a["kv_heads"], a["layers"], a["ffn"]
    pattern = a["kv_pattern"]
    s.key("block_count", layers)
    s.key("context_length", 4096)
    s.key("embedding_length", hidden)
    s.key("feed_forward_length", ffn)
    s.key("attention.head_count", heads)
    s.key("attention.head_count_kv", [kv if attention else 0 for attention in pattern])
    s.key("rope.freq_base", 1e7)
    s.key("attention.layer_norm_rms_epsilon", 1e-5)
    s.key("vocab_size", case.vocab)
    s.key("shortconv.l_cache", a["conv_width"])
    dense = layers
    if routed:
        dense = a["dense_layers"]
        for key, value in [("expert_count", a["experts"]), ("expert_used_count", a["experts_used"]),
                           ("expert_feed_forward_length", a["expert_ffn"]), ("leading_dense_block_count", dense),
                           ("expert_gating_func", 2)]:
            s.key(key, value)
    # A small tied embedding keeps the residual from being dominated by the input token (greedy
    # decoding of a weak random model would otherwise repeat the input token).
    s.tensor("token_embd.weight", s.normal(case.vocab, hidden, scale=0.2))
    s.norm("token_embd_norm.weight", hidden)
    dim = hidden // heads
    for layer, attention in enumerate(pattern):
        p = f"blk.{layer}."
        if attention:
            common_attention(s, p, hidden, heads, kv, dim, norms=True)
        else:
            s.norm(p + "attn_norm.weight", hidden)
            s.matrix(p + "shortconv.in_proj.weight", 3 * hidden, hidden)
            s.tensor(p + "shortconv.conv.weight", s.normal(hidden, a["conv_width"], scale=0.5), Q.F32)
            s.matrix(p + "shortconv.out_proj.weight", hidden, hidden)
        s.norm(p + "ffn_norm.weight", hidden)
        if layer < dense:
            swiglu(s, p, hidden, ffn)
        else:
            e, f = a["experts"], a["expert_ffn"]
            s.tensor(p + "ffn_gate_inp.weight", s.normal(e, hidden, scale=1 / math.sqrt(hidden)), Q.F32)
            s.tensor(p + "exp_probs_b.bias", s.normal(e, scale=0.05), Q.F32)
            s.experts(p + "ffn_gate_exps.weight", e, f, hidden)
            s.experts(p + "ffn_up_exps.weight", e, f, hidden)
            s.experts(p + "ffn_down_exps.weight", e, hidden, f)
    s.check_keys()
    return s


def muse(case: "Case") -> Synthetic:
    a = case.axes
    s = Synthetic(template("muse"), case.name, case.vocab, case.seed)
    hidden, heads, kv, dim, layers, ffn = a["hidden"], a["heads"], a["kv_heads"], a["head_dim"], a["layers"], a["ffn"]
    for key, value in [("block_count", layers), ("context_length", 4096), ("embedding_length", hidden),
                       ("feed_forward_length", ffn), ("attention.head_count", heads), ("attention.head_count_kv", kv),
                       ("rope.freq_base", 5e5), ("attention.layer_norm_rms_epsilon", 1e-5), ("attention.key_length", dim),
                       ("attention.value_length", dim), ("final_logit_softcapping", 20.0), ("logit_scale", a["logit_scale"]),
                       ("attention.sliding_window", a["window"]), ("attention.sliding_window_pattern", 4)]:
        s.key(key, value)
    head(s, hidden, tied=False, gain=12.0)
    for layer in range(layers):
        p = f"blk.{layer}."
        s.norm(p + "attn_norm.weight", hidden)
        s.matrix(p + "attn_q.weight", heads * dim, hidden)
        s.matrix(p + "attn_k.weight", kv * dim, hidden)
        s.matrix(p + "attn_v.weight", kv * dim, hidden)
        s.matrix(p + "attn_gate.weight", heads * dim, hidden)
        s.matrix(p + "attn_output.weight", hidden, heads * dim)
        s.tensor(p + "attn_q_norm.weight", np.full(dim, 3.87, np.float32), Q.F32)
        s.tensor(p + "attn_k_norm.weight", np.ones(dim, np.float32), Q.F32)
        s.norm(p + "post_attention_norm.weight", hidden)
        s.norm(p + "ffn_norm.weight", hidden)
        swiglu(s, p, hidden, ffn)
        s.norm(p + "post_ffw_norm.weight", hidden)
    s.check_keys()
    return s


def gemma4(case: "Case") -> Synthetic:
    a = case.axes
    s = Synthetic(template(a["template"]), case.name, case.vocab, case.seed)
    hidden, heads, layers, ple = a["hidden"], a["heads"], a["layers"], a.get("ple", 0)
    sliding = a["sliding"]
    swa_dim, full_dim = a["swa_head_dim"], a["full_head_dim"]
    kv = [a["swa_kv"] if s_ else a["full_kv"] for s_ in sliding]
    shared = a.get("shared", 0)
    widths = [a["ffn"] * (2 if a.get("double_wide") and layer >= layers - shared else 1) for layer in range(layers)]
    routed = a.get("experts", 0) > 0
    s.layered("feed_forward_length", widths)
    s.layered("attention.head_count_kv", kv)
    keys = [("block_count", layers), ("context_length", 4096), ("embedding_length", hidden),
            ("attention.head_count", heads), ("rope.freq_base", 1e6), ("rope.freq_base_swa", 1e4), ("attention.layer_norm_rms_epsilon", 1e-6),
            ("attention.key_length", full_dim), ("attention.value_length", full_dim), ("final_logit_softcapping", 30.0),
            ("attention.sliding_window", a["window"]), ("attention.shared_kv_layers", shared),
            ("embedding_length_per_layer_input", ple), ("attention.sliding_window_pattern", sliding),
            ("attention.key_length_swa", swa_dim), ("attention.value_length_swa", swa_dim),
            ("rope.dimension_count", full_dim), ("rope.dimension_count_swa", swa_dim)]
    if routed:
        keys += [("expert_count", a["experts"]), ("expert_used_count", a["experts_used"]),
                 ("expert_feed_forward_length", a["expert_ffn"])]
    for key, value in keys:
        s.key(key, value)
    s.tensor("token_embd.weight", s.normal(case.vocab, hidden, scale=1 / math.sqrt(hidden)))
    s.norm("output_norm.weight", hidden)
    rotated = int(full_dim // 2 * 0.25)
    s.tensor("rope_freqs.weight", np.array([1.0] * rotated + [1e30] * (full_dim // 2 - rotated), np.float32), Q.F32)
    if ple:
        s.tensor("per_layer_token_embd.weight", s.normal(case.vocab, layers * ple))
        s.matrix("per_layer_model_proj.weight", layers * ple, hidden)
        s.norm("per_layer_proj_norm.weight", ple)
    first_shared = layers - shared
    for layer in range(layers):
        p, dim = f"blk.{layer}.", swa_dim if sliding[layer] else full_dim
        s.norm(p + "attn_norm.weight", hidden)
        s.matrix(p + "attn_q.weight", heads * dim, hidden)
        s.norm(p + "attn_q_norm.weight", dim)
        if layer < first_shared:
            s.matrix(p + "attn_k.weight", kv[layer] * dim, hidden)
            s.norm(p + "attn_k_norm.weight", dim)
            if sliding[layer] or not a.get("k_eq_v"):
                s.matrix(p + "attn_v.weight", kv[layer] * dim, hidden)
        s.matrix(p + "attn_output.weight", hidden, heads * dim)
        s.norm(p + "post_attention_norm.weight", hidden)
        s.norm(p + "ffn_norm.weight", hidden)
        swiglu(s, p, hidden, widths[layer])
        s.norm(p + "post_ffw_norm.weight", hidden)
        s.tensor(p + "layer_output_scale.weight", s.rng.uniform(0.4, 1.0, 1).astype(np.float32), Q.F32)
        if ple:
            s.matrix(p + "inp_gate.weight", ple, hidden)
            s.matrix(p + "proj.weight", hidden, ple)
            s.norm(p + "post_norm.weight", hidden)
        if routed:
            e, f = a["experts"], a["expert_ffn"]
            s.tensor(p + "ffn_gate_inp.weight", s.normal(e, hidden, scale=1 / math.sqrt(hidden)), Q.F32)
            s.norm(p + "ffn_gate_inp.scale", hidden)
            s.tensor(p + "ffn_down_exps.scale", s.rng.uniform(0.5, 1.5, e).astype(np.float32), Q.F32)
            s.experts(p + "ffn_gate_up_exps.weight", e, 2 * f, hidden)
            s.experts(p + "ffn_down_exps.weight", e, hidden, f)
            for name in ("pre_ffw_norm_2", "post_ffw_norm_1", "post_ffw_norm_2"):
                s.norm(p + name + ".weight", hidden)
    s.check_keys()
    return s


def nemotron_h(case: "Case") -> Synthetic:
    a = case.axes
    s = Synthetic(template(a["template"]), case.name, case.vocab, case.seed)
    hidden, pattern = a["hidden"], a["pattern"]
    mtp = a.get("mtp", 0)
    layers = len(pattern) + mtp
    inner, heads_ssm, state, groups, conv = a["inner"], a["ssm_heads"], a["state"], a["groups"], a["conv"]
    heads, kv, dim = a["heads"], a["kv_heads"], a["head_dim"]
    e, f, shared, latent = a["experts"], a["expert_ffn"], a["shared_ffn"], a.get("latent", 0)
    ffn = [f if kind == "E" else 0 for kind in pattern] + [f] * mtp
    kv_array = [kv if kind == "*" else 0 for kind in pattern] + [kv] * mtp
    keys = [("block_count", layers), ("context_length", 8192), ("embedding_length", hidden), ("feed_forward_length", ffn),
            ("attention.head_count", heads), ("attention.head_count_kv", kv_array), ("rope.freq_base", 1e4),
            ("attention.layer_norm_rms_epsilon", 1e-5), ("attention.layer_norm_epsilon", 1e-5), ("expert_count", e),
            ("expert_used_count", a["experts_used"]), ("expert_group_count", 1), ("expert_group_used_count", 1),
            ("vocab_size", case.vocab), ("rope.dimension_count", dim), ("ssm.conv_kernel", conv), ("ssm.state_size", state),
            ("ssm.group_count", groups), ("ssm.inner_size", inner), ("ssm.time_step_rank", heads_ssm),
            ("attention.key_length", dim), ("attention.value_length", dim),
            ("expert_feed_forward_length", f), ("expert_shared_feed_forward_length", shared), ("expert_shared_count", 1),
            ("expert_weights_norm", True), ("expert_weights_scale", a["routed_scale"])]
    if mtp:
        keys.append(("nextn_predict_layers", mtp))
    if latent:
        keys.append(("moe_latent_size", latent))
    if s.has("rope.scaling.finetuned"):
        keys.append(("rope.scaling.finetuned", False))
    for key, value in keys:
        s.key(key, value)
    head(s, hidden, tied=False)
    channels = inner + 2 * groups * state

    def moe(p: str) -> None:
        width = latent or hidden
        s.tensor(p + "ffn_gate_inp.weight", s.normal(e, hidden, scale=1 / math.sqrt(hidden)), Q.F32)
        s.tensor(p + "exp_probs_b.bias", s.normal(e, scale=0.05), Q.F32)
        s.experts(p + "ffn_up_exps.weight", e, f, width)
        s.experts(p + "ffn_down_exps.weight", e, width, f)
        s.matrix(p + "ffn_up_shexp.weight", shared, hidden)
        s.matrix(p + "ffn_down_shexp.weight", hidden, shared)
        if latent:
            s.matrix(p + "ffn_latent_down.weight", latent, hidden)
            s.matrix(p + "ffn_latent_up.weight", hidden, latent)

    for layer, kind in enumerate(pattern):
        p = f"blk.{layer}."
        s.norm(p + "attn_norm.weight", hidden)
        if kind == "M":
            s.matrix(p + "ssm_in.weight", 2 * inner + 2 * groups * state + heads_ssm, hidden)
            s.tensor(p + "ssm_conv1d.weight", s.normal(channels, conv, scale=0.5), Q.F32)
            s.tensor(p + "ssm_conv1d.bias", s.normal(channels, scale=0.1), Q.F32)
            s.tensor(p + "ssm_a", -np.arange(1, heads_ssm + 1, dtype=np.float32)[:, None] / 2, Q.F32)
            s.tensor(p + "ssm_d", 1 + s.normal(heads_ssm, 1, scale=0.1), Q.F32)
            s.tensor(p + "ssm_dt.bias", np.log(np.expm1(s.rng.uniform(0.01, 0.2, heads_ssm))).astype(np.float32), Q.F32)
            s.tensor(p + "ssm_norm.weight", 1 + s.normal(groups, inner // groups, scale=0.1), Q.F32)
            s.matrix(p + "ssm_out.weight", hidden, inner)
        elif kind == "*":
            s.matrix(p + "attn_q.weight", heads * dim, hidden)
            s.matrix(p + "attn_k.weight", kv * dim, hidden)
            s.matrix(p + "attn_v.weight", kv * dim, hidden)
            s.matrix(p + "attn_output.weight", hidden, heads * dim)
        else:
            moe(p)
    for layer in range(len(pattern), layers):
        p = f"blk.{layer}."
        for name in ("nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm", "attn_norm", "post_attention_norm"):
            s.norm(p + name + ".weight", hidden)
        s.matrix(p + "nextn.eh_proj.weight", hidden, 2 * hidden)
        s.matrix(p + "attn_q.weight", heads * dim, hidden)
        s.matrix(p + "attn_k.weight", kv * dim, hidden)
        s.matrix(p + "attn_v.weight", kv * dim, hidden)
        s.matrix(p + "attn_output.weight", hidden, heads * dim)
        moe(p)
    s.check_keys()
    return s


# ---------------------------------------------------------------------------------------------
# Cases and schedules


@dataclass(frozen=True)
class Case:
    name: str
    build: object
    axes: dict
    covers: tuple[str, ...]
    vocab: int = 256
    seed: int = 1
    schedules: tuple[str, ...] = ("prefill-decode", "chunked", "verify")
    chunk: int = 5
    prompt: int = 20
    generate: int = 12


CASES = [
    Case("llama-gqa8", llama, {"hidden": 64, "heads": 8, "kv_heads": 1, "head_dim": 16, "layers": 3, "ffn": 96},
         ("plain GQA G=8, NORM rope rows permuted in the GGUF, untied head",)),
    Case("lfm2-shortconv", lfm2, {"hidden": 64, "heads": 4, "kv_heads": 2, "layers": 6, "ffn": 96, "conv_width": 3,
                                  "kv_pattern": [0, 0, 1, 0, 1, 0]},
         ("short conv (L=3) layers before and after attention", "per-layer head_count_kv array", "tied head")),
    Case("lfm2moe-routed", lfm2, {"hidden": 64, "heads": 4, "kv_heads": 2, "layers": 5, "ffn": 96, "conv_width": 3,
                                  "kv_pattern": [0, 1, 0, 0, 1], "experts": 8, "experts_used": 2, "expert_ffn": 32,
                                  "dense_layers": 1},
         ("leading dense layer then sigmoid-routed experts with selection bias",)),
    Case("muse-nope-sandwich", muse, {"hidden": 64, "heads": 4, "kv_heads": 1, "head_dim": 16, "layers": 8, "ffn": 96,
                                      "window": 6, "logit_scale": 0.35},
         ("NoPE full layers (3, 7) and sliding window 6 with NORM rope rows permuted", "sandwich post norms (eps 1e-8)",
          "weightless embedding RMS", "logit scale then softcap 20 (logits reach the cap)", "per-element sigmoid gate, G=4")),
    Case("gemma4-ple-shared", gemma4, {"template": "gemma4-e2b", "hidden": 64, "heads": 2, "layers": 6,
                                       "sliding": [True, True, False, True, True, False], "swa_head_dim": 32,
                                       "full_head_dim": 64, "swa_kv": 1, "full_kv": 1, "window": 5, "ffn": 48,
                                       "double_wide": True, "shared": 2, "ple": 16},
         ("per-layer embeddings (PLE)", "KV-shared layers (last 2 read layers 3 and 2)", "double-wide FFN on shared "
          "layers", "proportional rope (partial 0.25) on full layers", "sandwich norms, layer output scale", "softcap 30")),
    Case("gemma4-vk-512", gemma4, {"template": "gemma4-12b", "hidden": 64, "heads": 16, "layers": 6,
                                   "sliding": [True] * 5 + [False], "swa_head_dim": 256, "full_head_dim": 512,
                                   "swa_kv": 8, "full_kv": 1, "window": 4, "ffn": 64, "k_eq_v": True},
         ("head dim 256 (sliding) and 512 (full)", "V=K on the full layer (no attn_v)", "G=16 on the full layer",
          "window 4 smaller than a chunk")),
    Case("gemma4-moe", gemma4, {"template": "gemma4-26b", "hidden": 64, "heads": 4, "layers": 6,
                                "sliding": [True] * 5 + [False], "swa_head_dim": 32, "full_head_dim": 64,
                                "swa_kv": 2, "full_kv": 1, "window": 6, "ffn": 48, "k_eq_v": True,
                                "experts": 8, "experts_used": 3, "expert_ffn": 32},
         ("dense branch plus softmax-routed experts with per-expert scale", "fused gate|up experts", "V=K full layer")),
    Case("nemotron-hybrid", nemotron_h, {"template": "nemotron-lightning", "hidden": 64, "pattern": "MEM*EM*EME",
                                         "mtp": 1, "inner": 64, "ssm_heads": 8, "state": 16, "groups": 2, "conv": 4,
                                         "heads": 4, "kv_heads": 2, "head_dim": 16, "experts": 8, "experts_used": 2,
                                         "expert_ffn": 32, "shared_ffn": 48, "routed_scale": 2.5},
         ("Mamba-2 (gate before grouped norm, conv bias, D skip), float32 state", "lone Mamba blocks (M before *)",
          "NoPE attention without q/k norm", "sigmoid routing with relu² experts and shared expert", "an MTP block "
          "(not part of the forward)")),
    Case("nemotron-latent-e512", nemotron_h, {"template": "nemotron-super", "hidden": 64, "pattern": "MEM*EME",
                                              "inner": 64, "ssm_heads": 8, "state": 16, "groups": 2, "conv": 4,
                                              "heads": 4, "kv_heads": 2, "head_dim": 16, "experts": 512,
                                              "experts_used": 22, "expert_ffn": 16, "shared_ffn": 48, "latent": 32,
                                              "routed_scale": 5.0},
         ("latent MoE with E=512, K=22", "lone Mamba block")),
]


def dflash(case: "DraftCase", hidden: int) -> Synthetic:
    """A DFlash/DSpark draft for a synthetic target of width `hidden`."""
    a = case.axes
    s = Synthetic(template(a["template"]), case.name, case.vocab, case.seed)
    heads, kv, dim, layers, ffn, taps = a["heads"], a["kv_heads"], a["head_dim"], a["layers"], a["ffn"], a["target_layers"]
    context = int(a["yarn_factor"] * a["yarn_original"]) if "yarn_factor" in a else 4096
    keys = [("block_count", layers), ("context_length", context), ("embedding_length", hidden), ("feed_forward_length", ffn),
            ("attention.head_count", heads), ("attention.head_count_kv", kv), ("rope.freq_base", a.get("rope_base", 1e6)),
            ("attention.layer_norm_rms_epsilon", 1e-6), ("attention.key_length", dim), ("attention.value_length", dim),
            ("block_size", a["block_size"]), ("target_layers", list(taps))]
    optional = {"sample_from_anchor": True, "has_confidence_head": True, "attention.sliding_window": a.get("window"),
                "attention.sliding_window_pattern": a.get("sliding"), "rope.scaling.type": "yarn",
                "rope.scaling.factor": a.get("yarn_factor"), "rope.scaling.original_context_length": a.get("yarn_original")}
    keys += [(key, value) for key, value in optional.items() if s.has(key)]
    for key, value in keys:
        s.key(key, value)
    if a.get("own_embedding"):
        s.tensor("token_embd.weight", s.normal(case.vocab, hidden))
    s.matrix("fc.weight", hidden, len(taps) * hidden)
    s.norm("enc.output_norm.weight", hidden)
    s.norm("output_norm.weight", hidden)
    for layer in range(layers):
        p = f"blk.{layer}."
        common_attention(s, p, hidden, heads, kv, dim, norms=True)
        s.norm(p + "ffn_norm.weight", hidden)
        swiglu(s, p, hidden, ffn)
    if rank := a.get("markov_rank"):
        s.tensor("markov_w1.weight", s.normal(case.vocab, rank))
        s.tensor("markov_w2.weight", s.normal(case.vocab, rank, scale=1 / math.sqrt(rank)))
        width = hidden + rank
        conf = s.normal(width, scale=1 / math.sqrt(width))
        s.tensor("conf_proj.weight", conf if a["conf_vector"] else conf[None, :])
        s.tensor("conf_proj.bias", s.normal(1, scale=0.1), Q.F32)
    s.check_keys()
    return s


@dataclass(frozen=True)
class DraftCase:
    """A draft over the synthetic model of target case `target` (same vocabulary and width). Each anchor n
    is a draft pass after the target ran greedy[:n]; greedy[n] is the anchor."""
    name: str
    target: str
    axes: dict
    covers: tuple[str, ...]
    anchors: tuple[int, ...]
    vocab: int = 256
    seed: int = 2


DRAFT_CASES = [
    DraftCase("dspark-on-llama", "llama-gqa8",
              {"template": "dspark-minicpm5", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 5, "target_layers": (1, 2, 3), "markov_rank": 8, "conf_vector": False, "rope_base": 5e6},
              ("DSpark: sample from the anchor, Markov bias chained greedily, confidence head",
               "a tap at block_count (the final pre-norm residual)", "untied target head"), anchors=(20, 25)),
    DraftCase("dspark-on-lfm2", "lfm2-shortconv",
              {"template": "dspark-lfm2", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 4, "target_layers": (1, 3, 6), "markov_rank": 8, "conf_vector": True, "rope_base": 1e7},
              ("DSpark without a sample_from_anchor key (the DSpark default)", "tied target head, taps at conv and "
               "attention layer inputs"), anchors=(20, 27)),
    DraftCase("dflash-swa-on-muse", "muse-nope-sandwich",
              {"template": "dflash-muse", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 6, "target_layers": (2, 5, 8), "window": 3, "sliding": [True, False], "rope_base": 5e5},
              ("DFlash: mask slots predict positions n+1.., no logit scale or softcap on the target head",
               "a sliding draft layer with window 3 smaller than the context (|q-k| <= W, bidirectional)"),
              anchors=(20, 26)),
    DraftCase("dflash-yarn-on-nemotron", "nemotron-hybrid",
              {"template": "dflash-nemotron", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 4, "target_layers": (2, 6, 10), "own_embedding": True, "yarn_factor": 4.0,
               "yarn_original": 16, "rope_base": 1e4},
              ("DFlash with YaRN rope and its own embedding table", "taps across Mamba, attention and MoE layers"),
              anchors=(20,)),
]
TEMPLATES |= {
    "dspark-minicpm5": ("openbmb/MiniCPM5-2B-DSpark-GGUF", "a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3", "MiniCPM5-2.6B-DSpark.gguf"),
    "dspark-lfm2": ("LiquidAI/LFM2.5-2.6B-DSpark-GGUF", "7bc2896af56d82ccc7e156800197408db464d63b", "LFM2.5-2.6B-DSpark-Q8_0.gguf"),
    "dflash-muse": ("unsloth/Muse-Glimmer-30B-GGUF", "1afeb8e879f60116d206cf724425dbe1e1a2f7f5", "dflash-kquant.gguf"),
    "dflash-nemotron": ("magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash-GGUF",
                        "cb86a5267e902064e551f028e97abb11990c258d", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash.gguf"),
}


def run_draft_case(case: DraftCase, output: Path, transformers: bool) -> dict:
    target_dir = output / case.target
    target_record = json.loads((target_dir / "reference.json").read_text())
    directory = output / case.name
    directory.mkdir(parents=True, exist_ok=True)
    target = open_reference(target_dir / "model.gguf", torch.device("cpu"), cache=True)
    draft_path = directory / "draft.gguf"
    dflash(case, target.hidden_size).write(draft_path)
    draft = DFlashReference(Package.local(draft_path), Weights(Package.local(draft_path), torch.device("cpu"), True), target)
    greedy_tokens = target_record["greedy"]
    passes = []
    for anchor in case.anchors:
        with torch.no_grad():
            proposal = draft.propose(torch.tensor([greedy_tokens[:anchor + 1]]))
        logits_file = f"anchor-{anchor}.logits.npy"
        np.save(directory / logits_file, to_numpy(proposal.logits[0]))
        passes.append({"committed": anchor, "anchor": greedy_tokens[anchor], "positions": proposal.positions,
                       "proposals": proposal.tokens[0].tolist(), "logits": logits_file,
                       "confidence": None if proposal.confidence is None else proposal.confidence[0].tolist()})
    record = {
        "case": case.name, "target_case": case.target, "target_model": f"../{case.target}/model.gguf",
        "draft": "draft.gguf", "draft_sha256": hashlib.sha256(draft_path.read_bytes()).hexdigest(),
        "dspark": draft.dspark, "sample_from_anchor": draft.from_anchor, "block_size": draft.block_size,
        "target_layers": list(draft.target_layers), "mask_token": draft.mask_token, "covers": list(case.covers),
        "axes": case.axes, "passes": passes, "tensors_not_in_forward": draft.weights.unread(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((ROOT / "dflash_reference.py").read_bytes()).hexdigest(),
    }
    if transformers:
        features = torch.randn(1, 12, len(draft.target_layers) * draft.hidden_size, generator=torch.Generator().manual_seed(3))
        record["released_code_check"] = draft.check_released(features, torch.tensor([greedy_tokens[0]]))
    (directory / "reference.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def forward_rows(reference, prefix: list[int], rows: list[int]) -> np.ndarray:
    tokens = torch.tensor([prefix + rows], device=reference.weights.device)
    with torch.no_grad():
        logits = reference.forward(tokens)[0, len(prefix):]
    return to_numpy(logits)


def greedy(reference, prompt: list[int], count: int) -> list[int]:
    sequence = list(prompt)
    for _ in range(count):
        sequence.append(int(forward_rows(reference, sequence[:-1], sequence[-1:])[-1].argmax()))
    return sequence


def schedule(kind: str, case: Case, sequence: list[int], vocab: int) -> list[dict]:
    """Steps over the greedy sequence. A step feeds `tokens` at positions start.. and keeps the first
    `keep` rows in the state (the rest are rolled back); the next step starts at start + keep."""
    prompt = case.prompt
    if kind == "prefill-decode":
        steps = [{"start": 0, "tokens": sequence[:prompt]}]
        steps += [{"start": t, "tokens": [sequence[t]]} for t in range(prompt, len(sequence) - 1)]
        return [step | {"keep": len(step["tokens"])} for step in steps]
    if kind == "chunked":
        return [{"start": t, "tokens": sequence[t:min(t + case.chunk, len(sequence) - 1)],
                 "keep": min(case.chunk, len(sequence) - 1 - t)} for t in range(0, len(sequence) - 1, case.chunk)]
    if kind == "verify":
        steps = [{"start": 0, "tokens": sequence[:prompt], "keep": prompt}]
        position, accepted_runs = prompt, [3, 0, 2, 5]
        for run in accepted_runs:
            if position + run + 2 > len(sequence):
                break
            drafts = sequence[position + 1:position + 1 + run] + [(sequence[position + 1 + run] + 1) % vocab]
            steps.append({"start": position, "tokens": [sequence[position]] + drafts, "keep": run + 1})
            position += run + 1
        return steps
    raise ValueError(kind)


def run_case(case: Case, output: Path, transformers: bool) -> dict:
    directory = output / case.name
    directory.mkdir(parents=True, exist_ok=True)
    synthetic = case.build(case)
    model = directory / "model.gguf"
    synthetic.write(model)
    reference = open_reference(model, torch.device("cpu"), cache=True)
    rng = np.random.default_rng(case.seed + 1000)
    prompt = rng.integers(0, case.vocab, case.prompt).tolist()
    sequence = greedy(reference, prompt, case.generate)
    with torch.no_grad():
        tokens = torch.tensor([sequence])
        _, taps = reference.hidden(tokens, taps=tuple(range(reference.block_count + 1)))
    np.save(directory / "residuals.npy", np.stack([to_numpy(t[0]) for t in taps]))
    sequences = []
    for kind in case.schedules:
        steps = schedule(kind, case, sequence, case.vocab)
        committed, rows = [], []
        for index, step in enumerate(steps):
            if step["start"] != len(committed):
                raise AssertionError(f"{case.name}/{kind}: step {index} starts at {step['start']}, state holds {len(committed)}")
            logits = forward_rows(reference, committed, step["tokens"])
            step["rows"] = [len(rows) + r for r in range(len(step["tokens"]))]
            step["argmax"] = logits.argmax(-1).tolist()
            rows.extend(logits)
            committed += step["tokens"][:step["keep"]]
        np.save(directory / f"{kind}.logits.npy", np.stack(rows).astype(np.float32))
        sequences.append({"name": kind, "logits": f"{kind}.logits.npy", "steps": steps})
    unread = reference.weights.unread()
    record = {
        "case": case.name, "architecture": reference.architecture, "reference": type(reference).__module__,
        "covers": list(case.covers), "axes": case.axes, "vocab": case.vocab, "seed": case.seed,
        "model": "model.gguf", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "prompt": prompt, "greedy": sequence, "residuals": "residuals.npy",
        "residuals_layout": "[block_count + 1, T, hidden]: residual entering each layer, then the final pre-norm residual",
        "sequences": sequences, "tensors_not_in_forward": unread,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((ROOT / f"{type(reference).__module__}.py").read_bytes()).hexdigest(),
        "torch": torch.__version__,
    }
    if transformers:
        record["transformers_check"] = check_hf(reference, 2, 32, 7)
    (directory / "reference.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", help="comma-separated case names (default: all)")
    parser.add_argument("--no-transformers", action="store_true",
                        help="skip the per-case cross-check against transformers / the released draft code")
    parser.add_argument("--list", action="store_true")
    options = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    names = None if options.cases is None else set(options.cases.split(","))
    drafts = [case for case in DRAFT_CASES if names is None or case.name in names]
    # A draft case reads its target case's model and greedy sequence: select those targets too.
    wanted = None if names is None else names | {case.target for case in drafts}
    targets = [case for case in CASES if wanted is None or case.name in wanted]
    if options.list:
        for case in [*targets, *drafts]:
            print(f"{case.name}: {'; '.join(case.covers)}")
        return 0
    failures = []
    for case in targets:
        record = run_case(case, options.output, not options.no_transformers)
        check = record.get("transformers_check", {})
        if check and not check["pass"]:
            failures.append(case.name)
        print(f"{case.name}: {record['architecture']}, greedy {record['greedy'][case.prompt:]}, "
              f"transformers {check.get('pass', 'skipped')} (rel {check.get('relative_max_difference', float('nan')):.2e})",
              flush=True)
    for case in drafts:
        record = run_draft_case(case, options.output, not options.no_transformers)
        check = record.get("released_code_check", {})
        if check and not check["pass"]:
            failures.append(case.name)
        print(f"{case.name}: proposals {[p['proposals'] for p in record['passes']]}, released code "
              f"{check.get('pass', 'skipped')} {json.dumps({k: v for k, v in check.items() if k != 'pass'})}", flush=True)
    if failures:
        print(f"disagreement with the released code: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
