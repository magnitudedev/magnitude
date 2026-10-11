"""Synthetic variation models of the in-scope families, for validating the references themselves.

Each case is a small synthetic GGUF of one family, built from the metadata of a locked real GGUF of
that family (every `<architecture>.*` key of the real header is written, with synthetic values, and no
other architecture key; the tokenizer keeps the real model/pre/special-token keys over a small
vocabulary). `check_families` runs every case's reference against transformers (and every draft case
against the released draft code), covering the family variation axes listed in each case. Templates
are read once from Hugging Face by byte range and cached under `gguf.HEADER_CACHE`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

from magnitude_references.checks import check_hf
from magnitude_references.families import open_reference
from magnitude_references.families.dflash import DFlashReference, check_draft
from magnitude_references.gguf import HEADER_CACHE, Package, RemoteSource, Value, Writer, quantize
from magnitude_references.model import Weights

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
    "qwen35": ("unsloth/Qwen3.5-4B-MTP-GGUF", "86835bf9949e4d14d6860f7910b1340ad4f271a9", "Qwen3.5-4B-UD-Q4_K_XL.gguf"),
    "qwen35moe": ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
                  "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
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


def qwen35(case: "Case") -> Synthetic:
    a = case.axes
    routed = a.get("experts", 0) > 0
    s = Synthetic(template("qwen35moe" if routed else "qwen35"), case.name, case.vocab, case.seed)
    hidden, heads, kv, dim, layers, mtp = a["hidden"], a["heads"], a["kv_heads"], a["head_dim"], a["layers"], a["mtp"]
    k_heads, v_heads, k_width, v_width, conv = a["k_heads"], a["v_heads"], a["k_width"], a["v_width"], a["conv"]
    keys = [("block_count", layers + mtp), ("context_length", 4096), ("embedding_length", hidden),
            ("attention.head_count", heads), ("attention.head_count_kv", kv), ("rope.dimension_sections", [1, 1, 0, 0]),
            ("rope.freq_base", 1e7), ("attention.layer_norm_rms_epsilon", 1e-6), ("attention.key_length", dim),
            ("attention.value_length", dim), ("ssm.conv_kernel", conv), ("ssm.state_size", k_width),
            ("ssm.group_count", k_heads), ("ssm.time_step_rank", v_heads), ("ssm.inner_size", v_heads * v_width),
            ("full_attention_interval", a["interval"]), ("rope.dimension_count", dim // 4), ("nextn_predict_layers", mtp)]
    if routed:
        keys += [("expert_count", a["experts"]), ("expert_used_count", a["experts_used"]),
                 ("expert_feed_forward_length", a["expert_ffn"]), ("expert_shared_feed_forward_length", a["shared_ffn"])]
    else:
        keys.append(("feed_forward_length", a["ffn"]))
    for key, value in keys:
        s.key(key, value)
    s.tensor("token_embd.weight", s.normal(case.vocab, hidden))
    s.norm("output_norm.weight", hidden)
    if not a.get("tied"):
        s.matrix("output.weight", case.vocab, hidden, gain=3.0)

    def attention(p: str) -> None:
        s.matrix(p + "attn_q.weight", heads * dim * 2, hidden)
        s.matrix(p + "attn_k.weight", kv * dim, hidden)
        s.matrix(p + "attn_v.weight", kv * dim, hidden)
        s.matrix(p + "attn_output.weight", hidden, heads * dim)
        s.norm(p + "attn_q_norm.weight", dim)
        s.norm(p + "attn_k_norm.weight", dim)

    def feed_forward(p: str) -> None:
        if not routed:
            swiglu(s, p, hidden, a["ffn"])
            return
        e, f, shared = a["experts"], a["expert_ffn"], a["shared_ffn"]
        s.tensor(p + "ffn_gate_inp.weight", s.normal(e, hidden, scale=1 / math.sqrt(hidden)), Q.F32)
        s.experts(p + "ffn_gate_exps.weight", e, f, hidden)
        s.experts(p + "ffn_up_exps.weight", e, f, hidden)
        s.experts(p + "ffn_down_exps.weight", e, hidden, f)
        swiglu(s, p, hidden, shared, "_shexp")
        s.tensor(p + "ffn_gate_inp_shexp.weight", s.normal(hidden, scale=1 / math.sqrt(hidden)), Q.F32)

    channels = 2 * k_heads * k_width + v_heads * v_width
    for layer in range(layers + mtp):
        p = f"blk.{layer}."
        s.norm(p + "attn_norm.weight", hidden)
        s.norm(p + "post_attention_norm.weight", hidden)
        if layer >= layers or (layer + 1) % a["interval"] == 0:
            attention(p)
        else:
            s.matrix(p + "attn_qkv.weight", channels, hidden)
            s.matrix(p + "attn_gate.weight", v_heads * v_width, hidden)
            s.matrix(p + "ssm_beta.weight", v_heads, hidden)
            s.matrix(p + "ssm_alpha.weight", v_heads, hidden)
            s.tensor(p + "ssm_conv1d.weight", s.normal(channels, conv, scale=0.5), Q.F32)
            s.tensor(p + "ssm_a", -s.rng.uniform(0.5, 4.0, v_heads).astype(np.float32), Q.F32)
            s.tensor(p + "ssm_dt.bias", s.normal(v_heads, scale=0.5), Q.F32)
            s.norm(p + "ssm_norm.weight", v_width)
            s.matrix(p + "ssm_out.weight", hidden, v_heads * v_width)
        feed_forward(p)
        if layer >= layers:
            for name in ("nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"):
                s.norm(p + name + ".weight", hidden)
            s.matrix(p + "nextn.eh_proj.weight", hidden, 2 * hidden)
    s.check_keys()
    return s


# ---------------------------------------------------------------------------------------------
# Cases


@dataclass(frozen=True)
class Case:
    name: str
    build: object
    axes: dict
    covers: tuple[str, ...]
    vocab: int = 256
    seed: int = 1


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
    Case("muse-metal-window", muse, {"hidden": 256, "heads": 4, "kv_heads": 1, "head_dim": 128, "layers": 8, "ffn": 512,
                                      "window": 6, "logit_scale": 0.35},
         ("Metal-supported attention geometry for a small-window draft comparison",)),
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
    Case("qwen35-delta-net", qwen35, {"hidden": 64, "heads": 4, "kv_heads": 2, "head_dim": 32, "layers": 6, "mtp": 1,
                                      "interval": 3, "k_heads": 2, "v_heads": 6, "k_width": 16, "v_width": 8, "conv": 4,
                                      "ffn": 96, "tied": True},
         ("gated delta net with 3 v heads per k head (tiled v-head order in the GGUF)", "gated attention with q/k "
          "norm and partial rope (head 32, rope 8)", "zero-centered norms stored as 1 + w", "tied head",
          "an MTP block (not part of the forward)")),
    Case("qwen35moe-routed", qwen35, {"hidden": 64, "heads": 4, "kv_heads": 1, "head_dim": 32, "layers": 4, "mtp": 1,
                                      "interval": 4, "k_heads": 2, "v_heads": 4, "k_width": 16, "v_width": 16, "conv": 4,
                                      "experts": 16, "experts_used": 4, "expert_ffn": 32, "shared_ffn": 48},
         ("softmax-routed SwiGLU experts, renormalized top-k", "sigmoid-gated shared expert", "untied head")),
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
    optional = {"sample_from_anchor": a.get("sample_from_anchor", True), "has_confidence_head": True, "attention.sliding_window": a.get("window"),
                "attention.sliding_window_pattern": a.get("sliding"), "rope.scaling.type": "yarn",
                "rope.scaling.factor": a.get("yarn_factor"), "rope.scaling.original_context_length": a.get("yarn_original")}
    keys += [(key, value) for key, value in optional.items() if s.has(key)]
    for key, value in keys:
        s.key(key, value)
    if s.has("rope.dimension_sections"):
        s.key("rope.dimension_sections", [dim // 2, 0, 0])
    if a.get("conv_kernel_size"):
        for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            s.key(key, a[key])
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
        if kernel := a.get("conv_kernel_size"):
            groups = hidden // a["conv_group_size"]
            for operator in ("attn", "ffn"):
                base = s.normal(2, kernel, hidden, scale=0.1)
                base[:, 0] += 1.0
                s.tensor(p + operator + "_conv_base", base)
                s.matrix(p + operator + "_conv_proj.weight", 2 * kernel * groups, hidden, gain=0.1)
    if rank := a.get("selector_rank"):
        s.matrix("selector_hidden.weight", rank, hidden)
        s.tensor("selector_predecessor.weight", s.normal(case.vocab, rank))
        s.tensor("selector_successor.weight", s.normal(case.vocab, rank, scale=1 / math.sqrt(rank)))
    if rank := a.get("markov_rank"):
        s.tensor("markov_w1.weight", s.normal(case.vocab, rank))
        s.tensor("markov_w2.weight", s.normal(case.vocab, rank, scale=1 / math.sqrt(rank)))
        width = hidden + rank
        conf = s.normal(width, scale=1 / math.sqrt(width))
        s.tensor("conf_proj.weight", conf if a["conf_vector"] else conf[None, :])
        s.tensor("conf_proj.bias", s.normal(1, scale=0.1), Q.F32)
    # Publication metadata is explicit even when the pinned historical template omitted it.
    causal = a.get("causal", False)
    s.metadata["dflash.attention.causal"] = Value(V.ARRAY, causal, V.BOOL) if isinstance(causal, list) else Value(V.BOOL, causal)
    s.check_keys()
    return s


@dataclass(frozen=True)
class DraftCase:
    """A draft over the synthetic model of target case `target` (same vocabulary and width)."""
    name: str
    target: str
    axes: dict
    covers: tuple[str, ...]
    vocab: int = 256
    seed: int = 2


DRAFT_CASES = [
    DraftCase("dspark-on-llama", "llama-gqa8",
              {"template": "dspark-minicpm5", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 5, "target_layers": (1, 2, 3), "markov_rank": 8, "conf_vector": False, "rope_base": 5e6},
              ("DSpark: sample from the anchor, Markov bias chained greedily, confidence head",
               "a tap at block_count (the final pre-norm residual)", "untied target head")),
    DraftCase("dspark-on-lfm2", "lfm2-shortconv",
              {"template": "dspark-lfm2", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 4, "target_layers": (1, 3, 6), "markov_rank": 8, "conf_vector": True, "rope_base": 1e7},
              ("DSpark without a sample_from_anchor key (the DSpark default)", "tied target head, taps at conv and "
               "attention layer inputs")),
    DraftCase("dflash-swa-on-muse", "muse-nope-sandwich",
              {"template": "dflash-muse", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 6, "target_layers": (2, 5, 8), "window": 3, "sliding": [True, False], "rope_base": 5e5},
              ("DFlash: mask slots predict positions n+1.., no logit scale or softcap on the target head",
               "a sliding draft layer with window 3 smaller than the context (|q-k| < W, bidirectional)")),
    DraftCase("dflash-swa-metal", "muse-metal-window",
              {"template": "dflash-muse", "heads": 4, "kv_heads": 2, "head_dim": 128, "layers": 2, "ffn": 512,
               "block_size": 6, "target_layers": (2, 5, 8), "window": 3, "sliding": [True, False], "rope_base": 5e5},
              ("native Metal dimensions with a bidirectional window smaller than the physical block",)),
    DraftCase("dflash2-swa-metal", "muse-metal-window",
              {"template": "dflash2-qwen38", "heads": 4, "kv_heads": 2, "head_dim": 128, "layers": 2, "ffn": 512,
               "block_size": 6, "target_layers": (2, 5, 8), "window": 3, "sliding": [True, True],
               "conv_kernel_size": 4, "conv_group_size": 32, "selector_rank": 16, "selector_top_k": 8,
               "sample_from_anchor": False},
              ("DFlash2 grouped causal convolution across full and shortened blocks", "candidate selector chained from the anchor",
               "bidirectional window smaller than the physical block", "raw draft logits before selector bias despite target softcap")),
    DraftCase("dflash-yarn-on-nemotron", "nemotron-hybrid",
              {"template": "dflash-nemotron", "heads": 4, "kv_heads": 2, "head_dim": 16, "layers": 2, "ffn": 96,
               "block_size": 4, "target_layers": (2, 6, 10), "own_embedding": True, "yarn_factor": 4.0,
               "yarn_original": 16, "rope_base": 1e4},
              ("DFlash with YaRN rope and its own embedding table", "taps across Mamba, attention and MoE layers")),
]
TEMPLATES |= {
    "dflash2-qwen38": ("z-lab/Qwen3.8-27B-DFlash2-GGUF", "2d9571f8ce46e151f61c6499c99dee6079e1d610", "Qwen3.8-27B-DFlash2-Q8_0.gguf"),
    "dspark-minicpm5": ("openbmb/MiniCPM5-2B-DSpark-GGUF", "a261d2b4abc9c9ebfbad2af8a817a09802fc4ca3", "MiniCPM5-2.6B-DSpark.gguf"),
    "dspark-lfm2": ("LiquidAI/LFM2.5-2.6B-DSpark-GGUF", "7bc2896af56d82ccc7e156800197408db464d63b", "LFM2.5-2.6B-DSpark-Q8_0.gguf"),
    "dflash-muse": ("unsloth/Muse-Glimmer-30B-GGUF", "1afeb8e879f60116d206cf724425dbe1e1a2f7f5", "dflash-kquant.gguf"),
    "dflash-nemotron": ("magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash-GGUF",
                        "cb86a5267e902064e551f028e97abb11990c258d", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash.gguf"),
}


def selected(names: set[str] | None) -> tuple[list[Case], list[DraftCase]]:
    """The named cases (all when None); a draft case brings its target case."""
    drafts = [case for case in DRAFT_CASES if names is None or case.name in names]
    wanted = None if names is None else names | {case.target for case in drafts}
    targets = [case for case in CASES if wanted is None or case.name in wanted]
    known = {case.name for case in CASES} | {case.name for case in DRAFT_CASES}
    if names is not None and names - known:
        raise ValueError(f"unknown cases: {sorted(names - known)}")
    return targets, drafts


def check_families(names: set[str] | None, device: torch.device, zlab_source: Path | None) -> list[dict]:
    """Every selected case's reference against transformers and every draft case against the released
    draft code, over synthetic GGUFs written to a temporary directory. Each report carries `pass`."""
    targets, drafts = selected(names)
    if any(case.axes.get("conv_kernel_size") for case in drafts) and zlab_source is None:
        raise ValueError("DFlash2 released-code checks require a pinned Z-Lab model.py (zlab_source)")
    reports = []
    with tempfile.TemporaryDirectory(prefix="magnitude-synthetic-") as temporary:
        directory = Path(temporary)
        references = {}
        for case in targets:
            model = directory / f"{case.name}.gguf"
            case.build(case).write(model)
            reference = open_reference(model, device, cache=True)
            references[case.name] = reference
            report = check_hf(reference, 2, 32, 7)
            reports.append({"case": case.name, "architecture": reference.architecture, "covers": list(case.covers),
                            "tensors_not_in_forward": reference.weights.unread(), **report})
        for case in drafts:
            target = references[case.target]
            path = directory / f"{case.name}.gguf"
            dflash(case, target.hidden_size).write(path)
            package = Package.local(path)
            draft = DFlashReference(package, Weights(package, device, True), target)
            source = "dspark" if draft.dspark else "zlab" if draft.dflash2 else "muse"
            source_config = None
            if draft.dflash2:
                source_config = directory / f"{case.name}.source-config.json"
                source_config.write_text(json.dumps({"is_causal": case.axes.get("causal", False)}) + "\n")
            report = check_draft(draft, 12, 3, None, source, zlab_source if draft.dflash2 else None, source_config)
            reports.append({"case": case.name, "target_case": case.target, "covers": list(case.covers), **report})
    return reports
