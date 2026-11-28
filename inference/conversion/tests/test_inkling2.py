"""Tests for Inkling 2 lossless conversion: Inkling 2 Small (NVFP4 + MXFP8 + BF16, sharded),
Inkling 2 Mini (looped core + controllers), TML-d24-varloop-research (`is_varloop=true`),
bundled DSpark (`mtp.safetensors` -> `dflash`), and bundled vision (`model.visual.*` -> `mmproj`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conversion.base import gguf
from safetensors.torch import save_file
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from magnitude_conversion import __main__ as cli
from magnitude_conversion import check, inkling2

URL = "https://huggingface.co/thinkingmachines/Inkling-2-Small/tree/0000000000000000000000000000000000000000"
T = gguf.GGMLQuantizationType

HIDDEN = 128
INTER = 128
EXPERTS = 4
HEAD_DIM = 32
SWA_HEAD_DIM = 32
HEADS = 4
KV_HEADS = 2
SWA_HEADS = 4
SWA_KV_HEADS = 2
D_REL = 16
REL_EXTENT = 64
SWA_WINDOW = 32
CONV_K = 4
VOCAB_PADDED = 288
VOCAB_UNPADDED = 272
VISION_HIDDEN = 64
VISION_INTER = 128
VISION_HEADS = 4
VISION_LAYERS = 2
PATCH_SIZE = 4
GRID_SIZE = 4
MERGE_SIZE = 2
MTP_LAYERS = 2
MTP_HEADS = 4
MTP_KV_HEADS = 2
MTP_HEAD_DIM = 32
MTP_INTER = 128
MTP_MARKOV_RANK = 32
MTP_BLOCK_SIZE = 8


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


_E4M3 = _e4m3_table()
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def _write_tokenizer(path: Path, *, use_inkling2_pre: bool = True) -> None:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    extra_needed = VOCAB_UNPADDED - len(alphabet)
    special = ["<|endoftext|>"] + [f"<|special_{i}|>" for i in range(1, extra_needed)]
    tokenizer = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    if use_inkling2_pre:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(inkling2._INKLING2_PRE_REGEX), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(tok, special=True, normalized=False) for tok in special])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "backend": "tokenizers",
        "eos_token": "<|endoftext|>",
        "extra_special_tokens": special,
        "tokenizer_class": "TokenizersBackend",
    }))


def _make_nvfp4(rng: np.random.Generator, experts: int, rows: int, cols: int) -> dict[str, torch.Tensor]:
    packed = torch.from_numpy(rng.integers(0, 256, size=(experts, rows, cols // 2), dtype=np.uint8))
    # Positive finite E4M3 codes (0x01..0x7E)
    scale_u8 = rng.integers(1, 0x7F, size=(experts, rows, cols // 16), dtype=np.uint8)
    scale = torch.from_numpy(scale_u8).view(torch.float8_e4m3fn)
    amax = torch.from_numpy(rng.uniform(0.5, 4.0, size=(experts,)).astype(np.float32))
    denom = torch.tensor([6.0 * 448.0], dtype=torch.float32)
    return {
        "": packed,
        "_scale": scale,
        "_amax": amax,
        "_outer_scale_denominator": denom,
    }


def _make_mxfp8(rng: np.random.Generator, experts: int, rows: int, cols: int) -> dict[str, torch.Tensor]:
    # Finite E4M3 codes (exclude 0x7F and 0xFF NaN)
    codes = rng.integers(0, 256, size=(experts, rows, cols), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x38
    weight = torch.from_numpy(codes).view(torch.float8_e4m3fn)
    # Valid UE8M0 bytes in [115, 140] (well within <= 246)
    scale = torch.from_numpy(rng.integers(115, 141, size=(experts, rows, cols // 32), dtype=np.uint8))
    return {"": weight, "_scale": scale}


def _bf16(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    return torch.from_numpy(rng.normal(0.0, 0.1, size=shape).astype(np.float32)).to(torch.bfloat16)


def _f32(rng: np.random.Generator, *shape: int) -> torch.Tensor:
    return torch.from_numpy(rng.normal(0.0, 0.1, size=shape).astype(np.float32))


def _build_checkpoint(
    path: Path,
    *,
    n_layers: int = 4,
    local_layers: list[int] | None = None,
    layer_quant: dict[int, str] | None = None,
    loop_config: dict[str, Any] | None = None,
    include_vision: bool = True,
    include_mtp: bool = True,
    seed: int = 42,
) -> tuple[Path, dict[str, torch.Tensor]]:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if local_layers is None:
        local_layers = [i for i in range(n_layers) if i % 2 == 0]
    if layer_quant is None:
        layer_quant = {0: "nvfp4", 1: "mxfp8", 2: "bf16", 3: "nvfp4"}
    is_loop = loop_config is not None
    block_size = 2

    text_cfg: dict[str, Any] = {
        "model_type": "inkling2_text",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "moe_intermediate_size": INTER,
        "num_hidden_layers": n_layers,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "swa_num_attention_heads": SWA_HEADS,
        "swa_num_key_value_heads": SWA_KV_HEADS,
        "head_dim": HEAD_DIM,
        "swa_head_dim": SWA_HEAD_DIM,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "route_scale": 2.5,
        "vocab_size": VOCAB_PADDED,
        "unpadded_vocab_size": VOCAB_UNPADDED,
        "eos_token_id": 256,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "sliding_window_size": SWA_WINDOW,
        "local_layer_ids": local_layers,
        "sconv_kernel_size": CONV_K,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "n_layers_per_attn_res_block": block_size,
        "final_logit_softcapping": 30.0,
        "d_rel": D_REL,
        "rel_extent": REL_EXTENT,
        "qk_l2_clip_threshold": 50.0,
        "rel_l2_clip_threshold": 10.0,
        "log_scaling_n_floor": 512,
        "log_scaling_alpha": 0.25,
        "use_gate_bias": True,
    }
    if loop_config is not None:
        text_cfg["loop_config"] = loop_config

    root_cfg: dict[str, Any] = {
        "model_type": "inkling2_loop" if is_loop else "inkling2",
        "architectures": [
            "Inkling2LoopForConditionalGeneration" if is_loop else "Inkling2ForConditionalGeneration"
        ],
        "text_config": text_cfg,
    }

    if include_vision:
        root_cfg["vision_config"] = {
            "hidden_size": VISION_HIDDEN,
            "intermediate_size": VISION_INTER,
            "num_hidden_layers": VISION_LAYERS,
            "num_attention_heads": VISION_HEADS,
            "native_patch_size": PATCH_SIZE,
            "adapter_merge_size": MERGE_SIZE,
            "posemb_grid_size": GRID_SIZE,
            "layer_norm_eps": 1e-6,
            "num_channels": 3,
            "hidden_act": "gelu",
        }

    if include_mtp:
        n_visits = n_layers
        if loop_config is not None:
            n_visits += (loop_config["num_passes"] - 1) * (loop_config["core_end"] - loop_config["core_start"])
        root_cfg["mtp_config"] = {
            "num_hidden_layers": MTP_LAYERS,
            "intermediate_size": MTP_INTER,
            "num_attention_heads": MTP_HEADS,
            "num_key_value_heads": MTP_KV_HEADS,
            "head_dim": MTP_HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "block_size": MTP_BLOCK_SIZE,
            "markov_rank": MTP_MARKOV_RANK,
            "target_layer_ids": [0, 1, n_visits],
            "local_layer_ids": [0],
            "sliding_window_size": SWA_WINDOW,
        }

    (path / "config.json").write_text(json.dumps(root_cfg, indent=2))

    # Build text tensors (using Thinking Machines native `model.llm.*` prefix)
    tensors: dict[str, torch.Tensor] = {
        "model.llm.embed.weight": _bf16(rng, VOCAB_PADDED, HIDDEN),
        "model.llm.embed_norm.weight": _bf16(rng, HIDDEN),
        "model.llm.unembed.weight": _bf16(rng, VOCAB_PADDED, HIDDEN),
        "model.llm.norm.weight": _bf16(rng, HIDDEN),
        "model.llm.attn_res.output_query": _f32(rng, HIDDEN),
    }

    nvfp4_names: list[str] = []
    mxfp8_names: list[str] = []

    for i in range(n_layers):
        swa = i in local_layers
        lp = f"model.llm.layers.{i}."
        n_h = SWA_HEADS if swa else HEADS
        n_kv = SWA_KV_HEADS if swa else KV_HEADS
        hd = SWA_HEAD_DIM if swa else HEAD_DIM
        extent = SWA_WINDOW if swa else REL_EXTENT

        tensors[lp + "attn.wq_du.weight"] = _bf16(rng, n_h * hd, HIDDEN)
        tensors[lp + "attn.wk_dv.weight"] = _bf16(rng, n_kv * hd, HIDDEN)
        tensors[lp + "attn.wv_dv.weight"] = _bf16(rng, n_kv * hd, HIDDEN)
        tensors[lp + "attn.wo_ud.weight"] = _bf16(rng, HIDDEN, n_h * hd)
        tensors[lp + "attn.gate_attn.weight"] = _bf16(rng, n_h * hd, HIDDEN)
        tensors[lp + "attn.wr_du.weight"] = _bf16(rng, n_h * D_REL, HIDDEN)
        tensors[lp + "attn.attn_sink_proj.weight"] = _bf16(rng, n_h, HIDDEN)

        tensors[lp + "attn.q_l2_clip.scale"] = _f32(rng, n_h, hd)
        tensors[lp + "attn.rel_l2_clip.scale"] = _f32(rng, n_h, D_REL)
        tensors[lp + "attn.rel_logits_proj.proj"] = _f32(rng, D_REL, extent)

        tensors[lp + "attn_sconv.weight"] = _f32(rng, HIDDEN, 1, CONV_K)
        tensors[lp + "mlp_sconv.weight"] = _f32(rng, HIDDEN, 1, CONV_K)
        tensors[lp + "mlp_norm.weight"] = _bf16(rng, HIDDEN)
        tensors[lp + "mlp.gate.weight"] = _bf16(rng, EXPERTS, HIDDEN)
        tensors[lp + "mlp.gate.bias"] = _f32(rng, EXPERTS)
        tensors[f"model.llm.attn_res.mlp_queries.{i}"] = _f32(rng, HIDDEN)

        if i > 0:
            tensors[f"model.llm.attn_res.attn_queries.{i - 1}"] = _f32(rng, 1 if swa else 3, HIDDEN)

        if swa:
            tensors[lp + "attn_norm.weight"] = _bf16(rng, HIDDEN)
            tensors[lp + "attn.k_sconv.weight"] = _f32(rng, n_kv * hd, 1, CONV_K)
            tensors[lp + "attn.v_sconv.weight"] = _f32(rng, n_kv * hd, 1, CONV_K)
        else:
            for c in "qkv":
                tensors[lp + f"{c}_input_norm.weight"] = _bf16(rng, HIDDEN)
            for c in "kv":
                tensors[lp + f"attn.{c}_sconv.static_w"] = _f32(rng, CONV_K, n_kv * hd)
                tensors[lp + f"attn.{c}_sconv.weight_proj"] = _bf16(rng, CONV_K * n_kv, HIDDEN)

        scheme = layer_quant.get(i, "bf16")
        w13_name = lp + "mlp.experts.w13_weight"
        w2_name = lp + "mlp.experts.w2_weight"
        if scheme == "nvfp4":
            for suffix, t in _make_nvfp4(rng, EXPERTS, 2 * INTER, HIDDEN).items():
                tensors[w13_name + suffix] = t
            for suffix, t in _make_nvfp4(rng, EXPERTS, HIDDEN, INTER).items():
                tensors[w2_name + suffix] = t
            nvfp4_names.extend([w13_name, w2_name])
        elif scheme == "mxfp8":
            for suffix, t in _make_mxfp8(rng, EXPERTS, 2 * INTER, HIDDEN).items():
                tensors[w13_name + suffix] = t
            for suffix, t in _make_mxfp8(rng, EXPERTS, HIDDEN, INTER).items():
                tensors[w2_name + suffix] = t
            mxfp8_names.extend([w13_name, w2_name])
        else:
            tensors[w13_name] = _bf16(rng, EXPERTS, 2 * INTER, HIDDEN)
            tensors[w2_name] = _bf16(rng, EXPERTS, HIDDEN, INTER)

        if loop_config is not None and loop_config["core_start"] <= i < loop_config["core_end"]:
            rank = loop_config["gate_rank"]
            for m in ("attn", "k", "v", "mlp"):
                tensors[lp + f"controllers.{m}.down_weight"] = _bf16(rng, rank, 2 * HIDDEN)
                tensors[lp + f"controllers.{m}.gain_weight"] = _bf16(rng, HIDDEN, rank)
                tensors[lp + f"controllers.{m}.forget_weight"] = _bf16(rng, HIDDEN, rank)

    if loop_config is not None:
        n_q = 2 * (loop_config["core_end"] - loop_config["core_start"])
        tensors["model.llm.middle_cycle_cross_pass_queries"] = _f32(rng, n_q, HIDDEN)
        tensors["model.llm.middle_cycle_cross_pass_kv_queries"] = _f32(rng, n_q, HIDDEN)

    # Vision tensors (`model.visual.*`)
    if include_vision:
        tensors["model.visual.backbone.embeddings.class_embedding"] = _bf16(rng, VISION_HIDDEN)
        tensors["model.visual.backbone.embeddings.cls_position_embedding"] = _bf16(rng, VISION_HIDDEN)
        tensors["model.visual.backbone.embeddings.position_embedding"] = _bf16(rng, GRID_SIZE * GRID_SIZE, VISION_HIDDEN)
        tensors["model.visual.backbone.embeddings.patch_embedding.weight"] = _bf16(
            rng, VISION_HIDDEN, PATCH_SIZE * PATCH_SIZE * 3
        )
        tensors["model.visual.backbone.ln_pre.weight"] = _bf16(rng, VISION_HIDDEN)
        tensors["model.visual.backbone.ln_pre.bias"] = _bf16(rng, VISION_HIDDEN)
        for i in range(VISION_LAYERS):
            vp = f"model.visual.backbone.layers.{i}."
            tensors[vp + "ln_1.weight"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "ln_1.bias"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "ln_2.weight"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "ln_2.bias"] = _bf16(rng, VISION_HIDDEN)
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                tensors[vp + f"self_attn.{proj}.weight"] = _bf16(rng, VISION_HIDDEN, VISION_HIDDEN)
                tensors[vp + f"self_attn.{proj}.bias"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "mlp.c_fc.weight"] = _bf16(rng, VISION_INTER, VISION_HIDDEN)
            tensors[vp + "mlp.c_fc.bias"] = _bf16(rng, VISION_INTER)
            tensors[vp + "mlp.c_proj.weight"] = _bf16(rng, VISION_HIDDEN, VISION_INTER)
            tensors[vp + "mlp.c_proj.bias"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "ls_1.gamma"] = _bf16(rng, VISION_HIDDEN)
            tensors[vp + "ls_2.gamma"] = _bf16(rng, VISION_HIDDEN)
        tensors["model.visual.proj_in.weight"] = _bf16(rng, VISION_INTER, VISION_HIDDEN * MERGE_SIZE * MERGE_SIZE)
        tensors["model.visual.proj_in.bias"] = _bf16(rng, VISION_INTER)
        tensors["model.visual.proj_out.weight"] = _bf16(rng, HIDDEN, VISION_INTER)
        tensors["model.visual.proj_out.bias"] = _bf16(rng, HIDDEN)

    # Save main model safetensors (2 shards)
    keys = list(tensors)
    mid = len(keys) // 2
    s1, s2 = keys[:mid], keys[mid:]
    save_file({k: tensors[k].contiguous() for k in s1}, str(path / "model-00001-of-00002.safetensors"))
    save_file({k: tensors[k].contiguous() for k in s2}, str(path / "model-00002-of-00002.safetensors"))
    weight_map = {k: "model-00001-of-00002.safetensors" for k in s1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in s2})
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))

    # hf_quant_config.json matching sglang_quant_inkling2.py schema
    groups: list[dict[str, Any]] = []
    if nvfp4_names:
        groups.append({
            "tensors": nvfp4_names,
            "output_dtype": "uint8",
            "compute_dtype": "nvfp4",
            "weight_transform": "nvfp4_nearest",
            "weight_block_shape": [1, 16],
            "weight_scale_format": "amax_denominator",
            "required_activation_quantization": {
                "format": "nvfp4_e2m1",
                "block_size": 16,
                "scale_dtype": "float8_e4m3fn",
                "rounding": "nearest",
                "amax_type": "per_token",
                "scale_selection_rule": "fouroversix_mae",
            },
        })
    if mxfp8_names:
        groups.append({
            "tensors": mxfp8_names,
            "output_dtype": "float8_e4m3fn",
            "compute_dtype": "mxfp8",
            "weight_transform": "mxfp8_nearest",
            "weight_block_shape": [1, 32],
            "required_activation_quantization": {
                "format": "mxfp8_e4m3",
                "block_size": 32,
                "scale_dtype": "ue8m0",
                "rounding": "nearest",
            },
        })
    (path / "hf_quant_config.json").write_text(json.dumps({"model": {"parameter_groups": groups}}, indent=2))

    # Build mtp.safetensors for DSpark
    if include_mtp:
        n_ctx = len(root_cfg["mtp_config"]["target_layer_ids"])
        mtp_tensors: dict[str, torch.Tensor] = {
            "fuse_weight": _bf16(rng, MTP_LAYERS + 1, n_ctx),
            "mask_embed": _bf16(rng, HIDDEN),
            "norm.weight": _bf16(rng, HIDDEN),
            "markov_head.markov_w1.weight": _bf16(rng, VOCAB_PADDED, MTP_MARKOV_RANK),
            "markov_head.rnn_proj.weight": _bf16(rng, 3 * MTP_MARKOV_RANK, 2 * MTP_MARKOV_RANK + 2 * HIDDEN),
            "markov_head.rnn_proj.bias": _bf16(rng, 3 * MTP_MARKOV_RANK),
            "markov_head.hidden_out.weight": _bf16(rng, HIDDEN, MTP_MARKOV_RANK),
            "markov_head.hidden_out.bias": _bf16(rng, HIDDEN),
            "markov_head.dflare_norm.weight": _bf16(rng, HIDDEN),
        }
        for i in range(MTP_LAYERS):
            mp = f"layers.{i}."
            mtp_tensors[mp + "input_layernorm.weight"] = _bf16(rng, HIDDEN)
            mtp_tensors[mp + "dflare_norm.weight"] = _bf16(rng, HIDDEN)
            mtp_tensors[mp + "self_attn.q_proj.weight"] = _bf16(rng, MTP_HEADS * MTP_HEAD_DIM, HIDDEN)
            mtp_tensors[mp + "self_attn.k_proj.weight"] = _bf16(rng, MTP_KV_HEADS * MTP_HEAD_DIM, HIDDEN)
            mtp_tensors[mp + "self_attn.v_proj.weight"] = _bf16(rng, MTP_KV_HEADS * MTP_HEAD_DIM, HIDDEN)
            mtp_tensors[mp + "self_attn.o_proj.weight"] = _bf16(rng, HIDDEN, MTP_HEADS * MTP_HEAD_DIM)
            mtp_tensors[mp + "self_attn.q_norm.weight"] = _bf16(rng, MTP_HEAD_DIM)
            mtp_tensors[mp + "self_attn.k_norm.weight"] = _bf16(rng, MTP_HEAD_DIM)
            mtp_tensors[mp + "self_attn.k_proj_ctx.weight"] = _bf16(rng, MTP_KV_HEADS * MTP_HEAD_DIM, HIDDEN)
            mtp_tensors[mp + "self_attn.v_proj_ctx.weight"] = _bf16(rng, MTP_KV_HEADS * MTP_HEAD_DIM, HIDDEN)
            mtp_tensors[mp + "self_attn.k_norm_ctx.weight"] = _bf16(rng, MTP_HEAD_DIM)
            mtp_tensors[mp + "post_attention_layernorm.weight"] = _bf16(rng, HIDDEN)
            mtp_tensors[mp + "mlp.gate_proj.weight"] = _bf16(rng, MTP_INTER, HIDDEN)
            mtp_tensors[mp + "mlp.up_proj.weight"] = _bf16(rng, MTP_INTER, HIDDEN)
            mtp_tensors[mp + "mlp.down_proj.weight"] = _bf16(rng, HIDDEN, MTP_INTER)
        save_file(mtp_tensors, str(path / "mtp.safetensors"))

    _write_tokenizer(path)
    return path, tensors


def _read_fields(path: Path) -> dict[str, Any]:
    r = gguf.GGUFReader(path)
    return {k: f.contents() for k, f in r.fields.items()}


def _read_tensors(path: Path) -> dict[str, gguf.ReaderTensor]:
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def _form(t: gguf.ReaderTensor) -> tuple[T, tuple[int, ...]]:
    return t.tensor_type, tuple(int(n) for n in reversed(t.shape))


def _ref_nvfp4_dequant(
    packed: torch.Tensor, scale: torch.Tensor, amax: torch.Tensor, denom: torch.Tensor, *, w13: bool
) -> np.ndarray:
    """Reference NVFP4 dequantization matching `sglang_quant_inkling2.py` / `llamacpp_convert_inkling2.py`."""
    p = packed.view(torch.uint8).numpy()
    s = scale.view(torch.uint8).numpy()
    a = amax.numpy().astype(np.float32)
    d = denom.numpy().astype(np.float32)
    experts, rows, half = p.shape
    out = np.empty((experts, rows, half * 2), dtype=np.float32)
    for e in range(experts):
        pe = np.concatenate([p[e, 0::2], p[e, 1::2]], axis=0) if w13 else p[e]
        se = np.concatenate([s[e, 0::2], s[e, 1::2]], axis=0) if w13 else s[e]
        vals = np.empty((rows, half * 2), dtype=np.float32)
        vals[:, 0::2] = _E2M1[pe & 15]
        vals[:, 1::2] = _E2M1[pe >> 4]
        vals.reshape(rows, -1, 16)[:] *= _E4M3[se][:, :, None]
        vals *= np.float32(a[e]) / np.float32(d[0])
        out[e] = vals
    return out


def _ref_mxfp8_dequant(weight: torch.Tensor, scale: torch.Tensor, *, w13: bool) -> np.ndarray:
    """Reference MXFP8 dequantization matching `llamacpp_convert_inkling2.py`."""
    w = weight.view(torch.uint8).numpy()
    s = scale.view(torch.uint8).numpy()
    experts, rows, cols = w.shape
    out = np.empty((experts, rows, cols), dtype=np.float32)
    for e in range(experts):
        we = np.concatenate([w[e, 0::2], w[e, 1::2]], axis=0) if w13 else w[e]
        se = np.concatenate([s[e, 0::2], s[e, 1::2]], axis=0) if w13 else s[e]
        vals = _E4M3[we]
        bs = np.ldexp(np.ones(se.shape, dtype=np.float32), se.astype(np.int32) - 127)
        vals.reshape(rows, -1, 32)[:] *= bs[:, :, None]
        out[e] = vals
    return out


def test_inkling2_small_nvfp4_mxfp8_bf16_and_metadata(tmp_path: Path) -> None:
    ckpt_dir, src = _build_checkpoint(tmp_path / "small")
    out = tmp_path / "inkling2-small.gguf"
    with torch.inference_mode():
        inkling2.build("text", ckpt_dir, out, source_repo_url=URL).write()
        checked = check.check(inkling2.build("text", ckpt_dir, out, source_repo_url=URL), out)

    assert set(checked) == {"NVFP4", "F8_E4M3", "I8", "F32", "BF16"}

    fields = _read_fields(out)
    assert fields["general.architecture"] == "inkling2"
    assert fields["general.file_type"] == gguf.LlamaFileType.MOSTLY_NVFP4
    assert fields["general.alignment"] == 4096
    assert fields["general.source.repo_url"] == URL
    assert fields["inkling2.block_count"] == 4
    assert fields["inkling2.embedding_length"] == HIDDEN
    assert fields["inkling2.vocab_size"] == VOCAB_PADDED
    assert fields["inkling2.vocab_size_unpadded"] == VOCAB_UNPADDED
    assert len(fields["tokenizer.ggml.tokens"]) == VOCAB_UNPADDED
    assert fields["tokenizer.ggml.model"] == "gpt2"
    assert fields["tokenizer.ggml.pre"] == "inkling2"

    # Qualify actual tokenizer IDs (whitespace, non-ASCII, and controls) without stubbing
    tok = PreTrainedTokenizerFast(tokenizer_file=str(ckpt_dir / "tokenizer.json"))
    gguf_tokens = fields["tokenizer.ggml.tokens"]
    gguf_types = fields["tokenizer.ggml.token_type"]
    for sample, expected_type in (
        ("  \n\t", gguf.TokenType.NORMAL),
        ("🚀 Български 中文", gguf.TokenType.NORMAL),
        ("<|endoftext|><|special_1|>", gguf.TokenType.CONTROL),
    ):
        ids = tok.encode(sample, add_special_tokens=False)
        assert ids, f"empty tokenization for {sample!r}"
        for tid in ids:
            assert 0 <= tid < VOCAB_UNPADDED
            assert gguf_types[tid] == expected_type
            assert gguf_tokens[tid] == tok.convert_ids_to_tokens(tid)
        assert tok.decode(ids) == sample
    assert fields["inkling2.attention.sliding_window"] == SWA_WINDOW
    assert fields["inkling2.attention.sliding_window_pattern"] == [True, False, True, False]
    assert fields["inkling2.shortconv.l_cache"] == CONV_K
    assert fields["inkling2.activation.situ_beta"] == pytest.approx(4.0)
    assert fields["inkling2.activation.situ_linear_beta"] == pytest.approx(25.0)
    assert fields["inkling2.attn_res.block_size"] == 2
    assert fields["inkling2.final_logit_softcapping"] == pytest.approx(30.0)
    assert fields["inkling2.attention.relative_length"] == D_REL
    assert fields["inkling2.attention.relative_extent"] == REL_EXTENT
    assert fields["inkling2.attention.qk_clip"] == pytest.approx(50.0)
    assert fields["inkling2.attention.relative_clip"] == pytest.approx(10.0)
    assert fields["inkling2.attention.log_scale_floor"] == 512
    assert fields["inkling2.attention.log_scale_alpha"] == pytest.approx(0.25)
    assert fields["inkling2.rope.dimension_count"] == 0
    assert "inkling2.num_loops" not in fields

    ts = _read_tensors(out)
    # Layer 0: NVFP4 experts
    assert _form(ts["blk.0.ffn_gate_up_exps.weight"]) == (T.NVFP4, (EXPERTS, 2 * INTER, HIDDEN))
    assert _form(ts["blk.0.ffn_gate_up_exps.scale"]) == (T.F32, (EXPERTS,))
    assert _form(ts["blk.0.ffn_down_exps.weight"]) == (T.NVFP4, (EXPERTS, HIDDEN, INTER))
    assert _form(ts["blk.0.ffn_down_exps.scale"]) == (T.F32, (EXPERTS,))
    # Layer 1: MXFP8 experts
    assert _form(ts["blk.1.ffn_gate_up_exps.weight"]) == (T.F8_E4M3, (EXPERTS, 2 * INTER, HIDDEN))
    assert _form(ts["blk.1.ffn_gate_up_exps.scale"]) == (T.I8, (EXPERTS, 2 * INTER, HIDDEN // 32))
    assert _form(ts["blk.1.ffn_down_exps.weight"]) == (T.F8_E4M3, (EXPERTS, HIDDEN, INTER))
    assert _form(ts["blk.1.ffn_down_exps.scale"]) == (T.I8, (EXPERTS, HIDDEN, INTER // 32))
    # Layer 2: BF16 experts
    assert _form(ts["blk.2.ffn_gate_up_exps.weight"]) == (T.BF16, (EXPERTS, 2 * INTER, HIDDEN))
    assert "blk.2.ffn_gate_up_exps.scale" not in ts
    # Precision preservation: BF16 dense stays BF16, F32 queries/biases/scales stay F32
    assert _form(ts["token_embd.weight"]) == (T.BF16, (VOCAB_PADDED, HIDDEN))
    assert _form(ts["token_embd_norm.weight"]) == (T.BF16, (HIDDEN,))
    assert _form(ts["output_res_query.weight"]) == (T.F32, (HIDDEN,))
    assert _form(ts["blk.1.attn_res_query.weight"]) == (T.F32, (3, HIDDEN))
    assert _form(ts["blk.2.attn_res_query.weight"]) == (T.F32, (1, HIDDEN))
    assert _form(ts["blk.0.exp_probs_b.bias"]) == (T.F32, (EXPERTS,))
    assert _form(ts["blk.0.attn_rel_bias.weight"]) == (T.F32, (SWA_WINDOW, D_REL))
    assert _form(ts["blk.1.attn_rel_bias.weight"]) == (T.F32, (REL_EXTENT, D_REL))
    assert _form(ts["blk.1.attn_k_conv.weight"]) == (T.F32, (KV_HEADS * HEAD_DIM, CONV_K))
    assert "blk.0.attn_res_query.weight" not in ts

    # Bit-exact NVFP4 dequant parity against sglang_quant_inkling2.py / llamacpp_convert_inkling2.py
    for proj, gguf_stem, shape, w13 in (
        ("w13_weight", "blk.0.ffn_gate_up_exps", (EXPERTS, 2 * INTER, HIDDEN), True),
        ("w2_weight", "blk.0.ffn_down_exps", (EXPERTS, HIDDEN, INTER), False),
    ):
        prefix = f"model.llm.layers.0.mlp.experts.{proj}"
        expected = _ref_nvfp4_dequant(
            src[prefix], src[prefix + "_scale"], src[prefix + "_amax"], src[prefix + "_outer_scale_denominator"],
            w13=w13,
        )
        gguf_w = gguf.quants.dequantize(np.asarray(ts[gguf_stem + ".weight"].data), T.NVFP4).reshape(shape)
        gguf_s = np.asarray(ts[gguf_stem + ".scale"].data).view(np.float32)
        actual = gguf_w * gguf_s[:, None, None]
        assert np.array_equal(actual, expected), proj

    # Bit-exact MXFP8 dequant parity
    for proj, gguf_stem, shape, w13 in (
        ("w13_weight", "blk.1.ffn_gate_up_exps", (EXPERTS, 2 * INTER, HIDDEN), True),
        ("w2_weight", "blk.1.ffn_down_exps", (EXPERTS, HIDDEN, INTER), False),
    ):
        prefix = f"model.llm.layers.1.mlp.experts.{proj}"
        expected = _ref_mxfp8_dequant(src[prefix], src[prefix + "_scale"], w13=w13)
        codes_u8 = np.asarray(ts[gguf_stem + ".weight"].data).view(np.uint8).reshape(shape)
        scales_u8 = np.asarray(ts[gguf_stem + ".scale"].data).view(np.uint8).reshape(shape[0], shape[1], shape[2] // 32)
        actual = _E4M3[codes_u8]
        actual.reshape(shape[0], shape[1], -1, 32)[:] *= np.ldexp(
            np.ones(scales_u8.shape, dtype=np.float32), scales_u8.astype(np.int32) - 127
        )[:, :, :, None]
        assert np.array_equal(actual, expected), proj


def test_inkling2_small_sharded_output(tmp_path: Path) -> None:
    ckpt_dir, _ = _build_checkpoint(tmp_path / "small_sharded")
    out = tmp_path / "sharded" / "inkling2-small.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        # Small split threshold so synthetic model splits into multiple 4096-aligned shards
        inkling2.build("text", ckpt_dir, out, source_repo_url=URL, split_max_size=200_000).write()

    shard_files = sorted(out.parent.glob("inkling2-small-*-of-*.gguf"))
    assert len(shard_files) >= 2
    with torch.inference_mode():
        check.check(inkling2.build("text", ckpt_dir, out, source_repo_url=URL), shard_files[0])


def test_inkling2_mini_loop_and_controllers(tmp_path: Path) -> None:
    loop_cfg = {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 16, "is_varloop": False}
    ckpt_dir, _ = _build_checkpoint(
        tmp_path / "mini",
        n_layers=6,
        local_layers=[0, 2, 4],
        layer_quant={i: "nvfp4" if i % 2 == 0 else "mxfp8" for i in range(6)},
        loop_config=loop_cfg,
    )
    out = tmp_path / "inkling2-mini.gguf"
    with torch.inference_mode():
        inkling2.build("text", ckpt_dir, out, source_repo_url=URL).write()
        check.check(inkling2.build("text", ckpt_dir, out, source_repo_url=URL), out)

    fields = _read_fields(out)
    # 6 physical layers, core [2, 4) runs 3 passes -> visits = [0, 1, 2, 3, 2, 3, 2, 3, 4, 5] (10 visits)
    assert fields["inkling2.block_count"] == 10
    assert fields["inkling2.num_loops"] == 3
    assert fields["inkling2.loop.core_start"] == 2
    assert fields["inkling2.loop.core_end"] == 4
    assert fields["inkling2.loop.gate_rank"] == 16
    assert "inkling2.loop.is_varloop" not in fields
    assert fields["inkling2.attention.sliding_window_pattern"] == [
        True, False, True, False, True, False, True, False, True, False
    ]

    ts = _read_tensors(out)
    assert _form(ts["loop_res_query.weight"]) == (T.F32, (4, HIDDEN))
    assert _form(ts["loop_kv_res_query.weight"]) == (T.F32, (4, HIDDEN))
    for bid in (2, 3):
        for m in ("attn", "k", "v", "mlp"):
            assert _form(ts[f"blk.{bid}.loop_{m}_down.weight"]) == (T.BF16, (16, 2 * HIDDEN))
            assert _form(ts[f"blk.{bid}.loop_{m}_gain.weight"]) == (T.BF16, (HIDDEN, 16))
            assert _form(ts[f"blk.{bid}.loop_{m}_forget.weight"]) == (T.BF16, (HIDDEN, 16))
    assert "blk.0.loop_attn_down.weight" not in ts
    assert "blk.4.loop_attn_down.weight" not in ts


def test_inkling2_varloop_research(tmp_path: Path) -> None:
    loop_cfg = {"core_start": 2, "core_end": 4, "num_passes": 1, "gate_rank": 16, "is_varloop": True}
    ckpt_dir, _ = _build_checkpoint(
        tmp_path / "varloop",
        n_layers=6,
        local_layers=[0, 2, 4],
        layer_quant={i: "mxfp8" for i in range(6)},
        loop_config=loop_cfg,
    )
    out = tmp_path / "varloop.gguf"
    with torch.inference_mode():
        inkling2.build("text", ckpt_dir, out, source_repo_url=URL).write()
        check.check(inkling2.build("text", ckpt_dir, out, source_repo_url=URL), out)

    fields = _read_fields(out)
    assert fields["general.file_type"] == gguf.LlamaFileType.MOSTLY_F8_E4M3
    assert fields["inkling2.block_count"] == 6
    assert fields["inkling2.num_loops"] == 1
    assert fields["inkling2.loop.is_varloop"] is True
    assert fields["inkling2.loop.core_start"] == 2
    assert fields["inkling2.loop.core_end"] == 4
    assert fields["inkling2.loop.gate_rank"] == 16


def test_inkling2_dspark_mtp(tmp_path: Path) -> None:
    ckpt_dir, _ = _build_checkpoint(tmp_path / "with_mtp", include_mtp=True)
    out = tmp_path / "inkling2-dspark.gguf"
    with torch.inference_mode():
        inkling2.build("dspark", ckpt_dir, out, source_repo_url=URL).write()
        checked = check.check(inkling2.build("dspark", ckpt_dir, out, source_repo_url=URL), out)

    assert set(checked) == {"BF16"}
    fields = _read_fields(out)
    assert fields["general.architecture"] == "dflash"
    assert fields["general.file_type"] == gguf.LlamaFileType.MOSTLY_BF16
    assert fields["dflash.block_count"] == MTP_LAYERS
    assert fields["dflash.block_size"] == MTP_BLOCK_SIZE
    assert fields["dflash.target_layers"] == [0, 1, 4]
    assert fields["dflash.sample_from_anchor"] is False
    assert fields["dflash.attention.causal"] is False
    assert fields["dflash.markov_rank"] == MTP_MARKOV_RANK
    assert fields["dflash.attention.sliding_window_pattern"] == [True, False]

    ts = _read_tensors(out)
    assert _form(ts["fuse_weight.weight"]) == (T.BF16, (MTP_LAYERS + 1, 3))
    assert _form(ts["mask_embd.weight"]) == (T.BF16, (HIDDEN,))
    assert _form(ts["output_norm.weight"]) == (T.BF16, (HIDDEN,))
    assert _form(ts["markov_w1.weight"]) == (T.BF16, (VOCAB_PADDED, MTP_MARKOV_RANK))
    assert _form(ts["markov_rnn_proj.weight"]) == (T.BF16, (3 * MTP_MARKOV_RANK, 2 * MTP_MARKOV_RANK + 2 * HIDDEN))
    assert _form(ts["markov_hidden_out.weight"]) == (T.BF16, (HIDDEN, MTP_MARKOV_RANK))
    assert _form(ts["markov_dflare_norm.weight"]) == (T.BF16, (HIDDEN,))
    assert _form(ts["blk.0.dflare_norm.weight"]) == (T.BF16, (HIDDEN,))
    assert _form(ts["blk.0.attn_k_ctx.weight"]) == (T.BF16, (MTP_KV_HEADS * MTP_HEAD_DIM, HIDDEN))
    assert _form(ts["blk.0.attn_k_norm_ctx.weight"]) == (T.BF16, (MTP_HEAD_DIM,))


def test_inkling2_vision_mmproj(tmp_path: Path) -> None:
    ckpt_dir, src = _build_checkpoint(tmp_path / "with_vision", include_vision=True)
    out = tmp_path / "mmproj-BF16.gguf"
    with torch.inference_mode():
        inkling2.build("vision", ckpt_dir, out, source_repo_url=URL).write()
        checked = check.check(inkling2.build("vision", ckpt_dir, out, source_repo_url=URL), out)

    assert set(checked) == {"BF16"}
    fields = _read_fields(out)
    assert fields["general.architecture"] == "clip"
    assert fields["general.type"] == "mmproj"
    assert fields["clip.has_vision_encoder"] is True
    assert fields["clip.vision.projector_type"] == "inkling2v"
    assert fields["clip.vision.projection_dim"] == HIDDEN
    assert fields["clip.vision.patch_size"] == PATCH_SIZE
    assert fields["clip.vision.image_size"] == GRID_SIZE * PATCH_SIZE
    assert fields["clip.vision.spatial_merge_size"] == MERGE_SIZE
    assert fields["clip.vision.attention.layer_norm_epsilon"] == pytest.approx(1e-6)

    ts = _read_tensors(out)
    assert _form(ts["v.class_embd"]) == (T.BF16, (VISION_HIDDEN,))
    assert _form(ts["v.class_position_embd.weight"]) == (T.BF16, (VISION_HIDDEN,))
    assert _form(ts["v.position_embd.weight"]) == (T.BF16, (GRID_SIZE * GRID_SIZE, VISION_HIDDEN))
    assert _form(ts["v.patch_embd.weight"]) == (T.BF16, (VISION_HIDDEN, 3, PATCH_SIZE, PATCH_SIZE))
    assert _form(ts["v.blk.0.ls1.weight"]) == (T.BF16, (VISION_HIDDEN,))
    assert _form(ts["mm.1.weight"]) == (T.BF16, (VISION_INTER, VISION_HIDDEN * MERGE_SIZE * MERGE_SIZE))
    assert _form(ts["mm.2.weight"]) == (T.BF16, (HIDDEN, VISION_INTER))

    # Verify conv2d patch layout transform
    orig = src["model.visual.backbone.embeddings.patch_embedding.weight"]
    expected = orig.reshape(VISION_HIDDEN, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 3, 1, 2).contiguous()
    got = torch.from_numpy(np.asarray(ts["v.patch_embd.weight"].data).copy()).view(torch.bfloat16).reshape(expected.shape)
    assert torch.equal(got, expected)


def test_inkling2_refuses_invalid_inputs(tmp_path: Path) -> None:
    # 1. Non-varloop with num_passes=1 refused
    with pytest.raises(ValueError, match="invalid loop_config"):
        ckpt_dir, _ = _build_checkpoint(
            tmp_path / "bad_loop",
            n_layers=6,
            loop_config={"core_start": 2, "core_end": 4, "num_passes": 1, "gate_rank": 16, "is_varloop": False},
        )
        inkling2.build("text", ckpt_dir, tmp_path / "bad_loop.gguf", source_repo_url=URL)

    # 2. Negative NVFP4 block scale refused at write time (and no partial file left behind)
    ckpt_dir, _ = _build_checkpoint(tmp_path / "bad_nvfp4", n_layers=2, layer_quant={0: "nvfp4", 1: "bf16"})
    shard = ckpt_dir / "model-00001-of-00002.safetensors"
    with inkling2.safe_open(shard, framework="pt") as f:
        data = {k: f.get_tensor(k) for k in f.keys()}
    for shard_name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        p = ckpt_dir / shard_name
        with inkling2.safe_open(p, framework="pt") as f:
            d = {k: f.get_tensor(k) for k in f.keys()}
        key = "model.llm.layers.0.mlp.experts.w13_weight_scale"
        if key in d:
            bad = d[key].view(torch.uint8).clone()
            bad[0, 0, 0] = 0x80  # negative E4M3
            d[key] = bad.view(torch.float8_e4m3fn)
            save_file(d, str(p))
    out = tmp_path / "bad_nvfp4.gguf"
    with pytest.raises(ValueError, match="non-negative finite E4M3"):
        inkling2.build("text", ckpt_dir, out, source_repo_url=URL).write()
    assert not out.exists()

    # 3. Unknown pre-tokenizer fingerprint/config explicitly refused (not silently labeled "inkling2")
    bad_tok_dir, _ = _build_checkpoint(tmp_path / "bad_pretok", n_layers=2, layer_quant={0: "bf16", 1: "bf16"})
    _write_tokenizer(bad_tok_dir, use_inkling2_pre=False)
    bad_tok_out = tmp_path / "bad_pretok.gguf"
    with pytest.raises(ValueError, match="unrecognized BPE pre-tokenizer"):
        inkling2.build("text", bad_tok_dir, bad_tok_out, source_repo_url=URL).write()
    assert not bad_tok_out.exists()


def test_cli_dispatches_inkling2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt_dir, _ = _build_checkpoint(tmp_path / "cli_ckpt", n_layers=2, layer_quant={0: "nvfp4", 1: "mxfp8"})
    out = tmp_path / "cli.gguf"
    for cmd in ("convert", "check"):
        monkeypatch.setattr(sys, "argv", [
            "magnitude-conversion", cmd, str(ckpt_dir), str(out),
            "--part", "text",
            "--source-repo", "thinkingmachines/Inkling-2-Small",
            "--source-revision", "0" * 40,
        ])
        cli.main()
    assert out.is_file()
