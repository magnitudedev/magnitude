"""A tiny GLM-5.3-Flash checkpoint in a release's form: `config.json` shaped as the pin's (`zai-org/GLM-5.3-Flash`
at `46a050be`) at small sizes, KDA and DSA layers, dense and MoE layers, an MTP layer and a vision tensor (neither
in the text file), BF16 and F32 tensors as the release ships them, with a transformers 5 tokenizer
(`TokenizersBackend`), in one of two forms:

- `zai`, Z.ai's FP8 release: FP8 weights as E4M3 codes with F32 `weight_scale_inv` per 128x128 tile, in two
  safetensors shards;
- `modelopt`, NVIDIA's QAD release (`nvidia/GLM-5.3-Flash-FP4-Local`): routed experts NVFP4 with csf_32x2_v1 block
  scales and `input_scale` (the MTP layer's W4A16, with plain scales), MLA and shared experts FP8_PB_WO (F32 per
  128x128, in ModelOpt's broadcast form), KDA, dense FFN, the head and the MTP layer's other linears (its kv_b
  included) MXFP8, as its
  `quantized_layers` declares; its input scales and some BF16 tensors in files of their own, one in a subdirectory."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from csf_streams import encode
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

HIDDEN = 256
HEADS = 2
LINEAR_HEAD = 64
Q_LORA = KV_LORA = NOPE = V_HEAD = 128
INDEX_HEADS, INDEX_DIM, KPOOL = 2, 64, 4
HC = 4
DENSE_FF, MOE_FF, EXPERTS = 256, 128, 4
VOCAB = 288
LAYER_TYPES = ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
MLP_TYPES = ["dense", "dense", "sparse", "sparse"]
SPECIAL = ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>", "<|user|>",
           "<|assistant|>", "<|observation|>", "<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>",
           "<|end_of_video|>", "<|begin_of_audio|>", "<|end_of_audio|>", "<|begin_of_transcription|>",
           "<|end_of_transcription|>"]

# how each kind of linear is stored in each form
KINDS = {
    "kda": {"zai": "bf16", "modelopt": "mxfp8"},
    "mla": {"zai": "fp8", "modelopt": "fp8_pb_wo"},
    "dense": {"zai": "fp8", "modelopt": "mxfp8"},
    "experts": {"zai": "fp8", "modelopt": "nvfp4"},
    "shared": {"zai": "fp8", "modelopt": "fp8_pb_wo"},
    "head": {"zai": "bf16", "modelopt": "mxfp8"},
    "mtp_mla": {"zai": "fp8", "modelopt": "mxfp8_g32"},
    "mtp_kv_b": {"zai": "bf16", "modelopt": "mxfp8_g32"},
    "mtp_indexer": {"zai": "bf16", "modelopt": "mxfp8_g32"},
    "mtp_experts": {"zai": "fp8", "modelopt": "w4a16"},
    "mtp_shared": {"zai": "fp8", "modelopt": "mxfp8_g32"},
    "vision": {"zai": "bf16", "modelopt": "mxfp8_g32"},
}
# a ModelOpt quantized_layers entry per stored kind
ENTRIES = {"mxfp8": {"quant_algo": "MXFP8", "weight_block_size": [1, 32]},
           "mxfp8_g32": {"group_size": 32, "quant_algo": "MXFP8"},
           "fp8_pb_wo": {"quant_algo": "FP8_PB_WO", "weight_block_size": [128, 128]},
           "nvfp4": {"group_size": 16, "quant_algo": "NVFP4", "weight_scale_encoding": "csf_32x2_v1"},
           "w4a16": {"group_size": 16, "quant_algo": "W4A16_NVFP4"}}
ZAI_QUANTIZATION = {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                    "weight_block_size": [128, 128], "modules_to_not_convert": ["lm_head"]}


def config(form: str = "zai") -> dict[str, Any]:
    """The config of the checkpoint `tensors(form)` makes."""
    n = len(LAYER_TYPES)
    kda = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "attention_bias": False, "attention_dropout": 0.0, "dtype": "bfloat16", "eos_token_id": [256, 263, 265],
            "first_k_dense_replace": MLP_TYPES.count("dense"), "hc_eps": 1e-06, "hc_mult": HC,
            "hc_sinkhorn_iters": 20, "head_dim": 0, "hidden_act": "silu", "hidden_size": HIDDEN,
            "index_head_dim": INDEX_DIM, "index_kpool": KPOOL, "index_kpool_always_select_tail": True,
            "index_kpool_compress": True, "index_n_heads": INDEX_HEADS, "index_topk": 16,
            "index_share_for_mtp_iteration": True, "indexer_rope_interleave": True, "indexer_types": ["full"] * n,
            "initializer_range": 0.02, "intermediate_size": DENSE_FF, "kv_lora_rank": KV_LORA,
            "layer_types": list(LAYER_TYPES),
            "linear_attn_config": {"num_heads": HEADS, "gate_lower_bound": -5.0, "head_dim": LINEAR_HEAD,
                                   "short_conv_kernel_size": 4, "kda_layers": kda,
                                   "full_attn_layers": [i for i in range(n) if i not in kda]},
            "max_position_embeddings": 1048576, "mhc": True, "mla_use_nope": True,
            "mlp_layer_types": list(MLP_TYPES), "model_type": "glm5_next_text", "moe_intermediate_size": MOE_FF,
            "moe_router_dtype": "float32", "n_group": 1, "n_routed_experts": EXPERTS, "n_shared_experts": 1,
            "norm_topk_prob": True, "num_attention_heads": HEADS, "num_experts_per_tok": 2, "num_hidden_layers": n,
            "num_key_value_heads": HEADS, "num_nextn_predict_layers": 1, "output_router_logits": False,
            "pad_token_id": 256, "q_lora_rank": Q_LORA, "qk_head_dim": NOPE, "qk_nope_head_dim": NOPE,
            "qk_rope_head_dim": 0, "rms_norm_eps": 1e-05, "routed_scaling_factor": 2.5,
            "router_aux_loss_coef": 0.001, "scoring_func": "sigmoid", "swiglu_limit": 10.0,
            "tie_word_embeddings": False, "topk_group": 1, "topk_method": "noaux_tc", "use_cache": True,
            "v_head_dim": V_HEAD, "vocab_size": VOCAB,
        },
        "image_token_id": 272, "tie_word_embeddings": False, "model_type": "glm5_next",
        "transformers_version": "5.16.0",
        "quantization_config": ZAI_QUANTIZATION if form == "zai" else {
            "exclude_modules": [], "quant_algo": "MIXED_PRECISION", "quant_method": "modelopt",
            "quantized_layers": tensors(form)[1]},
    }


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(shape).to(torch.bfloat16)


def _f32(*shape: int) -> torch.Tensor:
    return torch.randn(shape)


def _e4m3_codes(rows: int, cols: int) -> torch.Tensor:
    codes = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3C  # no NaN codes
    return codes.view(torch.float8_e4m3fn)


def _fp8(tensors: dict[str, torch.Tensor], name: str, rows: int, cols: int) -> None:
    tensors[f"{name}.weight"] = _e4m3_codes(rows, cols)
    # amax/448 per tile in the release: not powers of two
    tensors[f"{name}.weight_scale_inv"] = torch.rand(-(-rows // 128), -(-cols // 128)) * 1e-2 + 1e-5


def _nvfp4_scales(rows: int, cols: int) -> np.ndarray:
    """E4M3 block scale bytes [rows, cols / 16]: positive, spread wider than a 4-bit code within a row."""
    return torch.randint(0x10, 0x70, (rows, cols // 16), dtype=torch.uint8).numpy()


def _linear(t: dict[str, torch.Tensor], layers: dict[str, Any], kind: str, name: str, rows: int, cols: int) -> None:
    """The weight `name` [rows, cols] stored as `kind`, and its ModelOpt entry when it has one."""
    if kind == "bf16":
        t[f"{name}.weight"] = _bf16(rows, cols)
    elif kind == "fp8":
        _fp8(t, name, rows, cols)
    elif kind in ("mxfp8", "mxfp8_g32"):
        t[f"{name}.weight"] = _e4m3_codes(rows, cols)
        t[f"{name}.weight_scale"] = torch.randint(100, 140, (rows, cols // 32), dtype=torch.uint8)
    elif kind == "fp8_pb_wo":
        t[f"{name}.weight"] = _e4m3_codes(rows, cols)
        t[f"{name}.weight_scale"] = torch.rand(rows // 128, 1, cols // 128, 1) * 1e-2 + 1e-5
    else:
        t[f"{name}.weight"] = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
        scales = _nvfp4_scales(rows, cols)
        if kind == "nvfp4":
            for stream, data in encode(scales).items():
                t[f"{name}.weight_scale.csf32_{stream}"] = torch.from_numpy(data)
            t[f"{name}.input_scale"] = torch.tensor(float(torch.rand(1)) + 0.5)
        else:
            t[f"{name}.weight_scale"] = torch.from_numpy(scales).view(torch.float8_e4m3fn)
        t[f"{name}.weight_scale_2"] = torch.tensor(float(torch.rand(1)) * 1e-3 + 1e-4)
    if kind in ENTRIES:
        # a layer's routed experts share one entry, `mlp.experts`
        layers[re.sub(r"(\.mlp\.experts)\.\d+\..*", r"\1", name)] = ENTRIES[kind]


def _experts(t: dict[str, torch.Tensor], layers: dict[str, Any], kind: str, m: str) -> None:
    for e in range(EXPERTS):
        _linear(t, layers, kind, f"{m}.experts.{e}.gate_proj", MOE_FF, HIDDEN)
        _linear(t, layers, kind, f"{m}.experts.{e}.up_proj", MOE_FF, HIDDEN)
        _linear(t, layers, kind, f"{m}.experts.{e}.down_proj", HIDDEN, MOE_FF)


def _moe(t: dict[str, torch.Tensor], layers: dict[str, Any], experts: str, shared: str, m: str) -> None:
    t[f"{m}.gate.weight"] = _bf16(EXPERTS, HIDDEN)
    t[f"{m}.gate.e_score_correction_bias"] = _f32(EXPERTS)
    _experts(t, layers, experts, m)
    _linear(t, layers, shared, f"{m}.shared_experts.gate_proj", MOE_FF, HIDDEN)
    _linear(t, layers, shared, f"{m}.shared_experts.up_proj", MOE_FF, HIDDEN)
    _linear(t, layers, shared, f"{m}.shared_experts.down_proj", HIDDEN, MOE_FF)


def _mla(t: dict[str, torch.Tensor], layers: dict[str, Any], kind: str, kv_b: str, indexer: str, a: str) -> None:
    _linear(t, layers, kind, f"{a}.q_a_proj", Q_LORA, HIDDEN)
    t[f"{a}.q_a_layernorm.weight"] = _bf16(Q_LORA)
    _linear(t, layers, kind, f"{a}.q_b_proj", HEADS * NOPE, Q_LORA)
    _linear(t, layers, kind, f"{a}.kv_a_proj_with_mqa", KV_LORA, HIDDEN)
    t[f"{a}.kv_a_layernorm.weight"] = _bf16(KV_LORA)
    _linear(t, layers, kv_b, f"{a}.kv_b_proj", HEADS * (NOPE + V_HEAD), KV_LORA)
    _linear(t, layers, kind, f"{a}.o_proj", HIDDEN, HEADS * V_HEAD)
    x = f"{a}.indexer"
    _linear(t, layers, indexer, f"{x}.wq_b", INDEX_HEADS * INDEX_DIM, Q_LORA)
    _linear(t, layers, indexer, f"{x}.wk", INDEX_DIM, HIDDEN)
    t[f"{x}.k_norm.weight"] = _bf16(INDEX_DIM)
    t[f"{x}.k_norm.bias"] = _bf16(INDEX_DIM)
    _linear(t, layers, indexer, f"{x}.weights_proj", INDEX_HEADS, HIDDEN)
    t[f"{x}.index_kpool_compress_ape"] = _bf16(KPOOL, INDEX_DIM)
    t[f"{x}.index_kpool_compress_gate"] = _bf16(INDEX_DIM, HIDDEN)


def tensors(form: str = "zai") -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """The checkpoint's tensors in `form`, and its ModelOpt `quantized_layers` (empty for Z.ai's)."""
    torch.manual_seed(0)
    kind = {role: forms[form] for role, forms in KINDS.items()}
    t: dict[str, torch.Tensor] = {}
    q: dict[str, Any] = {}
    lm = "model.language_model"
    t[f"{lm}.embed_tokens.weight"] = _bf16(VOCAB, HIDDEN)
    t[f"{lm}.norm.weight"] = _bf16(HIDDEN)
    _linear(t, q, kind["head"], "lm_head", VOCAB, HIDDEN)
    _linear(t, q, kind["vision"], "model.visual.blocks.0.attn.qkv", 3 * 64, 64)
    mix = (2 + HC) * HC
    qkv = HEADS * LINEAR_HEAD
    for i in range(len(LAYER_TYPES) + 1):  # the last is the MTP layer
        p = f"{lm}.layers.{i}"
        t[f"{p}.input_layernorm.weight"] = _bf16(HIDDEN)
        t[f"{p}.post_attention_layernorm.weight"] = _bf16(HIDDEN)
        if i == len(LAYER_TYPES):
            t[f"{p}.eh_proj.weight"] = _bf16(HIDDEN, 2 * HIDDEN)
            for norm in ("enorm", "hnorm", "shared_head.norm"):
                t[f"{p}.{norm}.weight"] = _bf16(HIDDEN)
            _mla(t, q, kind["mtp_mla"], kind["mtp_kv_b"], kind["mtp_indexer"], f"{p}.self_attn")
            _moe(t, q, kind["mtp_experts"], kind["mtp_shared"], f"{p}.mlp")
            continue
        for hc in ("attn", "ffn"):
            t[f"{p}.hc_{hc}_fn"] = _bf16(mix, HC * HIDDEN)
            t[f"{p}.hc_{hc}_base"] = _f32(mix)
            t[f"{p}.hc_{hc}_scale"] = _f32(3)
        a = f"{p}.self_attn"
        if LAYER_TYPES[i] == "linear_attention":
            for proj in ("q", "k", "v"):
                _linear(t, q, kind["kda"], f"{a}.{proj}_proj", qkv, HIDDEN)
                t[f"{a}.{proj}_conv1d.weight"] = _bf16(qkv, 1, 4)
            t[f"{a}.f_a_proj.weight"] = _bf16(LINEAR_HEAD, HIDDEN)
            t[f"{a}.f_b_proj.weight"] = _bf16(qkv, LINEAR_HEAD)
            t[f"{a}.g_a_proj.weight"] = _bf16(LINEAR_HEAD, HIDDEN)
            t[f"{a}.g_b_proj.weight"] = _bf16(qkv, LINEAR_HEAD)
            t[f"{a}.b_proj.weight"] = _bf16(HEADS, HIDDEN)
            t[f"{a}.A_log"] = _f32(HEADS)
            t[f"{a}.dt_bias"] = _f32(qkv)
            t[f"{a}.o_norm.weight"] = _bf16(LINEAR_HEAD)
            _linear(t, q, kind["kda"], f"{a}.o_proj", HIDDEN, qkv)
        else:
            _mla(t, q, kind["mla"], "bf16", "bf16", a)
        m = f"{p}.mlp"
        if MLP_TYPES[i] == "dense":
            _linear(t, q, kind["dense"], f"{m}.gate_proj", DENSE_FF, HIDDEN)
            _linear(t, q, kind["dense"], f"{m}.up_proj", DENSE_FF, HIDDEN)
            _linear(t, q, kind["dense"], f"{m}.down_proj", HIDDEN, DENSE_FF)
        else:
            _moe(t, q, kind["experts"], kind["shared"], m)
    return t, q


def _tokenizer(path: Path) -> None:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE({c: i for i, c in enumerate(alphabet)}, []))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in SPECIAL])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "backend": "tokenizers", "clean_up_tokenization_spaces": False, "do_lower_case": False,
        "eos_token": "<|endoftext|>", "extra_special_tokens": SPECIAL, "is_local": True,
        "model_max_length": 1048576, "model_specific_special_tokens": {}, "pad_token": "<|endoftext|>",
        "padding_side": "left", "remove_space": False, "tokenizer_class": "TokenizersBackend"}))


def _files(form: str, names: list[str]) -> dict[str, list[str]]:
    """The checkpoint's files and the tensors in each: two shards, and in NVIDIA's form its input scales and some
    BF16 tensors in files of their own, one in a subdirectory."""
    files: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        if form == "modelopt" and name.endswith(".input_scale"):
            file = "model-inputscales.safetensors"
        elif form == "modelopt" and re.search(r"\.(b_proj|f_a_proj|kv_b_proj)\.weight$", name):
            file = "tensors/native-QB.safetensors"
        else:
            file = f"model-0000{1 + 2 * i // len(names)}-of-00002.safetensors"
        files.setdefault(file, []).append(name)
    return files


def write(path: Path, *, form: str = "zai", edit_config: Callable[[dict[str, Any]], None] | None = None,
          edit_tensors: Callable[[dict[str, torch.Tensor]], None] | None = None) -> Path:
    """Writes the checkpoint in `form` into `path`, after the edits; returns `path`."""
    path.mkdir(parents=True, exist_ok=True)
    cfg, t = config(form), tensors(form)[0]
    if edit_config is not None:
        edit_config(cfg)
    if edit_tensors is not None:
        edit_tensors(t)
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    files = _files(form, list(t))
    for file, keys in files.items():
        (path / file).parent.mkdir(parents=True, exist_ok=True)
        save_file({k: t[k].contiguous() for k in keys}, str(path / file))
    weight_map = {k: file for file, keys in files.items() for k in keys}
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    _tokenizer(path)
    return path
