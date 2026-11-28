"""Inkling 2 checkpoints -> Magnitude GGUFs, at the checkpoint's own precision.

Supports:
- Inkling 2 Small (`Inkling2ForConditionalGeneration` / `Inkling2ForCausalLM`, `model_type` `inkling2` or
  `inkling2_text`, sharded output via `--split-max-size`);
- Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`, `model_type` `inkling2_loop`, single-file output);
- TML-d24-varloop-research (`model_type` `inkling2_loop` with `loop_config.is_varloop = true`);
- Bundled DSpark drafter (`--part dspark`, reading `mtp.safetensors` -> `dflash` GGUF);
- Bundled vision tower + merger (`--part vision`, reading `model.visual.*` / `model.vision_tower.*` -> `mmproj` GGUF).

Emits the standard llama.cpp `inkling2` GGUF schema (metadata keys and unfused tensor names) while preserving the
checkpoint's exact precision without dequantizing/re-quantizing:
- ModelOpt NVFP4 routed experts (`w13_weight`, `w2_weight` on NVFP4 layers): repacked bit-exact into ggml `NVFP4`
  (`block_nvfp4`, with `w13` gate/up rows de-interleaved `[0::2]` then `[1::2]`), beside F32 `.scale` [experts] equal
  to `amax / outer_scale_denominator`;
- MXFP8 routed experts (`w13_weight`, `w2_weight` on MXFP8 layers): `F8_E4M3` codes (with `w13` rows de-interleaved)
  beside `I8` UE8M0 `.scale` [experts, rows, cols / 32];
- BF16 dense weights in BF16, F32 queries/biases/scales/convs in F32 (each non-quantized tensor kept at its exact
  checkpoint dtype).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from conversion.base import TextModel, gguf
from safetensors import safe_open

from .native import add_file_metadata, check_ue8m0, scale_name, streamed

logger = logging.getLogger(__name__)

PARTS = ("text", "dspark", "vision")
MODEL_TYPES = ("inkling2", "inkling2_text", "inkling2_loop")
T = gguf.GGMLQuantizationType

_RAW_DTYPE: dict[torch.dtype, T] = {
    torch.bfloat16: T.BF16,
    torch.float16: T.F16,
    torch.float32: T.F32,
}

_SAFETENSORS_DTYPE: dict[str, torch.dtype] = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I8": torch.int8,
}

_EXPERT_COMPANIONS = ("_scale", "_amax", "_outer_scale_denominator")
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorRef:
    """Lazy reference to one tensor in a safetensors file (reads header metadata only until `load()`/`slice()`)."""

    file: Path
    source_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    def load(self) -> torch.Tensor:
        with safe_open(self.file, framework="pt") as f:
            return f.get_tensor(self.source_name)

    def load_expert(self, expert: int) -> torch.Tensor:
        with safe_open(self.file, framework="pt") as f:
            return f.get_slice(self.source_name)[expert]


def _index_safetensors(dir_model: Path, *, single_file: str | None = None) -> dict[str, TensorRef]:
    """Indexes safetensors in `dir_model` without reading tensor data."""
    if single_file is not None:
        path = dir_model / single_file
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        files = [path]
    else:
        index_path = dir_model / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            files = [dir_model / name for name in dict.fromkeys(weight_map.values())]
        else:
            single = dir_model / "model.safetensors"
            if single.is_file():
                files = [single]
            else:
                files = sorted(
                    p for p in dir_model.glob("*.safetensors")
                    if p.name != "mtp.safetensors"
                )
        if not files:
            raise FileNotFoundError(f"no model safetensors found in {dir_model}")

    refs: dict[str, TensorRef] = {}
    for path in files:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                sl = f.get_slice(name)
                dtype_str = sl.get_dtype()
                if dtype_str not in _SAFETENSORS_DTYPE:
                    raise ValueError(f"{path.name}: {name}: unsupported safetensors dtype {dtype_str!r}")
                refs[name] = TensorRef(path, name, tuple(sl.get_shape()), _SAFETENSORS_DTYPE[dtype_str])
    return refs


def repack_nvfp4_rows(packed: np.ndarray, scales: np.ndarray, name: str, expert: int) -> np.ndarray:
    """Repacks checkpoint NVFP4 rows (2 E2M1 nibbles/byte, low nibble first; 1 E4M3 scale per 16) into ggml
    `block_nvfp4` bytes (per 64 elements: 4 unsigned E4M3 scales, then 4 groups of 8 bytes holding j and j+8)."""
    n, half = packed.shape
    nb = scales.shape[1]
    if half * 2 != nb * 16 or nb % 4 != 0:
        raise ValueError(f"Inkling2: {name} expert {expert}: {half * 2} columns do not form NVFP4 blocks of 64")
    if (scales >= 0x7F).any():
        raise ValueError(
            f"Inkling2: {name}_scale expert {expert}: block scale byte {int(scales.max()):#04x} "
            f"is not a non-negative finite E4M3 value"
        )
    nibbles = np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(n, nb, 16)
    qs = (nibbles[:, :, :8] | (nibbles[:, :, 8:] << 4)).astype(np.uint8)
    blocks = np.concatenate([scales.reshape(n, nb // 4, 4), qs.reshape(n, nb // 4, 32)], axis=-1)
    return np.ascontiguousarray(blocks.reshape(n, nb // 4 * 36))


def _check_e4m3_finite(name: str, expert: int, codes: np.ndarray) -> None:
    if ((codes & 0x7F) == 0x7F).any():
        raise ValueError(f"Inkling2: {name} expert {expert}: E4M3FN NaN")


def _raw_bytes(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.contiguous().view(torch.uint8).numpy())


def _deinterleave_rows(arr: np.ndarray) -> np.ndarray:
    """Converts interleaved (gate, up) rows [2*inter, ...] to stacked [gate..., up...]."""
    return np.ascontiguousarray(np.concatenate([arr[0::2], arr[1::2]], axis=0))


def _normalize_text_name(name: str) -> str | None:
    """Normalizes Inkling 2 text checkpoint tensor names to canonical `model.*` form; returns None for non-text."""
    if any(p in name for p in ("visual.", "vision_tower.", "mtp.", "markov_head", "nextn_predict")):
        return None
    name = name.replace("model.language_model.", "model.").replace("model.llm.", "model.")
    # attn_queries.{i} feeds block i + 1
    name = re.sub(r"attn_res\.attn_queries\.(\d+)", lambda m: f"layers.{int(m[1]) + 1}.attention_input.query", name)
    name = name.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
    return name


# Exact llama.cpp Inkling2 text tensor mapping (normalized checkpoint name -> GGUF name)
_TEXT_GLOBAL_MAP: dict[str, str] = {
    "model.embed.weight": "token_embd.weight",
    "model.embed_tokens.weight": "token_embd.weight",
    "model.embed_norm.weight": "token_embd_norm.weight",
    "model.embed_tokens.embed_norm.weight": "token_embd_norm.weight",
    "model.unembed.weight": "output.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
    "model.attn_res.output_query": "output_res_query.weight",
    "model.final_input.query": "output_res_query.weight",
    "model.middle_cycle_cross_pass_queries": "loop_res_query.weight",
    "model.middle_cycle_cross_pass_kv_queries": "loop_kv_res_query.weight",
}

_TEXT_LAYER_MAP: dict[str, str] = {
    # Attention projections
    "attn.wq_du.weight": "attn_q.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "attn.wk_dv.weight": "attn_k.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "attn.wv_dv.weight": "attn_v.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "attn.wo_ud.weight": "attn_output.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "attn.gate_attn.weight": "attn_gate.weight",
    "self_attn.gate_attn.weight": "attn_gate.weight",
    "self_attn.o_gate.weight": "attn_gate.weight",
    "attn.wr_du.weight": "attn_rel.weight",
    "self_attn.r_proj.weight": "attn_rel.weight",
    "attn.attn_sink_proj.weight": "attn_sink_proj.weight",
    "self_attn.sink_proj.weight": "attn_sink_proj.weight",
    # Relative position bias & L2 clip scales
    "attn.rel_logits_proj.proj": "attn_rel_bias.weight",
    "self_attn.rel_logits_proj.proj": "attn_rel_bias.weight",
    "attn.q_l2_clip.scale": "attn_q_clip.weight",
    "self_attn.q_l2_clip.scale": "attn_q_clip.weight",
    "attn.rel_l2_clip.scale": "attn_rel_clip.weight",
    "self_attn.rel_l2_clip.scale": "attn_rel_clip.weight",
    # Norms
    "attn_norm.weight": "attn_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "q_input_norm.weight": "attn_q_inp_norm.weight",
    "q_input_layernorm.weight": "attn_q_inp_norm.weight",
    "k_input_norm.weight": "attn_k_inp_norm.weight",
    "k_input_layernorm.weight": "attn_k_inp_norm.weight",
    "v_input_norm.weight": "attn_v_inp_norm.weight",
    "v_input_layernorm.weight": "attn_v_inp_norm.weight",
    "mlp_norm.weight": "ffn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # Short convolutions
    "attn.k_sconv.weight": "attn_k_conv.weight",
    "self_attn.k_sconv.conv1d.weight": "attn_k_conv.weight",
    "attn.k_sconv.static_w": "attn_k_conv.weight",
    "self_attn.k_sconv.static_weight": "attn_k_conv.weight",
    "attn.v_sconv.weight": "attn_v_conv.weight",
    "self_attn.v_sconv.conv1d.weight": "attn_v_conv.weight",
    "attn.v_sconv.static_w": "attn_v_conv.weight",
    "self_attn.v_sconv.static_weight": "attn_v_conv.weight",
    "attn.k_sconv.weight_proj": "attn_k_conv_proj.weight",
    "attn.k_sconv.weight_proj.weight": "attn_k_conv_proj.weight",
    "self_attn.k_sconv.weight_proj": "attn_k_conv_proj.weight",
    "self_attn.k_sconv.weight_proj.weight": "attn_k_conv_proj.weight",
    "attn.v_sconv.weight_proj": "attn_v_conv_proj.weight",
    "attn.v_sconv.weight_proj.weight": "attn_v_conv_proj.weight",
    "self_attn.v_sconv.weight_proj": "attn_v_conv_proj.weight",
    "self_attn.v_sconv.weight_proj.weight": "attn_v_conv_proj.weight",
    "attn_sconv.weight": "attn_post_conv.weight",
    "post_attention_convolution.conv1d.weight": "attn_post_conv.weight",
    "mlp_sconv.weight": "ffn_post_conv.weight",
    "post_mlp_convolution.conv1d.weight": "ffn_post_conv.weight",
    # Attention residual queries
    "attention_input.query": "attn_res_query.weight",
    "mlp_input.query": "ffn_res_query.weight",
    # MoE
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.experts.w13_weight": "ffn_gate_up_exps.weight",
    "mlp.experts.w2_weight": "ffn_down_exps.weight",
}

for _mod in ("attn", "k", "v", "mlp"):
    for _part in ("down", "gain", "forget"):
        _TEXT_LAYER_MAP[f"controllers.{_mod}.{_part}_weight"] = f"loop_{_mod}_{_part}.weight"


def map_text_tensor_name(name: str) -> tuple[str, int | None]:
    """Maps a normalized Inkling 2 text tensor name to its standard llama.cpp GGUF name and block id."""
    if name in _TEXT_GLOBAL_MAP:
        return _TEXT_GLOBAL_MAP[name], None
    m = re.match(r"model\.attn_res\.mlp_queries\.(\d+)$", name)
    if m is not None:
        bid = int(m.group(1))
        return f"blk.{bid}.ffn_res_query.weight", bid
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
    if m is not None:
        bid, rest = int(m.group(1)), m.group(2)
        if rest in _TEXT_LAYER_MAP:
            return f"blk.{bid}.{_TEXT_LAYER_MAP[rest]}", bid
    raise ValueError(f"Inkling2: unrecognized text tensor name {name!r}")


def _layer_sort_key(name: str) -> int:
    m = _LAYER_RE.search(name)
    if m is not None:
        return int(m.group(1))
    m = re.match(r"model\.attn_res\.mlp_queries\.(\d+)$", name)
    if m is not None:
        return int(m.group(1))
    return -1


@contextmanager
def _maybe_tokenizer_from_json(dir_model: Path) -> Iterator[None]:
    """Handles transformers 5's `TokenizersBackend` under transformers 4.57.6, just like `glm5next.py`."""
    tc_path = dir_model / "tokenizer_config.json"
    if not tc_path.is_file():
        yield
        return
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    if tc.get("tokenizer_class") != "TokenizersBackend":
        yield
        return
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(dir_model / "tokenizer.json"))

    def from_pretrained(path: str | Path, *args: Any, **kwargs: Any) -> PreTrainedTokenizerFast:
        if Path(path) != dir_model:
            raise ValueError(f"a tokenizer of {path} asked for while converting {dir_model}")
        return tokenizer

    loader = AutoTokenizer.__dict__["from_pretrained"]
    AutoTokenizer.from_pretrained = from_pretrained
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = loader


_INKLING2_PRE_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _matches_inkling2_pre_tokenizer(dir_model: Path) -> bool:
    """Returns whether `tokenizer.json` explicitly specifies the Inkling 2 / GPT-4o byte-level BPE pre-tokenizer."""
    tok_path = dir_model / "tokenizer.json"
    if not tok_path.is_file():
        return False
    cfg = json.loads(tok_path.read_text(encoding="utf-8"))
    if cfg.get("normalizer") is not None:
        return False
    if (cfg.get("model") or {}).get("type") != "BPE":
        return False
    if (cfg.get("decoder") or {}).get("type") != "ByteLevel":
        return False
    pre = cfg.get("pre_tokenizer")
    if not isinstance(pre, dict) or pre.get("type") != "Sequence":
        return False
    steps = pre.get("pretokenizers")
    if not isinstance(steps, list) or len(steps) != 2:
        return False
    split, byte_level = steps
    return (
        isinstance(split, dict)
        and split.get("type") == "Split"
        and split.get("pattern") == {"Regex": _INKLING2_PRE_REGEX}
        and split.get("behavior") == "Isolated"
        and split.get("invert") is False
        and isinstance(byte_level, dict)
        and byte_level.get("type") == "ByteLevel"
        and byte_level.get("add_prefix_space") is False
        and byte_level.get("use_regex") is False
    )


def _write_inkling2_vocab(writer: gguf.GGUFWriter, dir_model: Path, hparams: dict[str, Any]) -> None:
    """Writes the Inkling 2 GPT-2 BPE vocabulary (cropped to `unpadded_vocab_size`) and special tokens."""
    helper = object.__new__(TextModel)
    helper.dir_model = dir_model
    helper.hparams = hparams

    def _get_pre(tokenizer: Any) -> str:
        try:
            return TextModel.get_vocab_base_pre(helper, tokenizer)
        except NotImplementedError as exc:
            if _matches_inkling2_pre_tokenizer(dir_model):
                return "inkling2"
            raise ValueError(
                f"Inkling2: unrecognized BPE pre-tokenizer in {dir_model} "
                f"(fingerprint not in upstream table and tokenizer.json does not match Inkling 2 pre-tokenizer)"
            ) from exc

    helper.get_vocab_base_pre = _get_pre  # type: ignore[method-assign]
    with _maybe_tokenizer_from_json(dir_model):
        tokens, types, pre = TextModel.get_vocab_base(helper)
    n = hparams["unpadded_vocab_size"]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(pre)
    writer.add_token_list(tokens[:n])
    writer.add_token_types(types[:n])
    special = gguf.SpecialVocab(dir_model, load_merges=True)
    eos = hparams["eos_token_id"]
    if isinstance(eos, list):
        eos = eos[0]
    special.special_token_ids["eos"] = eos
    special.special_token_ids["eot"] = eos
    special.add_special_token["bos"] = False
    special.add_to_gguf(writer)


def _find_hparam(hp: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in hp:
            return hp[k]
    raise KeyError(f"Inkling2: missing required config key (one of {list(keys)})")


def _load_raw_config(dir_model: Path) -> dict[str, Any]:
    config_path = dir_model / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _merge_text_config(raw: dict[str, Any]) -> dict[str, Any]:
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        return {**raw, **raw["text_config"]}
    return dict(raw)


def _load_expert_quant_config(dir_model: Path) -> dict[str, dict[str, Any]]:
    """Loads `hf_quant_config.json` expert quantization schemes keyed by normalized tensor name."""
    quant_path = dir_model / "hf_quant_config.json"
    if not quant_path.is_file():
        return {}
    data = json.loads(quant_path.read_text(encoding="utf-8"))
    groups = data.get("model", {}).get("parameter_groups", [])
    expert_quant: dict[str, dict[str, Any]] = {}
    for group in groups:
        compute = group.get("compute_dtype")
        if compute not in (None, "float32", "bfloat16", "mxfp8", "nvfp4"):
            raise ValueError(f"Inkling2: unsupported compute_dtype {compute!r}")
        for raw_name in group.get("tensors", []):
            norm = _normalize_text_name(raw_name)
            if norm is not None and norm.endswith((".w13_weight", ".w2_weight")):
                if norm in expert_quant:
                    raise ValueError(f"Inkling2: duplicate hf_quant_config.json scheme for {norm}")
                expert_quant[norm] = group
    return expert_quant


def _transform_dense_tensor(name: str, data: torch.Tensor) -> torch.Tensor:
    """Applies exact llama.cpp Inkling 2 structural transforms to a dense tensor without changing its dtype."""
    if name.endswith(".w13_weight"):
        e, rows, cols = data.shape
        return data.reshape(e, rows // 2, 2, cols).transpose(1, 2).reshape(e, rows, cols).contiguous()
    if name.endswith((".static_weight", ".static_w")):
        return data.flip(0).T.contiguous()
    if name.endswith((".conv1d.weight", "sconv.weight")):
        return data.squeeze(1).contiguous() if data.ndim == 3 else data.contiguous()
    if name.endswith(".rel_logits_proj.proj"):
        return data.T.contiguous()
    return data.contiguous()


def _dense_output_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Output shape after `_transform_dense_tensor`, without loading tensor data."""
    if name.endswith((".static_weight", ".static_w", ".rel_logits_proj.proj")):
        return (shape[1], shape[0])
    if name.endswith((".conv1d.weight", "sconv.weight")) and len(shape) == 3 and shape[1] == 1:
        return (shape[0], shape[2])
    return shape


class _WriterBase:
    """Shared GGUF writer lifecycle: 4096-aligned sharding (`AlignedShardsWriter`) and clean removal on error."""

    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        arch: str,
        *,
        source_repo_url: str,
        split_max_size: int = 0,
        shards: frozenset[int] | None = None,
    ):
        self.dir_model = dir_model
        self.fname_out = fname_out
        self.arch = arch
        self._source_repo_url = source_repo_url
        self._shards = shards
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch=arch,
            use_temp_file=False,
            split_max_size=split_max_size,
        )

    def _prepare_metadata(self) -> None:
        raise NotImplementedError

    def _prepare_tensors(self) -> None:
        raise NotImplementedError

    def write(self) -> None:
        try:
            self._prepare_tensors()
            self._prepare_metadata()
            self.gguf_writer.write_header_to_file(path=self.fname_out)
            self.gguf_writer.write_kv_data_to_file()
            self.gguf_writer.write_tensors_to_file(progress=True)
            self.gguf_writer.close()
        except BaseException:
            writer = self.gguf_writer
            if writer.fout is not None:
                opened = [Path(f.name) for f in writer.fout]
                writer.close()
                for path in opened:
                    path.unlink(missing_ok=True)
            raise


@dataclass(frozen=True)
class _Nvfp4ExpertSpec:
    name: str
    gguf_name: str
    w13: bool
    shape: tuple[int, int, int]  # (experts, rows, cols)
    weight: TensorRef
    scale: TensorRef
    amax: TensorRef
    denominator: TensorRef


@dataclass(frozen=True)
class _Mxfp8ExpertSpec:
    name: str
    gguf_name: str
    w13: bool
    shape: tuple[int, int, int]  # (experts, rows, cols)
    scale_shape: tuple[int, int, int]  # (experts, rows, cols // 32)
    weight: TensorRef
    scale: TensorRef


class Inkling2Text(_WriterBase):
    """Lossless Inkling 2 text converter (Small, Mini looped, and varloop research)."""

    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        *,
        source_repo_url: str,
        split_max_size: int = 0,
        shards: frozenset[int] | None = None,
    ):
        super().__init__(
            dir_model,
            fname_out,
            "inkling2",
            source_repo_url=source_repo_url,
            split_max_size=split_max_size,
            shards=shards,
        )
        raw = _load_raw_config(dir_model)
        self.hparams = _merge_text_config(raw)
        hp = self.hparams

        for key, default in (
            ("q_bias", False),
            ("o_bias", False),
            ("n_shared_experts", 0),
            ("use_value_norm_no_gamma", True),
            ("attn_res_normalize_values", True),
            ("scoring_func", "sigmoid"),
        ):
            if hp.get(key, default) != default:
                raise ValueError(f"Inkling2: unsupported {key}={hp[key]!r}")

        self.block_count: int = int(_find_hparam(hp, ("num_hidden_layers", "n_layers", "n_layer")))
        types = hp.get("layer_types")
        if types is None:
            local = set(hp["local_layer_ids"])
            types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.block_count)]
        if len(types) != self.block_count or set(types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError(f"Inkling2: invalid layer_types {types!r}")
        self.is_swa: list[bool] = [t == "hybrid_sliding" for t in types]

        # Loop / varloop config
        self.loop: dict[str, Any] | None = hp.get("loop_config")
        self.is_varloop: bool = False
        self.visits: list[int] = list(range(self.block_count))
        if self.loop is not None:
            loop = self.loop
            self.is_varloop = bool(loop.get("is_varloop", False))
            start, end, passes, rank = (loop.get(k) for k in ("core_start", "core_end", "num_passes", "gate_rank"))
            block = hp["n_layers_per_attn_res_block"]
            min_passes = 1 if self.is_varloop else 2
            if (
                any(type(v) is not int for v in (start, end, passes, rank))
                or not (0 < start < end <= self.block_count)
                or start % block
                or (end - start) % block
                or passes < min_passes
                or rank <= 0
            ):
                raise ValueError(f"Inkling2: invalid loop_config {loop!r}")
            self.visits = list(range(start)) + list(range(start, end)) * passes + list(range(end, self.block_count))

        self._expert_quant = _load_expert_quant_config(dir_model)
        self.tensors: dict[str, TensorRef] = {}
        self.nvfp4_experts: dict[str, _Nvfp4ExpertSpec] = {}
        self.mxfp8_experts: dict[str, _Mxfp8ExpertSpec] = {}
        self.dense_tensors: dict[str, TensorRef] = {}

        self._index_and_classify()

    def _index_and_classify(self) -> None:
        raw_refs = _index_safetensors(self.dir_model)
        normalized: dict[str, TensorRef] = {}
        for raw_name, ref in raw_refs.items():
            norm = _normalize_text_name(raw_name)
            if norm is None:
                continue
            if norm in normalized:
                raise ValueError(f"Inkling2: duplicate normalized tensor name {norm!r}")
            normalized[norm] = ref
        self.tensors = dict(sorted(normalized.items(), key=lambda item: _layer_sort_key(item[0])))

        consumed: set[str] = set()
        hp = self.hparams
        experts = int(hp["n_routed_experts"])
        hidden = int(hp["hidden_size"])
        inter = int(_find_hparam(hp, ("moe_intermediate_size", "intermediate_size")))

        for name, ref in self.tensors.items():
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue
            companions = [name + s for s in _EXPERT_COMPANIONS]
            if ref.dtype in _RAW_DTYPE:
                if any(c in self.tensors for c in companions):
                    raise ValueError(f"Inkling2: floating-point {name} has packed quantization companions")
                continue

            config = self._expert_quant.get(name, {})
            scheme = config.get("weight_transform")
            if scheme not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise ValueError(f"Inkling2: {name}: missing or unsupported hf_quant_config.json scheme {scheme!r}")
            nvfp4 = scheme == "nvfp4_nearest"
            w13 = name.endswith(".w13_weight")
            rows, cols = (2 * inter, hidden) if w13 else (hidden, inter)
            logical_shape = (experts, rows, cols)
            gguf_name, _ = map_text_tensor_name(name)

            if nvfp4:
                if config.get("weight_scale_format") != "amax_denominator":
                    raise ValueError(f"Inkling2: {name}: expected NVFP4 amax_denominator scale format")
                if config.get("weight_block_shape") != [1, 16]:
                    raise ValueError(f"Inkling2: {name}: expected NVFP4 weight_block_shape [1, 16]")
                for c in companions:
                    if c not in self.tensors:
                        raise ValueError(f"Inkling2: {name}: missing companion {c}")
                scale_ref = self.tensors[companions[0]]
                amax_ref = self.tensors[companions[1]]
                denom_ref = self.tensors[companions[2]]
                if ref.dtype != torch.uint8 or scale_ref.dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        f"Inkling2: {name}: expected uint8 weights and float8_e4m3fn scales, "
                        f"got {ref.dtype} and {scale_ref.dtype}"
                    )
                if cols % 64 != 0 or ref.shape != (experts, rows, cols // 2):
                    raise ValueError(
                        f"Inkling2: {name}: invalid NVFP4 packed shape {ref.shape} for logical shape {logical_shape}"
                    )
                if scale_ref.shape != (experts, rows, cols // 16):
                    raise ValueError(
                        f"Inkling2: {companions[0]}: shape {scale_ref.shape} != {(experts, rows, cols // 16)}"
                    )
                if amax_ref.dtype != torch.float32 or amax_ref.shape != (experts,):
                    raise ValueError(
                        f"Inkling2: {companions[1]}: expected F32 {(experts,)}, got {amax_ref.dtype} {amax_ref.shape}"
                    )
                if denom_ref.dtype != torch.float32 or denom_ref.shape != (1,):
                    raise ValueError(
                        f"Inkling2: {companions[2]}: expected F32 (1,), got {denom_ref.dtype} {denom_ref.shape}"
                    )
                self.nvfp4_experts[name] = _Nvfp4ExpertSpec(
                    name, gguf_name, w13, logical_shape, ref, scale_ref, amax_ref, denom_ref
                )
                consumed.add(name)
                consumed.update(companions)
            else:
                if companions[0] not in self.tensors:
                    raise ValueError(f"Inkling2: {name}: missing companion {companions[0]}")
                if any(c in self.tensors for c in companions[1:]):
                    raise ValueError(f"Inkling2: {name}: MXFP8 must not have amax or outer_scale_denominator")
                scale_ref = self.tensors[companions[0]]
                if ref.dtype != torch.float8_e4m3fn or scale_ref.dtype != torch.uint8:
                    raise ValueError(
                        f"Inkling2: {name}: expected float8_e4m3fn weights and uint8 scales, "
                        f"got {ref.dtype} and {scale_ref.dtype}"
                    )
                if cols % 32 != 0 or ref.shape != logical_shape:
                    raise ValueError(
                        f"Inkling2: {name}: invalid MXFP8 shape {ref.shape} for logical shape {logical_shape}"
                    )
                expected_scale_shape = (experts, rows, cols // 32)
                if scale_ref.shape != expected_scale_shape:
                    raise ValueError(
                        f"Inkling2: {companions[0]}: shape {scale_ref.shape} != {expected_scale_shape} "
                        f"(only (1, 32) stored block scales are supported)"
                    )
                self.mxfp8_experts[name] = _Mxfp8ExpertSpec(
                    name, gguf_name, w13, logical_shape, expected_scale_shape, ref, scale_ref
                )
                consumed.add(name)
                consumed.add(companions[0])

        for name, ref in self.tensors.items():
            if name in consumed:
                continue
            if ref.dtype not in _RAW_DTYPE or name.endswith(_EXPERT_COMPANIONS):
                raise ValueError(f"Inkling2: unexpected packed tensor or orphan companion {name!r} ({ref.dtype})")
            # Validate tensor name is known before writing anything
            map_text_tensor_name(name)
            self.dense_tensors[name] = ref

    @property
    def file_type(self) -> gguf.LlamaFileType:
        if self.nvfp4_experts:
            return gguf.LlamaFileType.MOSTLY_NVFP4
        if self.mxfp8_experts:
            return gguf.LlamaFileType.MOSTLY_F8_E4M3
        return gguf.LlamaFileType.MOSTLY_BF16

    def _prepare_metadata(self) -> None:
        hp = self.hparams
        w = self.gguf_writer
        arch = self.arch

        n_heads_phys = [
            int(hp.get("swa_num_attention_heads", hp["num_attention_heads"])) if s else int(hp["num_attention_heads"])
            for s in self.is_swa
        ]
        n_heads_kv_phys = [
            int(hp.get("swa_num_key_value_heads", hp["num_key_value_heads"])) if s else int(hp["num_key_value_heads"])
            for s in self.is_swa
        ]
        n_heads = [n_heads_phys[p] for p in self.visits]
        n_heads_kv = [n_heads_kv_phys[p] for p in self.visits]
        swa_pattern = [self.is_swa[p] for p in self.visits]

        head_dim = int(hp["head_dim"])
        swa_head_dim = int(hp.get("swa_head_dim", head_dim))
        inter = int(_find_hparam(hp, ("moe_intermediate_size", "intermediate_size")))
        conv_k = int(_find_hparam(hp, ("conv_kernel_size", "sconv_kernel_size")))

        w.add_file_type(self.file_type)
        w.add_name(self.dir_model.name)
        w.add_type(gguf.GGUFType.MODEL)
        w.add_block_count(len(self.visits))
        w.add_context_length(int(_find_hparam(hp, ("max_position_embeddings", "model_max_length"))))
        w.add_embedding_length(int(hp["hidden_size"]))
        w.add_feed_forward_length(int(hp.get("intermediate_size", inter)))
        w.add_head_count(n_heads)
        w.add_head_count_kv(n_heads_kv)
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        w.add_key_length_swa(swa_head_dim)
        w.add_value_length_swa(swa_head_dim)
        w.add_layer_norm_rms_eps(float(hp["rms_norm_eps"]))
        w.add_rope_dimension_count(0)
        w.add_expert_count(int(hp["n_routed_experts"]))
        w.add_expert_used_count(int(hp["num_experts_per_tok"]))
        w.add_expert_feed_forward_length(inter)
        w.add_expert_weights_scale(float(hp["route_scale"]))
        w.add_vocab_size(int(hp["vocab_size"]))
        w.add_uint32(f"{arch}.vocab_size_unpadded", int(hp["unpadded_vocab_size"]))
        w.add_sliding_window(int(hp["sliding_window_size"]))
        w.add_sliding_window_pattern(swa_pattern)
        w.add_shortconv_l_cache(conv_k)
        w.add_activation_situ_beta(float(hp["mlp_gate_softcap"]))
        w.add_activation_situ_linear_beta(float(hp["mlp_up_softcap"]))
        w.add_attn_res_block_size(int(hp["n_layers_per_attn_res_block"]))
        w.add_final_logit_softcapping(float(hp["final_logit_softcapping"]))
        w.add_uint32(f"{arch}.attention.relative_length", int(hp["d_rel"]))
        w.add_uint32(f"{arch}.attention.relative_extent", int(hp["rel_extent"]))
        w.add_float32(f"{arch}.attention.qk_clip", float(hp["qk_l2_clip_threshold"]))
        w.add_float32(f"{arch}.attention.relative_clip", float(hp["rel_l2_clip_threshold"]))
        w.add_uint32(f"{arch}.attention.log_scale_floor", int(hp["log_scaling_n_floor"]))
        w.add_float32(f"{arch}.attention.log_scale_alpha", float(hp["log_scaling_alpha"]))

        if self.loop is not None:
            w.add_num_loops(int(self.loop["num_passes"]))
            w.add_uint32(f"{arch}.loop.core_start", int(self.loop["core_start"]))
            w.add_uint32(f"{arch}.loop.core_end", int(self.loop["core_end"]))
            w.add_uint32(f"{arch}.loop.gate_rank", int(self.loop["gate_rank"]))
            if self.is_varloop:
                w.add_bool(f"{arch}.loop.is_varloop", True)

        _write_inkling2_vocab(w, self.dir_model, hp)
        add_file_metadata(w, self._source_repo_url, self._shards)

    def _prepare_tensors(self) -> None:
        w = self.gguf_writer
        for name in self.tensors:
            if name in self.nvfp4_experts:
                self._add_nvfp4_expert(self.nvfp4_experts[name])
            elif name in self.mxfp8_experts:
                self._add_mxfp8_expert(self.mxfp8_experts[name])
            elif name in self.dense_tensors:
                ref = self.dense_tensors[name]
                gguf_name, _ = map_text_tensor_name(name)
                qtype = _RAW_DTYPE[ref.dtype]
                out_shape = _dense_output_shape(name, ref.shape)

                def _read_dense(r: TensorRef = ref, n: str = name) -> Iterable[np.ndarray]:
                    yield _raw_bytes(_transform_dense_tensor(n, r.load()))

                w.add_tensor(gguf_name, streamed(out_shape, qtype, _read_dense), raw_dtype=qtype)

    def _add_nvfp4_expert(self, spec: _Nvfp4ExpertSpec) -> None:
        w = self.gguf_writer
        experts = spec.shape[0]

        def _weight_chunks(s: _Nvfp4ExpertSpec = spec) -> Iterable[np.ndarray]:
            for e in range(experts):
                packed = s.weight.load_expert(e).view(torch.uint8).numpy()
                scales = s.scale.load_expert(e).view(torch.uint8).numpy()
                if s.w13:
                    yield np.concatenate([
                        repack_nvfp4_rows(packed[0::2], scales[0::2], s.name, e),
                        repack_nvfp4_rows(packed[1::2], scales[1::2], s.name, e),
                    ], axis=0)
                else:
                    yield repack_nvfp4_rows(packed, scales, s.name, e)

        def _scale_chunks(s: _Nvfp4ExpertSpec = spec) -> Iterable[np.ndarray]:
            a = s.amax.load().numpy().astype(np.float32)
            d = s.denominator.load().numpy().astype(np.float32)
            if not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f"Inkling2: {s.name}_amax: expected finite nonnegative F32 values")
            if not np.isfinite(d).all() or (d <= 0).any():
                raise ValueError(f"Inkling2: {s.name}_outer_scale_denominator: expected finite positive F32 value")
            yield np.ascontiguousarray((a / d).astype(np.float32))

        w.add_tensor(spec.gguf_name, streamed(spec.shape, T.NVFP4, _weight_chunks), raw_dtype=T.NVFP4)
        w.add_tensor(scale_name(spec.gguf_name), streamed((experts,), T.F32, _scale_chunks), raw_dtype=T.F32)

    def _add_mxfp8_expert(self, spec: _Mxfp8ExpertSpec) -> None:
        w = self.gguf_writer
        experts = spec.shape[0]

        def _weight_chunks(s: _Mxfp8ExpertSpec = spec) -> Iterable[np.ndarray]:
            for e in range(experts):
                codes = s.weight.load_expert(e).view(torch.uint8).numpy()
                _check_e4m3_finite(s.name, e, codes)
                yield _deinterleave_rows(codes) if s.w13 else np.ascontiguousarray(codes)

        def _scale_chunks(s: _Mxfp8ExpertSpec = spec) -> Iterable[np.ndarray]:
            for e in range(experts):
                sc = s.scale.load_expert(e).view(torch.uint8).numpy()
                check_ue8m0(f"{s.name}_scale[{e}]", sc)
                out = _deinterleave_rows(sc) if s.w13 else np.ascontiguousarray(sc)
                yield out.view(np.int8)

        w.add_tensor(spec.gguf_name, streamed(spec.shape, T.F8_E4M3, _weight_chunks), raw_dtype=T.F8_E4M3)
        w.add_tensor(scale_name(spec.gguf_name), streamed(spec.scale_shape, T.I8, _scale_chunks), raw_dtype=T.I8)


# ---------------------------------------------------------------------------
# DSpark drafter (`mtp.safetensors` -> `dflash` GGUF)
# ---------------------------------------------------------------------------

_DSPARK_GLOBAL_MAP: dict[str, str] = {
    "fuse_weight": "fuse_weight.weight",
    "mask_embed": "mask_embd.weight",
    "norm.weight": "output_norm.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.rnn_proj.weight": "markov_rnn_proj.weight",
    "markov_head.rnn_proj.bias": "markov_rnn_proj.bias",
    "markov_head.hidden_out.weight": "markov_hidden_out.weight",
    "markov_head.hidden_out.bias": "markov_hidden_out.bias",
    "markov_head.dflare_norm.weight": "markov_dflare_norm.weight",
}

_DSPARK_LAYER_MAP: dict[str, str] = {
    "input_layernorm.weight": "attn_norm.weight",
    "dflare_norm.weight": "dflare_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.k_proj_ctx.weight": "attn_k_ctx.weight",
    "self_attn.v_proj_ctx.weight": "attn_v_ctx.weight",
    "self_attn.k_norm_ctx.weight": "attn_k_norm_ctx.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def map_dspark_tensor_name(name: str) -> str:
    """Maps a tensor name in Inkling 2's `mtp.safetensors` to its `dflash` GGUF name."""
    norm = name
    for prefix in ("model.", "mtp."):
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
    if norm in _DSPARK_GLOBAL_MAP:
        return _DSPARK_GLOBAL_MAP[norm]
    m = re.match(r"layers\.(\d+)\.(.+)$", norm)
    if m is not None:
        bid, rest = int(m.group(1)), m.group(2)
        if rest in _DSPARK_LAYER_MAP:
            return f"blk.{bid}.{_DSPARK_LAYER_MAP[rest]}"
    raise ValueError(f"Inkling2 DSpark: unrecognized MTP tensor {name!r}")


class Inkling2DSpark(_WriterBase):
    """Inkling 2 DSpark speculative drafter (`mtp.safetensors`) -> `dflash` GGUF (BF16)."""

    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        *,
        source_repo_url: str,
        split_max_size: int = 0,
        shards: frozenset[int] | None = None,
    ):
        super().__init__(
            dir_model,
            fname_out,
            "dflash",
            source_repo_url=source_repo_url,
            split_max_size=split_max_size,
            shards=shards,
        )
        raw = _load_raw_config(dir_model)
        self.text_config = _merge_text_config(raw)
        mtp = raw.get("mtp_config")
        if not isinstance(mtp, dict):
            raise ValueError("Inkling2 DSpark: config.json is missing mtp_config")
        self.mtp_config = mtp
        self.hparams = self.text_config

        refs = _index_safetensors(dir_model, single_file="mtp.safetensors")
        self.tensors: dict[str, TensorRef] = dict(sorted(refs.items(), key=lambda item: _layer_sort_key(item[0])))
        for name, ref in self.tensors.items():
            if ref.dtype != torch.bfloat16:
                raise ValueError(f"Inkling2 DSpark: {name}: expected bfloat16, got {ref.dtype}")
            map_dspark_tensor_name(name)

    def _prepare_metadata(self) -> None:
        tc = self.text_config
        mc = self.mtp_config
        w = self.gguf_writer
        arch = self.arch

        n_layers = int(mc["num_hidden_layers"])
        head_dim = int(mc["head_dim"])

        w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
        w.add_name(f"{self.dir_model.name}-dspark")
        w.add_type(gguf.GGUFType.MODEL)
        w.add_block_count(n_layers)
        w.add_context_length(int(_find_hparam(mc, ("max_position_embeddings", "model_max_length")) if "max_position_embeddings" in mc or "model_max_length" in mc else _find_hparam(tc, ("max_position_embeddings", "model_max_length"))))
        w.add_embedding_length(int(tc["hidden_size"]))
        w.add_feed_forward_length(int(mc["intermediate_size"]))
        w.add_head_count(int(mc["num_attention_heads"]))
        w.add_head_count_kv(int(mc["num_key_value_heads"]))
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        w.add_layer_norm_rms_eps(float(mc.get("rms_norm_eps", tc["rms_norm_eps"])))
        w.add_bool(f"{arch}.attention.causal", False)
        w.add_rope_dimension_count(int(mc.get("rope_dim", head_dim)))
        w.add_rope_freq_base(float(mc.get("rope_theta", 10000.0)))
        w.add_block_size(int(mc["block_size"]))
        w.add_target_layers([int(i) for i in mc["target_layer_ids"]])
        w.add_sample_from_anchor(False)
        w.add_uint32(f"{arch}.markov_rank", int(mc["markov_rank"]))
        w.add_vocab_size(int(tc["vocab_size"]))
        w.add_uint32(f"{arch}.vocab_size_unpadded", int(tc["unpadded_vocab_size"]))

        if (swa := mc.get("sliding_window_size")) is not None:
            w.add_sliding_window(int(swa))
        if (local_ids := mc.get("local_layer_ids")) is not None:
            local = set(local_ids)
            w.add_sliding_window_pattern([i in local for i in range(n_layers)])
        if (softcap := tc.get("final_logit_softcapping")) is not None:
            w.add_final_logit_softcapping(float(softcap))

        _write_inkling2_vocab(w, self.dir_model, tc)
        add_file_metadata(w, self._source_repo_url, self._shards)

    def _prepare_tensors(self) -> None:
        w = self.gguf_writer
        for name, ref in self.tensors.items():
            gguf_name = map_dspark_tensor_name(name)

            def _read(r: TensorRef = ref) -> Iterable[np.ndarray]:
                yield _raw_bytes(r.load())

            w.add_tensor(gguf_name, streamed(ref.shape, T.BF16, _read), raw_dtype=T.BF16)


# ---------------------------------------------------------------------------
# Vision encoder + projector (`model.visual.*` / `model.vision_tower.*` -> `mmproj` GGUF)
# ---------------------------------------------------------------------------

_VISION_GLOBAL_MAP: dict[str, str] = {
    "model.visual.backbone.embeddings.class_embedding": "v.class_embd",
    "model.vision_tower.embeddings.class_embedding": "v.class_embd",
    "model.visual.backbone.embeddings.cls_position_embedding": "v.class_position_embd.weight",
    "model.vision_tower.embeddings.cls_position_embedding": "v.class_position_embd.weight",
    "model.visual.backbone.embeddings.position_embedding": "v.position_embd.weight",
    "model.vision_tower.embeddings.position_embedding.weight": "v.position_embd.weight",
    "model.visual.backbone.embeddings.patch_embedding.weight": "v.patch_embd.weight",
    "model.vision_tower.embeddings.patch_embedding.weight": "v.patch_embd.weight",
    "model.visual.backbone.ln_pre.weight": "v.pre_ln.weight",
    "model.visual.backbone.ln_pre.bias": "v.pre_ln.bias",
    "model.vision_tower.pre_layernorm.weight": "v.pre_ln.weight",
    "model.vision_tower.pre_layernorm.bias": "v.pre_ln.bias",
    "model.visual.proj_in.weight": "mm.1.weight",
    "model.visual.proj_in.bias": "mm.1.bias",
    "model.visual.proj_out.weight": "mm.2.weight",
    "model.visual.proj_out.bias": "mm.2.bias",
    "model.vision_tower.merger.linear_1.weight": "mm.1.weight",
    "model.vision_tower.merger.linear_1.bias": "mm.1.bias",
    "model.vision_tower.merger.linear_2.weight": "mm.2.weight",
    "model.vision_tower.merger.linear_2.bias": "mm.2.bias",
}

_VISION_LAYER_MAP: dict[str, str] = {
    "ln_1.weight": "ln1.weight",
    "ln_1.bias": "ln1.bias",
    "layernorm_before.weight": "ln1.weight",
    "layernorm_before.bias": "ln1.bias",
    "ln_2.weight": "ln2.weight",
    "ln_2.bias": "ln2.bias",
    "layernorm_after.weight": "ln2.weight",
    "layernorm_after.bias": "ln2.bias",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.out_proj.weight": "attn_out.weight",
    "self_attn.out_proj.bias": "attn_out.bias",
    "self_attn.o_proj.weight": "attn_out.weight",
    "self_attn.o_proj.bias": "attn_out.bias",
    "mlp.c_fc.weight": "ffn_up.weight",
    "mlp.c_fc.bias": "ffn_up.bias",
    "mlp.fc1.weight": "ffn_up.weight",
    "mlp.fc1.bias": "ffn_up.bias",
    "mlp.c_proj.weight": "ffn_down.weight",
    "mlp.c_proj.bias": "ffn_down.bias",
    "mlp.fc2.weight": "ffn_down.weight",
    "mlp.fc2.bias": "ffn_down.bias",
    "ls_1.gamma": "ls1.weight",
    "lambda_1": "ls1.weight",
    "ls_2.gamma": "ls2.weight",
    "lambda_2": "ls2.weight",
}


def map_vision_tensor_name(name: str) -> str:
    """Maps a `model.visual.*` or `model.vision_tower.*` tensor name to its llama.cpp `mmproj` name."""
    if name in _VISION_GLOBAL_MAP:
        return _VISION_GLOBAL_MAP[name]
    m = re.match(r"model\.(?:visual\.backbone|vision_tower)\.layers\.(\d+)\.(.+)$", name)
    if m is not None:
        bid, rest = int(m.group(1)), m.group(2)
        if rest in _VISION_LAYER_MAP:
            return f"v.blk.{bid}.{_VISION_LAYER_MAP[rest]}"
    raise ValueError(f"Inkling2 Vision: unrecognized vision tensor {name!r}")


class Inkling2Vision(_WriterBase):
    """Inkling 2 vision tower + merger -> `mmproj` GGUF (`clip` arch, `inkling2v` projector, BF16)."""

    def __init__(
        self,
        dir_model: Path,
        fname_out: Path,
        *,
        source_repo_url: str,
        split_max_size: int = 0,
        shards: frozenset[int] | None = None,
    ):
        super().__init__(
            dir_model,
            fname_out,
            "clip",
            source_repo_url=source_repo_url,
            split_max_size=split_max_size,
            shards=shards,
        )
        raw = _load_raw_config(dir_model)
        self.text_config = _merge_text_config(raw)
        vc = raw.get("vision_config")
        if not isinstance(vc, dict):
            raise ValueError("Inkling2 Vision: config.json is missing vision_config")
        vc = dict(vc)
        vc["patch_size"] = int(vc.get("native_patch_size", vc.get("patch_size", 20)))
        vc["spatial_merge_size"] = int(vc.get("adapter_merge_size", vc.get("spatial_merge_size", 2)))
        vc["intermediate_size"] = int(vc.get("intermediate_size") or int(vc["hidden_size"] * vc.get("mlp_ratio", 4.0)))
        vc["image_size"] = int(vc["posemb_grid_size"]) * vc["patch_size"]
        if vc.get("num_channels", 3) != 3 or vc.get("hidden_act", "gelu") != "gelu":
            raise ValueError("Inkling2 vision requires RGB patches and exact GELU")
        rope = vc.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope["rope_type"] != "axial" or float(rope["rope_theta"]) != 10000.0:
            raise ValueError("Inkling2 vision requires axial RoPE with theta 10000")
        self.vision_config = vc
        self.hparams = vc

        preproc_path = dir_model / "preprocessor_config.json"
        self.preprocessor_config: dict[str, Any] = (
            json.loads(preproc_path.read_text(encoding="utf-8")) if preproc_path.is_file() else {}
        )
        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])

        raw_refs = _index_safetensors(dir_model)
        vision_refs = {
            k: v for k, v in raw_refs.items()
            if k.startswith(("model.visual.", "model.vision_tower."))
        }
        if not vision_refs:
            raise ValueError(f"Inkling2 Vision: no model.visual.* or model.vision_tower.* tensors in {dir_model}")
        self.tensors: dict[str, TensorRef] = dict(
            sorted(vision_refs.items(), key=lambda item: _layer_sort_key(item[0]))
        )
        for name, ref in self.tensors.items():
            if ref.dtype not in _RAW_DTYPE:
                raise ValueError(f"Inkling2 Vision: {name}: unexpected dtype {ref.dtype}")
            map_vision_tensor_name(name)

    def _prepare_metadata(self) -> None:
        vc = self.vision_config
        w = self.gguf_writer

        w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
        w.add_name(f"{self.dir_model.name}-mmproj")
        w.add_type(gguf.GGUFType.MMPROJ)
        w.add_clip_has_vision_encoder(True)
        w.add_clip_vision_projector_type("inkling2v")
        w.add_vision_projection_dim(int(self.text_config["hidden_size"]))
        w.add_vision_image_size(int(vc["image_size"]))
        w.add_vision_patch_size(int(vc["patch_size"]))
        w.add_vision_embedding_length(int(vc["hidden_size"]))
        w.add_vision_feed_forward_length(int(vc["intermediate_size"]))
        w.add_vision_block_count(int(vc["num_hidden_layers"]))
        w.add_vision_head_count(int(vc["num_attention_heads"]))
        w.add_vision_image_mean([float(x) for x in self.preprocessor_config["image_mean"]])
        w.add_vision_image_std([float(x) for x in self.preprocessor_config["image_std"]])
        w.add_vision_attention_layernorm_eps(float(vc["layer_norm_eps"]))
        w.add_vision_spatial_merge_size(int(vc["spatial_merge_size"]))
        add_file_metadata(w, self._source_repo_url, self._shards)

    def _prepare_tensors(self) -> None:
        w = self.gguf_writer
        ps = int(self.vision_config["patch_size"])
        for name, ref in self.tensors.items():
            gguf_name = map_vision_tensor_name(name)
            qtype = _RAW_DTYPE[ref.dtype]
            if name.endswith(".patch_embedding.weight"):
                out_shape: tuple[int, ...] = (ref.shape[0], 3, ps, ps)

                def _read_patch(r: TensorRef = ref, p: int = ps) -> Iterable[np.ndarray]:
                    t = r.load()
                    yield _raw_bytes(t.reshape(t.shape[0], p, p, 3).permute(0, 3, 1, 2))

                w.add_tensor(gguf_name, streamed(out_shape, qtype, _read_patch), raw_dtype=qtype)
            else:
                def _read(r: TensorRef = ref) -> Iterable[np.ndarray]:
                    yield _raw_bytes(r.load())

                w.add_tensor(gguf_name, streamed(ref.shape, qtype, _read), raw_dtype=qtype)


def build(
    part: str,
    checkpoint: Path,
    out: Path,
    *,
    source_repo_url: str,
    layers: list[int] | None = None,
    split_max_size: int = 0,
    shards: frozenset[int] | None = None,
) -> Inkling2Text | Inkling2DSpark | Inkling2Vision:
    """Constructs the converter for one Inkling 2 GGUF (`text`, `dspark`, or `vision`)."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}: {part!r}")
    if layers is not None:
        raise ValueError("Inkling2 does not support --layers slicing")
    common: dict[str, Any] = {
        "source_repo_url": source_repo_url,
        "split_max_size": split_max_size,
        "shards": shards,
    }
    if part == "text":
        return Inkling2Text(checkpoint, out, **common)
    if part == "dspark":
        return Inkling2DSpark(checkpoint, out, **common)
    return Inkling2Vision(checkpoint, out, **common)
