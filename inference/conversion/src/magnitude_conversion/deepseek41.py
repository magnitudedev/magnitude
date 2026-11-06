"""DeepSeek V4.1 checkpoint -> Magnitude GGUF, at the checkpoint's own precision.

Upstream llama.cpp's converter (`conversion/deepseek41.py` at the pinned release) is exact but widens the FP8
weights and the Engram tables to BF16, and the router to F32. These classes keep upstream's names, metadata,
MXFP4 expert repack, DSpark and vision paths, and store what upstream widens as the checkpoint ships it:

- an FP8 weight `X` is written as `X.weight` (F8_E4M3 codes) plus `X.scale` (I8 holding the checkpoint's UE8M0
  bytes, one per block, in the checkpoint's block grid), declared by `magnitude.scale_encoding = "ue8m0"`
  (the convention is `inference/docs/precision.md`'s "UE8M0-scaled FP8 weights");
- the Engram tables are `per_layer_token_embd.weight` (F8_E4M3) plus `per_layer_token_embd.scale` (I8, one per 32);
- BF16 and F32 tensors keep their type. The Engram gate stays upstream's F32 product of two BF16 tensors (exact).

A slice (`layers`) is a sub-model of the checkpoint: the kept layers renumbered from 0, with every layer-indexed
config value remapped, so a slice converts like any model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from conversion.base import LazyTorchTensor, ModelBase, gguf
from conversion.deepseek41 import (
    DeepseekV41DSparkModel,
    DeepseekV41Model,
    DeepseekV41VisionModel,
    _StreamedBytes,
    engram_multipliers,
    engram_primes,
)

SCALE_ENCODING = "magnitude.scale_encoding"
UE8M0 = "ue8m0"
MAX_UE8M0 = 246  # the largest byte whose E4M3 products are exact in F32 (`inference/docs/precision.md`)
ALIGNMENT = 4096
_EXPERT_SCALE = re.compile(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.scale$")


def scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".scale"


def check_ue8m0(name: str, scales: np.ndarray) -> None:
    """Raises unless every UE8M0 byte of `scales` keeps its E4M3 products exact in F32."""
    if scales.size and (largest := int(scales.view(np.uint8).max())) > MAX_UE8M0:
        raise ValueError(f"{name}: UE8M0 byte {largest} is above {MAX_UE8M0}, where E4M3 products leave F32's range")


def _add_file_metadata(writer: gguf.GGUFWriter, source_repo_url: str) -> None:
    writer.add_custom_alignment(ALIGNMENT)
    writer.add_key_value(gguf.Keys.General.SOURCE_REPO_URL, source_repo_url, gguf.GGUFValueType.STRING)


class _NativeFp8:
    """Keeps FP8 weights as F8_E4M3 codes with their block scales beside them, and stored types as they are."""

    gguf_writer: gguf.GGUFWriter
    model_tensors: dict[str, Callable[[], torch.Tensor]]
    _fp8_block: tuple[int, int]
    _fp8_e4m3_preserved: set[str]
    _dsv4_bf16_tensors: set[str]
    _dsv4_f32_tensors: set[str]
    _ENGRAM_EMBED_RE: Any

    def __init__(self, *args, source_repo_url: str, **kwargs):
        self._fp8_scales: dict[str, Callable[[], torch.Tensor]] = {}
        self._source_repo_url = source_repo_url
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        _add_file_metadata(self.gguf_writer, self._source_repo_url)
        self.gguf_writer.add_key_value(SCALE_ENCODING, UE8M0, gguf.GGUFValueType.STRING)

    def dequant_model(self):
        bo, bi = self._fp8_block
        for name in [n for n in self.model_tensors if n.endswith(".scale") and not self._ENGRAM_EMBED_RE.match(n)]:
            if _EXPERT_SCALE.match(name):
                continue  # MXFP4 expert scales, consumed with their experts
            weight_name = name.removesuffix(".scale") + ".weight"
            weight = self.model_tensors.get(weight_name)
            if weight is None or weight().dtype != torch.float8_e4m3fn:
                found = "missing" if weight is None else weight().dtype
                raise ValueError(f"{name}: its weight {weight_name} is {found}, not E4M3 FP8")
            out_f, in_f = weight().shape
            scale = self.model_tensors.pop(name)
            if tuple(scale().shape) != (-(-out_f // bo), -(-in_f // bi)):
                raise ValueError(f"{weight_name}: scale shape {tuple(scale().shape)} does not match {(bo, bi)} blocks")
            check_ue8m0(name, LazyTorchTensor.to_eager(scale()).view(torch.uint8).numpy())
            self._fp8_scales[weight_name] = scale
            self._fp8_e4m3_preserved.add(weight_name)

    def modify_tensors(self, data_torch, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        for new_name, data in super().modify_tensors(data_torch, name, bid):
            yield new_name, data
            # resumed after the weight is added, so its scale follows it in the file
            if (scale := self._fp8_scales.get(name)) is not None:
                codes = LazyTorchTensor.to_eager(scale()).view(torch.uint8).numpy().view(np.int8)
                self.gguf_writer.add_tensor(scale_name(new_name), codes, raw_dtype=gguf.GGMLQuantizationType.I8)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if name in self._fp8_e4m3_preserved:
            return gguf.GGMLQuantizationType.F8_E4M3
        if new_name.endswith(".engram_gate.weight") or name in self._dsv4_f32_tensors:
            return gguf.GGMLQuantizationType.F32
        if name in self._dsv4_bf16_tensors and n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16
        return False


class Slice:
    """Layers kept from the checkpoint's main model, in order; kept layer `keep[i]` becomes layer `i`."""

    def __init__(self, keep: list[int], text_config: dict[str, Any]):
        n = text_config["num_hidden_layers"]
        if not keep or sorted(set(keep)) != keep or keep[-1] >= n:
            raise ValueError(f"layers must be distinct, ascending and below {n}: {keep}")
        self.keep = keep
        self.new_id = {old: new for new, old in enumerate(keep)}
        self.n_main = n

    def remap_config(self, tc: dict[str, Any]) -> dict[str, Any]:
        """The kept layers' config. Every kept compressed layer must keep the source layers it reads."""
        ratios = tc["compress_ratios"]
        tc = dict(tc)

        def governing(sources: list[int], layer: int) -> int:
            return max(s for s in sources if s <= layer)

        for key in ("kv_source_layer_ids", "index_source_layer_ids"):
            for layer in self.keep:
                if ratios[layer] > 0 and (source := governing(tc[key], layer)) not in self.new_id:
                    raise ValueError(f"layer {layer} reads {key} layer {source}, which the slice drops")
            tc[key] = [self.new_id[s] for s in tc[key] if s in self.new_id]
        candidate = tc.get("candidate_source_layer_id", -1)
        if candidate >= 0:
            readers = [layer for layer in self.keep if ratios[layer] == 1]
            if readers and candidate not in self.new_id:
                raise ValueError(f"layers {readers} read candidate source layer {candidate}, which the slice drops")
            tc["candidate_source_layer_id"] = self.new_id[candidate] if readers else -1
        tc["compress_ratios"] = [ratios[layer] for layer in self.keep] + ratios[self.n_main:]
        tc["num_hidden_layers"] = len(self.keep)
        if not self.engram_layers(tc.get("engram_layer_ids") or []):
            # the source Engram layer list stays when any is kept: its hashing constants depend on all of them
            tc["engram_layer_ids"], tc["engram_num_embeddings"] = [], []
        return tc

    def engram_layers(self, engram_layer_ids: list[int]) -> list[int]:
        return [layer for layer in engram_layer_ids if layer in self.new_id]


class DeepseekV41(_NativeFp8, DeepseekV41Model):
    """The text model. With a `Slice`, only its layers, renumbered."""

    model_arch = gguf.MODEL_ARCH.DEEPSEEK41

    def __init__(self, *args, slice: Slice | None = None, **kwargs):
        self._slice = slice
        self._source_name: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id)
        if self._slice is None:
            return tensors
        kept: dict[str, Callable[[], torch.Tensor]] = {}
        for name, gen in tensors.items():
            parts = name.split(".", 2)
            if parts[0] == "layers":
                if (new := self._slice.new_id.get(int(parts[1]))) is None:
                    continue
                renamed = f"layers.{new}.{parts[2]}"
                self._source_name[renamed] = name
                name = renamed
            kept[name] = gen
        return kept

    def _tensor_part(self, name: str) -> Path:
        return super()._tensor_part(self._source_name.get(name, name))

    def _engram_layers(self) -> tuple[list[int], list[int]]:
        """(source ids, ids in this file) of the Engram layers this file holds."""
        source = list(self.hparams.get("engram_layer_ids") or [])
        if self._slice is None:
            return source, source
        kept = self._slice.engram_layers(source)
        return kept, [self._slice.new_id[layer] for layer in kept]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self._slice is None:
            return
        # GGUF cannot hold empty arrays: a layer list the kept layers leave empty (no source layers) is omitted
        w = self.gguf_writer
        for key in [k for k, v in w.kv_data[0].items() if isinstance(v.value, list) and not v.value]:
            w.remove_key(key)
        source, kept = self._engram_layers()
        if not kept:
            return
        # Hashing constants follow the source model's Engram layers (primes are unique across all of them and
        # the multipliers are seeded by source layer id); the table holds only the kept layers' rows.
        hp = self.hparams
        all_ids = list(hp["engram_layer_ids"])
        primes = engram_primes(len(all_ids), hp["engram_max_ngram_size"], hp["engram_n_heads"], hp["engram_vocab_size"])
        rows = engram_multipliers(all_ids, hp["engram_max_ngram_size"], hp["engram_compressed_vocab_size"])
        positions = [all_ids.index(layer) for layer in source]
        sizes = [p for i in positions for per in primes[i] for p in per]
        keys = gguf.Keys.PerLayerEmbedding
        for key in (keys.LAYERS, keys.LAYER_MULTIPLIERS, keys.HEAD_OFFSETS, keys.HEAD_VOCAB_SIZES):
            w.remove_key(key.format(arch=w.arch))
        w.add_key_value(keys.LAYERS.format(arch=w.arch), kept, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        w.add_ple_layer_multipliers([m for i in positions for m in rows[i]])
        w.add_ple_head_offsets([sum(sizes[:i]) for i in range(len(sizes))])
        w.add_ple_head_vocab_sizes(sizes)

    def generate_extra_tensors(self) -> Iterable[tuple[str, torch.Tensor]]:
        if not self._dsv4_mxfp4_generated:
            for bid in range(self.block_count):
                self._write_mxfp4_experts(bid, "w1", gguf.MODEL_TENSOR.FFN_GATE_EXP)
                self._write_mxfp4_experts(bid, "w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP)
                self._write_mxfp4_experts(bid, "w3", gguf.MODEL_TENSOR.FFN_UP_EXP)
            source, kept = self._engram_layers()
            if kept:
                self._write_engram_tables(source, kept)
            self._dsv4_mxfp4_generated = True
        return ()

    def _write_engram_tables(self, source: list[int], kept: list[int]) -> None:  # type: ignore[override]
        """The kept layers' tables, concatenated in layer order: E4M3 codes and their UE8M0 scales, verbatim."""
        all_ids = list(self.hparams["engram_layer_ids"])
        parts, dim = [], None
        for layer, bid in zip(source, kept):
            wn, sn = f"layers.{bid}.engram.embed.weight", f"layers.{bid}.engram.embed.scale"
            rows, d = self.model_tensors.pop(wn)().shape
            if d % 32 != 0 or tuple(self.model_tensors.pop(sn)().shape) != (rows, d // 32) or d != (dim or d):
                raise ValueError(f"{wn}: expected one e8m0 scale per 32 elements and the same width in every layer")
            if rows != self.hparams["engram_num_embeddings"][all_ids.index(layer)]:
                raise ValueError(f"{wn}: {rows} rows, config says otherwise")
            dim = d
            parts.append((self._source_name.get(wn, wn), self._source_name.get(sn, sn), rows,
                          self._tensor_part(wn), self._tensor_part(sn)))
        assert dim is not None
        n_rows = sum(p[2] for p in parts)

        def chunks(tensor: int) -> Callable[[], Iterable[np.ndarray]]:
            def read() -> Iterable[np.ndarray]:
                from safetensors import safe_open
                for wn, sn, rows, wpart, spart in parts:
                    name, part = (wn, wpart) if tensor == 0 else (sn, spart)
                    with safe_open(part, framework="pt") as f:
                        view = f.get_slice(name)
                        for r0 in range(0, rows, 1 << 18):
                            chunk = view[r0:r0 + (1 << 18)].view(torch.uint8).numpy()
                            if tensor == 1:
                                check_ue8m0(name, chunk)
                            yield chunk
            return read

        name = self.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
        self.gguf_writer.add_tensor(name, cast(np.ndarray, _StreamedBytes((n_rows, dim), chunks(0))),
                                    raw_dtype=gguf.GGMLQuantizationType.F8_E4M3)
        self.gguf_writer.add_tensor(scale_name(name), cast(np.ndarray, _StreamedBytes((n_rows, dim // 32), chunks(1))),
                                    raw_dtype=gguf.GGMLQuantizationType.I8)


class DeepseekV41DSpark(_NativeFp8, DeepseekV41DSparkModel):
    """The DSpark drafter (`mtp.*`) as a `dflash` GGUF."""

    model_arch = gguf.MODEL_ARCH.DFLASH


class DeepseekV41Vision(DeepseekV41VisionModel):
    """The vision tower and aligner as a projector GGUF. It has no FP8 tensors, so only the file metadata differs."""

    model_arch = gguf.MODEL_ARCH.MMPROJ

    def __init__(self, *args, source_repo_url: str, **kwargs):
        self._source_repo_url = source_repo_url
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        _add_file_metadata(self.gguf_writer, self._source_repo_url)


PARTS = ("text", "dspark", "vision")


def build(part: str, checkpoint: Path, out: Path, *, source_repo_url: str, layers: list[int] | None = None,
          split_max_size: int = 0) -> DeepseekV41 | DeepseekV41DSpark | DeepseekV41Vision:
    """The converter for one file of the model: the text model (or a slice of it), the drafter or the projector."""
    hparams = ModelBase.load_hparams(checkpoint, False)
    common: dict[str, Any] = {"hparams": hparams, "split_max_size": split_max_size, "source_repo_url": source_repo_url}
    # BF16 is the type of every tensor nothing else claims; FP8 and MXFP4 tensors are claimed explicitly
    args = (checkpoint, gguf.LlamaFileType.MOSTLY_BF16, out)
    if part == "text":
        slice = None
        if layers is not None:
            slice = Slice(layers, hparams["text_config"])
            hparams["text_config"] = slice.remap_config(hparams["text_config"])
        return DeepseekV41(*args, slice=slice, **common)
    if layers is not None:
        raise ValueError("only the text model is sliced")
    if part == "dspark":
        return DeepseekV41DSpark(*args, **common)
    if part == "vision":
        return DeepseekV41Vision(*args, **common)
    raise ValueError(f"part must be one of {PARTS}: {part!r}")
