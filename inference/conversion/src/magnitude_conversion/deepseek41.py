"""DeepSeek V4.1 checkpoint -> Magnitude GGUF, at the checkpoint's own precision.

Upstream llama.cpp's converter (`conversion/deepseek41.py` at the pinned release) is exact but widens the FP8
weights and the Engram tables to BF16, and the router to F32. These classes keep upstream's names, metadata,
MXFP4 expert repack, DSpark and vision paths, and store what upstream widens as the checkpoint ships it:

- an FP8 weight `X` is written as `X.weight` (F8_E4M3 codes) plus `X.scale` (I8 holding the checkpoint's UE8M0
  bytes, one per block, in the checkpoint's block grid; the convention is `inference/docs/precision.md`'s
  "Block-scaled FP8 weights");
- the Engram tables are `per_layer_token_embd.weight` (F8_E4M3) plus `per_layer_token_embd.scale` (I8, one per 32);
- BF16 and F32 tensors keep their type, vectors and norms included (upstream's base writes those F32). The Engram
  gate stays upstream's F32 product of two BF16 tensors (exact).

A slice (`layers`) is a sub-model of the checkpoint: the kept layers renumbered from 0, with every layer-indexed
config value remapped, so a slice converts like any model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from conversion.base import ModelBase, gguf
from conversion.deepseek41 import (
    DeepseekV41DSparkModel,
    DeepseekV41Model,
    DeepseekV41VisionModel,
    engram_multipliers,
    engram_primes,
)

from .native import UE8M0, Fp8Grid, KeepBf16, Native, Slice, add_file_metadata, check_ue8m0, scale_name, streamed

_EXPERT_SCALE = re.compile(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.scale$")
# The checkpoint states its chat format only as Python (`encoding/encoding.py`), which no GGUF reader runs. The text
# file carries its Jinja translation, which the engine's chat tests render equal to the encoding
# (`engine/chat/tests/real_tokenizer.rs`, against `magnitude_references.deepseek41_chat`).
CHAT_TEMPLATE = (Path(__file__).parent / "deepseek41.jinja").read_text(encoding="utf-8")


class _Native(Native):
    """V4.1's FP8 weights: `X.weight` beside its UE8M0 `X.scale`. The Engram tables' scales and the MXFP4 experts'
    are not FP8 block scales of a weight; both are written whole by their own paths."""

    _ENGRAM_EMBED_RE: Any
    _fp8_block: tuple[int, int]

    def _format(self, module: str) -> Fp8Grid | None:
        scale = module + ".scale"
        if scale not in self.model_tensors or self._ENGRAM_EMBED_RE.match(scale) or _EXPERT_SCALE.match(scale):
            return None
        return Fp8Grid(scale, self._fp8_block, UE8M0)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if new_name.endswith(".engram_gate.weight"):
            # upstream's F32 product of two BF16 tensors, exact only in F32: not a BF16 source to keep
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)


class Deepseek41Slice(Slice):
    """A V4.1 slice: the kept layers' compression sources, candidate source and Engram layers remapped."""

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


class DeepseekV41(_Native, DeepseekV41Model):
    """The text model. With a `Slice`, only its layers, renumbered."""

    model_arch = gguf.MODEL_ARCH.DEEPSEEK41

    def __init__(self, *args, slice: Deepseek41Slice | None = None, **kwargs):
        self._slice = slice
        self._source_name: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id)
        if self._slice is None:
            return tensors
        tensors, self._source_name = self._slice.tensors(tensors)
        return tensors

    def _tensor_part(self, name: str) -> Path:
        return super()._tensor_part(self._source_name.get(name, name))

    def set_vocab(self):
        super().set_vocab()
        # The encoding defines the format, so its translation replaces any template the checkpoint ships
        self.gguf_writer.remove_key(gguf.Keys.Tokenizer.CHAT_TEMPLATE)
        self.gguf_writer.add_chat_template(CHAT_TEMPLATE)

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
        T = gguf.GGMLQuantizationType
        self.gguf_writer.add_tensor(name, streamed((n_rows, dim), T.F8_E4M3, chunks(0)), raw_dtype=T.F8_E4M3)
        self.gguf_writer.add_tensor(scale_name(name), streamed((n_rows, dim // 32), T.I8, chunks(1)), raw_dtype=T.I8)


class DeepseekV41DSpark(_Native, DeepseekV41DSparkModel):
    """The DSpark drafter (`mtp.*`) as a `dflash` GGUF."""

    model_arch = gguf.MODEL_ARCH.DFLASH


class DeepseekV41Vision(KeepBf16, DeepseekV41VisionModel):
    """The vision tower and aligner as a projector GGUF. It has no FP8 tensors: the BF16 tensors the base would widen
    (vectors, norms, the patch embedding) and the file metadata are what differ."""

    model_arch = gguf.MODEL_ARCH.MMPROJ

    def __init__(self, *args, source_repo_url: str, shards: frozenset[int] | None = None, **kwargs):
        self._source_repo_url = source_repo_url
        self._shards = shards
        super().__init__(*args, **kwargs)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        add_file_metadata(self.gguf_writer, self._source_repo_url, self._shards)


PARTS = ("text", "dspark", "vision")


def small_pool(text_config: dict[str, Any], blocks: int, block_rows: int, top_k: int) -> dict[str, Any]:
    """The text config with its indexer's candidate pool of `blocks` blocks of `block_rows` rows and its `top_k`:
    a test definition whose pool binds over short prompts. The engine and the reference read these keys alone."""
    if not (blocks > 0 and blocks % 8 == 0 and 0 < block_rows <= 16 and 0 < top_k <= blocks * block_rows):
        raise ValueError(f"a candidate pool takes a multiple of 8 blocks of at most 16 rows holding at least top_k "
                         f"rows: {blocks} blocks of {block_rows}, top_k {top_k}")
    return {**text_config, "candidate_topk_blocks": blocks, "candidate_block_size": block_rows, "index_topk": top_k}


def build(part: str, checkpoint: Path, out: Path, *, source_repo_url: str, layers: list[int] | None = None,
          pool: tuple[int, int, int] | None = None,
          split_max_size: int = 0, shards: frozenset[int] | None = None
          ) -> DeepseekV41 | DeepseekV41DSpark | DeepseekV41Vision:
    """The converter for one file of the model: the text model (or a slice of it, or with a small candidate pool),
    the drafter or the projector. `shards` limits writing to those shards of the split file (see
    `AlignedShardsWriter`)."""
    hparams = ModelBase.load_hparams(checkpoint, False)
    if pool is not None:
        if part != "text":
            raise ValueError("only the text model has a candidate pool")
        hparams["text_config"] = small_pool(hparams["text_config"], *pool)
    common: dict[str, Any] = {"hparams": hparams, "split_max_size": split_max_size, "source_repo_url": source_repo_url,
                              "shards": shards}
    # BF16 is the type of every tensor nothing else claims; FP8 and MXFP4 tensors are claimed explicitly
    args = (checkpoint, gguf.LlamaFileType.MOSTLY_BF16, out)
    if part == "text":
        slice = None
        if layers is not None:
            slice = Deepseek41Slice(layers, hparams["text_config"]["num_hidden_layers"])
            hparams["text_config"] = slice.remap_config(hparams["text_config"])
        return DeepseekV41(*args, slice=slice, **common)
    if layers is not None:
        raise ValueError("only the text model is sliced")
    if part == "dspark":
        return DeepseekV41DSpark(*args, **common)
    if part == "vision":
        return DeepseekV41Vision(*args, **common)
    raise ValueError(f"part must be one of {PARTS}: {part!r}")
