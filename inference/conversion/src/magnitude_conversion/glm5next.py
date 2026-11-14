"""GLM-5.3-Flash checkpoint (`glm5_next`) -> Magnitude GGUF, at the checkpoint's own precision.

Upstream llama.cpp's converter (`conversion/glm.py` `Glm5NextModel` at the pinned release) dequantizes the FP8
weights to BF16 and widens the router, the convolutions, the indexer's head weights and the mHC and pool-gate weights
to F32. This class keeps upstream's names, metadata and transforms, and stores what upstream widens as the checkpoint
ships it (`native`). Two releases are read, each by its quantization config:

- Z.ai's FP8 release: an FP8 weight is its F8_E4M3 codes plus a `.scale` of the checkpoint's F32 `weight_scale_inv`
  values, one per 128x128 tile, verbatim, as an F32 grid; routed experts are stacked [experts, rows, cols] as
  upstream stacks them, their scales [experts, rows/128, cols/128] beside them;
- a ModelOpt release (NVIDIA's QAD checkpoint): MXFP8 weights beside I8 UE8M0 grids, FP8_PB_WO weights beside F32
  grids, routed experts as stacked NVFP4 with F32 `.scale` [experts] and, for W4A4 experts, `.input_scale`
  [experts]; the projector's NVFP4 and MXFP8 weights likewise; the MTP layer's MXFP8 `kv_b` is split as upstream
  splits it, `v_b` MXFP8 verbatim and `k_b` (transposed, so no grid form fits it) MXFP8 dequantized exactly to BF16;
- BF16 and F32 tensors keep their type. Upstream's transforms are exact or one F32 operation the model performs: the
  `kv_b` split into `k_b` (transposed) and `v_b`, the convolution reshape, `A_log` written as -exp(A_log) in F32 (the
  value the model computes from it, exact negation), and renames.

The file is the text model with its MTP layer as upstream writes it (llama.cpp's nextn layout: the last block), in
layer-major order. A slice keeps layers, renumbered, with the per-layer config remapped, and no MTP layer. A config
that contradicts what upstream's conversion hard-codes is refused.

`upstream_f32` is upstream's own conversion of the same layers with every tensor F32, for upstream llama.cpp to run as
the references' independent producer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from conversion.base import LazyTorchTensor, ModelBase, TextModel, gguf
from conversion.glm import Glm5NextModel
from conversion.qwen3vl import Glm5NextVisionModel

from .exact import negative_exp_f32
from .native import (
    F32,
    UE8M0,
    Format,
    Fp8Grid,
    ModelOpt,
    Native,
    Slice,
    exact_bf16,
    layer_major,
    scale_name,
    streamed,
)

PARTS = ("text", "vision")
_LAYER_ID = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\.")
# Values upstream's `set_gguf_parameters` writes regardless of the config, and the config keys that would state them
_HARD_CODED = {("layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"): 1e-6}


def check_config(tc: dict[str, Any]) -> None:
    """Refuses a text config that contradicts what upstream's conversion writes or encodes without reading it."""
    for keys, value in _HARD_CODED.items():
        for key in keys:
            if key in tc and tc[key] != value:
                raise ValueError(f"config {key} = {tc[key]}, but upstream's conversion writes {value}")
    if tc.get("index_kpool_compress") is False:
        raise ValueError("config index_kpool_compress = false, but upstream's conversion writes the pool compressor")
    n, k = tc["num_hidden_layers"], tc["first_k_dense_replace"]
    if tc["mlp_layer_types"] != ["dense"] * k + ["sparse"] * (n - k):
        raise ValueError(f"config mlp_layer_types is not {k} dense layers then sparse ones, which is all "
                         f"first_k_dense_replace can state")


class Glm5NextSlice(Slice):
    """A GLM-5.x slice: the per-layer type lists keep the kept layers' entries, and the leading dense layers that
    remain are counted."""

    def remap_config(self, tc: dict[str, Any]) -> dict[str, Any]:
        indexer = tc.get("indexer_types")
        if indexer is not None:
            for layer in self.keep:
                # a shared layer reuses the selection of the layer just before it
                if indexer[layer] == "shared" and layer - 1 not in self.new_id:
                    raise ValueError(f"layer {layer} reuses layer {layer - 1}'s indexer selection, which the slice "
                                     f"drops")
        tc = dict(tc)
        for key in ("layer_types", "mlp_layer_types", "indexer_types"):
            if key in tc:
                tc[key] = self.kept(tc[key])
        tc["first_k_dense_replace"] = sum(layer < tc["first_k_dense_replace"] for layer in self.keep)
        tc["num_hidden_layers"] = len(self.keep)
        tc["num_nextn_predict_layers"] = 0  # the MTP layer follows the source's last layer, which no slice keeps
        return tc


@contextmanager
def _tokenizer_from_json(dir_model: Path) -> Iterator[None]:
    """Within the block, transformers' AutoTokenizer loads the checkpoint's tokenizer from its tokenizer.json.

    The checkpoint's tokenizer_config.json is transformers 5's: class `TokenizersBackend` with a list of
    `extra_special_tokens`, which the lock's transformers 4.57.6 can't read. `TokenizersBackend` is the tokenizers
    library over tokenizer.json, which 4.57.6 loads as `PreTrainedTokenizerFast`; tokenizer.json carries every
    added token with its special flag, so the vocabulary, token types and pre-tokenizer hash are the same."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer_class = json.loads((dir_model / "tokenizer_config.json").read_text(encoding="utf-8"))["tokenizer_class"]
    if tokenizer_class != "TokenizersBackend":
        raise ValueError(f"tokenizer class {tokenizer_class!r}: the tokenizer is read from tokenizer.json only as "
                         f"transformers 5's TokenizersBackend")
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


class Glm5NextText(Glm5NextModel):
    """Upstream's text model with its MTP layer, or with a `Glm5NextSlice` only its layers, renumbered; the tokenizer
    read from tokenizer.json. Written as it is, it is upstream's own conversion (`upstream_f32`)."""

    model_arch = gguf.MODEL_ARCH.GLM5_NEXT

    def filter_tensors(self, item):  # type: ignore[override]
        # upstream's filter is a classmethod that tells the MTP layer by `cls._n_main_layers`, which `index_tensors`
        # writes to the class; run on the converter itself it reads the converter's own, so two do not share it
        return super().filter_tensors.__func__(self, item)

    def __init__(self, *args, slice: Glm5NextSlice | None = None, **kwargs):
        self._slice = slice
        super().__init__(*args, **kwargs)

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], torch.Tensor]]:
        # upstream's filter tells the MTP layer by the source model's layer count, which a slice's config no longer
        # states; num_hidden_layers lives in text_config, which is merged only after indexing
        n_layers = self.hparams["text_config"]["num_hidden_layers"]
        self._n_main_layers = n_layers if self._slice is None else self._slice.n_main
        tensors = TextModel.index_tensors(self, remote_hf_model_id)
        if self._slice is not None:
            tensors = self._slice.tensors(tensors)[0]
        # the check compares the tensors the checkpoint has, so a block the checkpoint lacks is refused here
        blocks = n_layers + self.hparams["text_config"].get("num_nextn_predict_layers", 0)
        present = {int(m.group(1)) for name in tensors if (m := _LAYER_ID.search(name)) is not None}
        if missing := sorted(set(range(blocks)) - present):
            raise ValueError(f"layers {missing} have no tensors in the checkpoint")
        return layer_major(tensors)

    def set_vocab(self):
        with _tokenizer_from_json(self.dir_model):
            super().set_vocab()


def declared_formats(quantization: dict[str, Any], filter_tensors: Callable,
                     tensors: dict[str, Callable[[], torch.Tensor]]) -> Callable[[str], Format | None]:
    """The formats a GLM-5.x checkpoint's quantization config declares: Z.ai's FP8 release (each weight that has a
    `weight_scale_inv`, under F32 tiles of its `weight_block_size`) or a ModelOpt release (`ModelOpt`)."""
    method = quantization.get("quant_method")
    if method == "modelopt":
        return ModelOpt(quantization, filter_tensors, tensors)
    if (method, quantization.get("fmt")) != ("fp8", "e4m3"):
        raise ValueError(f"quantization {method!r} {quantization.get('fmt')!r}, expected fp8 e4m3 or modelopt")
    block = tuple(quantization["weight_block_size"])

    def fp8(module: str) -> Fp8Grid | None:
        scale = module + ".weight_scale_inv"
        return Fp8Grid(scale, block, F32) if scale in tensors else None
    return fp8


class _Quantized(Native):
    """A GLM-5.x file's quantized weights, as its checkpoint's `quantization_config` declares them."""

    def __init__(self, *args, quantization: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self._declared = declared_formats(quantization, self.filter_tensors, self.model_tensors)

    def _format(self, module: str) -> Format | None:
        return self._declared(module)


class Glm5Next(_Quantized, Glm5NextText):
    """The text model at the checkpoint's own precision."""

    model_arch = gguf.MODEL_ARCH.GLM5_NEXT  # upstream requires every model class to state its own

    def modify_tensors(self, data_torch, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        if name.endswith(".kv_b_proj.weight") and name in self._fp8:
            self._write_fp8_kv_b(name)
            return
        if name.endswith(".A_log"):
            # upstream's `-torch.exp` differs by a ulp between host architectures; its result is replaced by the
            # correctly rounded one, so the slice's bytes do not depend on the host
            stored = LazyTorchTensor.to_eager(data_torch)
            heads = self.hparams["num_attention_heads"]
            if stored.dtype != torch.float32 or stored.numel() != heads:
                raise ValueError(f"{name}: {stored.dtype} {list(stored.shape)}, expected float32 of {heads} heads")
            source = stored.flatten()
            for new_name, _ in super().modify_tensors(data_torch, name, bid):
                yield new_name, torch.from_numpy(negative_exp_f32(source.numpy()))
            return
        yield from super().modify_tensors(data_torch, name, bid)

    def _write_fp8_kv_b(self, name: str) -> None:
        """An FP8 `kv_b` under a row grid (NVIDIA's MTP layer: MXFP8), split as upstream splits it: `v_b`, each head's
        value rows, stays FP8 with its grid's rows, verbatim; `k_b`, each head's key rows transposed, would run its
        blocks down the columns, a grid no engine form has, so it is written as BF16, each value the exact product
        code x 2^(b - 127), refused unless BF16 holds every one. Its bytes are read at write time, as every quantized
        weight's are."""
        fp8 = self._fp8.pop(name)
        if (fp8.grid.block, fp8.grid.encoding) != ((1, 32), UE8M0):
            raise ValueError(f"{name}: kv_b under a {fp8.grid.block} grid; only MXFP8's row grid splits into k_b "
                             f"and v_b")
        hp = self.hparams
        heads, nope, v = hp["num_attention_heads"], hp["qk_nope_head_dim"], hp["v_head_dim"]
        source = self.model_tensors[name]
        rows, rank = source().shape
        if rows != heads * (nope + v):
            raise ValueError(f"{name}: {rows} rows, not {heads} heads of {nope} key and {v} value rows")
        k_b = self.map_tensor_name(name.replace("kv_b_proj", "k_b_proj"))
        v_b = self.map_tensor_name(name.replace("kv_b_proj", "v_b_proj"))
        T = gguf.GGMLQuantizationType

        def codes() -> torch.Tensor:
            return LazyTorchTensor.to_eager(source()).view(heads, nope + v, rank)

        def grid() -> np.ndarray:
            return fp8.stored().view(np.uint8).reshape(heads, nope + v, rank // 32)

        def key() -> np.ndarray:
            values = exact_bf16(k_b, codes()[:, :nope], grid()[:, :nope]).transpose(0, 2, 1)
            return gguf.quants.quantize(np.ascontiguousarray(values), T.BF16)
        w = self.gguf_writer
        w.add_tensor(k_b, streamed((heads, rank, nope), T.BF16, lambda: (key(),)), raw_dtype=T.BF16)
        w.add_tensor(v_b, streamed((heads, v, rank), T.F8_E4M3,
                                   lambda: (codes()[:, nope:].contiguous().view(torch.uint8).numpy(),)),
                     raw_dtype=T.F8_E4M3)
        w.add_tensor(scale_name(v_b), streamed((heads, v, rank // 32), T.I8,
                                               lambda: (grid()[:, nope:],)),
                     raw_dtype=T.I8)

    def _sources(self, name: str, new_name: str, bid: int | None) -> list[str]:
        if not _EXPERT.match(name):
            return [name]
        # upstream stacks a layer's experts per projection, in expert order
        for proj in ("down_proj", "gate_proj", "up_proj"):
            if self.map_tensor_name(f"model.layers.{bid}.mlp.experts.{proj}.weight") == new_name:
                return [f"model.layers.{bid}.mlp.experts.{e}.{proj}.weight"
                        for e in range(self.hparams["n_routed_experts"])]
        raise ValueError(f"{name}: no stacked projection is written as {new_name}")


def _config(checkpoint: Path, layers: list[int] | None) -> tuple[dict[str, Any], Glm5NextSlice | None]:
    """The checkpoint's config, checked, and with `layers` the slice of them, its config and quantized layers
    remapped."""
    hparams = ModelBase.load_hparams(checkpoint, False)
    check_config(hparams["text_config"])
    if layers is None:
        return hparams, None
    slice = Glm5NextSlice(layers, hparams["text_config"]["num_hidden_layers"])
    hparams["text_config"] = slice.remap_config(hparams["text_config"])
    if "quantized_layers" in (qc := hparams["quantization_config"]):
        hparams["quantization_config"] = {**qc, "quantized_layers": slice.remap_modules(qc["quantized_layers"])}
    return hparams, slice


class Glm5NextVision(_Quantized, Glm5NextVisionModel):
    """The vision tower and merger as a projector GGUF: Z.ai's in BF16 (its quantization skips `visual.*`), or
    ModelOpt's with its MXFP8 and NVFP4 weights kept. The BF16 tensors the base would widen, the quantized weights and
    the file metadata are what differ from upstream's."""

    model_arch = gguf.MODEL_ARCH.MMPROJ


def build(part: str, checkpoint: Path, out: Path, *, source_repo_url: str, layers: list[int] | None = None,
          split_max_size: int = 0, shards: frozenset[int] | None = None) -> Glm5Next | Glm5NextVision:
    """The converter for one file of the model: the text model (or a slice of it) or the projector. `shards` limits
    writing to those shards of the split file (see `AlignedShardsWriter`)."""
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}: {part!r}")
    # BF16 is the type of every tensor nothing else claims; quantized tensors are claimed explicitly
    args = (checkpoint, gguf.LlamaFileType.MOSTLY_BF16, out)
    common: dict[str, Any] = {"split_max_size": split_max_size, "source_repo_url": source_repo_url, "shards": shards}
    if part == "vision":
        if layers is not None:
            raise ValueError("only the text model is sliced")
        hparams = ModelBase.load_hparams(checkpoint, False)
        return Glm5NextVision(*args, hparams=hparams, quantization=hparams["quantization_config"], **common)
    hparams, slice = _config(checkpoint, layers)
    return Glm5Next(*args, hparams=hparams, quantization=hparams["quantization_config"], slice=slice, **common)


def upstream_f32(checkpoint: Path, out: Path, *, layers: list[int] | None = None) -> Glm5NextText:
    """Upstream's own conversion of the text model, or a slice of it, with every tensor F32: FP8 weights dequantized
    as upstream does it, the code times its tile's scale in one F32 product, and BF16 widened exactly. It is the file
    upstream llama.cpp runs on the values the reference computes with (llama.cpp's BF16 dequantization isn't exact);
    never published."""
    hparams, slice = _config(checkpoint, layers)
    return Glm5NextText(checkpoint, gguf.LlamaFileType.ALL_F32, out, hparams=hparams, slice=slice)
