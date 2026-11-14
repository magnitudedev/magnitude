"""What every family's converter shares: the file Magnitude publishes and the checkpoint's own precision in it.

Upstream llama.cpp's converters (at the pinned release) widen what the checkpoint ships narrow. The classes here are
mixed in before an upstream model class and keep its names, metadata and transforms, while storing:

- an FP8 weight `X` as `X.weight` (F8_E4M3 codes) plus `X.scale`, the checkpoint's block scales in its own block grid,
  the grid's encoding being the scale tensor's type (`inference/docs/precision.md`), so one file holds several:
  - I8 holding UE8M0 bytes, each at most 246 so that every E4M3 product is exact in F32;
  - F32 holding the scale values verbatim, each finite and either zero or at least 2^-117 in magnitude, so that
    the engine's `(s * 2^-9) * code` stays one F32 product;
  a weight several source weights are stacked into (routed experts) carries their scale grids stacked the same way;
- an NVFP4 weight `X` (ModelOpt's) as ggml's `block_nvfp4`, its codes and E4M3 block scales repacked bit for bit, plus
  `X.scale`, its F32 second-level scale, one per matrix ([1], or [experts] for stacked experts), and, for a weight
  calibrated for FP4 activations (W4A4), `X.input_scale`, its static activation scale, likewise; block scales stored
  compressed (csf_32x2_v1) are decoded to their exact bytes first;
- BF16 and F32 tensors at their type, vectors, norms and the tensors upstream forces wide included.

A family declares which of its checkpoint's weights are FP8 and under which grid, and which are NVFP4 (`_format`);
`ModelOpt` declares them as a ModelOpt checkpoint's config does. Any other weight not stored as BF16, F16 or F32 is
refused.

A `Slice` is a sub-model of the checkpoint: the kept layers renumbered from 0, every layer-indexed config value
remapped by its family, so a slice converts like any model.
"""

from __future__ import annotations

import errno
import math
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from conversion.base import LazyTorchTensor, ModelBase, gguf

from . import csf

# safetensors' U16 and U32, which upstream's loader leaves out of its type maps (pytorch/pytorch#58734: few torch ops
# take them); ModelOpt stores compressed scale streams in them, which are only ever read whole
for _code, _dtype, _numpy in (("U16", torch.uint16, np.uint16), ("U32", torch.uint32, np.uint32)):
    LazyTorchTensor._dtype_str_map[_code] = _dtype
    LazyTorchTensor._dtype_map[_dtype] = _numpy
    LazyTorchTensor._dtype_byteswap_map[_dtype] = _numpy

ALIGNMENT = 4096
MAX_UE8M0 = 246  # the largest byte whose E4M3 products are exact in F32 (`inference/docs/precision.md`)
MIN_F32_SCALE = 2.0**-117  # the smallest magnitude s for which s * 2^-9 is a normal F32


def scale_name(weight_name: str) -> str:
    return weight_name.removesuffix(".weight") + ".scale"


def check_ue8m0(name: str, scales: np.ndarray) -> None:
    """Raises unless every UE8M0 byte of `scales` keeps its E4M3 products exact in F32."""
    if scales.size and (largest := int(scales.view(np.uint8).max())) > MAX_UE8M0:
        raise ValueError(f"{name}: UE8M0 byte {largest} is above {MAX_UE8M0}, where E4M3 products leave F32's range")


def check_f32(name: str, scales: np.ndarray) -> None:
    """Raises unless every F32 scale is finite, and zero or at least `MIN_F32_SCALE` in magnitude."""
    if not np.isfinite(scales).all():
        raise ValueError(f"{name}: a scale is not finite")
    tiny = (scales != 0) & (np.abs(scales) < MIN_F32_SCALE)
    if tiny.any():
        raise ValueError(f"{name}: scale {scales[tiny].flat[0]!r} is below 2^-117, where s * 2^-9 is subnormal")


@dataclass(frozen=True)
class ScaleEncoding:
    """How a grid of block scales is stored: the stored type, the checkpoint types it is stored from, and the stored
    array of one scale tensor (raising when the engine can't use it exactly)."""

    qtype: gguf.GGMLQuantizationType
    sources: frozenset[torch.dtype]
    stored: Callable[[str, torch.Tensor], np.ndarray]

    def check_source(self, name: str, dtype: torch.dtype) -> None:
        if dtype not in self.sources:
            raise ValueError(f"{name}: a {self.qtype.name} grid is stored from {sorted(map(str, self.sources))} "
                             f"scales, not {dtype}")

    def of(self, name: str, scale: torch.Tensor) -> np.ndarray:
        self.check_source(name, scale.dtype)
        return self.stored(name, scale)


def _ue8m0(name: str, scale: torch.Tensor) -> np.ndarray:
    codes = scale.view(torch.uint8).numpy()
    check_ue8m0(name, codes)
    return codes.view(np.int8)


def _f32(name: str, scale: torch.Tensor) -> np.ndarray:
    values = scale.numpy()
    check_f32(name, values)
    return values


UE8M0 = ScaleEncoding(gguf.GGMLQuantizationType.I8, frozenset({torch.uint8, torch.float8_e8m0fnu}), _ue8m0)
F32 = ScaleEncoding(gguf.GGMLQuantizationType.F32, frozenset({torch.float32}), _f32)


@dataclass(frozen=True)
class Fp8Grid:
    """An FP8 weight's block scales: the checkpoint tensor holding them, the block [rows, columns] each scales, and
    their encoding. The checkpoint stores the grid [rows / r, columns / c], or with a unit axis after each (ModelOpt's
    broadcast form); the file stores it [rows / r, columns / c]."""

    scale: str
    block: tuple[int, int]
    encoding: ScaleEncoding


@dataclass(frozen=True)
class Nvfp4:
    """An NVFP4 weight `<m>.weight` as ModelOpt stores it: U8 codes [rows, columns / 2], two E2M1 codes a byte, low
    nibble first; E4M3 block scales `<m>.weight_scale`, one per 16 along a row, stored plain or (`compressed`) as
    csf_32x2_v1 streams; its F32 second-level scale `<m>.weight_scale_2`; and, when it was calibrated for FP4
    activations (`activations`, W4A4), its F32 static activation scale `<m>.input_scale`."""

    compressed: bool
    activations: bool


Format = Fp8Grid | Nvfp4


class ModelOpt:
    """The formats a ModelOpt checkpoint declares in `quantization_config.quantized_layers`, keyed by module, each
    covering the modules under it (a layer's `mlp.experts` covers every expert's projections): `MXFP8` (UE8M0 per
    1x32), `FP8_PB_WO` (F32 per block), `NVFP4` (W4A4) and `W4A16_NVFP4`, NVFP4's block scales plain or compressed
    (`weight_scale_encoding: csf_32x2_v1`). Module names go through the model's `filter_tensors`, as upstream's do; an
    entry it keeps that covers no weight of the model is refused."""

    def __init__(self, quantization: dict[str, Any], filter_tensors: Callable[[tuple[str, Callable]], Any],
                 tensors: Iterable[str]):
        covered = {name.rpartition(".")[0] for name in tensors if name.endswith(".weight")}
        covered |= {module[:i] for module in list(covered) for i, c in enumerate(module) if c == "."}
        self._entries: dict[str, Nvfp4 | tuple[tuple[int, int], ScaleEncoding]] = {}
        for layer, entry in quantization["quantized_layers"].items():
            if (item := filter_tensors((layer + ".weight", lambda: torch.empty(0)))) is None:
                continue
            module = item[0].removesuffix(".weight")
            if module not in covered:
                raise ValueError(f"quantized_layers names {layer}, which has no weight in the checkpoint")
            self._entries[module] = _modelopt_format(layer, entry)

    def __call__(self, module: str) -> Format | None:
        prefix = module
        while prefix:
            if (entry := self._entries.get(prefix)) is not None:
                return entry if isinstance(entry, Nvfp4) else Fp8Grid(module + ".weight_scale", *entry)
            prefix = prefix.rpartition(".")[0]
        return None


def _modelopt_format(layer: str, entry: dict[str, Any]) -> Nvfp4 | tuple[tuple[int, int], ScaleEncoding]:
    algo = entry.get("quant_algo")
    if algo == "MXFP8":
        if entry.get("weight_block_size", [1, 32]) != [1, 32] or entry.get("group_size", 32) != 32:
            raise ValueError(f"{layer}: MXFP8 is one scale per 32 along a row, not {entry}")
        return (1, 32), UE8M0
    if algo == "FP8_PB_WO":
        rows, cols = entry["weight_block_size"]
        return (rows, cols), F32
    if algo in ("NVFP4", "W4A16_NVFP4"):
        encoding = entry.get("weight_scale_encoding")
        if entry.get("group_size") != 16 or encoding not in (None, "csf_32x2_v1"):
            raise ValueError(f"{layer}: NVFP4 is one E4M3 scale per 16, plain or csf_32x2_v1, not {entry}")
        return Nvfp4(compressed=encoding is not None, activations=algo == "NVFP4")
    raise ValueError(f"{layer}: no rule for quant_algo {algo!r}")


def holds_data(path: Path) -> bool:
    """Whether a GGUF file has any data allocated past its header: a header-only shard is a hole there."""
    offset = gguf.GGUFReader(path, "r").data_offset
    with open(path, "rb") as file:
        try:
            os.lseek(file.fileno(), offset, os.SEEK_DATA)
        except OSError as error:
            if error.errno == errno.ENXIO:
                return False
            raise
    return True


class AlignedShardsWriter(gguf.GGUFWriter):
    """gguf-py pads every shard's data to `general.alignment` but writes the key into the first shard only, so a
    reader of any later shard falls back to the default 32 and misplaces every tensor. Each shard carries it here.

    With `shards`, only those shards (numbered from 1) are written whole, and the others' tensors are never read: a
    change to one shard's metadata rewrites that shard alone. The split is planned from every tensor as before, so a
    written shard is the one a whole conversion writes. Every other shard is written as its header alone, at its full
    size with its data a hole (a sparse file), so the written shards check and verify in their own directory, every
    header of the file beside them. A file holding data where such a header would go is refused, never overwritten:
    it is a shard written whole."""

    shards: frozenset[int] | None
    _held: dict[int, int]  # an unwritten shard's data bytes, by index

    def format_shard_names(self, path: Path) -> list[Path]:
        names = super().format_shard_names(path)
        if self.shards is None:
            return names
        if not self.shards <= set(range(1, len(names) + 1)):
            raise ValueError(f"shards {sorted(self.shards)}: the file has {len(names)}")
        for i, name in enumerate(names, 1):
            if i not in self.shards and name.exists() and holds_data(name):
                raise ValueError(f"{name} holds a shard's data; --shards {','.join(map(str, sorted(self.shards)))} "
                                 f"would write only its header there")
        return names

    def write_ti_data_to_file(self) -> None:
        super().write_ti_data_to_file()
        self._held = {}
        if self.shards is not None:
            for i, tensors in enumerate(self.tensors):
                if i + 1 not in self.shards:
                    self._held[i] = sum(self.ggml_pad(ti.nbytes, self.data_alignment) for ti in tensors.values())
                    tensors.clear()

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        if self.temp_file is not None:
            raise ValueError("a shard's header is sized from its tensors' plan, not from a temporary file")
        super().write_tensors_to_file(progress=progress)
        assert self.fout is not None
        for i, held in self._held.items():
            self.fout[i].truncate(self.fout[i].tell() + held)

    def add_shard_kv_data(self) -> None:
        super().add_shard_kv_data()
        key = gguf.Keys.General.ALIGNMENT
        for kv_data in self.kv_data[1:]:
            kv_data[key] = self.kv_data[0][key]


def add_file_metadata(writer: gguf.GGUFWriter, source_repo_url: str, shards: frozenset[int] | None) -> None:
    writer.__class__ = AlignedShardsWriter  # the writer is made by ModelBase; only its sharding differs
    cast(AlignedShardsWriter, writer).shards = shards
    writer.add_custom_alignment(ALIGNMENT)
    writer.add_key_value(gguf.Keys.General.SOURCE_REPO_URL, source_repo_url, gguf.GGUFValueType.STRING)


class KeepBf16:
    """Writes a BF16 tensor the base would widen as BF16: a vector or a norm (which the base casts to F32 before any
    type choice), or a tensor the base forces to F32 or F16 (the router and the drafter's confidence projection,
    which upstream writes F32, a convolution or the indexer's head weights, which upstream's base widens, and the
    projector's patch embedding, for ggml's convolution)."""

    gguf_writer: gguf.GGUFWriter
    model_tensors: dict[str, Callable[[], torch.Tensor]]

    def _bf16(self, name: str) -> bool:
        return name in self.model_tensors and self.model_tensors[name]().dtype == torch.bfloat16

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        forced = super().tensor_force_quant(name, new_name, bid, n_dims)  # type: ignore[misc]
        # `False` (no type forced) equals F32 (0) as an integer, so membership alone would match it
        if forced is False:
            wider = self._tensor_requires_f32(new_name, bid, n_dims)  # type: ignore[attr-defined]
        else:
            wider = forced in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        return gguf.GGMLQuantizationType.BF16 if wider and self._bf16(name) else forced

    def modify_tensors(self, data_torch, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        bf16 = self._bf16(name)
        for new_name, data in super().modify_tensors(data_torch, name, bid):  # type: ignore[misc]
            if bf16 and (data.ndim <= 1 or new_name.endswith("_norm.weight")):
                # the base casts sources to F32 first, which BF16 survives exactly
                values = LazyTorchTensor.to_eager(data).float().numpy()
                self.gguf_writer.add_tensor(new_name, gguf.quants.quantize(values, gguf.GGMLQuantizationType.BF16),
                                            raw_dtype=gguf.GGMLQuantizationType.BF16)
                continue
            yield new_name, data


@dataclass(frozen=True)
class Fp8Weight:
    """A claimed FP8 weight's block scales: their checkpoint tensor and grid."""

    scale: Callable[[], torch.Tensor]
    grid: Fp8Grid
    shape: tuple[int, int]  # the stored grid

    def stored(self) -> np.ndarray:
        return self.grid.encoding.of(self.grid.scale, LazyTorchTensor.to_eager(self.scale())).reshape(self.shape)


@dataclass(frozen=True)
class Nvfp4Weight:
    """A claimed NVFP4 weight [rows, columns]: its codes, its E4M3 block scale bytes [rows, columns / 16] (decoded
    when compressed), and its F32 second-level and, for W4A4, activation scales, each one value."""

    name: str
    shape: tuple[int, int]
    codes: Callable[[], torch.Tensor]
    scales: Callable[[], np.ndarray]
    scale_2: Callable[[], torch.Tensor]
    input_scale: Callable[[], torch.Tensor] | None

    def block_scales(self) -> np.ndarray:
        """The block scale bytes; raises unless each is a positive finite E4M3 value or zero, which is what ggml's
        unsigned E4M3 block scale holds."""
        scales = self.scales()
        if (scales >= 0x7F).any():
            raise ValueError(f"{self.name}: block scale byte {int(scales.max()):#04x} is not a positive E4M3 value")
        return scales


def exact_bf16(name: str, codes: torch.Tensor, ue8m0: np.ndarray) -> np.ndarray:
    """The values of MXFP8 codes [..., columns] under their UE8M0 bytes [..., columns / 32], each the exact product
    code x 2^(b - 127), in F32; raises naming the tensor and the value unless every one is a BF16 value too, so that
    BF16 holds them exactly (an E4M3 code's 4 significant bits fit BF16's 8, except below BF16's range)."""
    exponents = torch.from_numpy(ue8m0.view(np.uint8).astype(np.int64)).repeat_interleave(32, -1) - 127
    exact = torch.ldexp(codes.to(torch.float64), exponents)
    values = exact.float()
    wrong = (values.double() != exact) | (values.to(torch.bfloat16).float() != values)
    if wrong.any():
        i = tuple(int(x) for x in wrong.nonzero()[0])
        raise ValueError(f"{name}: {float(codes[i].float())} x 2^{int(exponents[i])} = {float(exact[i])!r} at {i} is "
                         f"not a BF16 value")
    return values.numpy()


def _scalar(name: str, gen: Callable[[], torch.Tensor]) -> Callable[[], torch.Tensor]:
    """An NVFP4 weight's F32 per-tensor scale, as one value, read when called; raises unless it is one F32 value
    (at once) and positive and finite (when read)."""
    if gen().dtype != torch.float32 or gen().numel() != 1:
        raise ValueError(f"{name}: {gen().dtype} {list(gen().shape)}, not one F32 value")

    def read() -> torch.Tensor:
        value = LazyTorchTensor.to_eager(gen()).reshape(1)
        if not (math.isfinite(v := float(value)) and v > 0):
            raise ValueError(f"{name}: {v!r}, not a positive finite F32 value")
        return value
    return read


class _StreamedBytes:
    """A tensor's raw bytes [shape], as much of the ndarray interface as `GGUFWriter` reads, produced chunk by chunk
    when the tensor is written (an Engram table is ~200 GB in BF16)."""

    def __init__(self, shape: tuple[int, ...], chunks: Callable[[], Iterable[np.ndarray]]):
        self.shape = shape
        self.dtype = np.dtype(np.uint8)
        self.nbytes = math.prod(shape)
        self._chunks = chunks

    def tofile(self, fout) -> None:
        written = 0
        for chunk in self._chunks():
            data = np.ascontiguousarray(chunk).view(np.uint8)
            data.tofile(fout)
            written += data.nbytes
        if written != self.nbytes:
            raise ValueError(f"streamed {written} bytes, expected {self.nbytes}")

    def byteswap(self, inplace: bool = False):
        raise NotImplementedError("streamed tensors are written little-endian only")


def streamed(shape: tuple[int, ...], qtype: gguf.GGMLQuantizationType,
             chunks: Callable[[], Iterable[np.ndarray]]) -> np.ndarray:
    """A tensor of `shape` (in elements, whole `qtype` blocks along the last axis) whose bytes `chunks` produces, in
    order, when it is written, so that a shard not written (`AlignedShardsWriter`) never reads them, nor refuses
    their contents. Each chunk's bytes are taken as they lie in memory."""
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    assert shape[-1] % block == 0, f"{shape}: not whole {qtype.name} blocks of {block}"  # refused when claimed
    return cast(np.ndarray, _StreamedBytes((*shape[:-1], shape[-1] // block * size), chunks))


class Native(KeepBf16):
    """Keeps FP8 weights as F8_E4M3 codes with their block scales beside them, NVFP4 weights as `block_nvfp4` with
    their scales beside them, and stored types as they are.

    A family declares each quantized weight's format (`_format`) and, where upstream stacks several source weights
    into one tensor, which sources a written tensor holds (`_sources`)."""

    gguf_writer: gguf.GGUFWriter
    model_tensors: dict[str, Callable[[], torch.Tensor]]
    _fp8_e4m3_preserved: set[str]

    def __init__(self, *args, source_repo_url: str, shards: frozenset[int] | None = None, **kwargs):
        self._fp8: dict[str, Fp8Weight] = {}
        self._nvfp4: dict[str, Nvfp4Weight] = {}
        self._source_repo_url = source_repo_url
        self._shards = shards
        super().__init__(*args, **kwargs)

    def _format(self, module: str) -> Format | None:
        """The format of the quantized weight `<module>.weight`, or None if the checkpoint stores it unquantized."""
        raise NotImplementedError

    def _sources(self, name: str, new_name: str, bid: int | None) -> list[str]:
        """The source weights a tensor upstream writes as `new_name` (from source `name`) holds, in order: the source
        itself, unless upstream stacks several into it."""
        return [name]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # type: ignore[misc]
        add_file_metadata(self.gguf_writer, self._source_repo_url, self._shards)

    def _take(self, name: str, of: str) -> Callable[[], torch.Tensor]:
        if name not in self.model_tensors:
            raise ValueError(f"{of}: {name} is missing")
        return self.model_tensors.pop(name)

    def _generate_nvfp4_tensors(self):
        """Upstream writes NVFP4 weights ahead of every other tensor; here `dequant_model` claims them and
        `modify_tensors` writes each in its place."""

    def dequant_model(self):
        """Claims each quantized weight's companions, refusing (before anything is written) a form that can't be
        stored exactly: types, shapes, layouts. Contents are refused as they are read, at write time, so a shard not
        written reads none of them."""
        for name in list(self.model_tensors):
            if not name.endswith(".weight") or (f := self._format(module := name.removesuffix(".weight"))) is None:
                continue
            if isinstance(f, Fp8Grid):
                self._claim_fp8(name, f)
            else:
                self._claim_nvfp4(name, module, f)

    def _claim_fp8(self, name: str, grid: Fp8Grid) -> None:
        weight = self.model_tensors[name]
        if weight().dtype != torch.float8_e4m3fn or weight().ndim != 2:
            raise ValueError(f"{name}: {weight().dtype} {list(weight().shape)}, not an E4M3 FP8 matrix")
        scale = self._take(grid.scale, name)
        (rows, cols), (r, c) = weight().shape, grid.block
        shape = (rows // r, cols // c)
        # whole blocks only: a reader infers the block from the two shapes, which a partial last block makes ambiguous
        if rows % r or cols % c or tuple(scale().shape) not in (shape, (shape[0], 1, shape[1], 1)):
            raise ValueError(f"{name} {(rows, cols)}: scale shape {tuple(scale().shape)} is not whole {grid.block} "
                             f"blocks")
        grid.encoding.check_source(grid.scale, scale().dtype)
        self._fp8[name] = Fp8Weight(scale, grid, shape)
        self._fp8_e4m3_preserved.add(name)

    def _claim_nvfp4(self, name: str, module: str, f: Nvfp4) -> None:
        codes = self.model_tensors[name]
        if codes().dtype != torch.uint8 or codes().ndim != 2:
            raise ValueError(f"{name}: {codes().dtype} {list(codes().shape)}, not a matrix of packed E2M1 codes")
        rows, cols = codes().shape[0], 2 * codes().shape[1]
        block, _ = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.NVFP4]
        if cols % block:
            raise ValueError(f"{name}: {cols} columns, not whole {block}-weight NVFP4 blocks")
        scale = module + ".weight_scale"
        if f.compressed:
            streams = {s: self._take(csf.stream_name(scale, s), name) for s in csf.STREAMS}
            csf.check_layout(scale, rows, cols // 16, {s: (_STREAM_TYPES.get(g().dtype, g().dtype), tuple(g().shape))
                                                       for s, g in streams.items()})

            def scales() -> np.ndarray:
                arrays = {s: LazyTorchTensor.to_eager(g()).numpy() for s, g in streams.items()}
                return csf.decode(scale, rows, cols // 16, arrays)
        else:
            plain = self._take(scale, name)
            if plain().dtype != torch.float8_e4m3fn or tuple(plain().shape) != (rows, cols // 16):
                raise ValueError(f"{scale}: {plain().dtype} {list(plain().shape)}, not E4M3 [{rows}, {cols // 16}]")

            def scales() -> np.ndarray:
                return LazyTorchTensor.to_eager(plain()).view(torch.uint8).numpy()
        input_scale = module + ".input_scale"
        if not f.activations and input_scale in self.model_tensors:
            raise ValueError(f"{name}: W4A16, yet the checkpoint stores {input_scale}")
        self._nvfp4[name] = Nvfp4Weight(
            name, (rows, cols), codes, scales, _scalar(module + ".weight_scale_2", self._take(
                module + ".weight_scale_2", name)),
            _scalar(input_scale, self._take(input_scale, name)) if f.activations else None)

    def modify_tensors(self, data_torch, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        source = self.model_tensors[name]().dtype if name in self.model_tensors else data_torch.dtype
        if name not in self._fp8_e4m3_preserved and name not in self._nvfp4 and source not in _WIDE:
            raise ValueError(f"{name}: {source}, which no declared format stores")
        for new_name, data in super().modify_tensors(data_torch, name, bid):
            sources = self._sources(name, new_name, bid)
            if any(s in self._nvfp4 for s in sources):
                self._write_nvfp4(name, new_name, sources, tuple(data.shape))
                continue
            if data.dtype != torch.float8_e4m3fn:
                yield new_name, data
                continue
            # each source is written once and whole: a split or reshaped FP8 weight no longer matches its grid
            if any(s not in self._fp8 for s in sources):
                raise ValueError(f"{name}: an FP8 weight written as {new_name} without its whole block scales")
            shape = tuple(self.model_tensors[sources[0]]().shape)
            whole = shape if len(sources) == 1 else (len(sources), *shape)
            if tuple(data.shape) != whole or any(tuple(self.model_tensors[s]().shape) != shape for s in sources):
                raise ValueError(f"{name}: an FP8 weight written as {new_name} {tuple(data.shape)}, not whole")
            fp8 = [self._fp8.pop(s) for s in sources]
            if len({(f.grid.block, f.grid.encoding) for f in fp8}) != 1:
                raise ValueError(f"{name}: the weights stacked into {new_name} have different grids")
            yield new_name, data
            # resumed after the weight is added, so its scale follows it in the file
            self.gguf_writer.add_tensor(scale_name(new_name), streamed(
                fp8[0].shape if len(fp8) == 1 else (len(fp8), *fp8[0].shape), fp8[0].grid.encoding.qtype,
                lambda fp8=fp8: (f.stored() for f in fp8)),
                raw_dtype=fp8[0].grid.encoding.qtype)

    def _write_nvfp4(self, name: str, new_name: str, sources: list[str], shape: tuple[int, ...]) -> None:
        """Writes the NVFP4 weights `sources` (stacked when several) as `new_name`, its `block_nvfp4` bytes streamed
        from the checkpoint at write time, then its scales."""
        if any(s not in self._nvfp4 for s in sources):
            raise ValueError(f"{name}: {new_name} would hold NVFP4 and other weights")
        nvfp4 = [self._nvfp4.pop(s) for s in sources]
        rows, cols = nvfp4[0].shape
        # each source is written once and whole: a split or reshaped one is no longer whole blocks
        whole = (rows, cols // 2) if len(nvfp4) == 1 else (len(nvfp4), rows, cols // 2)
        if shape != whole or any(n.shape != nvfp4[0].shape for n in nvfp4):
            raise ValueError(f"{name}: an NVFP4 weight written as {new_name} {shape}, not whole")
        if len({n.input_scale is None for n in nvfp4}) != 1:
            raise ValueError(f"{name}: {new_name} stacks W4A4 and W4A16 weights")

        def blocks() -> Iterable[np.ndarray]:
            for n in nvfp4:
                codes = LazyTorchTensor.to_eager(n.codes())
                yield ModelBase._nvfp4_pack(codes, torch.from_numpy(n.block_scales()))[0]
        nvfp4_type = gguf.GGMLQuantizationType.NVFP4
        self.gguf_writer.add_tensor(new_name, streamed(
            (rows, cols) if len(nvfp4) == 1 else (len(nvfp4), rows, cols), nvfp4_type, blocks), raw_dtype=nvfp4_type)
        stem = new_name.removesuffix(".weight")
        f32 = gguf.GGMLQuantizationType.F32
        self.gguf_writer.add_tensor(stem + ".scale", streamed(
            (len(nvfp4),), f32, lambda: (n.scale_2().numpy() for n in nvfp4)), raw_dtype=f32)
        if nvfp4[0].input_scale is not None:
            self.gguf_writer.add_tensor(stem + ".input_scale", streamed((len(nvfp4),), f32, lambda: (
                s().numpy() for n in nvfp4 if (s := n.input_scale) is not None)), raw_dtype=f32)
        self._tag_prec_a4(name, new_name)  # type: ignore[attr-defined]  # upstream's W4A16 key, from the config

    def write(self) -> None:
        """A refused or failed conversion leaves no file: what this write opened holds nothing a reader can use (a
        content refusal comes as the refused tensor is read, after the headers are written)."""
        try:
            super().write()  # type: ignore[misc]
        except BaseException:
            writer = self.gguf_writer
            if writer.fout is not None:
                opened = [Path(f.name) for f in writer.fout]
                writer.close()
                for path in opened:
                    path.unlink(missing_ok=True)
            raise

    def prepare_tensors(self):
        super().prepare_tensors()  # type: ignore[misc]
        if unwritten := sorted(self._fp8) + sorted(self._nvfp4):
            raise ValueError(f"quantized weights not written with their scales: {unwritten[:5]}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int):
        if name in self._fp8_e4m3_preserved:
            return gguf.GGMLQuantizationType.F8_E4M3
        return super().tensor_force_quant(name, new_name, bid, n_dims)


_WIDE = (torch.bfloat16, torch.float16, torch.float32)  # types stored as they are
_STREAM_TYPES = {torch.uint8: np.dtype(np.uint8), torch.uint16: np.dtype(np.uint16), torch.uint32: np.dtype(np.uint32)}
_LAYER = re.compile(r"((?:model\.)?(?:language_model\.)?layers)\.(\d+)\.")


class Slice:
    """Layers kept from the checkpoint's main model, in order; kept layer `keep[i]` becomes layer `i`. A family
    remaps its layer-indexed config (`remap_config`)."""

    def __init__(self, keep: list[int], n_layers: int):
        if not keep or sorted(set(keep)) != keep or keep[0] < 0 or keep[-1] >= n_layers:
            raise ValueError(f"layers must be distinct, ascending and below {n_layers}: {keep}")
        self.keep = keep
        self.new_id = {old: new for new, old in enumerate(keep)}
        self.n_main = n_layers

    def kept(self, per_layer: list[Any]) -> list[Any]:
        """The kept layers' entries of a per-layer config list."""
        if len(per_layer) != self.n_main:
            raise ValueError(f"a per-layer list of {len(per_layer)}, the model has {self.n_main} layers")
        return [per_layer[layer] for layer in self.keep]

    def remap_config(self, tc: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def rename(self, name: str) -> str | None:
        """A checkpoint name in the slice: a kept layer's renamed to its new id, None for a dropped layer's, any
        other as it is."""
        if (m := _LAYER.match(name)) is None:
            return name
        if (new := self.new_id.get(int(m.group(2)))) is None:
            return None
        return f"{m.group(1)}.{new}.{name[m.end():]}"

    def remap_modules(self, per_module: dict[str, Any]) -> dict[str, Any]:
        """A config keyed by module name (a ModelOpt `quantized_layers`), its modules renamed as `rename` does."""
        return {new: value for name, value in per_module.items() if (new := self.rename(name)) is not None}

    def tensors(self, tensors: dict[str, Callable[[], torch.Tensor]]
                ) -> tuple[dict[str, Callable[[], torch.Tensor]], dict[str, str]]:
        """The kept layers' tensors renamed to their new ids, with every other tensor, and each renamed tensor's
        source name."""
        kept: dict[str, Callable[[], torch.Tensor]] = {}
        source: dict[str, str] = {}
        for name, gen in tensors.items():
            if (renamed := self.rename(name)) is None:
                continue
            if renamed != name:
                source[renamed] = name
            kept[renamed] = gen
        return kept, source


def layer_major(tensors: dict[str, Callable[[], torch.Tensor]]) -> dict[str, Callable[[], torch.Tensor]]:
    """The tensors in file order: those of no layer first, then each layer's in turn, each group in source order."""
    def layer(name: str) -> int:
        return -1 if (m := _LAYER.match(name)) is None else int(m.group(2))
    return dict(sorted(tensors.items(), key=lambda item: layer(item[0])))
