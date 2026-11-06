"""Byte-exact check of a converted file against the checkpoint it came from.

Tensor names come from upstream's mapping, the one the converter writes by. Every tensor's bytes are rebuilt from the
safetensors by this module's rules, independently of the converter's data path:

- F8_E4M3 codes, I8 scales, BF16 and F32 tensors equal the checkpoint's bytes, at the checkpoint's type; a source
  upstream splits into several tensors is compared run of rows by run of rows;
- the Engram gate equals the F32 product of `q_weight` and `k_weight`;
- MXFP4 experts equal the ggml `block_mxfp4` layout built from the checkpoint's codes and UE8M0 scales;
- the Engram table equals the kept layers' code and scale rows, concatenated.

The file must hold exactly the expected tensors, each at the expected type, shape and alignment.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from conversion.base import LazyTorchTensor, gguf

from .deepseek41 import (ALIGNMENT, SCALE_ENCODING, UE8M0, DeepseekV41, DeepseekV41DSpark, DeepseekV41Vision,
                         scale_name)

logger = logging.getLogger(__name__)

T = gguf.GGMLQuantizationType
_CHUNK = 1 << 28
_RAW = {torch.bfloat16: T.BF16, torch.float32: T.F32}  # checkpoint types stored as they are


@dataclass
class Expected:
    qtype: T
    shape: tuple[int, ...]
    chunks: Callable[[], Iterable[np.ndarray]]  # the tensor's bytes, in order


def _raw(tensor: torch.Tensor) -> np.ndarray:
    return tensor.contiguous().view(torch.uint8).numpy().reshape(-1)


def _eager(gen: Callable[[], torch.Tensor]) -> torch.Tensor:
    return LazyTorchTensor.to_eager(gen())


def _once(make: Callable[[], np.ndarray]) -> Callable[[], Iterable[np.ndarray]]:
    return lambda: (make(),)


def mxfp4_blocks(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """ggml block_mxfp4 bytes of one expert matrix: per 32 values, the UE8M0 byte, then 16 bytes holding value j in
    the low nibble and value j + 16 in the high one. Source codes are packed two per byte, value 2i low."""
    rows = packed.shape[0]
    codes = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    codes = codes.reshape(rows, -1, 32)
    qs = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return np.concatenate((scale.reshape(rows, -1, 1), qs), axis=2).reshape(-1)


def expected_tensors(model: DeepseekV41 | DeepseekV41DSpark | DeepseekV41Vision) -> dict[str, Expected]:
    out: dict[str, Expected] = {}

    def add(name: str, e: Expected) -> None:
        if name in out:
            raise ValueError(f"two sources for {name}")
        out[name] = e

    # experts and Engram tables are written whole by the converter, not tensor by tensor; checked below
    separate = re.compile(r"layers\.\d+\.(ffn\.experts\.\d+\.w[123]|engram\.embed)\.(weight|scale)$")
    # upstream's modify_tensors: names only, without the writes this package's classes make inside it
    upstream = next(c for c in type(model).__mro__ if c.__module__.startswith("conversion."))
    if not isinstance(model, DeepseekV41Vision):
        model.dequant_model()  # the converter's bookkeeping only: which weights keep FP8 codes, and their scales
    for name, gen in model.model_tensors.items():
        if separate.match(name):
            continue
        source = gen()
        bid = next((int(p) for p in name.split(".") if p.isdecimal()), None)
        outputs = list(upstream.modify_tensors(model, source, name, bid))
        if len(outputs) > 1:
            # one source written as several tensors (the vision MLP's w1 as gate and up): each must be the next run of
            # the source's rows, at the source's type
            shapes = [tuple(data.shape) for _, data in outputs]
            if (source.dtype not in _RAW or sum(s[0] for s in shapes) != source.shape[0]
                    or any(s[1:] != tuple(source.shape[1:]) for s in shapes)):
                raise ValueError(f"{name}: no rule for {source.dtype} {tuple(source.shape)} written as {shapes}")
            qtype = _RAW[source.dtype]
            r0 = 0
            for (new_name, _), shape in zip(outputs, shapes):
                add(new_name, Expected(qtype, shape, _once(
                    lambda g=gen, r0=r0, n=shape[0]: _raw(_eager(g)[r0:r0 + n]))))
                r0 += shape[0]
            continue
        for new_name, data in outputs:
            shape = tuple(data.shape)
            if new_name.endswith(".engram_gate.weight"):
                k = model.model_tensors[name.removesuffix("q_weight") + "k_weight"]
                add(new_name, Expected(T.F32, shape, _once(
                    lambda g=gen, k=k: _raw(_eager(g).float() * _eager(k).float()))))
            elif source.dtype == torch.float8_e4m3fn:
                add(new_name, Expected(T.F8_E4M3, shape, _once(lambda g=gen: _raw(_eager(g)))))
                scale = model._fp8_scales[name]
                add(scale_name(new_name), Expected(T.I8, tuple(scale().shape), _once(lambda s=scale: _raw(_eager(s)))))
            elif source.dtype in _RAW:
                add(new_name, Expected(_RAW[source.dtype], shape, _once(lambda g=gen: _raw(_eager(g)))))
            else:
                raise ValueError(f"{name}: no rule for {source.dtype}")

    if isinstance(model, DeepseekV41Vision):
        return out

    n_expert = model.hparams["dspark_n_routed_experts" if isinstance(model, DeepseekV41DSpark) else "n_routed_experts"]
    first = model.block_count - model._n_stage if isinstance(model, DeepseekV41DSpark) else 0
    for bid in range(first, model.block_count):
        for proj, key in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP), ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                          ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
            names = [f"layers.{bid}.ffn.experts.{e}.{proj}" for e in range(n_expert)]
            rows, packed_cols = model.model_tensors[names[0] + ".weight"]().shape

            def experts(names=names) -> Iterator[np.ndarray]:
                for n in names:
                    w = _eager(model.model_tensors[n + ".weight"]).view(torch.uint8).numpy()
                    s = _eager(model.model_tensors[n + ".scale"]).view(torch.uint8).numpy()
                    yield mxfp4_blocks(w, s)
            add(model.format_tensor_name(key, bid), Expected(T.MXFP4, (n_expert, rows, packed_cols * 2), experts))

    if isinstance(model, DeepseekV41):
        source, kept = model._engram_layers()
        if kept:
            tables = [(f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale") for layer in source]
            parts = [(wn, sn, model._tensor_part(f"layers.{bid}.engram.embed.weight"),
                      model._tensor_part(f"layers.{bid}.engram.embed.scale")) for (wn, sn), bid in zip(tables, kept)]

            def rows(column: int) -> Callable[[], Iterator[np.ndarray]]:
                def read() -> Iterator[np.ndarray]:
                    from safetensors import safe_open
                    for wn, sn, wpart, spart in parts:
                        tensor, part = (wn, wpart) if column == 0 else (sn, spart)
                        with safe_open(part, framework="pt") as f:
                            view = f.get_slice(tensor)
                            n, width = view.get_shape()
                            step = max(1, _CHUNK // width)
                            for r0 in range(0, n, step):
                                yield _raw(view[r0:r0 + step])
                return read

            codes = [model.model_tensors[f"layers.{bid}.engram.embed.weight"]().shape for bid in kept]
            scales = [model.model_tensors[f"layers.{bid}.engram.embed.scale"]().shape for bid in kept]
            name = model.format_tensor_name(gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD)
            add(name, Expected(T.F8_E4M3, (sum(c[0] for c in codes), codes[0][1]), rows(0)))
            add(scale_name(name), Expected(T.I8, (sum(c[0] for c in scales), scales[0][1]), rows(1)))
    return out


def shards(first: Path) -> list[Path]:
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if m is None:
        return [first]
    return [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf") for i in range(1, int(m.group(2)) + 1)]


def _compare(name: str, e: Expected, data: np.ndarray) -> None:
    offset = 0
    for chunk in e.chunks():
        chunk = np.ascontiguousarray(chunk, dtype=np.uint8).reshape(-1)
        got = data[offset:offset + chunk.size]
        if got.size != chunk.size or not np.array_equal(got, chunk):
            first = offset + (int(np.argmax(got != chunk[:got.size])) if got.size else 0)
            raise ValueError(f"{name}: bytes differ from the checkpoint at byte {first}")
        offset += chunk.size
    if offset != data.size:
        raise ValueError(f"{name}: {data.size} bytes in the file, {offset} expected")


def check(model: DeepseekV41 | DeepseekV41DSpark | DeepseekV41Vision, first_shard: Path) -> dict[str, int]:
    """Raises on the first difference; returns the bytes checked per type."""
    expected = expected_tensors(model)
    readers = [gguf.GGUFReader(path) for path in shards(first_shard)]
    head = readers[0]

    def field(key: str):
        f = head.get_field(key)
        return None if f is None else f.contents()

    if field(gguf.Keys.General.SOURCE_REPO_URL) != model._source_repo_url:
        raise ValueError(f"general.source.repo_url is {field(gguf.Keys.General.SOURCE_REPO_URL)!r}")
    for path, reader in zip(shards(first_shard), readers):
        if (f := reader.get_field(gguf.Keys.General.ALIGNMENT)) is None or f.contents() != ALIGNMENT:
            raise ValueError(f"{path.name}: general.alignment is {f and f.contents()}, expected {ALIGNMENT}")
    if not isinstance(model, DeepseekV41Vision):
        if field(SCALE_ENCODING) != UE8M0:
            raise ValueError(f"{SCALE_ENCODING} is {field(SCALE_ENCODING)!r}")
        if field(gguf.Keys.General.FILE_TYPE) != gguf.LlamaFileType.MOSTLY_MXFP4_MOE:
            raise ValueError(f"general.file_type is {field(gguf.Keys.General.FILE_TYPE)}")

    found: dict[str, tuple[gguf.GGUFReader, gguf.ReaderTensor]] = {}
    for reader in readers:
        for t in reader.tensors:
            if t.name in found:
                raise ValueError(f"{t.name} appears twice")
            found[t.name] = (reader, t)
    if missing := sorted(set(expected) - set(found)):
        raise ValueError(f"{len(missing)} tensors missing, e.g. {missing[:5]}")
    if extra := sorted(set(found) - set(expected)):
        raise ValueError(f"{len(extra)} unexpected tensors, e.g. {extra[:5]}")

    checked: dict[str, int] = {}
    for i, (name, e) in enumerate(expected.items()):
        reader, t = found[name]
        shape = tuple(int(n) for n in reversed(t.shape))
        if t.tensor_type != e.qtype or shape != e.shape:
            raise ValueError(f"{name}: {t.tensor_type.name} {shape} in the file, {e.qtype.name} {e.shape} expected")
        if t.data_offset % ALIGNMENT:
            raise ValueError(f"{name}: data at {t.data_offset}, not {ALIGNMENT}-aligned")
        _compare(name, e, np.asarray(reader.data[t.data_offset:t.data_offset + t.n_bytes]))
        checked[e.qtype.name] = checked.get(e.qtype.name, 0) + t.n_bytes
        if i % 100 == 0:
            logger.info("checked %d/%d tensors", i, len(expected))
    return checked
