"""Byte-exact check of a converted file against the checkpoint it came from.

Tensor names come from upstream's mapping, the one the converter writes by. Every tensor's bytes are rebuilt from the
safetensors by this module's rules, independently of the converter's data path:

- F8_E4M3 codes, their scales (I8 UE8M0 bytes or F32 values, each grid at its own type), BF16 and F32 tensors equal
  the checkpoint's bytes, at the checkpoint's type; a source upstream splits into several tensors is compared run of
  rows by run of rows;
- stacked FP8 experts (GLM) equal their experts' codes and scales, concatenated in expert order;
- GLM's `kv_b` equals `k_b` (each head's key rows, transposed) and `v_b` (each head's value rows), an MXFP8 one's `k_b`
  MXFP8 dequantized exactly to BF16; its `A_log` is stored as -exp(A_log) in F32;
- the Engram gate equals the F32 product of `q_weight` and `k_weight`;
- MXFP4 experts equal the ggml `block_mxfp4` layout built from the checkpoint's codes and UE8M0 scales;
- NVFP4 weights (stacked when experts) equal the ggml `block_nvfp4` layout built from the checkpoint's codes and E4M3
  block scales (compressed ones decoded by `csf`, whose oracle is vLLM's decoder), their `.scale` and `.input_scale`
  the checkpoint's `weight_scale_2` and `input_scale` values, one per matrix;
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

from .exact import negative_exp_f32
from .deepseek41 import DeepseekV41, DeepseekV41DSpark, DeepseekV41Vision
from .glm5next import Glm5Next, Glm5NextVision
from .native import ALIGNMENT, Native, scale_name

logger = logging.getLogger(__name__)

T = gguf.GGMLQuantizationType
_CHUNK = 1 << 28
_RAW = {torch.bfloat16: T.BF16, torch.float32: T.F32}  # checkpoint types stored as they are
Model = DeepseekV41 | DeepseekV41DSpark | DeepseekV41Vision | Glm5Next | Glm5NextVision


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


def nvfp4_blocks(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """ggml block_nvfp4 bytes of one matrix: per 64 values, their four E4M3 block scales, then per 16 values 8 bytes
    holding value j in the low nibble and value j + 8 in the high one. Source codes are packed two per byte, value 2i
    low; source scales are one per 16 values."""
    rows = packed.shape[0]
    codes = np.empty((rows, packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    codes = codes.reshape(rows, -1, 16)
    qs = (codes[:, :, :8] | (codes[:, :, 8:] << 4)).reshape(rows, -1, 32)
    return np.concatenate((scales.reshape(rows, -1, 4), qs), axis=2).reshape(-1)


def _nvfp4(model: Native, names: list[str], name: str, add: Callable[[str, Expected], None]) -> None:
    """The NVFP4 weights `names` written as `name`, stacked when several, and their scales."""
    weights = [model._nvfp4[n] for n in names]
    stack = (len(names),) if len(names) > 1 else ()
    rows, cols = weights[0].shape
    if any(w.shape != (rows, cols) for w in weights):
        raise ValueError(f"{name}: its NVFP4 weights differ in shape")
    add(name, Expected(T.NVFP4, (*stack, rows, cols),
                       lambda: (nvfp4_blocks(_eager(w.codes).numpy(), w.block_scales()) for w in weights)))
    stem = name.removesuffix(".weight")
    add(stem + ".scale", Expected(T.F32, (len(names),), lambda: (_raw(w.scale_2()) for w in weights)))
    if (input_scales := [w.input_scale for w in weights if w.input_scale is not None]):
        add(stem + ".input_scale", Expected(T.F32, (len(names),), lambda: (_raw(s()) for s in input_scales)))


def _kv_b(model: Glm5Next, name: str, gen: Callable[[], torch.Tensor], names: list[str],
          add: Callable[[str, Expected], None]) -> None:
    """`kv_b` holds, per head, its key rows then its value rows: `k_b` is each head's key rows transposed
    [head, kv_lora_rank, key dim], `v_b` each head's value rows [head, value dim, kv_lora_rank]. An MXFP8 `kv_b`'s
    `v_b` is its codes' and grid's value rows, and its `k_b` is MXFP8 dequantized exactly to BF16: each code times
    2^(b - 127), which must be a BF16 value."""
    hp = model.hparams
    heads, nope, v = hp["num_attention_heads"], hp["qk_nope_head_dim"], hp["v_head_dim"]
    source = gen()
    if source.dtype == torch.float8_e4m3fn and len(names) == 2 and source.shape[0] == heads * (nope + v):
        fp8 = model._fp8[name]
        rank = source.shape[1]
        if fp8.grid.block != (1, 32) or fp8.grid.encoding.qtype != T.I8:
            raise ValueError(f"{name}: no rule for an FP8 kv_b under a {fp8.grid.block} grid")

        def codes() -> torch.Tensor:
            return _eager(gen).reshape(heads, nope + v, rank)

        def grid() -> torch.Tensor:
            return _eager(fp8.scale).view(torch.uint8).reshape(heads, nope + v, rank // 32)

        def k_b() -> np.ndarray:
            exponents = grid()[:, :nope].to(torch.float64).repeat_interleave(32, -1) - 127
            exact = codes()[:, :nope].to(torch.float64) * torch.pow(2.0, exponents)
            values = exact.to(torch.bfloat16)
            if not torch.equal(values.double(), exact):
                raise ValueError(f"{names[0]}: a product of {name} is not a BF16 value")
            return _raw(values.transpose(1, 2))
        add(names[0], Expected(T.BF16, (heads, rank, nope), _once(k_b)))
        add(names[1], Expected(T.F8_E4M3, (heads, v, rank), _once(lambda: _raw(codes()[:, nope:]))))
        add(scale_name(names[1]), Expected(T.I8, (heads, v, rank // 32), _once(lambda: _raw(grid()[:, nope:]))))
        return
    if source.dtype not in _RAW or len(names) != 2 or source.shape[0] != heads * (nope + v):
        raise ValueError(f"{name}: no rule for {source.dtype} {tuple(source.shape)} written as {names}")
    rank = source.shape[1]
    qtype = _RAW[source.dtype]

    def per_head() -> torch.Tensor:
        return _eager(gen).reshape(heads, nope + v, rank)
    add(names[0], Expected(qtype, (heads, rank, nope), _once(lambda: _raw(per_head()[:, :nope].transpose(1, 2)))))
    add(names[1], Expected(qtype, (heads, v, rank), _once(lambda: _raw(per_head()[:, nope:]))))


def _stacked_experts(model: Glm5Next, add: Callable[[str, Expected], None]) -> None:
    """Each MoE layer's routed experts, one tensor per projection: the experts' codes in expert order, and beside it
    their scale grids, or their NVFP4 scales, in the same order."""
    n_expert = model.hparams["n_routed_experts"]
    for bid in range(model.block_count):
        if f"model.layers.{bid}.mlp.experts.0.down_proj.weight" not in model.model_tensors:
            continue
        for proj in ("down_proj", "gate_proj", "up_proj"):
            names = [f"model.layers.{bid}.mlp.experts.{e}.{proj}.weight" for e in range(n_expert)]
            name = model.map_tensor_name(f"model.layers.{bid}.mlp.experts.{proj}.weight")
            if all(n in model._nvfp4 for n in names):
                _nvfp4(model, names, name, add)
                continue
            weights = [model.model_tensors[n] for n in names]
            if any(w().dtype != torch.float8_e4m3fn for w in weights):
                raise ValueError(f"layer {bid} {proj}: an expert is neither E4M3 FP8 nor NVFP4")
            fp8 = [model._fp8[n] for n in names]
            add(name, Expected(T.F8_E4M3, (n_expert, *weights[0]().shape),
                               lambda ws=weights: (_raw(_eager(w)) for w in ws)))
            add(scale_name(name), Expected(fp8[0].grid.encoding.qtype, (n_expert, *fp8[0].shape),
                                           lambda fs=fp8: (_raw(_eager(f.scale)) for f in fs)))


def _glm5next_tensor(model: Glm5Next, name: str, gen: Callable[[], torch.Tensor], source: torch.Tensor,
                     outputs: list, add: Callable[[str, Expected], None]) -> bool:
    """GLM's own tensor rules: the kv_b projection split, and `A_log` stored as the model's `-exp(A_log)`. Whether
    `name` was handled."""
    if name.endswith(".kv_b_proj.weight"):
        _kv_b(model, name, gen, [new_name for new_name, _ in outputs], add)
        return True
    if name.endswith(".A_log"):
        if source.dtype != torch.float32:
            raise ValueError(f"{name}: {source.dtype}, the model computes exp(A_log) in F32 from F32")
        # every stored value is the model's -exp(A_log) of one source value
        for new_name, _ in outputs:
            add(new_name, Expected(T.F32, (source.numel(),), _once(lambda g=gen: _raw(torch.from_numpy(negative_exp_f32(_eager(g).numpy()))))))
        return True
    return False


def _engram_gate(model: Model, name: str, gen: Callable[[], torch.Tensor], source: torch.Tensor,
                 outputs: list, add: Callable[[str, Expected], None]) -> bool:
    """DeepSeek V4.1's own tensor rule: the Engram gate, stored as the F32 product of `q_weight` and `k_weight`.
    Whether `name` was handled."""
    if len(outputs) != 1 or not outputs[0][0].endswith(".engram_gate.weight"):
        return False
    (new_name, data), = outputs
    k = model.model_tensors[name.removesuffix("q_weight") + "k_weight"]
    add(new_name, Expected(T.F32, tuple(data.shape), _once(
        lambda g=gen, k=k: _raw(_eager(g).float() * _eager(k).float()))))
    return True


def _mxfp4_experts(model: Model, layers: range, n_expert: int, add: Callable[[str, Expected], None]) -> None:
    """Each of `layers`' routed experts, one MXFP4 tensor per projection, in expert order."""
    for bid in layers:
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


def _deepseek41_written_whole(model: DeepseekV41, add: Callable[[str, Expected], None]) -> None:
    """Every layer's MXFP4 experts, then the Engram table: the kept layers' code and scale rows, concatenated."""
    _mxfp4_experts(model, range(model.block_count), model.hparams["n_routed_experts"], add)
    source, kept = model._engram_layers()
    if not kept:
        return
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


def _dspark_written_whole(model: DeepseekV41DSpark, add: Callable[[str, Expected], None]) -> None:
    """The MXFP4 experts of its stage layers, the last `_n_stage`."""
    _mxfp4_experts(model, range(model.block_count - model._n_stage, model.block_count),
                   model.hparams["dspark_n_routed_experts"], add)


@dataclass(frozen=True)
class WrittenWhole:
    """Source tensors a family writes whole rather than tensor by tensor (experts and Engram tables), checked on
    their own: which they are, and what they are written as (`expected(model, add)`)."""
    sources: re.Pattern[str]
    expected: Callable[..., None]


@dataclass(frozen=True)
class Rules:
    """A family's own rules for its file, beside the rules every tensor follows."""
    file_type: gguf.LlamaFileType | None = None  # the file's `general.file_type`, where the family states one
    whole: WrittenWhole | None = None
    # its own rule for one source tensor (`(model, name, gen, source, outputs, add)`): whether it handled it
    tensor: Callable[..., bool] | None = None


_DEEPSEEK41_WHOLE = re.compile(r"layers\.\d+\.(ffn\.experts\.\d+\.w[123]|engram\.embed)\.(weight|scale)$")
RULES: dict[type, Rules] = {
    DeepseekV41: Rules(gguf.LlamaFileType.MOSTLY_MXFP4_MOE, WrittenWhole(_DEEPSEEK41_WHOLE, _deepseek41_written_whole),
                       _engram_gate),
    DeepseekV41DSpark: Rules(gguf.LlamaFileType.MOSTLY_MXFP4_MOE, WrittenWhole(_DEEPSEEK41_WHOLE, _dspark_written_whole),
                             _engram_gate),
    DeepseekV41Vision: Rules(),
    Glm5Next: Rules(gguf.LlamaFileType.MOSTLY_F8_E4M3,
                    WrittenWhole(re.compile(r"model\.layers\.\d+\.mlp\.experts\.\d+\."), _stacked_experts),
                    _glm5next_tensor),
    Glm5NextVision: Rules(),
}


def expected_tensors(model: Model) -> dict[str, Expected]:
    out: dict[str, Expected] = {}

    def add(name: str, e: Expected) -> None:
        if name in out:
            raise ValueError(f"two sources for {name}")
        out[name] = e

    rules = RULES[type(model)]
    # upstream's modify_tensors: names only, without the writes this package's classes make inside it
    upstream = next(c for c in type(model).__mro__ if c.__module__.startswith("conversion."))
    if isinstance(model, Native):
        model.dequant_model()  # the converter's bookkeeping only: which weights are quantized, and their scales
    for name, gen in model.model_tensors.items():
        if rules.whole is not None and rules.whole.sources.match(name):
            continue
        source = gen()
        bid = next((int(p) for p in name.split(".") if p.isdecimal()), None)
        outputs = list(upstream.modify_tensors(model, source, name, bid))
        if rules.tensor is not None and rules.tensor(model, name, gen, source, outputs, add):
            continue
        if len(outputs) > 1:
            # one source written as several tensors, at the source's type: each the next run of its rows (the vision
            # MLP's w1 as gate and up), or each one index of one axis (a temporal patch embedding as one per frame)
            shapes = [tuple(data.shape) for _, data in outputs]
            full = tuple(source.shape)
            axes = [a for a in range(len(full)) if len(outputs) == full[a]
                    and all(s == full[:a] + full[a + 1:] for s in shapes)]
            if source.dtype not in _RAW:
                raise ValueError(f"{name}: no rule for {source.dtype} {full} written as {shapes}")
            qtype = _RAW[source.dtype]
            if sum(s[0] for s in shapes) == full[0] and all(s[1:] == full[1:] for s in shapes):
                r0 = 0
                for (new_name, _), shape in zip(outputs, shapes):
                    add(new_name, Expected(qtype, shape, _once(
                        lambda g=gen, r0=r0, n=shape[0]: _raw(_eager(g)[r0:r0 + n]))))
                    r0 += shape[0]
            elif len(axes) == 1:
                for k, ((new_name, _), shape) in enumerate(zip(outputs, shapes)):
                    add(new_name, Expected(qtype, shape, _once(
                        lambda g=gen, a=axes[0], k=k: _raw(_eager(g).select(a, k)))))
            else:
                raise ValueError(f"{name}: no rule for {source.dtype} {full} written as {shapes}")
            continue
        for new_name, data in outputs:
            shape = tuple(data.shape)
            if isinstance(model, Native) and name in model._nvfp4:
                _nvfp4(model, [name], new_name, add)
            elif source.dtype == torch.float8_e4m3fn:
                assert isinstance(model, Native)
                add(new_name, Expected(T.F8_E4M3, shape, _once(lambda g=gen: _raw(_eager(g)))))
                fp8 = model._fp8[name]
                add(scale_name(new_name), Expected(fp8.grid.encoding.qtype, fp8.shape,
                                                   _once(lambda s=fp8.scale: _raw(_eager(s)))))
            elif source.dtype in _RAW:
                add(new_name, Expected(_RAW[source.dtype], shape, _once(lambda g=gen: _raw(_eager(g)))))
            else:
                raise ValueError(f"{name}: no rule for {source.dtype}")

    if rules.whole is not None:
        rules.whole.expected(model, add)
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


def shard_facts(first_shard: Path) -> list[gguf.GGUFReader]:
    """Every shard of the file reads on its own: it carries `general.alignment`, its split keys number it within the
    file, and its tensor data sits at the alignment. Raises naming the shard and the key; returns the readers."""
    paths = shards(first_shard)
    readers = [gguf.GGUFReader(path) for path in paths]
    total = sum(len(reader.tensors) for reader in readers)
    for i, (path, reader) in enumerate(zip(paths, readers)):
        if (f := reader.get_field(gguf.Keys.General.ALIGNMENT)) is None or f.contents() != ALIGNMENT:
            raise ValueError(f"{path.name}: general.alignment is {f and f.contents()}, expected {ALIGNMENT}")
        if len(paths) > 1:
            for key, value in ((gguf.Keys.Split.LLM_KV_SPLIT_NO, i), (gguf.Keys.Split.LLM_KV_SPLIT_COUNT, len(paths)),
                               (gguf.Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT, total)):
                if (f := reader.get_field(key)) is None or f.contents() != value:
                    raise ValueError(f"{path.name}: {key} is {f and f.contents()}, expected {value}")
        for t in reader.tensors:
            if t.data_offset % ALIGNMENT:
                raise ValueError(f"{path.name}: {t.name} data at {t.data_offset}, not {ALIGNMENT}-aligned")
    return readers


def check(model: Model, first_shard: Path, only: frozenset[int] | None = None) -> dict[str, int]:
    """Raises on the first difference; returns the bytes checked per type. Every shard's header and the file's tensor
    set are checked; with `only`, the bytes of those shards' tensors alone (numbered from 1), so one process per shard
    checks the whole file between them."""
    expected = expected_tensors(model)
    readers = shard_facts(first_shard)
    head = readers[0]

    def field(key: str):
        f = head.get_field(key)
        return None if f is None else f.contents()

    if field(gguf.Keys.General.SOURCE_REPO_URL) != model._source_repo_url:
        raise ValueError(f"general.source.repo_url is {field(gguf.Keys.General.SOURCE_REPO_URL)!r}")
    if (file_type := RULES[type(model)].file_type) is not None:
        # upstream's rule: a ModelOpt config with any NVFP4 layer makes the file NVFP4
        layers = model.hparams.get("quantization_config", {}).get("quantized_layers", {})
        if any(str(entry.get("quant_algo")).endswith("NVFP4") for entry in layers.values()):
            file_type = gguf.LlamaFileType.MOSTLY_NVFP4
        if field(gguf.Keys.General.FILE_TYPE) != file_type:
            raise ValueError(f"general.file_type is {field(gguf.Keys.General.FILE_TYPE)}, {file_type!r} expected")

    if only is not None and not only <= set(range(1, len(readers) + 1)):
        raise ValueError(f"shards {sorted(only)}: the file has {len(readers)}")
    found: dict[str, tuple[int, gguf.GGUFReader, gguf.ReaderTensor]] = {}
    for number, reader in enumerate(readers, 1):
        for t in reader.tensors:
            if t.name in found:
                raise ValueError(f"{t.name} appears twice")
            found[t.name] = (number, reader, t)
    if missing := sorted(set(expected) - set(found)):
        raise ValueError(f"{len(missing)} tensors missing, e.g. {missing[:5]}")
    if extra := sorted(set(found) - set(expected)):
        raise ValueError(f"{len(extra)} unexpected tensors, e.g. {extra[:5]}")

    checked: dict[str, int] = {}
    for i, (name, e) in enumerate(expected.items()):
        number, reader, t = found[name]
        shape = tuple(int(n) for n in reversed(t.shape))
        if t.tensor_type != e.qtype or shape != e.shape:
            raise ValueError(f"{name}: {t.tensor_type.name} {shape} in the file, {e.qtype.name} {e.shape} expected")
        if only is not None and number not in only:
            continue
        _compare(name, e, np.asarray(reader.data[t.data_offset:t.data_offset + t.n_bytes]))
        checked[e.qtype.name] = checked.get(e.qtype.name, 0) + t.n_bytes
        if i % 100 == 0:
            logger.info("checked %d/%d tensors", i, len(expected))
    return checked
