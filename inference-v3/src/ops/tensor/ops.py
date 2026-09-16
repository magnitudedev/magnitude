"""Composable tensor operations used by inference model functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .primitive import NumericalContract, primitive
from .tracing import Tensor, active_trace
from .types import DType, TensorSpec, broadcast_shape, normalize_axes
from ..formula import formula
from ..kv import KVRepresentation
from ..performance import semantics as useful


def _np():
    import numpy as np

    return np


def _one(inputs: tuple[TensorSpec, ...], count: int, name: str) -> None:
    if len(inputs) != count:
        raise ValueError(f"{name} expects {count} inputs, got {len(inputs)}")


def _emit(name: str, *inputs: Tensor, **attributes: Any) -> Any:
    trace = active_trace()
    outputs = trace.emit(name, tuple(inputs), attributes)
    return outputs[0] if len(outputs) == 1 else outputs


def _tensor(value: Tensor | int | float | bool, like: Tensor | None = None) -> Tensor:
    if isinstance(value, Tensor):
        return value
    dtype = like.dtype if like is not None and like.dtype.floating else None
    return scalar(value, dtype=dtype)


@primitive(
    "scalar",
    work=useful.no_arithmetic,
    reference=lambda _inputs, attrs: (_np().asarray(attrs["value"]),),
    tags=frozenset({"cheap"}),
)
def _scalar(_inputs, attrs):
    dtype = attrs["dtype"]
    if not isinstance(dtype, DType):
        raise TypeError("scalar dtype must be a DType")
    return (TensorSpec((), dtype),)


def scalar(value: int | float | bool, *, dtype: DType | None = None) -> Tensor:
    if dtype is None:
        dtype = (
            DType.BOOL
            if isinstance(value, bool)
            else DType.I32
            if isinstance(value, int)
            else DType.F32
        )
    return _emit("scalar", value=value, dtype=dtype)


def _binary_abstract(inputs, _attrs):
    _one(inputs, 2, "binary primitive")
    left, right = inputs
    if left.dtype != right.dtype:
        raise ValueError("binary tensor dtypes must agree explicitly")
    return (TensorSpec(broadcast_shape(left.shape, right.shape), left.dtype),)


def _binary_ref(operator):
    return lambda inputs, _attrs: (operator(inputs[0], inputs[1]),)


primitive("add", reference=_binary_ref(lambda a, b: a + b), tags=frozenset({"cheap"}), work=useful.elementwise(floating=1))(
    _binary_abstract
)
primitive("subtract", reference=_binary_ref(lambda a, b: a - b), tags=frozenset({"cheap"}), work=useful.elementwise(floating=1))(
    _binary_abstract
)
primitive("multiply", reference=_binary_ref(lambda a, b: a * b), tags=frozenset({"cheap"}), work=useful.elementwise(floating=1))(
    _binary_abstract
)
primitive("divide", reference=_binary_ref(lambda a, b: a / b), tags=frozenset({"cheap"}), work=useful.elementwise(floating=1))(
    _binary_abstract
)


@primitive("less", reference=_binary_ref(lambda a, b: a < b), tags=frozenset({"cheap"}),
           work=useful.elementwise(comparisons=1))
def _less(inputs, attrs):
    result, = _binary_abstract(inputs, attrs)
    return (TensorSpec(result.shape, DType.BOOL),)


def less(left: Tensor, right: Tensor) -> Tensor:
    return _emit("less", left, right)


def add(left, right) -> Tensor:
    left = _tensor(left, right if isinstance(right, Tensor) else None)
    return _emit("add", left, _tensor(right, left))


def subtract(left, right) -> Tensor:
    left = _tensor(left, right if isinstance(right, Tensor) else None)
    return _emit("subtract", left, _tensor(right, left))


def multiply(left, right) -> Tensor:
    left = _tensor(left, right if isinstance(right, Tensor) else None)
    return _emit("multiply", left, _tensor(right, left))


def divide(left, right) -> Tensor:
    left = _tensor(left, right if isinstance(right, Tensor) else None)
    return _emit("divide", left, _tensor(right, left))


@primitive(
    "cast",
    work=useful.no_arithmetic,
    reference=lambda inputs, attrs: (inputs[0],),
    tags=frozenset({"cheap"}),
)
def _cast(inputs, attrs):
    _one(inputs, 1, "cast")
    return (TensorSpec(inputs[0].shape, attrs["dtype"], inputs[0].layout),)


def cast(value: Tensor, dtype: DType) -> Tensor:
    return _emit("cast", value, dtype=dtype)


def _decode_bfloat16_reference(inputs, attrs):
    np = _np()
    bits = inputs[0].astype(np.uint32) << 16
    return (bits.view(np.float32),)


@primitive(
    "decode_bfloat16",
    work=useful.elementwise(integer=1),
    reference=_decode_bfloat16_reference,
    tags=frozenset({"cheap", "representation"}),
)
def _decode_bfloat16(inputs, attrs):
    _one(inputs, 1, "decode_bfloat16")
    dtype = attrs["dtype"]
    if inputs[0].dtype != DType.U16 or dtype not in (DType.F16, DType.F32):
        raise ValueError("bfloat16 decoding requires uint16 storage and an F16/F32 result")
    return (TensorSpec(inputs[0].shape, dtype, inputs[0].layout),)


def decode_bfloat16(value: Tensor, dtype: DType) -> Tensor:
    return _emit("decode_bfloat16", value, dtype=dtype)


def _unpack_words_reference(inputs, attrs):
    np = _np()
    words = np.asarray(inputs[0], dtype=np.uint32)
    offset, outputs = 0, []
    for shape, dtype in attrs['fields']:
        count = math.prod(shape)
        outputs.append(words[offset:offset + count].view(dtype.value).reshape(shape))
        offset += count
    return tuple(outputs)


@primitive('unpack_words', work=useful.no_arithmetic, reference=_unpack_words_reference)
def _unpack_words(inputs, attrs):
    _one(inputs, 1, 'unpack_words')
    source = inputs[0]
    if (source.rank != 1 or source.dtype != DType.U32 or source.representation is not None
            or source.layout.strides is not None):
        raise ValueError('word records require a dense one-dimensional uint32 input')
    outputs = tuple(TensorSpec(tuple(shape), dtype) for shape, dtype in attrs['fields'])
    if not outputs or any(not spec.static or spec.dtype not in (DType.I32, DType.U32) for spec in outputs):
        raise ValueError('word record fields must be concrete 32-bit integer tensors')
    if source.shape[0] != sum(spec.elements for spec in outputs):
        raise ValueError('word record fields must cover the input exactly')
    return outputs


def unpack_words(value: Tensor, specs: Sequence[TensorSpec]) -> tuple[Tensor, ...]:
    """Materialize typed integer fields from one contiguous word record.

    Signed fields reinterpret bits, preserving negative indices and all unsigned
    random-draw bits. Field storage is independent of the record's input lease.
    """
    if any(spec.representation is not None or spec.layout.strides is not None for spec in specs):
        raise ValueError('word record fields require dense logical storage')
    result = _emit('unpack_words', value, fields=tuple((spec.shape, spec.dtype) for spec in specs))
    return (result,) if len(specs) == 1 else result


@primitive(
    "reshape",
    work=useful.no_arithmetic,
    aliases=((0, 0),),
    reference=lambda inputs, attrs: (_np().reshape(inputs[0], attrs["shape"]),),
    tags=frozenset({"view"}),
)
def _reshape(inputs, attrs):
    _one(inputs, 1, "reshape")
    shape = tuple(attrs["shape"])
    if any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("reshape needs positive static extents")
    source = inputs[0]
    if source.static and math.prod(shape) != source.elements:
        raise ValueError("reshape changes element count")
    return (TensorSpec(shape, source.dtype, representation=source.representation),)


def reshape(value: Tensor, shape: Sequence[int]) -> Tensor:
    return _emit("reshape", value, shape=tuple(shape))


@primitive(
    "transpose",
    work=useful.no_arithmetic,
    reference=lambda inputs, attrs: (_np().transpose(inputs[0], attrs["axes"]),),
    tags=frozenset({"view"}),
)
def _transpose(inputs, attrs):
    _one(inputs, 1, "transpose")
    axes = tuple(attrs["axes"])
    if sorted(axes) != list(range(inputs[0].rank)):
        raise ValueError("transpose axes must be a permutation")
    source = inputs[0]
    return (TensorSpec(tuple(source.shape[axis] for axis in axes), source.dtype),)


def transpose(value: Tensor, axes: Sequence[int]) -> Tensor:
    return _emit("transpose", value, axes=tuple(axes))


@primitive(
    "concatenate", reference=lambda inputs, attrs: (_np().concatenate(inputs, axis=attrs["axis"]),)
    , work=useful.no_arithmetic
)
def _concatenate(inputs, attrs):
    if not inputs:
        raise ValueError("concatenate needs inputs")
    axis = normalize_axes(inputs[0].rank, attrs["axis"])[0]
    first = inputs[0]
    if any(item.rank != first.rank or item.dtype != first.dtype for item in inputs):
        raise ValueError("concatenated tensors must have equal rank and dtype")
    shape = list(first.shape)
    if any(type(item.shape[axis]) is not int for item in inputs):
        raise ValueError("concatenated axis must be static")
    for item in inputs[1:]:
        if any(item.shape[i] != first.shape[i] for i in range(first.rank) if i != axis):
            raise ValueError("concatenated non-axis dimensions must agree")
    shape[axis] = sum(item.shape[axis] for item in inputs)  # type: ignore[misc]
    return (TensorSpec(tuple(shape), first.dtype),)


def concatenate(values: Sequence[Tensor], axis: int = 0) -> Tensor:
    return _emit("concatenate", *tuple(values), axis=axis)


@primitive(
    "take_rows",
    work=useful.no_arithmetic,
    reference=lambda inputs, _attrs: (inputs[0][inputs[1]],),
    tags=frozenset({"indexing"}),
)
def _take_rows(inputs, _attrs):
    _one(inputs, 2, "take_rows")
    value, indices = inputs
    if value.rank < 1 or indices.rank != 1 or not indices.dtype.integer:
        raise ValueError("take_rows expects a tensor and one-dimensional integer indices")
    return (TensorSpec((indices.shape[0], *value.shape[1:]), value.dtype),)


def take_rows(value: Tensor, indices: Tensor) -> Tensor:
    return _emit("take_rows", value, indices)


@primitive(
    "overlay_rows",
    work=useful.no_arithmetic,
    reference=lambda inputs, _attrs: (_overlay_rows_reference(inputs),),
    tags=frozenset({"indexing"}),
)
def _overlay_rows(inputs, _attrs):
    _one(inputs, 3, "overlay_rows")
    value, replacement, indices = inputs
    if (
        value.rank < 1
        or replacement.shape != (indices.shape[0], *value.shape[1:])
        or replacement.dtype != value.dtype
        or indices.rank != 1
        or not indices.dtype.integer
    ):
        raise ValueError("overlay_rows replacement geometry differs from its destination")
    return (value,)


def _overlay_rows_reference(inputs):
    result = inputs[0].copy()
    result[inputs[2]] = inputs[1]
    return result


def overlay_rows(value: Tensor, replacement: Tensor, indices: Tensor) -> Tensor:
    return _emit("overlay_rows", value, replacement, indices)


@primitive(
    "quantized_import",
    work=useful.no_arithmetic,
    resource_writes=(1,),
    aliases=((0, 1),),
    tags=frozenset({"residency"}),
)
def _quantized_import(inputs, attrs):
    _one(inputs, 3, "quantized_import")
    source, target, extent = inputs
    if (
        source.rank != 1
        or source.dtype != DType.U8
        or target.representation is None
        or extent != TensorSpec((2,), DType.I32)
        or attrs["staged_tiles"] <= 0
    ):
        raise ValueError("invalid quantized residency import geometry")
    return (target,)


def quantized_import(
    source: Tensor,
    target: Tensor,
    extent: Tensor,
    *,
    codec: object,
    staged_tiles: int,
) -> Tensor:
    """Relayout format bytes into a canonical encoded resource."""

    return _emit(
        "quantized_import",
        source,
        target,
        extent,
        codec=codec,
        staged_tiles=staged_tiles,
    )


def _byte_copy_reference(inputs, attrs):
    source, destination, extent = inputs
    offset, count = map(int, extent)
    result = destination.copy()
    result[offset:offset + count] = source[:count]
    return (result,)


@primitive("byte_copy", reference=_byte_copy_reference, work=useful.no_arithmetic,
           resource_writes=(1,), aliases=((0, 1),), tags=frozenset({"residency"}))
def _byte_copy(inputs, attrs):
    _one(inputs, 3, "byte_copy")
    source, destination, extent = inputs
    if (source.rank != 1 or source.dtype != DType.U8 or destination.rank != 1
            or destination.dtype != DType.U8 or extent != TensorSpec((2,), DType.I64)):
        raise ValueError("byte_copy requires byte buffers and a 64-bit offset/count pair")
    return (destination,)


@formula(id="byte_copy", version=1)
def byte_copy(source: Tensor, destination: Tensor, extent: Tensor) -> Tensor:
    return _emit("byte_copy", source, destination, extent)


@primitive(
    "sample",
    work=useful.sampling,
    reference=lambda inputs, _attrs: (_sample_reference(inputs[0], inputs[1]),),
    tags=frozenset({"sampling", "host-output"}),
    host_observation=True,
)
def _sample(inputs, _attrs):
    _one(inputs, 2, "sample")
    logits, draws = inputs
    if (
        logits.rank != 2
        or logits.dtype != DType.F32
        or draws != TensorSpec((logits.shape[0], 6), DType.U32)
    ):
        raise ValueError("sampling expects FP32 logits and six uint32 draw words per row")
    return (TensorSpec((logits.shape[0], 2), DType.I32),)


@formula(id="sample", version=1)
def sample(logits: Tensor, draws: Tensor) -> Tensor:
    """Select token/status rows using position-addressed deterministic draws."""

    return _emit("sample", logits, draws)


def _constrained_sample_reference(inputs, _attrs):
    logits, draws, masks, mask_rows = inputs
    np = _np()
    masked = logits.copy()
    for row, mask_row in enumerate(mask_rows):
        if mask_row < -1 or mask_row >= len(masks):
            masked[row] = np.nan
            continue
        if mask_row >= 0:
            allowed = np.asarray([
                bool(int(masks[mask_row, token // 32]) & (1 << (token % 32)))
                for token in range(logits.shape[1])
            ])
            masked[row, ~allowed] = -np.inf
    result = _sample_reference(masked, draws)
    # Masking cannot turn an invalid source distribution into a valid one.
    result[np.any(np.isnan(logits) | np.isposinf(logits), axis=1)] = (-1, 2)
    return (result,)


@primitive(
    "sample_constrained",
    work=useful.constrained_sampling,
    reference=_constrained_sample_reference,
    tags=frozenset({"sampling", "host-output"}),
    host_observation=True,
)
def _sample_constrained(inputs, _attrs):
    _one(inputs, 4, "sample_constrained")
    logits, draws, masks, mask_rows = inputs
    output = _sample((logits, draws), {})
    if (masks.rank != 2 or masks.dtype != DType.U32
            or masks.shape[1] != (logits.shape[1] + 31) // 32
            or mask_rows != TensorSpec((logits.shape[0],), DType.I32)):
        raise ValueError("constrained sampling requires packed uint32 masks and signed row indices")
    return output


@formula(id="sample_constrained", version=1)
def sample_constrained(logits: Tensor, draws: Tensor, masks: Tensor, mask_rows: Tensor) -> Tensor:
    """Sample allowed logits; -1 mask rows preserve ordinary selection.

    Each mask bit denotes one vocabulary ID. Unset logits are negative infinity
    for selection. NaN/positive infinity in the original row still fail that row.
    """
    return _emit("sample_constrained", logits, draws, masks, mask_rows)


def _sample_reference(logits, draws):
    np = _np()
    output = np.empty((logits.shape[0], 2), dtype=np.int32)
    for row in range(logits.shape[0]):
        values = logits[row].astype(np.float32)
        invalid = bool(np.isnan(values).any() or np.isposinf(values).any())
        finite = ~np.isneginf(values)
        if invalid:
            output[row] = (-1, 2)
            continue
        if not finite.any():
            output[row] = (-1, 1)
            continue
        scores = values.copy()
        if int(draws[row, 0]) == 1:
            for token in np.flatnonzero(finite):
                word = _philox_reference(
                    int(token),
                    int(draws[row, 3]),
                    int(draws[row, 4]),
                    int(draws[row, 5]),
                    int(draws[row, 1]),
                    int(draws[row, 2]),
                )
                uniform = ((word >> 9) + 0.5) * (2**-23)
                scores[token] -= np.log(-np.log(uniform))
        scores[~finite] = -np.inf
        output[row] = (int(np.argmax(scores)), 0)
    return output


def _philox_reference(c0, c1, c2, c3, k0, k1):
    mask = 0xFFFFFFFF
    counter = [c0 & mask, c1 & mask, c2 & mask, c3 & mask]
    key = [k0 & mask, k1 & mask]
    for _ in range(10):
        product0 = 0xD2511F53 * counter[0]
        product1 = 0xCD9E8D57 * counter[2]
        hi0, lo0 = (product0 >> 32) & mask, product0 & mask
        hi1, lo1 = (product1 >> 32) & mask, product1 & mask
        counter = [
            (hi1 ^ counter[1] ^ key[0]) & mask,
            lo1,
            (hi0 ^ counter[3] ^ key[1]) & mask,
            lo0,
        ]
        key[0] = (key[0] + 0x9E3779B9) & mask
        key[1] = (key[1] + 0xBB67AE85) & mask
    return counter[0]


@primitive(
    "matmul",
    work=useful.contraction,
    reference=lambda inputs, attrs: (_matmul_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
)
def _matmul(inputs, attrs):
    _one(inputs, 2, "matmul")
    left, right = inputs
    if left.rank < 2 or right.rank < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError("invalid matrix product geometry")
    if left.dtype != right.dtype or not left.dtype.floating:
        raise ValueError("matrix inputs must have the same floating dtype")
    if attrs["accumulate"] != DType.F32:
        raise ValueError("matmul currently specifies FP32 accumulation")
    if not attrs["output"].floating:
        raise ValueError("matmul output must be floating point")
    batch = broadcast_shape(left.shape[:-2], right.shape[:-2])
    return (TensorSpec((*batch, left.shape[-2], right.shape[-1]), attrs["output"]),)


@formula(id="matmul", version=1)
def matmul(left: Tensor, right: Tensor, *, accumulate: DType = DType.F32,
           output: DType | None = None) -> Tensor:
    return _emit("matmul", left, right, accumulate=accumulate, output=output or left.dtype)


def _matmul_reference(inputs, attrs):
    np = _np()
    result = np.matmul(inputs[0].astype(np.float32), inputs[1].astype(np.float32))
    return result


def _unary(name, function, *, floating=0, special=0):
    @primitive(
        name,
        work=useful.elementwise(floating=floating, special=special),
        reference=lambda inputs, _attrs: (function(_np(), inputs[0]),),
        tags=frozenset({"cheap"}),
    )
    def abstract(inputs, _attrs):
        _one(inputs, 1, name)
        if not inputs[0].dtype.floating:
            raise ValueError(f"{name} requires floating input")
        return (inputs[0],)

    return formula(lambda value: _emit(name, value), id=name, version=1)


exp = _unary("exp", lambda np, x: np.exp(x), special=1)
sigmoid = _unary("sigmoid", lambda np, x: 1 / (1 + np.exp(-x)), floating=3, special=1)
silu = _unary("silu", lambda np, x: x / (1 + np.exp(-x)), floating=3, special=1)
tanh = _unary("tanh", lambda np, x: np.tanh(x), special=1)


@primitive(
    "softmax",
    work=useful.softmax,
    reference=lambda inputs, attrs: (_softmax_reference(inputs[0], attrs["axis"]),),
    numerical=NumericalContract(DType.F32),
)
def _softmax(inputs, attrs):
    _one(inputs, 1, "softmax")
    if not inputs[0].dtype.floating:
        raise ValueError("softmax requires floating input")
    normalize_axes(inputs[0].rank, attrs["axis"])
    return (inputs[0],)


def _softmax_reference(value, axis):
    np = _np()
    accumulated = value.astype(np.float32)
    shifted = accumulated - np.max(accumulated, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return (values / np.sum(values, axis=axis, keepdims=True)).astype(value.dtype)


@formula(id="softmax", version=1)
def softmax(value: Tensor, axis: int = -1) -> Tensor:
    return _emit("softmax", value, axis=axis)


@primitive(
    "rms_norm",
    work=useful.normalization,
    reference=lambda inputs, attrs: (_rms_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
)
def _rms_norm(inputs, attrs):
    if len(inputs) not in (1, 2):
        raise ValueError("rms_norm expects input and optional weight")
    value = inputs[0]
    if not value.dtype.floating or value.rank < 1:
        raise ValueError("rms_norm needs a floating tensor")
    if len(inputs) == 2 and (inputs[1].shape != (value.shape[-1],) or not inputs[1].dtype.floating):
        raise ValueError("rms_norm weight geometry differs from the last axis")
    if attrs["epsilon"] <= 0:
        raise ValueError("rms_norm epsilon must be positive")
    output_dtype = attrs.get("output_dtype") or value.dtype
    if not output_dtype.floating:
        raise ValueError("rms_norm output must be floating point")
    return (TensorSpec(value.shape, output_dtype, value.layout),)


def _rms_reference(inputs, attrs):
    np = _np()
    value = inputs[0]
    result = value / np.sqrt(
        np.mean(value.astype(np.float32) ** 2, axis=-1, keepdims=True) + attrs["epsilon"]
    )
    if len(inputs) == 2:
        result = result * inputs[1]
    return result


@formula(id="rms_norm", version=1)
def rms_norm(
    value: Tensor,
    weight: Tensor | None = None,
    *,
    epsilon: float = 1e-6,
    output_dtype: DType | None = None,
) -> Tensor:
    inputs = (value,) if weight is None else (value, weight)
    return _emit("rms_norm", *inputs, epsilon=epsilon, output_dtype=output_dtype)


@primitive(
    "linear",
    work=useful.contraction,
    reference=lambda inputs, attrs: (_linear_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"projection"}),
)
def _linear(inputs, attrs):
    if len(inputs) not in (2, 3):
        raise ValueError("linear expects input, weight and optional bias")
    value, weight = inputs[:2]
    if value.rank < 1 or weight.rank != 2 or value.shape[-1] != weight.shape[-1]:
        raise ValueError("linear input width differs from weight width")
    if len(inputs) == 3 and inputs[2].shape != (weight.shape[0],):
        raise ValueError("linear bias width differs from output width")
    output_dtype = attrs.get("output_dtype") or value.dtype
    return (TensorSpec((*value.shape[:-1], weight.shape[0]), output_dtype),)


def _linear_reference(inputs, attrs):
    np = _np()
    result = np.matmul(inputs[0].astype(np.float32), np.swapaxes(inputs[1].astype(np.float32), -1, -2))
    if len(inputs) == 3:
        result = result + inputs[2]
    return result


@formula(id="linear", version=1)
def linear(
    value: Tensor, weight: Tensor, bias: Tensor | None = None, *, output_dtype: DType | None = None
) -> Tensor:
    inputs = (value, weight) if bias is None else (value, weight, bias)
    return _emit("linear", *inputs, output_dtype=output_dtype or value.dtype)


@primitive(
    "row_dot",
    work=useful.contraction,
    reference=lambda inputs, attrs: (_row_dot_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"projection"}),
)
def _row_dot(inputs, attrs):
    _one(inputs, 2, "row_dot")
    value, weight = inputs
    if (
        value.rank != 2
        or weight.rank != 1
        or value.shape[-1] != weight.shape[0]
        or not value.dtype.floating
        or not weight.dtype.floating
    ):
        raise ValueError("row_dot expects floating rows and one matching vector")
    output_dtype = attrs.get("output_dtype") or value.dtype
    return (TensorSpec((*value.shape[:-1], 1), output_dtype),)


def _row_dot_reference(inputs, attrs):
    np = _np()
    output_dtype = attrs.get("output_dtype") or inputs[0].dtype
    result = np.sum(
        inputs[0].astype(np.float32) * inputs[1].astype(np.float32),
        axis=-1,
        keepdims=True,
    )
    return result


@formula(id="row_dot", version=1)
def row_dot(value: Tensor, weight: Tensor, *, output_dtype: DType | None = None) -> Tensor:
    return _emit("row_dot", value, weight, output_dtype=output_dtype)


@primitive(
    "embedding",
    work=useful.no_arithmetic,
    reference=lambda inputs, _attrs: (_embedding_reference(*inputs),),
    tags=frozenset({"embedding"}),
)
def _embedding(inputs, _attrs):
    _one(inputs, 2, "embedding")
    indices, table = inputs
    if not indices.dtype.integer or table.rank != 2:
        raise ValueError("embedding expects integer indices and rank-two table")
    return (TensorSpec((*indices.shape, table.shape[1]), table.dtype),)


@formula(id="embedding", version=1)
def embedding(indices: Tensor, table: Tensor) -> Tensor:
    """Look up vocabulary rows; indices must be nonnegative and below table rows."""
    return _emit("embedding", indices, table)


def _embedding_reference(indices, table):
    if _np().any(indices < 0) or _np().any(indices >= table.shape[0]):
        raise ValueError("embedding index is outside the vocabulary")
    return table[indices]


@primitive(
    "rotary",
    work=lambda inputs, attrs, outputs, *, values=None: useful.work(floating=3 * inputs[0].elements, special=inputs[0].elements),
    reference=lambda inputs, attrs: _rotary_reference(inputs, attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"position"}),
)
def _rotary(inputs, attrs):
    if len(inputs) not in (2, 3):
        raise ValueError("rotary expects q, k and optional coordinates")
    q, k = inputs[:2]
    if q.dtype != k.dtype or q.shape[-1] != k.shape[-1] or q.shape[-1] % 2:
        raise ValueError("rotary q/k width must agree and be even")
    if attrs["dimensions"] <= 0 or attrs["dimensions"] > q.shape[-1] or attrs["dimensions"] % 2:
        raise ValueError("invalid rotary dimensions")
    if len(inputs) == 3 and inputs[2].shape != q.shape[:-2]:
        raise ValueError("rotary coordinates must match the leading token geometry")
    return (q, k)


def _rotary_reference(inputs, attrs):
    np = _np()
    dimensions = attrs["dimensions"]
    half = dimensions // 2
    leading = inputs[0].shape[:-2]
    positions = inputs[2] if len(inputs) == 3 else np.arange(math.prod(leading)).reshape(leading)
    frequency = attrs["base"] ** (-np.arange(0, dimensions, 2, dtype=np.float32) / dimensions)
    angles = positions[..., None, None] * frequency

    def apply(value):
        rotated = value.copy()
        first = value[..., :half]
        second = value[..., half:dimensions]
        rotated[..., :half] = first * np.cos(angles) - second * np.sin(angles)
        rotated[..., half:dimensions] = second * np.cos(angles) + first * np.sin(angles)
        return rotated

    return apply(inputs[0]), apply(inputs[1])


@formula(id="rotary", version=1)
def rotary(
    q: Tensor,
    k: Tensor,
    coordinates: Tensor | None = None,
    *,
    dimensions: int | None = None,
    base: float = 1_000_000.0,
):
    if dimensions is None:
        if not isinstance(q.shape[-1], int):
            raise ValueError("rotary width must be specialized")
        dimensions = q.shape[-1]
    inputs = (q, k) if coordinates is None else (q, k, coordinates)
    return _emit("rotary", *inputs, dimensions=dimensions, base=base)


@primitive(
    "attention_prepare",
    work=useful.attention_prepare,
    reference=lambda inputs, attrs: _attention_prepare_reference(inputs, attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"attention", "normalization", "position"}),
)
def _attention_prepare(inputs, attrs):
    _one(inputs, 5, "attention_prepare")
    query_gate, keys, query_norm, key_norm, coordinates = inputs
    rows = query_gate.shape[0] if query_gate.rank == 2 else None
    query_heads = attrs["query_heads"]
    kv_heads = attrs["kv_heads"]
    width = attrs["width"]
    rotary_width = attrs["rotary_width"]
    sections = attrs["sections"]
    if (
        rows is None
        or query_gate.shape != (rows, query_heads * 2 * width)
        or keys.shape != (rows, kv_heads * width)
        or query_norm.shape != (width,)
        or key_norm.shape != (width,)
        or coordinates.shape != (rows, 3)
        or coordinates.dtype != DType.I32
        or len({query_gate.dtype, keys.dtype}) != 1
        or not query_gate.dtype.floating
        or query_norm.dtype != DType.F32
        or key_norm.dtype != DType.F32
        or rotary_width <= 0
        or rotary_width > width
        or rotary_width % 2
        or len(sections) != 4
        or any(type(value) is not int or value < 0 for value in sections)
        or sum(sections) * 2 != rotary_width
        or attrs["base"] <= 0
        or attrs["epsilon"] <= 0
    ):
        raise ValueError("invalid normalized rotary attention geometry")
    dtype = query_gate.dtype
    return (
        TensorSpec((rows, query_heads, width), dtype),
        TensorSpec((rows, kv_heads, width), dtype),
        TensorSpec((rows, query_heads, width), dtype),
    )


def _attention_prepare_reference(inputs, attrs):
    np = _np()
    query_gate, keys, query_norm, key_norm, coordinates = inputs
    rows = query_gate.shape[0]
    query_heads = attrs["query_heads"]
    kv_heads = attrs["kv_heads"]
    width = attrs["width"]
    rotary_width = attrs["rotary_width"]
    half = rotary_width // 2
    sections = attrs["sections"]

    def normalize(value, weight):
        inverse = 1 / np.sqrt(
            np.mean(value.astype(np.float32) ** 2, axis=-1, keepdims=True) + attrs["epsilon"]
        )
        return value.astype(np.float32) * inverse * weight.astype(np.float32)

    query_gate = query_gate.reshape(rows, query_heads, 2, width)
    query = normalize(query_gate[:, :, 0], query_norm)
    key = normalize(keys.reshape(rows, kv_heads, width), key_norm)
    frequency = attrs["base"] ** (-np.arange(0, rotary_width, 2, dtype=np.float32) / rotary_width)
    index = np.arange(half)
    axis = np.where(
        (index % 3 == 1) & (index < sections[1] * 3),
        1,
        np.where((index % 3 == 2) & (index < sections[2] * 3), 2, 0),
    )
    angles = coordinates[:, axis].astype(np.float32) * frequency

    def rotate(value):
        result = value.copy()
        first = value[..., :half]
        second = value[..., half:rotary_width]
        cosine = np.cos(angles)[:, None, :]
        sine = np.sin(angles)[:, None, :]
        result[..., :half] = first * cosine - second * sine
        result[..., half:rotary_width] = second * cosine + first * sine
        return result.astype(query_gate.dtype)

    return rotate(query), rotate(key), query_gate[:, :, 1].astype(query_gate.dtype)


@formula(id="attention_prepare", version=1)
def attention_prepare(
    query_gate: Tensor,
    keys: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    coordinates: Tensor,
    *,
    query_heads: int,
    kv_heads: int,
    width: int,
    rotary_width: int,
    base: float,
    sections: tuple[int, int, int, int],
    epsilon: float,
):
    return _emit(
        "attention_prepare",
        query_gate,
        keys,
        query_norm,
        key_norm,
        coordinates,
        query_heads=query_heads,
        kv_heads=kv_heads,
        width=width,
        rotary_width=rotary_width,
        base=base,
        sections=sections,
        epsilon=epsilon,
    )


@primitive(
    "kv_append",
    work=useful.no_arithmetic,
    reference=lambda inputs, attrs: (_kv_append_reference(inputs, attrs),),
    resource_reads=(0,),
    resource_writes=(0,),
    aliases=((0, 0),),
    tags=frozenset({"state"}),
)
def _kv_append(inputs, attrs):
    if len(inputs) != 4:
        raise ValueError("kv_append expects resource, keys, values and destinations")
    resource, keys, values, destinations = inputs
    if keys.shape[:2] != values.shape[:2] or keys.dtype != values.dtype:
        raise ValueError("KV keys and values must agree")
    if isinstance(resource.representation, KVRepresentation):
        representation = resource.representation
        if (resource.rank != 3 or keys.rank != 3 or values.rank != 3
                or resource.shape[1] != keys.shape[1]
                or resource.shape[2] != representation.logical_width
                or keys.shape[2] != representation.key_width
                or values.shape[2] != representation.value_width):
            raise ValueError("appended KV vectors disagree with persistent representation")
    elif keys.rank != 3 or resource.rank != 4 or resource.shape[0] != 2:
        raise ValueError("KV storage uses [2, capacity, head, channel] geometry")
    elif resource.shape[2:] != keys.shape[1:] or keys.shape != values.shape:
        raise ValueError("KV storage head geometry differs from appended values")
    if resource.dtype != keys.dtype or not keys.dtype.floating:
        raise ValueError("KV storage and appended values require the same floating dtype")
    if destinations.shape != (keys.shape[0],) or destinations.dtype != DType.I32:
        raise ValueError("KV destinations require one int32 index per appended row")
    return (resource,)


def _kv_append_reference(inputs, attrs):
    resource = inputs[0].copy()
    destinations = inputs[3]
    valid = destinations >= 0
    written = destinations[valid]
    representation = attrs.get("representation")
    if isinstance(representation, KVRepresentation):
        if _np().any(written >= resource.shape[0]) or len(_np().unique(written)) != len(written):
            raise ValueError("KV destinations must be distinct in-capacity rows or negative padding")
        if len(written) == 0:
            return resource
        from ..kv_codecs import quantize_reference, dequantize_reference
        resource[written, :, :representation.key_width] = dequantize_reference(
            quantize_reference(inputs[1][valid], representation.key), representation.key)
        resource[written, :, representation.key_width:] = dequantize_reference(
            quantize_reference(inputs[2][valid], representation.value), representation.value)
        return resource
    if _np().any(written >= resource.shape[1]) or len(_np().unique(written)) != len(written):
        raise ValueError("KV destinations must be distinct in-capacity rows or negative padding")
    resource[0, destinations[valid]] = inputs[1][valid]
    resource[1, destinations[valid]] = inputs[2][valid]
    return resource


@formula(id="kv_append", version=1)
def kv_append(resource: Tensor, keys: Tensor, values: Tensor, destinations: Tensor, *,
              reserved: bool = False) -> Tensor:
    """Publish distinct cache rows; negative destinations denote padding.

    The engine supplies valid, nonoverlapping destination rows. The independent
    reference checks this input precondition before a measurement is qualified.
    ``reserved=True`` additionally promises that destinations are outside every
    committed-history interval consumed by this advance. A reservation owner
    must establish that promise before constructing the formula; it permits
    producer persistence to overlap pre-advance history consumption.
    """
    if type(reserved) is not bool:
        raise TypeError("KV reservation declaration must be boolean")
    return _emit("kv_append", resource, keys, values, destinations,
                 representation=resource.spec.representation, reserved=reserved)


@primitive(
    "kv_copy",
    work=useful.no_arithmetic,
    reference=lambda inputs, attrs: (_kv_copy_reference(inputs, attrs),),
    resource_reads=(0,),
    resource_writes=(0,),
    aliases=((0, 0),),
    tags=frozenset({"state"}),
)
def _kv_copy(inputs, attrs):
    _one(inputs, 2, "kv_copy")
    resource, ranges = inputs
    if (
        (not isinstance(resource.representation, KVRepresentation)
         and (resource.rank != 4 or resource.shape[0] != 2))
        or ranges.rank != 2
        or ranges.shape[1] != 3
        or ranges.dtype != DType.I32
        or attrs["max_count"] <= 0
    ):
        raise ValueError("invalid KV copy geometry")
    return (resource,)


def _kv_copy_reference(inputs, attrs):
    source, ranges = inputs
    result = source.copy()
    typed = isinstance(attrs.get("representation"), KVRepresentation)
    capacity = source.shape[0 if typed else 1]
    reads, writes = [], []
    for start, destination, count in ranges:
        start, destination, count = int(start), int(destination), int(count)
        if (min(start, destination, count) < 0 or count > attrs["max_count"]
                or max(start, destination) + count > capacity):
            raise ValueError("KV copy range lies outside its declared capacity")
        if count:
            reads.append((start, start + count))
            writes.append((destination, destination + count))
        if typed:
            result[destination:destination + count] = source[start:start + count]
        else:
            result[:, destination:destination + count] = source[:, start:start + count]
    for index, target in enumerate(writes):
        if any(max(target[0], other[0]) < min(target[1], other[1])
               for other in writes[:index]):
            raise ValueError("KV copy destinations must be disjoint")
        if any(max(target[0], other[0]) < min(target[1], other[1])
               and not (other == target and reads[index] == target)
               for other in reads):
            raise ValueError("KV copy destinations must not overwrite source ranges")
    return result


def kv_copy(resource: Tensor, ranges: Tensor, *, max_count: int) -> Tensor:
    """Copy disjoint ranges, preserving the entire represented plane bundle."""
    return _emit("kv_copy", resource, ranges, max_count=max_count,
                 representation=resource.spec.representation)


@primitive(
    "persistent_attention",
    work=useful.persistent_attention,
    reference=lambda inputs, attrs: (_persistent_attention_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    resource_reads=(1,),
    tags=frozenset({"attention"}),
)
def _persistent_attention(inputs, attrs):
    _one(inputs, 5, "persistent_attention")
    queries, history, keys, values, visible = inputs
    representation = history.representation
    if not isinstance(representation, KVRepresentation) or history.rank != 3:
        raise ValueError("persistent attention requires typed KV state")
    if (queries.rank != 3 or keys.rank != 3 or values.rank != 3
            or not queries.dtype.floating
            or queries.dtype != keys.dtype or keys.dtype != values.dtype
            or keys.shape[:2] != values.shape[:2]
            or keys.shape[1] != history.shape[1]
            or queries.shape[1] % keys.shape[1]
            or queries.shape[2] != representation.key_width
            or keys.shape[2] != representation.key_width
            or values.shape[2] != representation.value_width
            or history.shape[2] != representation.logical_width):
        raise ValueError("persistent attention source geometry disagrees")
    if visible.shape != (queries.shape[0], 4) or visible.dtype != DType.I32:
        raise ValueError("visibility requires history start/count and current start/count")
    if not math.isfinite(attrs["scale"]) or attrs["scale"] <= 0:
        raise ValueError("attention scale must be finite and positive")
    return (TensorSpec((*queries.shape[:2], values.shape[2]), queries.dtype),)


def _persistent_attention_reference(inputs, attrs):
    """Logical reference: represented history is decoded by reference binding.

    Current rows are supplied independently and never round-trip through the
    persistence codec. Range counts carry causal visibility for each query.
    """
    np = _np()
    queries, history, keys, values, visible = inputs
    width = queries.shape[-1]
    result = np.zeros((*queries.shape[:2], values.shape[-1]), dtype=np.float32)
    group = queries.shape[1] // keys.shape[1]
    for row, ranges in enumerate(visible):
        start, count, current, current_count = map(int, ranges)
        if (min(start, count, current, current_count) < 0
                or start + count > history.shape[0]
                or current + current_count > keys.shape[0]):
            raise ValueError("persistent attention visibility lies outside its sources")
        if count + current_count == 0:
            continue
        for head in range(queries.shape[1]):
            kv_head = head // group
            k = np.concatenate((history[start:start + count, kv_head, :width],
                                keys[current:current + current_count, kv_head]), axis=0).astype(np.float32)
            v = np.concatenate((history[start:start + count, kv_head, width:],
                                values[current:current + current_count, kv_head]), axis=0).astype(np.float32)
            logits = queries[row, head].astype(np.float32) @ k.T * attrs["scale"]
            result[row, head] = _softmax_reference(logits, -1) @ v
    return result


@formula(id="persistent_attention", version=1)
def persistent_attention(queries: Tensor, history: Tensor, keys: Tensor, values: Tensor,
                         visible: Tensor, *, scale: float | None = None,
                         sequence_count: int | None = None) -> Tensor:
    """Attend to committed history and dense current rows in a single softmax.

    Visibility rows are [history_start, history_count, current_start,
    current_count]. The caller excludes newly reserved cache destinations from
    history and supplies current causal ranges explicitly.
    """
    return _emit("persistent_attention", queries, history, keys, values, visible,
                 scale=queries.shape[-1] ** -0.5 if scale is None else scale,
                 sequence_count=sequence_count)


@primitive(
    "causal_attention",
    work=useful.attention,
    reference=lambda inputs, attrs: (_attention_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    resource_reads=(1,),
    tags=frozenset({"attention"}),
)
def _causal_attention(inputs, attrs):
    if len(inputs) < 2:
        raise ValueError("causal_attention expects queries and history resource")
    queries = inputs[0]
    if queries.rank < 3 or not queries.dtype.floating:
        raise ValueError("attention queries require token, head and channel axes")
    history = inputs[1]
    if history.rank != 4 or history.shape[0] != 2 or history.shape[-1] != queries.shape[-1]:
        raise ValueError("attention history uses [2, capacity, head, channel] geometry")
    if queries.shape[-2] % history.shape[-2]:
        raise ValueError("query heads must be grouped over KV heads")
    if len(inputs) == 3 and inputs[2].shape not in (
        queries.shape[:-2],
        (*queries.shape[:-2], 2),
    ):
        raise ValueError("attention visibility must provide counts or start/count ranges")
    if len(inputs) == 3 and not inputs[2].dtype.integer:
        raise ValueError("attention visibility requires integer ranges")
    if attrs["sequence_count"] is not None and (
        type(attrs["sequence_count"]) is not int or attrs["sequence_count"] <= 0
    ):
        raise ValueError("attention sequence count must be positive")
    return (queries,)


def _attention_reference(inputs, attrs):
    np = _np()
    queries, history = inputs[:2]
    visible = inputs[2] if len(inputs) == 3 else np.full(queries.shape[:-2], history.shape[1])
    result = np.empty_like(queries)
    kv_heads = history.shape[-2]
    group = queries.shape[-2] // kv_heads
    ranges = tuple((int(item[0]), int(item[1])) if visible.ndim == 2 else (0, int(item))
                   for item in visible)
    if any(start < 0 or count < 0 or start + count > history.shape[1] for start, count in ranges):
        raise ValueError("attention visibility lies outside cache capacity")
    # Prepare one KV head once, not a fresh full-history FP32 copy for every
    # query/head dot product. Keep the same FP32 vector contractions and rounding.
    for head in range(queries.shape[1]):
        if head % group == 0:
            kv_head = head // group
            keys = np.ascontiguousarray(history[0, :, kv_head], dtype=np.float32)
            values = np.ascontiguousarray(history[1, :, kv_head], dtype=np.float32)
        for token, (start, count) in enumerate(ranges):
            if count == 0:
                result[token, head] = 0
                continue
            logits = queries[token, head].astype(np.float32, copy=False) @ keys[start : start + count].T
            probabilities = _softmax_reference(logits * attrs["scale"], -1)
            result[token, head] = probabilities @ values[start : start + count]
    return result


@formula(id="causal_attention", version=1)
def causal_attention(
    queries: Tensor,
    history: Tensor,
    reads: Tensor | None = None,
    *,
    scale: float | None = None,
    sequence_count: int | None = None,
) -> Tensor:
    inputs = (queries, history) if reads is None else (queries, history, reads)
    if scale is None:
        width = queries.shape[-1]
        if not isinstance(width, int):
            raise ValueError("attention width must be specialized")
        scale = width**-0.5
    return _emit(
        "causal_attention",
        *inputs,
        scale=scale,
        sequence_count=sequence_count,
    )


@primitive(
    "delta_recurrence",
    work=useful.recurrence,
    reference=lambda inputs, attrs: _recurrence_reference(inputs, attrs),
    numerical=NumericalContract(DType.F32),
    resource_reads=(1,),
    resource_writes=(1,),
    aliases=((1, 1),),
    tags=frozenset({"recurrence", "state"}),
)
def _delta_recurrence(inputs, attrs):
    if len(inputs) not in (2, 3):
        raise ValueError("delta_recurrence expects values, recurrent resource and optional decay")
    values, state = inputs[:2]
    if values.rank != 2 or state.shape not in ((values.shape[1],), (1, values.shape[1])):
        raise ValueError("recurrent state width differs from values")
    if len(inputs) == 3 and inputs[2].shape not in ((values.shape[1],), values.shape):
        raise ValueError("recurrence decay must be channel- or token-channel-shaped")
    return values, state


def _recurrence_reference(inputs, attrs):
    values, state = inputs[:2]
    result = _np().empty_like(values)
    recurrent = state.reshape(-1, values.shape[-1])[0].astype(_np().float32).copy()
    for token in range(values.shape[0]):
        decay = 0 if len(inputs) == 2 else inputs[2][token] if inputs[2].ndim == 2 else inputs[2]
        recurrent = recurrent * decay + values[token]
        result[token] = recurrent
    updated = state.copy()
    updated.reshape(-1, values.shape[-1])[0] = recurrent
    return result, updated


@formula(id="delta_recurrence", version=1)
def delta_recurrence(values: Tensor, state: Tensor, *parameters: Tensor, **attributes: Any):
    return _emit("delta_recurrence", values, state, *parameters, **attributes)


@primitive(
    "gated_delta_recurrence",
    work=useful.gated_delta,
    reference=lambda inputs, attrs: _gated_delta_reference(inputs, attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"recurrence", "state"}),
)
def _gated_delta_recurrence(inputs, attrs):
    _one(inputs, 7, "gated_delta_recurrence")
    query, key, value, decay, beta, previous, offsets = inputs
    if (
        query.rank != 3
        or key.shape != query.shape
        or value.rank != 3
        or value.shape[0] != query.shape[0]
        or decay.shape != value.shape[:2]
        or beta.shape != value.shape[:2]
        or previous.rank != 4
        or previous.shape[1:] != (value.shape[1], value.shape[2], query.shape[2])
        or query.shape[1] > value.shape[1]
        or value.shape[1] % query.shape[1]
        or attrs["mapping"] not in {"tiled", "grouped"}
        or len({query.dtype, key.dtype, value.dtype, beta.dtype}) != 1
        or decay.dtype != DType.F32
        or previous.dtype != DType.F32
        or offsets.shape != (previous.shape[0] + 1,)
        or offsets.dtype != DType.I32
    ):
        raise ValueError("invalid gated delta recurrence geometry")
    length = attrs.get("sequence_length")
    if length is not None and (
        type(length) is not int or previous.shape[0] != 1 or not 0 <= length <= query.shape[0]
    ):
        raise ValueError("static recurrence length requires one sequence within input capacity")
    return value, previous


def _gated_delta_reference(inputs, attrs):
    np = _np()
    query, key, value, decay, beta, previous, offsets = inputs
    length = attrs.get("sequence_length")
    if length is not None and tuple(offsets) != (0, length):
        raise ValueError("recurrent offsets do not match the declared static sequence length")
    batch, value_heads, value_width, key_width = previous.shape
    key_heads = query.shape[1]
    state = previous.astype(np.float32).copy()
    output = np.empty_like(value)
    for sequence in range(batch):
        for head in range(value_heads):
            key_head = (
                head % key_heads
                if attrs["mapping"] == "tiled"
                else head // (value_heads // key_heads)
            )
            for row in range(int(offsets[sequence]), int(offsets[sequence + 1])):
                state[sequence, head] *= decay[row, head]
                remembered = state[sequence, head] @ key[row, key_head].astype(np.float32)
                residual = (value[row, head].astype(np.float32) - remembered) * beta[row, head]
                state[sequence, head] += (
                    residual[:, None] * key[row, key_head].astype(np.float32)[None, :]
                )
                output[row, head] = (
                    state[sequence, head] @ query[row, key_head].astype(np.float32)
                ).astype(value.dtype)
    return output, state


@formula(id="gated_delta_recurrence", version=1)
def gated_delta_recurrence(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    decay: Tensor,
    beta: Tensor,
    previous: Tensor,
    offsets: Tensor,
    *,
    mapping: str = "tiled",
    sequence_length: int | None = None,
):
    """Apply recurrence; a static length declares offsets exactly (0, length).

    Callers providing that specialization must establish it from invocation
    metadata, not from the allocated input capacity alone.
    """
    return _emit(
        "gated_delta_recurrence",
        query,
        key,
        value,
        decay,
        beta,
        previous,
        offsets,
        mapping=mapping,
        sequence_length=sequence_length,
    )


@primitive(
    "recurrent_prepare",
    work=useful.recurrent_prepare,
    reference=lambda inputs, attrs: _recurrent_prepare_reference(inputs, attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"recurrence", "normalization", "state"}),
)
def _recurrent_prepare(inputs, attrs):
    _one(inputs, 8, "recurrent_prepare")
    projected, convolution, previous, alpha, beta_input, rate, bias, offsets = inputs
    batch = previous.shape[0] if previous.rank == 3 else None
    rows = projected.shape[0] if projected.rank == 2 else None
    key_heads = attrs["key_heads"]
    value_heads = attrs["value_heads"]
    width = attrs["width"]
    channels = (2 * key_heads + value_heads) * width
    history = attrs["convolution_width"] - 1
    if (
        batch is None
        or rows is None
        or projected.shape != (rows, channels)
        or convolution.shape != (channels, history + 1)
        or previous.shape != (batch, channels, history)
        or alpha.shape != (rows, value_heads)
        or beta_input.shape != alpha.shape
        or rate.shape != (value_heads,)
        or bias.shape != (value_heads,)
        or not projected.dtype.floating
        or convolution.dtype != DType.F32
        or previous.dtype != projected.dtype
        or alpha.dtype != projected.dtype
        or beta_input.dtype != projected.dtype
        or rate.dtype != DType.F32
        or bias.dtype != DType.F32
        or offsets.shape != (batch + 1,)
        or offsets.dtype != DType.I32
        or attrs["epsilon"] <= 0
    ):
        raise ValueError("invalid recurrent preparation geometry")
    dtype = projected.dtype
    return (
        TensorSpec((rows, key_heads, width), dtype),
        TensorSpec((rows, key_heads, width), dtype),
        TensorSpec((rows, value_heads, width), dtype),
        TensorSpec((rows, value_heads), dtype),
        TensorSpec((rows, value_heads), DType.F32),
        previous,
    )


def _recurrent_prepare_reference(inputs, attrs):
    np = _np()
    projected, convolution, previous, alpha, beta_input, rate, bias, offsets = inputs
    batch, channels, history = previous.shape
    rows = projected.shape[0]
    key_heads = attrs["key_heads"]
    value_heads = attrs["value_heads"]
    width = attrs["width"]
    convolved = np.empty((rows, channels), dtype=np.float32)
    following = np.empty_like(previous)
    for sequence in range(batch):
        start, end = int(offsets[sequence]), int(offsets[sequence + 1])
        joined = np.concatenate(
            (
                previous[sequence].astype(np.float32),
                projected[start:end].T.astype(np.float32),
            ),
            axis=1,
        )
        for step, row in enumerate(range(start, end)):
            convolved[row] = np.sum(joined[:, step : step + history + 1] * convolution, axis=1)
        following[sequence] = joined[:, -history:]
    convolved = convolved / (1 + np.exp(-convolved))
    heads = convolved.reshape(rows, 2 * key_heads + value_heads, width)

    def normalized(value, gain):
        inverse = 1 / np.sqrt(np.sum(value * value, axis=-1, keepdims=True) + attrs["epsilon"])
        return (value * inverse * gain).astype(projected.dtype)

    query = normalized(heads[:, :key_heads], 1 / math.sqrt(width))
    key = normalized(heads[:, key_heads : 2 * key_heads], 1)
    value = heads[:, 2 * key_heads :].astype(projected.dtype)
    beta = (1 / (1 + np.exp(-beta_input.astype(np.float32)))).astype(projected.dtype)
    shifted = alpha.astype(np.float32) + bias
    softplus = np.maximum(shifted, 0) + np.log1p(np.exp(-np.abs(shifted)))
    decay = np.exp(rate * softplus).astype(np.float32)
    return query, key, value, beta, decay, following


@formula(id="recurrent_prepare", version=1)
def recurrent_prepare(
    projected: Tensor,
    convolution: Tensor,
    previous: Tensor,
    alpha: Tensor,
    beta_input: Tensor,
    rate: Tensor,
    bias: Tensor,
    offsets: Tensor,
    *,
    key_heads: int,
    value_heads: int,
    width: int,
    convolution_width: int,
    epsilon: float,
):
    return _emit(
        "recurrent_prepare",
        projected,
        convolution,
        previous,
        alpha,
        beta_input,
        rate,
        bias,
        offsets,
        key_heads=key_heads,
        value_heads=value_heads,
        width=width,
        convolution_width=convolution_width,
        epsilon=epsilon,
    )


@primitive(
    "route_topk",
    work=useful.routing,
    reference=lambda inputs, attrs: _route_reference(inputs[0], attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"routing"}),
)
def _route_topk(inputs, attrs):
    _one(inputs, 1, "route_topk")
    logits = inputs[0]
    if (
        logits.rank != 2
        or not logits.dtype.floating
        or attrs["scoring"] not in {"softmax", "sigmoid"}
        or not 0 < attrs["k"] <= logits.shape[1]
    ):
        raise ValueError("invalid routed expert count")
    shape = (logits.shape[0], attrs["k"])
    return TensorSpec(shape, DType.I32), TensorSpec(shape, DType.F32)


def _route_reference(logits, attrs):
    np = _np()
    scores = (
        1 / (1 + np.exp(-logits))
        if attrs["scoring"] == "sigmoid"
        else _softmax_reference(logits, -1)
    )
    # The architecture observes selected routes in ascending probability/index
    # order. Stable ascending sorting also makes the larger expert ID win an
    # exact-score tie at the cutoff.
    indices = np.argsort(scores, axis=-1, kind="stable")[:, -attrs["k"] :]
    selected = np.take_along_axis(scores, indices, axis=-1)
    if attrs["normalize"]:
        selected = selected / np.sum(selected, axis=-1, keepdims=True)
    return indices.astype(np.int32), selected.astype(np.float32)


@formula(id="route_topk", version=1)
def route_topk(logits: Tensor, k: int, *, scoring: str = "softmax", normalize: bool = True):
    return _emit("route_topk", logits, k=k, scoring=scoring, normalize=normalize)


@primitive(
    "routed_experts",
    work=useful.experts,
    reference=lambda inputs, attrs: (_experts_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"experts"}),
)
def _routed_experts(inputs, attrs):
    if len(inputs) != 6:
        raise ValueError("routed_experts expects hidden, routes, scores and expert weights")
    hidden, routes, scores = inputs[:3]
    if attrs["storage_dtype"] != hidden.dtype or attrs["activation"] not in {"silu", "tanh"}:
        raise ValueError("invalid routed expert publication precision or activation")
    if (
        hidden.rank != 2
        or not hidden.dtype.floating
        or routes.shape != scores.shape
        or routes.shape[0] != hidden.shape[0]
        or not routes.dtype.integer
        or scores.dtype != DType.F32
    ):
        raise ValueError("invalid routed expert geometry")
    gate, up, down = inputs[3:6]
    if (
        gate.rank != 3
        or up.shape != gate.shape
        or down.rank != 3
        or not gate.dtype.floating
        or not up.dtype.floating
        or not down.dtype.floating
    ):
        raise ValueError("expert weights require [expert, output, input] geometry")
    if gate.shape[0] != down.shape[0] or gate.shape[2] != hidden.shape[1]:
        raise ValueError("expert input geometry differs from hidden state")
    if down.shape[1] != hidden.shape[1] or down.shape[2] != gate.shape[1]:
        raise ValueError("expert down projection geometry is invalid")
    return (hidden,)


def _experts_reference(inputs, attrs):
    from .primitive import round_reference

    np = _np()
    hidden, routes, scores, gate, up, down = inputs[:6]
    dtype = DType(attrs["storage_dtype"])
    result = np.zeros(hidden.shape, dtype=np.float32)
    for token in range(hidden.shape[0]):
        for choice in range(routes.shape[1]):
            expert = int(routes[token, choice])
            if not 0 <= expert < gate.shape[0]:
                raise ValueError("expert route is outside the logical bank")
            gated = round_reference(gate[expert].astype(np.float32) @ hidden[token].astype(np.float32), dtype).astype(np.float32)
            expanded = round_reference(up[expert].astype(np.float32) @ hidden[token].astype(np.float32), dtype).astype(np.float32)
            if attrs["activation"] == "silu":
                gated = gated / (1 + np.exp(-gated))
            elif attrs["activation"] == "tanh":
                gated = np.tanh(gated)
            else:
                raise ValueError(f"unsupported expert activation {attrs['activation']!r}")
            activated = round_reference(gated * expanded, dtype).astype(np.float32)
            projected = round_reference(down[expert].astype(np.float32) @ activated, dtype).astype(np.float32)
            result[token] += scores[token, choice] * projected
    return result


@formula(id="routed_experts", version=1)
def routed_experts(
    hidden: Tensor,
    routes: Tensor,
    scores: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    *,
    activation: str = "silu",
) -> Tensor:
    return _emit("routed_experts", hidden, routes, scores, gate, up, down,
                 activation=activation, storage_dtype=hidden.dtype)
