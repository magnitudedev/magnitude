"""Composable tensor operations used by inference model functions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .operation import NumericalContract, operation
from .tracing import Tensor, active_trace
from .types import DType, TensorSpec, broadcast_shape, normalize_axes


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


@operation(
    "scalar",
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
    _one(inputs, 2, "binary operation")
    left, right = inputs
    if left.dtype != right.dtype:
        raise ValueError("binary tensor dtypes must agree explicitly")
    return (TensorSpec(broadcast_shape(left.shape, right.shape), left.dtype),)


def _binary_ref(operator):
    return lambda inputs, _attrs: (operator(inputs[0], inputs[1]),)


operation("add", reference=_binary_ref(lambda a, b: a + b), tags=frozenset({"cheap"}))(
    _binary_abstract
)
operation("subtract", reference=_binary_ref(lambda a, b: a - b), tags=frozenset({"cheap"}))(
    _binary_abstract
)
operation("multiply", reference=_binary_ref(lambda a, b: a * b), tags=frozenset({"cheap"}))(
    _binary_abstract
)
operation("divide", reference=_binary_ref(lambda a, b: a / b), tags=frozenset({"cheap"}))(
    _binary_abstract
)


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


@operation(
    "cast",
    reference=lambda inputs, attrs: (inputs[0].astype(attrs["dtype"].value),),
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
    return (bits.view(np.float32).astype(attrs["dtype"].value),)


@operation(
    "decode_bfloat16",
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


@operation(
    "reshape",
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


@operation(
    "transpose",
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


@operation(
    "concatenate", reference=lambda inputs, attrs: (_np().concatenate(inputs, axis=attrs["axis"]),)
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


@operation(
    "take_rows",
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


@operation(
    "overlay_rows",
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


@operation(
    "quantized_import",
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


@operation(
    "sample",
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


def sample(logits: Tensor, draws: Tensor) -> Tensor:
    """Select token/status rows using position-addressed deterministic draws."""

    return _emit("sample", logits, draws)


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


@operation(
    "matmul",
    reference=lambda inputs, _attrs: (_np().matmul(*inputs),),
    numerical=NumericalContract(DType.F32),
)
def _matmul(inputs, _attrs):
    _one(inputs, 2, "matmul")
    left, right = inputs
    if left.rank < 2 or right.rank < 2 or left.shape[-1] != right.shape[-2]:
        raise ValueError("invalid matrix product geometry")
    if left.dtype != right.dtype or not left.dtype.floating:
        raise ValueError("matrix inputs must have the same floating dtype")
    batch = broadcast_shape(left.shape[:-2], right.shape[:-2])
    return (TensorSpec((*batch, left.shape[-2], right.shape[-1]), left.dtype),)


def matmul(left: Tensor, right: Tensor) -> Tensor:
    return _emit("matmul", left, right)


def _unary(name, function):
    @operation(
        name,
        reference=lambda inputs, _attrs: (function(_np(), inputs[0]),),
        tags=frozenset({"cheap"}),
    )
    def abstract(inputs, _attrs):
        _one(inputs, 1, name)
        if not inputs[0].dtype.floating:
            raise ValueError(f"{name} requires floating input")
        return (inputs[0],)

    return lambda value: _emit(name, value)


exp = _unary("exp", lambda np, x: np.exp(x))
sigmoid = _unary("sigmoid", lambda np, x: 1 / (1 + np.exp(-x)))
silu = _unary("silu", lambda np, x: x / (1 + np.exp(-x)))
tanh = _unary("tanh", lambda np, x: np.tanh(x))


@operation(
    "softmax",
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
    shifted = value - np.max(value, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=axis, keepdims=True)


def softmax(value: Tensor, axis: int = -1) -> Tensor:
    return _emit("softmax", value, axis=axis)


@operation(
    "rms_norm",
    reference=lambda inputs, attrs: (_rms_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
)
def _rms_norm(inputs, attrs):
    if len(inputs) not in (1, 2):
        raise ValueError("rms_norm expects input and optional weight")
    value = inputs[0]
    if not value.dtype.floating or value.rank < 1:
        raise ValueError("rms_norm needs a floating tensor")
    if len(inputs) == 2 and (
        inputs[1].shape != (value.shape[-1],) or inputs[1].dtype != value.dtype
    ):
        raise ValueError("rms_norm weight geometry differs from the last axis")
    if attrs["epsilon"] <= 0:
        raise ValueError("rms_norm epsilon must be positive")
    return (value,)


def _rms_reference(inputs, attrs):
    np = _np()
    value = inputs[0]
    result = value / np.sqrt(
        np.mean(value.astype(np.float32) ** 2, axis=-1, keepdims=True) + attrs["epsilon"]
    )
    return result if len(inputs) == 1 else result * inputs[1]


def rms_norm(value: Tensor, weight: Tensor | None = None, *, epsilon: float = 1e-6) -> Tensor:
    inputs = (value,) if weight is None else (value, weight)
    return _emit("rms_norm", *inputs, epsilon=epsilon)


@operation(
    "linear",
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
    result = _np().matmul(inputs[0], _np().swapaxes(inputs[1], -1, -2))
    if len(inputs) == 3:
        result = result + inputs[2]
    return result.astype((attrs.get("output_dtype") or DType.F32).value)


def linear(
    value: Tensor, weight: Tensor, bias: Tensor | None = None, *, output_dtype: DType | None = None
) -> Tensor:
    inputs = (value, weight) if bias is None else (value, weight, bias)
    return _emit("linear", *inputs, output_dtype=output_dtype)


@operation(
    "row_dot",
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
    return result.astype(output_dtype.value)


def row_dot(value: Tensor, weight: Tensor, *, output_dtype: DType | None = None) -> Tensor:
    return _emit("row_dot", value, weight, output_dtype=output_dtype)


@operation(
    "embedding",
    reference=lambda inputs, _attrs: (inputs[1][inputs[0]],),
    tags=frozenset({"embedding"}),
)
def _embedding(inputs, _attrs):
    _one(inputs, 2, "embedding")
    indices, table = inputs
    if not indices.dtype.integer or table.rank != 2:
        raise ValueError("embedding expects integer indices and rank-two table")
    return (TensorSpec((*indices.shape, table.shape[1]), table.dtype),)


def embedding(indices: Tensor, table: Tensor) -> Tensor:
    return _emit("embedding", indices, table)


@operation(
    "rotary",
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


@operation(
    "attention_prepare",
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
    frequency = attrs["base"] ** (-np.arange(0, rotary_width, 2, dtype=np.float32) / half)
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


@operation(
    "kv_append",
    reference=lambda inputs, attrs: (_kv_append_reference(inputs, attrs),),
    resource_reads=(0,),
    resource_writes=(0,),
    aliases=((0, 0),),
    tags=frozenset({"state"}),
)
def _kv_append(inputs, attrs):
    if len(inputs) != 4:
        raise ValueError("kv_append expects resource, keys, values and destinations")
    resource, keys, values = inputs[:3]
    if keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError("KV keys and values must agree")
    if keys.rank != 3 or resource.rank != 4 or resource.shape[0] != 2:
        raise ValueError("KV storage uses [2, capacity, head, channel] geometry")
    if resource.shape[2:] != keys.shape[1:]:
        raise ValueError("KV storage head geometry differs from appended values")
    return (resource,)


def _kv_append_reference(inputs, attrs):
    resource = inputs[0].copy()
    destinations = inputs[3]
    valid = destinations >= 0
    resource[0, destinations[valid]] = inputs[1][valid]
    resource[1, destinations[valid]] = inputs[2][valid]
    return resource


def kv_append(resource: Tensor, keys: Tensor, values: Tensor, destinations: Tensor) -> Tensor:
    return _emit("kv_append", resource, keys, values, destinations)


@operation(
    "kv_copy",
    resource_reads=(0,),
    resource_writes=(0,),
    aliases=((0, 0),),
    tags=frozenset({"state"}),
)
def _kv_copy(inputs, attrs):
    _one(inputs, 2, "kv_copy")
    resource, ranges = inputs
    if (
        resource.rank != 4
        or resource.shape[0] != 2
        or ranges.rank != 2
        or ranges.shape[1] != 3
        or ranges.dtype != DType.I32
        or attrs["max_count"] <= 0
    ):
        raise ValueError("invalid KV copy geometry")
    return (resource,)


def kv_copy(resource: Tensor, ranges: Tensor, *, max_count: int) -> Tensor:
    return _emit("kv_copy", resource, ranges, max_count=max_count)


@operation(
    "causal_attention",
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
    for token in range(queries.shape[0]):
        for head in range(queries.shape[1]):
            if visible.ndim == 2:
                start, count = map(int, visible[token])
            else:
                start, count = 0, int(visible[token])
            kv_head = head // group
            logits = (
                queries[token, head].astype(np.float32)
                @ history[0, start : start + count, kv_head].astype(np.float32).T
            )
            probabilities = _softmax_reference(logits * attrs["scale"], -1)
            result[token, head] = probabilities @ history[
                1, start : start + count, kv_head
            ].astype(np.float32)
    return result


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


@operation(
    "delta_recurrence",
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


def delta_recurrence(values: Tensor, state: Tensor, *parameters: Tensor, **attributes: Any):
    return _emit("delta_recurrence", values, state, *parameters, **attributes)


@operation(
    "gated_delta_recurrence",
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
    return value, previous


def _gated_delta_reference(inputs, attrs):
    np = _np()
    query, key, value, decay, beta, previous, offsets = inputs
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
):
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
    )


@operation(
    "recurrent_prepare",
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
            convolved[row] = np.sum(
                joined[:, step : step + history + 1] * convolution, axis=1
            )
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


@operation(
    "route_topk",
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


def route_topk(logits: Tensor, k: int, *, scoring: str = "softmax", normalize: bool = True):
    return _emit("route_topk", logits, k=k, scoring=scoring, normalize=normalize)


@operation(
    "routed_experts",
    reference=lambda inputs, attrs: (_experts_reference(inputs, attrs),),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"experts"}),
)
def _routed_experts(inputs, attrs):
    if len(inputs) < 6:
        raise ValueError("routed_experts expects hidden, routes, scores and expert weights")
    hidden, routes, scores = inputs[:3]
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
    np = _np()
    hidden, routes, scores, gate, up, down = inputs[:6]
    result = np.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for choice in range(routes.shape[1]):
            expert = int(routes[token, choice])
            gated = gate[expert].astype(np.float32) @ hidden[token].astype(np.float32)
            expanded = up[expert].astype(np.float32) @ hidden[token].astype(np.float32)
            if attrs["activation"] == "silu":
                gated = gated / (1 + np.exp(-gated))
            elif attrs["activation"] == "tanh":
                gated = np.tanh(gated)
            else:
                raise ValueError(f"unsupported expert activation {attrs['activation']!r}")
            result[token] += scores[token, choice] * (
                down[expert].astype(np.float32) @ (gated * expanded)
            )
    return result


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
    return _emit("routed_experts", hidden, routes, scores, gate, up, down, activation=activation)
