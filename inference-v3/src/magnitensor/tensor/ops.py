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
    resource[0, destinations] = inputs[1]
    resource[1, destinations] = inputs[2]
    return resource


def kv_append(resource: Tensor, keys: Tensor, values: Tensor, destinations: Tensor) -> Tensor:
    return _emit("kv_append", resource, keys, values, destinations)


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
    if len(inputs) == 3 and inputs[2].shape != queries.shape[:-2]:
        raise ValueError("attention visibility must match token geometry")
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
            count = int(visible[token])
            kv_head = head // group
            logits = (
                queries[token, head].astype(np.float32)
                @ history[0, :count, kv_head].astype(np.float32).T
            )
            probabilities = _softmax_reference(logits * attrs["scale"], -1)
            result[token, head] = probabilities @ history[1, :count, kv_head].astype(np.float32)
    return result


def causal_attention(
    queries: Tensor, history: Tensor, reads: Tensor | None = None, *, scale: float | None = None
) -> Tensor:
    inputs = (queries, history) if reads is None else (queries, history, reads)
    if scale is None:
        width = queries.shape[-1]
        if not isinstance(width, int):
            raise ValueError("attention width must be specialized")
        scale = width**-0.5
    return _emit("causal_attention", *inputs, scale=scale)


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
    "route_topk",
    reference=lambda inputs, attrs: _route_reference(inputs[0], attrs),
    numerical=NumericalContract(DType.F32),
    tags=frozenset({"routing"}),
)
def _route_topk(inputs, attrs):
    _one(inputs, 1, "route_topk")
    logits = inputs[0]
    if logits.rank != 2 or not 0 < attrs["k"] <= logits.shape[1]:
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
    indices = np.argsort(scores, axis=-1, kind="stable")[:, -attrs["k"] :][:, ::-1]
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
    if hidden.rank != 2 or routes.shape != scores.shape or routes.shape[0] != hidden.shape[0]:
        raise ValueError("invalid routed expert geometry")
    gate, up, down = inputs[3:6]
    if gate.rank != 3 or up.shape != gate.shape or down.rank != 3:
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
