"""Bind Qwen text positions to the canonical MLX-VLM frequency convention."""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

from magnitude_engine.models.transforms import PositionTransform


class QwenRotary:
    """Rotate Q/K with shared text positions and explicit FP32 inverse frequencies.

    The default MLX RoPE primitive uses a different frequency calculation, whose
    rounding can change Qwen logits. Canonical text rotation follows MLX-VLM;
    specialized library operators retain their own declared scaling semantics.
    """

    def __init__(self, operation: PositionTransform | Qwen3_5RotaryEmbedding):
        self.operation = operation
        self.rotation: Qwen3_5RotaryEmbedding | None = None
        if isinstance(operation, Qwen3_5RotaryEmbedding):
            self.rotation = operation
        elif type(operation) is nn.RoPE and not operation.traditional and operation.scale == 1:
            dims = operation.dims
            if dims < 2 or dims % 2:
                raise ValueError("Qwen rotary dimensions must be positive and even")
            self.rotation = Qwen3_5RotaryEmbedding(
                dims,
                base=operation.base,
                mrope_section=[dims // 2, 0, 0],
            )

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        *,
        offset: int | mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.rotation is None:
            if isinstance(offset, mx.array) and offset.ndim > 1:
                raise ValueError("this rotary binding does not support explicit coordinates")
            operation = cast(PositionTransform, self.operation)
            return operation(queries, offset=offset), operation(keys, offset=offset)
        if (
            queries.ndim != 4
            or keys.ndim != 4
            or queries.shape[0] != keys.shape[0]
            or queries.shape[2:] != keys.shape[2:]
            or min(*queries.shape, *keys.shape) < 1
            or self.rotation.dim > queries.shape[3]
            or queries.dtype != keys.dtype
            or queries.dtype not in (mx.float32, mx.float16, mx.bfloat16)
        ):
            raise ValueError("Qwen rotary requires aligned floating query/key tensors")
        if isinstance(offset, mx.array) and offset.ndim == 3:
            if offset.shape != (3, queries.shape[0], queries.shape[2]) or offset.dtype != mx.int32:
                raise ValueError("Qwen multimodal coordinates must have shape [3, rows, tokens]")
            return cast(
                tuple[mx.array, mx.array], self.rotation.apply_rotary(queries, keys, offset)
            )
        offsets = mx.array([offset], mx.int32) if isinstance(offset, int) else offset.reshape(-1)
        if offsets.size not in (1, queries.shape[0]) or offsets.dtype != mx.int32:
            raise ValueError("Qwen rotary positions must be scalar or per-row int32 offsets")
        if offsets.size == 1 and queries.shape[0] > 1:
            offsets = mx.broadcast_to(offsets, (queries.shape[0],))
        positions = offsets[:, None] + mx.arange(queries.shape[2], dtype=mx.int32)[None, :]
        return cast(tuple[mx.array, mx.array], self.rotation.apply_rotary(queries, keys, positions))
