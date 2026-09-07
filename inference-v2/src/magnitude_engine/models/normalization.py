"""Fused residual transitions with the MLX RMS reduction and dtype boundaries."""

from functools import cache
from typing import Any

import mlx.core as mx
import mlx.nn as nn

_RESIDUAL_SOURCE = """
    // Match MLX RMS: four consecutive values per thread, then SIMD reductions.
    // Wider rows use 1024 threads and retain the same ordered chunk traversal.
    constexpr int ELEMS = 4;
    constexpr int CHUNK = THREADS * ELEMS;
    uint t   = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float part[32];
    threadgroup float inv_sh[1];
    const size_t rowbase = (size_t)row * D;
    float xs[NCHUNK * ELEMS];
    float ss2 = 0.0f;
    _Pragma("clang loop unroll(full)") for (int c = 0; c < NCHUNK; ++c) {
        const uint idx = c * CHUNK + t * ELEMS;
        if (idx + ELEMS <= D) {
            _Pragma("clang loop unroll(full)") for (int i = 0; i < ELEMS; ++i) {
                float v = float(T(float(x[rowbase + idx + i]) + float(a[rowbase + idx + i])));
                xs[c * ELEMS + i] = v;
                xnew[rowbase + idx + i] = static_cast<T>(v);
                ss2 += v * v;
            }
        }
    }
    ss2 = simd_sum(ss2);
    if (sg == 0) part[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) part[sg] = ss2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float tot2 = simd_sum(part[lane]);
        if (lane == 0) inv_sh[0] = metal::precise::rsqrt(tot2 / float(D) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv2 = inv_sh[0];
    _Pragma("clang loop unroll(full)") for (int c = 0; c < NCHUNK; ++c) {
        const uint idx = c * CHUNK + t * ELEMS;
        if (idx + ELEMS <= D) {
            _Pragma("clang loop unroll(full)") for (int i = 0; i < ELEMS; ++i) {
                float n = static_cast<float>(static_cast<T>(xs[c * ELEMS + i] * inv2));
                normalized[rowbase + idx + i] = static_cast<T>(n * static_cast<float>(w1[idx + i]));
            }
        }
    }
"""


@cache
def _residual_kernel(eps: float) -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_residual_norm",
        input_names=["x", "a", "w1"],
        output_names=["xnew", "normalized"],
        source=_RESIDUAL_SOURCE.replace("EPS", f"{eps:.10e}f"),
    )


def residual_norm(x: mx.array, update: mx.array, norm) -> tuple[mx.array, mx.array]:
    width = x.shape[-1]
    if not isinstance(norm, nn.RMSNorm) or width % 128 or x.dtype != update.dtype:
        residual = x + update
        return residual, norm(residual)
    threads = min(width // 4, 1024)
    result = _residual_kernel(norm.eps)(
        inputs=[x, update, norm.weight],
        template=[
            ("T", x.dtype),
            ("D", width),
            ("THREADS", threads),
            ("NCHUNK", (width + threads * 4 - 1) // (threads * 4)),
        ],
        grid=(threads, x.size // width, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return result[0], result[1]


_GATED_SOURCE = """
    uint t = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    threadgroup float partial[32];
    threadgroup float inverse;
    float values[4];
    float sum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        values[i] = float(x[row * D + t * 4 + i]);
        sum += values[i] * values[i];
    }
    sum = simd_sum(sum);
    if (sg == 0) partial[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum(partial[lane]);
        if (lane == 0) inverse = metal::precise::rsqrt(total / float(D) + EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint column = t * 4 + i;
        uint index = row * D + column;
        T normed = T(float(T(values[i] * inverse)) * float(w[column]));
        float z = float(gate[index]);
        float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
        float sigmoid = z < 0.0f ? e : 1.0f - e;
        out[index] = T((z * sigmoid) * float(normed));
    }
"""


@cache
def _gated_kernel(eps: float) -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_gated_norm",
        input_names=["x", "gate", "w"],
        output_names=["out"],
        source=_GATED_SOURCE.replace("EPS", f"{eps:.10e}f"),
    )


class GatedRMSNorm(nn.Module):
    def __init__(self, weight: mx.array, eps: float):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        width = hidden.shape[-1]
        if width % 128 or width > 4096:
            normalized = mx.fast.rms_norm(hidden, self.weight, self.eps)
            return (nn.silu(gate.astype(mx.float32)) * normalized.astype(mx.float32)).astype(
                hidden.dtype
            )
        return _gated_kernel(self.eps)(
            inputs=[hidden, gate, self.weight],
            template=[("T", hidden.dtype), ("D", width)],
            grid=(width // 4, hidden.size // width, 1),
            threadgroup=(width // 4, 1, 1),
            output_shapes=[hidden.shape],
            output_dtypes=[hidden.dtype],
        )[0]
