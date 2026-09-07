"""Packed gated-delta preparation: causal convolution, head norms and gates.

The short-sequence kernel preserves the MLX equation's intermediate dtype rounding.
One group owns a key-width channel block; retained convolution state is bounded.
Based on the independently qualified preparation equation in mlx-poc/kernels/gdn.py.
"""

from functools import cache
from typing import Any

import mlx.core as mx


@cache
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_delta_prepare",
        input_names=["proj", "state", "w", "A_log", "dt_bias"],
        output_names=["y", "new_state", "z", "beta", "g"],
        source="""
    constexpr int TG = DK;                       // one q/k head per threadgroup
    uint c   = thread_position_in_grid.x;         // channel in [0, CK)
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.y;    // batch row
    const uint P = CK + CV + 2 * HV;
    const device T* st = state + row * (K - 1) * CK;
    device T* ns = new_state + row * (K - 1) * CK;
    threadgroup float part[TG / 32];
    // window of the last K-1 raw inputs for this channel (starts as the carried state)
    float win[K - 1];
    for (int t = 0; t < K - 1; ++t) win[t] = static_cast<float>(st[t * CK + c]);
    for (int tt = 0; tt < TT; ++tt) {
        const device T* pr = proj + (row * TT + tt) * P;
        float xin = static_cast<float>(pr[c]);
        float acc = 0.0f;
        for (int t = 0; t < K - 1; ++t) acc += static_cast<float>(w[c * K + t]) * win[t];
        acc += static_cast<float>(w[c * K + (K - 1)]) * xin;
        float r = static_cast<float>(static_cast<T>(acc));
        // Match MLX SiLU with input-dtype rounding at each sigmoid/product stage.
        float val;
        {
            T e = static_cast<T>(metal::exp(metal::abs(r)));
            T dd = static_cast<T>(1.0f + static_cast<float>(e));
            T yy = static_cast<T>(1.0f / static_cast<float>(dd));
            T sgm = r < 0.0f ? yy : static_cast<T>(1.0f - static_cast<float>(yy));
            val = static_cast<float>(static_cast<T>(r * static_cast<float>(sgm)));
        }
        for (int t = 0; t < K - 2; ++t) win[t] = win[t + 1];
        win[K - 2] = xin;
        float ss = simd_sum(val * val);
        if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (int s = 0; s < TG / 32; ++s) tot += part[s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool is_q = c < KEYDIM;
        bool is_k = (c >= KEYDIM) && (c < 2 * KEYDIM);
        float out = val;
        if (is_q || is_k) {
            float inv = metal::rsqrt(tot / float(TG) + 1e-6f);
            float normed = static_cast<float>(static_cast<T>(val * inv));
            // MLX converts scalar factors to the activation dtype before multiplying.
            float scale = is_q ? (1.0f / float(DK))
                : float(T(metal::rsqrt(float(DK))));
            out = static_cast<float>(static_cast<T>(normed * scale));
        }
        y[(row * TT + tt) * CK + c] = static_cast<T>(out);
        if (c >= 2 * KEYDIM) {
            uint zi = c - 2 * KEYDIM;
            z[(row * TT + tt) * CV + zi] = pr[CK + zi];
        }
        if (c < HV) {
            float bb = static_cast<float>(pr[CK + CV + c]);
            {   // mx.sigmoid on T: 1/(1+precise::exp(|x|)) with per-op rounding, flipped for x >= 0
                T e = static_cast<T>(metal::precise::exp(metal::abs(bb)));
                T dd = static_cast<T>(1.0f + static_cast<float>(e));
                T yy = static_cast<T>(1.0f / static_cast<float>(dd));
                beta[(row * TT + tt) * HV + c] = bb < 0.0f ? yy : T(1.0f - float(yy));
            }
            float aa = float(T(float(pr[CK + CV + HV + c]) + float(dt_bias[c])));
            // Match MLX LogAddExp's typed operators and overloads, including half exp/log1p.
            T av = T(aa);
            T maximum = metal::max(av, T(0));
            T minimum = metal::min(av, T(0));
            T sp = maximum + log1p(metal::exp(minimum - maximum));
            g[(row * TT + tt) * HV + c] = metal::precise::exp(-metal::precise::exp(A_log[c]) * sp);
        }
    }
    for (int t = 0; t < K - 1; ++t) ns[t * CK + c] = static_cast<T>(win[t]);
""",
    )


def prepare(
    projection: mx.array,
    state: mx.array,
    convolution: mx.array,
    log_rates: mx.array,
    time_bias: mx.array,
    *,
    key_heads: int,
    key_width: int,
    value_heads: int,
    value_width: int,
) -> tuple[mx.array, ...]:
    batch, count, width = projection.shape
    channels = 2 * key_heads * key_width + value_heads * value_width
    value_dim = value_heads * value_width
    if (
        key_width not in (32, 64, 128, 256)
        or channels % key_width
        or value_heads > channels
        or state.shape[0] != batch
        or state.shape[2] != channels
        or state.shape[1] < 1
        or width != channels + value_dim + 2 * value_heads
        or convolution.shape != (channels, state.shape[1] + 1, 1)
        or projection.dtype != state.dtype
        or convolution.dtype != projection.dtype
    ):
        raise ValueError("incompatible packed gated-delta preparation geometry")
    result = _kernel()(
        inputs=[projection, state, convolution, log_rates.astype(mx.float32), time_bias],
        template=[
            ("T", projection.dtype),
            ("CK", channels),
            ("CV", value_dim),
            ("HV", value_heads),
            ("K", state.shape[1] + 1),
            ("KEYDIM", key_heads * key_width),
            ("DK", key_width),
            ("TT", count),
        ],
        grid=(channels, batch, 1),
        threadgroup=(key_width, 1, 1),
        output_shapes=[
            (batch, count, channels),
            state.shape,
            (batch, count, value_dim),
            (batch, count, value_heads),
            (batch, count, value_heads),
        ],
        output_dtypes=[
            projection.dtype,
            state.dtype,
            projection.dtype,
            projection.dtype,
            mx.float32,
        ],
    )
    return tuple(result)
