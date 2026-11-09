// A head row held by a warp: its RMS norm (the portable `head_norm`), shared
// by the attention library, the pooled rows and the indexers. Its rotation is
// `rotary::layout`'s, its rounding `rounding::round_lanes`.
//
// This file is independent of any entry ABI.

#pragma once

namespace head {

// A head row of 32 E columns over a warp, lane l holding columns
// [l E, (l + 1) E), RMS-normalized with `weight`: x * inv * weight,
// inv = rsqrt(sum(x^2) / W + epsilon). `weight` steps by `weight_stride`. The
// whole warp calls it.
template <int E>
__device__ __forceinline__ void rms_norm(float (&x)[E], const float *weight, unsigned long long weight_stride,
                                         float epsilon, int lane) {
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < E; ++i) squares = __fmaf_rn(x[i], x[i], squares);
    squares = seismic_warp_sum_f32(squares);
    const float inverse = rsqrtf(squares / static_cast<float>(32 * E) + epsilon);
#pragma unroll
    for (int i = 0; i < E; ++i)
        x[i] = x[i] * inverse * weight[static_cast<unsigned long long>(lane * E + i) * weight_stride];
}

}  // namespace head
