// The residual streams' collapse on CUDA (`streams.seismic`): a row's S
// streams summed column by column, each weighted by the row's weight for it,
// in stream order. Shared by `stream_collapse` and `stream_mix_sinkhorn`'s
// sublayer input.
//
// This file is independent of any entry ABI.

#pragma once

#include <seismic/element.cuh>

namespace streams {

// Row `x`'s S streams (`stream_stride` apart, columns `column_stride` apart)
// collapsed by the per-stream weights `carried` a thread holds into the h
// columns of `out` (`out_stride` apart): columns first, first + step, ..
// below h. When `MEAN` is true, each collapsed sum is divided by `float(S)`
// before storing.
template <unsigned S, bool MEAN = false>
__device__ __forceinline__ void collapse(const float *x, unsigned long long stream_stride,
                                         unsigned long long column_stride, const float (&carried)[S],
                                         const element::Out<element::F32> &out, unsigned long long out_stride,
                                         unsigned h, unsigned first, unsigned step) {
    typedef unsigned long long u64;
    if (column_stride == 1ull && out_stride == 1ull &&
        ((reinterpret_cast<u64>(x) | reinterpret_cast<u64>(out.stored())) & 15ull) == 0ull &&
        (S == 1u || (stream_stride & 3ull) == 0ull)) {
        const unsigned vecs = h / 4u;
        for (unsigned v = first; v < vecs; v += step) {
            const u64 base = (u64)v * 4ull;
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (unsigned stream = 0; stream < S; ++stream) {
                const float w = carried[stream];
                const float4 vx = *reinterpret_cast<const float4 *>(x + (u64)stream * stream_stride + base);
                sum[0] += w * vx.x;
                sum[1] += w * vx.y;
                sum[2] += w * vx.z;
                sum[3] += w * vx.w;
            }
            if constexpr (MEAN) {
                sum[0] = __fdiv_rn(sum[0], float(S));
                sum[1] = __fdiv_rn(sum[1], float(S));
                sum[2] = __fdiv_rn(sum[2], float(S));
                sum[3] = __fdiv_rn(sum[3], float(S));
            }
            out.put_span<4>(base, sum);
        }
        for (unsigned column = vecs * 4u + first; column < h; column += step) {
            float sum = 0.0f;
#pragma unroll
            for (unsigned stream = 0; stream < S; ++stream)
                sum += carried[stream] * x[(u64)stream * stream_stride + (u64)column];
            if constexpr (MEAN)
                sum = __fdiv_rn(sum, float(S));
            out.put((u64)column, sum);
        }
        return;
    }
    for (unsigned column = first; column < h; column += step) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned stream = 0; stream < S; ++stream)
            sum += carried[stream] * x[(u64)stream * stream_stride + (u64)column * column_stride];
        if constexpr (MEAN)
            sum = __fdiv_rn(sum, float(S));
        out.put((u64)column * out_stride, sum);
    }
}

// As above, by the weights `weights` holds (`weight_stride` apart).
template <unsigned S, bool MEAN = false>
__device__ __forceinline__ void collapse(const float *x, unsigned long long stream_stride,
                                         unsigned long long column_stride, const float *weights,
                                         unsigned long long weight_stride, const element::Out<element::F32> &out,
                                         unsigned long long out_stride, unsigned h, unsigned first, unsigned step) {
    float carried[S];
#pragma unroll
    for (unsigned stream = 0; stream < S; ++stream)
        carried[stream] = weights[(unsigned long long)stream * weight_stride];
    collapse<S, MEAN>(x, stream_stride, column_stride, carried, out, out_stride, h, first, step);
}

}  // namespace streams
