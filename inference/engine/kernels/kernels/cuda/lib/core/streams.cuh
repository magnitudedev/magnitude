// The residual streams' collapse on CUDA (`streams.seismic`): a row's S
// streams summed column by column, each weighted by the row's weight for it,
// in stream order. Shared by `stream_collapse` and `stream_mix_sinkhorn`'s
// sublayer input.
//
// This file is independent of any entry ABI.

#pragma once

namespace streams {

// Row `x`'s S streams (`stream_stride` apart, columns `column_stride` apart)
// collapsed by the per-stream weights `carried` a thread holds into the h
// columns of `out` (`out_stride` apart): columns first, first + step, ..
// below h.
template <unsigned S>
__device__ __forceinline__ void collapse(const float *x, unsigned long long stream_stride,
                                         unsigned long long column_stride, const float (&carried)[S], float *out,
                                         unsigned long long out_stride, unsigned h, unsigned first, unsigned step) {
    typedef unsigned long long u64;
    for (unsigned column = first; column < h; column += step) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned stream = 0; stream < S; ++stream)
            sum += carried[stream] * x[(u64)stream * stream_stride + (u64)column * column_stride];
        out[(u64)column * out_stride] = sum;
    }
}

// As above, by the weights `weights` holds (`weight_stride` apart).
template <unsigned S>
__device__ __forceinline__ void collapse(const float *x, unsigned long long stream_stride,
                                         unsigned long long column_stride, const float *weights,
                                         unsigned long long weight_stride, float *out, unsigned long long out_stride,
                                         unsigned h, unsigned first, unsigned step) {
    float carried[S];
#pragma unroll
    for (unsigned stream = 0; stream < S; ++stream)
        carried[stream] = weights[(unsigned long long)stream * weight_stride];
    collapse<S>(x, stream_stride, column_stride, carried, out, out_stride, h, first, step);
}

}  // namespace streams
