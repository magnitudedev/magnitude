// The residual streams' collapse on CUDA (`streams.seismic`): a row's S
// streams summed column by column, each weighted by the row's weight for it,
// in stream order. Shared by `stream_collapse` and `stream_mix_sinkhorn`'s
// sublayer input.
//
// This file is independent of any entry ABI.

#pragma once

namespace streams {

// Row `x`'s S streams (`stream_stride` apart, columns `column_stride` apart)
// collapsed by `weights` (`weight_stride` apart) into the h columns of `out`
// (`out_stride` apart), the block's threads taking every `step`-th column.
template <unsigned S>
__device__ __forceinline__ void collapse(const float *x, unsigned long long stream_stride,
                                         unsigned long long column_stride, const float *weights,
                                         unsigned long long weight_stride, float *out, unsigned long long out_stride,
                                         unsigned h, unsigned step) {
    typedef unsigned long long u64;
    float carried[S];
#pragma unroll
    for (unsigned stream = 0; stream < S; ++stream)
        carried[stream] = weights[(u64)stream * weight_stride];
    for (unsigned column = threadIdx.x; column < h; column += step) {
        float sum = 0.0f;
#pragma unroll
        for (unsigned stream = 0; stream < S; ++stream)
            sum += carried[stream] * x[(u64)stream * stream_stride + (u64)column * column_stride];
        out[(u64)column * out_stride] = sum;
    }
}

}  // namespace streams
