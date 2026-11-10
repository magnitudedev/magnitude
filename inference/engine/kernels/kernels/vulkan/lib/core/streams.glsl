// The residual streams' collapse on Vulkan (`streams.seismic`): a row's S
// streams summed column by column, each weighted by the row's weight for it,
// in stream order. Shared by `stream_collapse` and `stream_mix_sinkhorn`'s
// sublayer input. The counterpart of `cuda/lib/core/streams.cuh`.
//
// This file is independent of any entry ABI. Addresses are bytes; strides are
// in F32 elements.

// Row `x`'s `streams` streams (`stream_stride` apart, columns
// `column_stride` apart) collapsed by `weights` (`weight_stride` apart) into
// the h columns of `result` (`result_stride` apart), the invocations taking every
// `step`-th column from `first`.
void streams_collapse(uint64_t x, uint64_t stream_stride, uint64_t column_stride, uint64_t weights,
    uint64_t weight_stride, uint64_t result, uint64_t result_stride, const uint streams, uint h, uint first,
    uint step) {
    for (uint column = first; column < h; column += step) {
        float sum = 0.0;
        for (uint stream = 0u; stream < streams; ++stream)
            sum += element_f32_at(weights + uint64_t(stream) * weight_stride * 4ul)
                * element_f32_at(x + (uint64_t(stream) * stream_stride + uint64_t(column) * column_stride) * 4ul);
        element_f32_put(result + uint64_t(column) * result_stride * 4ul, sum);
    }
}
