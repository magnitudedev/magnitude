// The residual streams' collapse on Vulkan (`streams.seismic`): a row's S
// streams summed column by column, each weighted by the row's weight for it,
// in stream order. Shared by `stream_collapse` and `stream_mix_sinkhorn`'s
// sublayer input. The counterpart of `cuda/lib/core/streams.cuh`.
//
// This file is independent of any entry ABI. Addresses are bytes; strides are
// in F32 elements.

// The collapse loop of every form, stream s contributing TERM(s, v) (with v a
// `vec4` on the unit-stride aligned fast path and a `float` on the scalar tail
// and strided fallback). When `MEAN` is true, each collapsed sum is divided by
// `float(streams)` via `seismic_div_rn` before storing.
#define STREAMS_COLLAPSE(TERM, MEAN)                                                                            \
    if (column_stride == 1ul && result_stride == 1ul && (x & 15ul) == 0ul                                       \
        && (streams == 1u || (stream_stride & 3ul) == 0ul)) {                                                   \
        const uint vecs = h >> 2u;                                                                              \
        for (uint v = first; v < vecs; v += step) {                                                             \
            const uint64_t base = uint64_t(v) * 4ul;                                                            \
            vec4 sum = vec4(0.0);                                                                               \
            for (uint stream = 0u; stream < streams; ++stream) {                                                \
                const vec4 vx = element_vec4_at(x + (uint64_t(stream) * stream_stride + base) * 4ul);           \
                sum += TERM(stream, vx);                                                                        \
            }                                                                                                   \
            if (MEAN) {                                                                                         \
                const float denom = float(streams);                                                             \
                sum = vec4(seismic_div_rn(sum.x, denom), seismic_div_rn(sum.y, denom),                          \
                    seismic_div_rn(sum.z, denom), seismic_div_rn(sum.w, denom));                                \
            }                                                                                                   \
            out_put2(result, base, sum.x, sum.y);                                                               \
            out_put2(result, base + 2ul, sum.z, sum.w);                                                         \
        }                                                                                                       \
        for (uint column = (vecs << 2u) + first; column < h; column += step) {                                  \
            float sum = 0.0;                                                                                    \
            for (uint stream = 0u; stream < streams; ++stream) {                                                \
                const float vx = element_f32_at(x + (uint64_t(stream) * stream_stride + uint64_t(column)) * 4ul); \
                sum += TERM(stream, vx);                                                                        \
            }                                                                                                   \
            if (MEAN)                                                                                           \
                sum = seismic_div_rn(sum, float(streams));                                                      \
            out_put(result, uint64_t(column), sum);                                                             \
        }                                                                                                       \
        return;                                                                                                 \
    }                                                                                                           \
    for (uint column = first; column < h; column += step) {                                                     \
        float sum = 0.0;                                                                                        \
        for (uint stream = 0u; stream < streams; ++stream) {                                                    \
            const float vx = element_f32_at(                                                                    \
                x + (uint64_t(stream) * stream_stride + uint64_t(column) * column_stride) * 4ul);               \
            sum += TERM(stream, vx);                                                                            \
        }                                                                                                       \
        if (MEAN)                                                                                               \
            sum = seismic_div_rn(sum, float(streams));                                                          \
        out_put(result, uint64_t(column) * result_stride, sum);                                                 \
    }

// Row `x`'s `streams` streams (`stream_stride` apart, columns `column_stride`
// apart) collapsed by the per-stream weights `carried` an invocation holds (to
// 8 streams) into the h columns of `result` (`result_stride` apart), the
// invocations taking every `step`-th column from `first`.
void streams_collapse(uint64_t x, uint64_t stream_stride, uint64_t column_stride, const float carried[8],
    Out result, uint64_t result_stride, const uint streams, uint h, uint first, uint step) {
#define STREAMS_CARRIED_TERM(s, v) (carried[s] * (v))
    STREAMS_COLLAPSE(STREAMS_CARRIED_TERM, false)
#undef STREAMS_CARRIED_TERM
}

// As above, by the weights `weights` holds (`weight_stride` apart): hoists
// weights into registers for up to 8 streams, falling back to memory loads
// above 8.
void streams_collapse(uint64_t x, uint64_t stream_stride, uint64_t column_stride, uint64_t weights,
    uint64_t weight_stride, Out result, uint64_t result_stride, const uint streams, uint h, uint first,
    uint step) {
    if (streams <= 8u) {
        float carried[8];
        for (uint stream = 0u; stream < streams; ++stream)
            carried[stream] = element_f32_at(weights + uint64_t(stream) * weight_stride * 4ul);
        streams_collapse(x, stream_stride, column_stride, carried, result, result_stride, streams, h, first, step);
        return;
    }
#define STREAMS_STORED_TERM(s, v) (element_f32_at(weights + uint64_t(s) * weight_stride * 4ul) * (v))
    STREAMS_COLLAPSE(STREAMS_STORED_TERM, false)
#undef STREAMS_STORED_TERM
}

// As above, every stream at weight 1 divided by `streams`: the streams' mean
// in stream order, fused into one pass.
void streams_mean(uint64_t x, uint64_t stream_stride, uint64_t column_stride, Out result,
    uint64_t result_stride, const uint streams, uint h, uint first, uint step) {
#define STREAMS_UNIT_TERM(s, v) (v)
    STREAMS_COLLAPSE(STREAMS_UNIT_TERM, true)
#undef STREAMS_UNIT_TERM
}

#undef STREAMS_COLLAPSE
