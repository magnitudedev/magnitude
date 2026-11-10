// The output entries' joins of an F32 output into the hyper-connection
// streams (`scatter`, functions.seismic): the inject scatter and, when the
// inject row carries one, its comb. Apart from the projection family so an
// entry with no activation element or weights (`stream_join`) joins through
// the same code. The counterpart of `cuda/lib/projection/join.cuh`.
//
// This file is independent of any entry ABI. Addresses are bytes; strides are
// in F32 elements.

// Column n of one row's `streams` streams of `width` columns: each target
// stream d of `result` (columns `result_stride` apart) the residual stream d of
// `residual` (columns `residual_stride` apart) plus y inject[d], or, `combed`,
// the residual streams mixed by the comb after the inject values
// (sum_s comb[s, d] residual[s], sources ascending, the comb source major)
// plus y inject[d]. `inject` is the row's inject values, contiguous.
void join_scatter(uint64_t result, uint64_t result_stride, uint64_t residual, uint64_t residual_stride, uint64_t inject,
    uint streams, uint width, const bool combed, uint64_t n, float y) {
    const uint64_t comb = inject + uint64_t(streams) * 4ul;
    for (uint d = 0u; d < streams; ++d) {
        float mixed;
        if (combed) {
            mixed = 0.0;
            for (uint s = 0u; s < streams; ++s)
                mixed = mixed + element_f32_at(comb + uint64_t(s * streams + d) * 4ul)
                    * element_f32_at(residual + (uint64_t(s * width) + n) * residual_stride * 4ul);
        } else {
            mixed = element_f32_at(residual + (uint64_t(d * width) + n) * residual_stride * 4ul);
        }
        element_f32_put(result + (uint64_t(d * width) + n) * result_stride * 4ul,
            mixed + y * element_f32_at(inject + uint64_t(d) * 4ul));
    }
}
