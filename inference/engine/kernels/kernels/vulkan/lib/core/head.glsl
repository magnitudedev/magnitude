// A head row held by a logical subgroup: its RMS norm (the portable
// `head_norm`), shared by the pooled rows and the indexers. Its rotation is
// `rotary_lane_rotate`'s, its rounding `rounding_lanes`'s. The counterpart
// of `metal/lib/core/head.h` and `cuda/lib/core/head.cuh`.
//
// This file is independent of any entry ABI. Lane l holds columns
// [l E, (l + 1) E), E = `LANE_E`, which the including entry defines.

// The row RMS-normalized with `weight` (F32, stepping by `weight_stride`
// elements): x * inv * weight, inv = rsqrt(sum(x^2) / W + epsilon). The whole
// logical subgroup calls it.
void head_rms_norm(inout float x[LANE_E], uint64_t weight, uint64_t weight_stride, float epsilon, uint lane) {
    float squares = 0.0;
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        squares = seismic_fma_rn(x[i], x[i], squares);
    squares = seismic_subgroup_sum_f32(squares);
    const float inverse = inversesqrt(seismic_div_rn(squares, float(32u * LANE_E)) + epsilon);
    [[unroll]] for (uint i = 0u; i < LANE_E; ++i)
        x[i] = x[i] * inverse * element_f32_at(weight + uint64_t(lane * LANE_E + i) * weight_stride * 4ul);
}
