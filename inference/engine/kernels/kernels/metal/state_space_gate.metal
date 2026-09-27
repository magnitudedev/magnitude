// state_space_gate (contract and portable body in state_space.seismic): the
// state-space output rows, y * SiLU(z) RMS-normalized over each group of U
// heads and scaled by the state norm. A threadgroup of 256 threads owns (row,
// group): each thread sums the squares of its strided share of the group's
// gated values, a simdgroup reduction and the eight simdgroup sums in order
// give the group's sum, and the threads write their share of normalized
// values (the gate recomputed with the same arithmetic).

#include "lib/core/activation.h"

// The gated value of `channel` (in the row's flat head-major order).
inline float gated_value(device const element::Act::storage *mixed, device const element::Act::storage *projection,
    ulong row, ulong channel, constant ulong *seismic_words) {
    const ulong head = channel / SEISMIC_DIM_P;
    const float gate = element::Act::load(projection[row * SEISMIC_PROJECTION_STRIDE_0
        + channel * SEISMIC_PROJECTION_STRIDE_1]);
    return element::Act::load(mixed[row * SEISMIC_MIXED_STRIDE_0 + head * SEISMIC_MIXED_STRIDE_1
               + (channel % SEISMIC_DIM_P) * SEISMIC_MIXED_STRIDE_2])
        * (gate / (1.0f + metal::exp(-gate)));
}

kernel void state_space_gate(
    device const element::Act::storage *mixed [[buffer(SEISMIC_BUFFER_MIXED)]],
    device const element::Act::storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *state_norm [[buffer(SEISMIC_BUFFER_STATE_NORM)]],
    device element::Act::storage *normalized [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[32];
    const ulong row = group.x;
    const ulong width = SEISMIC_DIM_U * SEISMIC_DIM_P;
    const ulong first = ulong(group.y) * width;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    float squares = 0.0f;
    for (ulong column = thread_index; column < width; column += threadgroup_shape.x) {
        const float value = gated_value(mixed, projection, row, first + column, seismic_words);
        squares = metal::fma(value, value, squares);
    }
    const float total = simd_sum(squares);
    if (lane == 0)
        partials[simdgroup] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint index = 0; index < simdgroups; ++index)
        sum += partials[index];
    const float inverse = metal::rsqrt(sum / float(width) + epsilon);
    for (ulong column = thread_index; column < width; column += threadgroup_shape.x) {
        const ulong channel = first + column;
        const ulong local = column / SEISMIC_DIM_P;
        const float weight = state_norm[ulong(group.y) * SEISMIC_STATE_NORM_STRIDE_0 + local * SEISMIC_STATE_NORM_STRIDE_1
            + (column % SEISMIC_DIM_P) * SEISMIC_STATE_NORM_STRIDE_2];
        const float value = gated_value(mixed, projection, row, channel, seismic_words);
        normalized[row * SEISMIC_RESULT_0_STRIDE_0 + (channel / SEISMIC_DIM_P) * SEISMIC_RESULT_0_STRIDE_1
            + (channel % SEISMIC_DIM_P) * SEISMIC_RESULT_0_STRIDE_2] = element::Act::store(value * inverse * weight);
    }
}
