// state_space_gate (contract and portable body in state_space.seismic): the
// state-space output rows, y * SiLU(z) RMS-normalized over each group of U
// heads and scaled by the state norm. A threadgroup of 256 threads owns (row,
// group): `vec4` lane chains on canonical strides, `reduce::group_sum`, and
// `row_norm` RMS normalization (gated vec4 values register-cached across the
// two passes for widths up to 4096).

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"
#include "lib/core/norm.h"
#include "lib/core/reduce.h"

namespace state_space_gate_detail {

typedef activation::Act Act;

inline float silu_mul(float value, float gate) {
    return value * (gate / (1.0f + metal::exp(-gate)));
}

inline float4 silu_mul4(float4 value, float4 gate) {
    return value * (gate / (1.0f + metal::exp(-gate)));
}

// The gated value of `column` within `(row, group)` (head = group * U + column / P).
inline float gated_value(device const Act::storage *mixed, device const Act::storage *projection, ulong mixed_row,
    ulong proj_row, uint head_base, uint first, uint p_dim, uint column, constant ulong *seismic_words) {
    const uint local = column / p_dim;
    const uint p = column - local * p_dim;
    const uint head = head_base + local;
    const uint channel = first + column;
    const float gate = Act::load(projection[proj_row + ulong(channel) * SEISMIC_PROJECTION_STRIDE_1]);
    const float value =
        Act::load(mixed[mixed_row + ulong(head) * SEISMIC_MIXED_STRIDE_1 + ulong(p) * SEISMIC_MIXED_STRIDE_2]);
    return silu_mul(value, gate);
}

} // namespace state_space_gate_detail

kernel void state_space_gate(
    device const activation::Act::storage *mixed [[buffer(SEISMIC_BUFFER_MIXED)]],
    device const activation::Act::storage *projection [[buffer(SEISMIC_BUFFER_PROJECTION)]],
    device const float *state_norm [[buffer(SEISMIC_BUFFER_STATE_NORM)]],
    SEISMIC_RESULT_0_PARAM(normalized),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    using namespace state_space_gate_detail;
    threadgroup float partials[8];
    const ulong row = group.x;
    const uint g = group.y;
    const uint u_dim = uint(SEISMIC_DIM_U);
    const uint p_dim = uint(SEISMIC_DIM_P);
    const uint width = u_dim * p_dim;
    const uint vec_width = width & ~3u;
    const uint head_base = g * u_dim;
    const uint first = g * width;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const ulong mixed_row = row * SEISMIC_MIXED_STRIDE_0;
    const ulong proj_row = row * SEISMIC_PROJECTION_STRIDE_0;
    device const Act::storage *mixed_group = mixed + mixed_row + first;
    device const Act::storage *proj_group = projection + proj_row + first;
    const bool vec_in =
        SEISMIC_PROJECTION_STRIDE_1 == 1 && SEISMIC_MIXED_STRIDE_1 == SEISMIC_DIM_P && SEISMIC_MIXED_STRIDE_2 == 1;
    constexpr uint MAX_CACHE = 4u;
    constexpr uint vec_stride = 4u * 256u;
    float4 cached[MAX_CACHE];
    float cached_tail = 0.0f;
    const uint tail_col = vec_width + thread_index;
    float squares = 0.0f;
    if (vec_in) {
        float4 sq4 = float4(0.0f);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0u; k < MAX_CACHE; ++k) {
            const uint column = 4u * thread_index + k * vec_stride;
            if (column < vec_width) {
                const float4 value = silu_mul4(Act::load4(mixed_group + column), Act::load4(proj_group + column));
                cached[k] = value;
                sq4 = row_norm::square4(value, sq4);
            }
        }
        for (uint column = 4u * thread_index + MAX_CACHE * vec_stride; column < vec_width; column += vec_stride) {
            const float4 value = silu_mul4(Act::load4(mixed_group + column), Act::load4(proj_group + column));
            sq4 = row_norm::square4(value, sq4);
        }
        squares = row_norm::sum4(sq4);
        if (tail_col < width) {
            cached_tail = silu_mul(Act::load(mixed_group[tail_col]), Act::load(proj_group[tail_col]));
            squares = row_norm::square(cached_tail, squares);
        }
    } else {
        for (uint column = thread_index; column < width; column += 256u) {
            const float value =
                gated_value(mixed, projection, mixed_row, proj_row, head_base, first, p_dim, column, seismic_words);
            squares = row_norm::square(value, squares);
        }
    }
    const float inverse = row_norm::inverse(reduce::group_sum<8>(squares, partials, simdgroup, lane), width, epsilon);
    device const float *norm_group = state_norm + ulong(g) * SEISMIC_STATE_NORM_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<Act> out(normalized);
    const bool vec_out = vec_in && SEISMIC_STATE_NORM_STRIDE_1 == SEISMIC_DIM_P && SEISMIC_STATE_NORM_STRIDE_2 == 1
        && SEISMIC_RESULT_0_STRIDE_1 == SEISMIC_DIM_P && SEISMIC_RESULT_0_STRIDE_2 == 1;
    if (vec_out) {
        const element::Out<Act> out_group = out.offset(out_row + first);
        _Pragma("clang loop unroll(full)")
        for (uint k = 0u; k < MAX_CACHE; ++k) {
            const uint column = 4u * thread_index + k * vec_stride;
            if (column < vec_width) {
                const float4 weight = float4(*reinterpret_cast<device const packed_float4 *>(norm_group + column));
                out_group.put4(column, row_norm::apply4<true, false>(cached[k], inverse, weight, float4(0.0f)));
            }
        }
        for (uint column = 4u * thread_index + MAX_CACHE * vec_stride; column < vec_width; column += vec_stride) {
            const float4 weight = float4(*reinterpret_cast<device const packed_float4 *>(norm_group + column));
            const float4 value = silu_mul4(Act::load4(mixed_group + column), Act::load4(proj_group + column));
            out_group.put4(column, row_norm::apply4<true, false>(value, inverse, weight, float4(0.0f)));
        }
        if (tail_col < width) {
            out_group.put(tail_col, row_norm::apply<true, false>(cached_tail, inverse, norm_group[tail_col], 0.0f));
        }
    } else {
        for (uint column = thread_index; column < width; column += 256u) {
            const uint local = column / p_dim;
            const uint p = column - local * p_dim;
            const uint head = head_base + local;
            const float weight = norm_group[ulong(local) * SEISMIC_STATE_NORM_STRIDE_1 + ulong(p) * SEISMIC_STATE_NORM_STRIDE_2];
            const float value =
                gated_value(mixed, projection, mixed_row, proj_row, head_base, first, p_dim, column, seismic_words);
            out.put(out_row + ulong(head) * SEISMIC_RESULT_0_STRIDE_1 + ulong(p) * SEISMIC_RESULT_0_STRIDE_2,
                row_norm::apply<true, false>(value, inverse, weight, 0.0f));
        }
    }
}
