// state_space_gate (contract and portable body in state_space.seismic): the
// state-space output rows, y * SiLU(z) RMS-normalized over each group of U
// heads and scaled by the state norm. The CUDA form of
// `metal/state_space_gate.metal`: a block of 256 threads per (row, group),
// `vec4` lane chains on canonical strides, `reduce::group_sum`, and
// `row_norm` RMS normalization.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"
#include "lib/core/norm.cuh"
#include "lib/core/reduce.cuh"

namespace state_space_gate_detail {

using activation::Act;
using element::u32;
using element::u64;

__device__ __forceinline__ float silu_mul(float value, float gate) {
    return seismic_mul_rn(value, gate / (1.0f + expf(-gate)));
}

// The gated value of `column` within `(row, group)` (head = group * U + column / P).
__device__ __forceinline__ float gated_value(const element::u8 *mixed, const element::u8 *projection, u64 mixed_row,
                                             u64 proj_row, u32 head_base, u32 first, u32 p_dim, u32 column,
                                             const seismic_words_t &seismic_words_value) {
    const u32 local = column / p_dim;
    const u32 p = column - local * p_dim;
    const u32 head = head_base + local;
    const u32 channel = first + column;
    const float gate =
        element::at<Act>(projection, proj_row + static_cast<u64>(channel) * SEISMIC_PROJECTION_STRIDE_1);
    const float value = element::at<Act>(
        mixed, mixed_row + static_cast<u64>(head) * SEISMIC_MIXED_STRIDE_1 + static_cast<u64>(p) * SEISMIC_MIXED_STRIDE_2);
    return silu_mul(value, gate);
}

// Four contiguous gated values starting at `column` (canonical strides, aligned).
__device__ __forceinline__ float4 gated_value4(const element::u8 *mixed, const element::u8 *projection, u64 mixed_base,
                                               u64 proj_base, u32 column) {
    float g[4], v[4];
    element::span<Act, 4, false>(projection, proj_base + column, g);
    element::span<Act, 4, false>(mixed, mixed_base + column, v);
    return make_float4(silu_mul(v[0], g[0]), silu_mul(v[1], g[1]), silu_mul(v[2], g[2]), silu_mul(v[3], g[3]));
}

} // namespace state_space_gate_detail

extern "C" __global__ void state_space_gate(SEISMIC_KERNEL_PARAMS) {
    using namespace state_space_gate_detail;
    const element::u8 *mixed = SEISMIC_PTR(SEISMIC_BUFFER_MIXED);
    const element::u8 *projection = SEISMIC_PTR(SEISMIC_BUFFER_PROJECTION);
    const float *state_norm = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_STATE_NORM));
    const element::Out<Act> normalized = SEISMIC_RESULT_0_OUT;
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    const u32 group = blockIdx.y;
    const u32 u_dim = static_cast<u32>(SEISMIC_DIM_U);
    const u32 p_dim = static_cast<u32>(SEISMIC_DIM_P);
    const u32 width = u_dim * p_dim;
    const u32 vec_width = width & ~3u;
    const u32 head_base = group * u_dim;
    const u32 first = group * width;
    const u32 tid = threadIdx.x;
    const u32 threads = blockDim.x;
    const float epsilon = element::word_f32(SEISMIC_PARAM_EPSILON);
    const u64 mixed_row = row * SEISMIC_MIXED_STRIDE_0;
    const u64 proj_row = row * SEISMIC_PROJECTION_STRIDE_0;
    const u64 mixed_base = mixed_row + first;
    const u64 proj_base = proj_row + first;
    constexpr u32 act_align = 4u * Act::bytes - 1u;
    const bool vec_in = SEISMIC_PROJECTION_STRIDE_1 == 1 && SEISMIC_MIXED_STRIDE_1 == SEISMIC_DIM_P
                        && SEISMIC_MIXED_STRIDE_2 == 1
                        && (((reinterpret_cast<u64>(mixed + mixed_base * Act::bytes)
                              | reinterpret_cast<u64>(projection + proj_base * Act::bytes))
                             & act_align)
                            == 0u);
    constexpr u32 MAX_CACHE = 4u;
    const u32 vec_stride = 4u * threads;
    float4 cached[MAX_CACHE];
    float cached_tail = 0.0f;
    float squares = 0.0f;
    if (vec_in) {
        float4 sq4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (u32 k = 0u; k < MAX_CACHE; ++k) {
            const u32 column = 4u * tid + k * vec_stride;
            if (column < vec_width) {
                const float4 value = gated_value4(mixed, projection, mixed_base, proj_base, column);
                cached[k] = value;
                sq4 = row_norm::square4(value, sq4);
            }
        }
        for (u32 column = 4u * tid + MAX_CACHE * vec_stride; column < vec_width; column += vec_stride)
            sq4 = row_norm::square4(gated_value4(mixed, projection, mixed_base, proj_base, column), sq4);
        squares = row_norm::sum4(sq4);
        u32 tail_idx = 0u;
        for (u32 column = vec_width + tid; column < width; column += threads, ++tail_idx) {
            const float value =
                silu_mul(element::at<Act>(mixed, mixed_base + column), element::at<Act>(projection, proj_base + column));
            if (tail_idx == 0u) cached_tail = value;
            squares = row_norm::square(value, squares);
        }
    } else {
        for (u32 column = tid; column < width; column += threads) {
            const float value =
                gated_value(mixed, projection, mixed_row, proj_row, head_base, first, p_dim, column, seismic_words_value);
            squares = row_norm::square(value, squares);
        }
    }
    const float inverse = row_norm::inverse(reduce::group_sum(squares, partials), width, epsilon);
    const float *norm_group = state_norm + static_cast<u64>(group) * SEISMIC_STATE_NORM_STRIDE_0;
    const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<Act> out_group = normalized.offset(out_row + first);
    const bool vec_out = vec_in && SEISMIC_STATE_NORM_STRIDE_1 == SEISMIC_DIM_P && SEISMIC_STATE_NORM_STRIDE_2 == 1
                         && SEISMIC_RESULT_0_STRIDE_1 == SEISMIC_DIM_P && SEISMIC_RESULT_0_STRIDE_2 == 1
                         && ((reinterpret_cast<u64>(norm_group) & 15u) == 0u)
                         && ((reinterpret_cast<u64>(out_group.stored(0)) & act_align) == 0u);
    if (vec_out) {
        const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (u32 k = 0u; k < MAX_CACHE; ++k) {
            const u32 column = 4u * tid + k * vec_stride;
            if (column < vec_width) {
                const float4 weight = *reinterpret_cast<const float4 *>(norm_group + column);
                const float4 scaled = row_norm::apply4<true, false>(cached[k], inverse, weight, zero);
                const float out[4] = {scaled.x, scaled.y, scaled.z, scaled.w};
                out_group.put_span<4>(column, out);
            }
        }
        for (u32 column = 4u * tid + MAX_CACHE * vec_stride; column < vec_width; column += vec_stride) {
            const float4 weight = *reinterpret_cast<const float4 *>(norm_group + column);
            const float4 value = gated_value4(mixed, projection, mixed_base, proj_base, column);
            const float4 scaled = row_norm::apply4<true, false>(value, inverse, weight, zero);
            const float out[4] = {scaled.x, scaled.y, scaled.z, scaled.w};
            out_group.put_span<4>(column, out);
        }
        u32 tail_idx = 0u;
        for (u32 column = vec_width + tid; column < width; column += threads, ++tail_idx) {
            const float value = (tail_idx == 0u)
                ? cached_tail
                : silu_mul(element::at<Act>(mixed, mixed_base + column), element::at<Act>(projection, proj_base + column));
            out_group.put(column, row_norm::apply<true, false>(value, inverse, norm_group[column], 0.0f));
        }
    } else {
        for (u32 column = tid; column < width; column += threads) {
            const u32 local = column / p_dim;
            const u32 p = column - local * p_dim;
            const u32 head = head_base + local;
            const float weight = norm_group[static_cast<u64>(local) * SEISMIC_STATE_NORM_STRIDE_1
                                            + static_cast<u64>(p) * SEISMIC_STATE_NORM_STRIDE_2];
            const float value =
                gated_value(mixed, projection, mixed_row, proj_row, head_base, first, p_dim, column, seismic_words_value);
            normalized.put(out_row + static_cast<u64>(head) * SEISMIC_RESULT_0_STRIDE_1
                               + static_cast<u64>(p) * SEISMIC_RESULT_0_STRIDE_2,
                           row_norm::apply<true, false>(value, inverse, weight, 0.0f));
        }
    }
}
