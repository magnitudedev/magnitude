// vision_pool: four contiguous columns per thread on a 2-D (column, row) grid:
// the G member rows times `weight`, summed in order, times `scale`, then
// standardized with the NS rows.
#include <seismic/element.cuh>

using element::u64;
using element::u8;
typedef ELEMENT_OF(SEISMIC_STANDARD_BIAS) StandardBias;
typedef ELEMENT_OF(SEISMIC_STANDARD_SCALE) StandardScale;

extern "C" __global__ void vision_pool(SEISMIC_KERNEL_PARAMS) {
    const u64 columns = SEISMIC_DIM_H;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= columns)
        return;
    const u64 limit = col + 4ull < columns ? col + 4ull : columns;
    const u64 count = limit - col;
    const float *source = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    const auto result = SEISMIC_RESULT_0_OUT;
    const float weight = __uint_as_float((unsigned)SEISMIC_PARAM_WEIGHT);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    float sb[4];
    float ss[4];
    if (SEISMIC_DIM_NS == 1) {
        const u8 *standard_bias = SEISMIC_PTR(SEISMIC_BUFFER_STANDARD_BIAS);
        const u8 *standard_scale = SEISMIC_PTR(SEISMIC_BUFFER_STANDARD_SCALE);
        constexpr u64 sb_align_mask = 4ull * StandardBias::bytes - 1ull;
        constexpr u64 ss_align_mask = 4ull * StandardScale::bytes - 1ull;
        if (count == 4ull && SEISMIC_STANDARD_BIAS_STRIDE_1 == 1ull &&
            (reinterpret_cast<u64>(standard_bias) & sb_align_mask) == 0ull) {
            element::span<StandardBias, 4, false>(standard_bias, col, sb);
        } else {
            for (u64 k = 0; k < count; ++k)
                sb[k] = element::at<StandardBias>(standard_bias, (col + k) * SEISMIC_STANDARD_BIAS_STRIDE_1);
        }
        if (count == 4ull && SEISMIC_STANDARD_SCALE_STRIDE_1 == 1ull &&
            (reinterpret_cast<u64>(standard_scale) & ss_align_mask) == 0ull) {
            element::span<StandardScale, 4, false>(standard_scale, col, ss);
        } else {
            for (u64 k = 0; k < count; ++k)
                ss[k] = element::at<StandardScale>(standard_scale, (col + k) * SEISMIC_STANDARD_SCALE_STRIDE_1);
        }
    }
    const bool member_aligned = SEISMIC_DIM_G <= 1 || (SEISMIC_SOURCE_STRIDE_1 & 3ull) == 0ull;
    const bool full_vec = count == 4ull &&
                          SEISMIC_SOURCE_STRIDE_2 == 1ull &&
                          SEISMIC_RESULT_0_STRIDE_1 == 1ull &&
                          member_aligned;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 src_row = row * SEISMIC_SOURCE_STRIDE_0;
        const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
        const bool aligned = ((reinterpret_cast<u64>(source + src_row) |
                               reinterpret_cast<u64>(result.stored(out_row))) & 15ull) == 0ull;
        if (full_vec && aligned) {
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u64 member = 0; member < SEISMIC_DIM_G; ++member) {
                const float4 sv = *reinterpret_cast<const float4 *>(
                    source + src_row + member * SEISMIC_SOURCE_STRIDE_1 + col);
                sum[0] = __fadd_rn(sum[0], __fmul_rn(sv.x, weight));
                sum[1] = __fadd_rn(sum[1], __fmul_rn(sv.y, weight));
                sum[2] = __fadd_rn(sum[2], __fmul_rn(sv.z, weight));
                sum[3] = __fadd_rn(sum[3], __fmul_rn(sv.w, weight));
            }
            float out[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float value = sum[k] * scale;
                if (SEISMIC_DIM_NS == 1)
                    value = (value - sb[k]) * ss[k];
                out[k] = value;
            }
            result.put_span<4>(out_row + col, out);
        } else {
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u64 member = 0; member < SEISMIC_DIM_G; ++member) {
                const u64 member_row = src_row + member * SEISMIC_SOURCE_STRIDE_1;
                for (u64 k = 0; k < count; ++k)
                    sum[k] = __fadd_rn(sum[k], __fmul_rn(source[member_row + (col + k) * SEISMIC_SOURCE_STRIDE_2], weight));
            }
            for (u64 k = 0; k < count; ++k) {
                float value = sum[k] * scale;
                if (SEISMIC_DIM_NS == 1)
                    value = (value - sb[k]) * ss[k];
                result.put(out_row + (col + k) * SEISMIC_RESULT_0_STRIDE_1, value);
            }
        }
    }
}
