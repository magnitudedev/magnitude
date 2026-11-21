// draft_convolve_residual: DFlash2's half-1 grouped dynamic causal convolution
// of a sublayer's F32 output rows within each draft block, plus the residual;
// 4 columns per thread on a (column, row) grid (y capped at 65535 and strided),
// summing taps in offset order.
#include <seismic/element.cuh>

extern "C" __global__ void draft_convolve_residual(SEISMIC_KERNEL_PARAMS) {
    using element::u32;
    using element::u64;

    const u64 channels = SEISMIC_DIM_C;
    const u64 width = SEISMIC_DIM_G * channels;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= width)
        return;

    const float *residual = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RESIDUAL));
    const float *output = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUTPUT));
    const float *dynamic = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_DYNAMIC));
    const float *base = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE));
    const auto result = SEISMIC_RESULT_0_OUT;

    const u32 channels32 = static_cast<u32>(channels);
    const u32 col32 = static_cast<u32>(col);
    const u32 group = col32 / channels32;
    const u32 lane = col32 - group * channels32;
    const u32 block = static_cast<u32>(SEISMIC_PARAM_BLOCK);
    const u32 k_dim = static_cast<u32>(SEISMIC_DIM_K);

    const float *base1 = base + SEISMIC_BASE_STRIDE_0;
    const bool vec4 =
        col + 4ull <= width &&
        lane + 4u <= channels32 &&
        SEISMIC_RESIDUAL_STRIDE_1 == 1ull &&
        (SEISMIC_RESIDUAL_STRIDE_0 & 3ull) == 0ull &&
        (reinterpret_cast<u64>(residual) & 15ull) == 0ull &&
        SEISMIC_OUTPUT_STRIDE_1 == 1ull &&
        (SEISMIC_OUTPUT_STRIDE_0 & 3ull) == 0ull &&
        (reinterpret_cast<u64>(output) & 15ull) == 0ull &&
        SEISMIC_BASE_STRIDE_2 == 1ull &&
        (SEISMIC_BASE_STRIDE_1 & 3ull) == 0ull &&
        (reinterpret_cast<u64>(base1) & 15ull) == 0ull &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ull &&
        (SEISMIC_RESULT_0_STRIDE_0 & 3ull) == 0ull &&
        (reinterpret_cast<u64>(result.stored(0)) & 15ull) == 0ull;

    if (vec4) {
        const float *base_col = base1 + col;
        const float *dyn_group = dynamic + SEISMIC_DYNAMIC_STRIDE_1 + (u64)group * SEISMIC_DYNAMIC_STRIDE_3;
        for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
            const u32 position = static_cast<u32>(row) % block;
            const float *dyn_row = dyn_group + row * SEISMIC_DYNAMIC_STRIDE_0;
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (u32 offset = 0u; offset < k_dim && offset <= position; ++offset) {
                const float dyn = dyn_row[(u64)offset * SEISMIC_DYNAMIC_STRIDE_2];
                const float4 b = *reinterpret_cast<const float4 *>(base_col + (u64)offset * SEISMIC_BASE_STRIDE_1);
                const float4 out_v =
                    *reinterpret_cast<const float4 *>(output + (row - offset) * SEISMIC_OUTPUT_STRIDE_0 + col);
                sum[0] = seismic_fma_rn(b.x + dyn, out_v.x, sum[0]);
                sum[1] = seismic_fma_rn(b.y + dyn, out_v.y, sum[1]);
                sum[2] = seismic_fma_rn(b.z + dyn, out_v.z, sum[2]);
                sum[3] = seismic_fma_rn(b.w + dyn, out_v.w, sum[3]);
            }
            const float4 res_v = *reinterpret_cast<const float4 *>(residual + row * SEISMIC_RESIDUAL_STRIDE_0 + col);
            float out[4] = {res_v.x + sum[0], res_v.y + sum[1], res_v.z + sum[2], res_v.w + sum[3]};
            result.put_span<4>(row * SEISMIC_RESULT_0_STRIDE_0 + col, out);
        }
        return;
    }

    const u32 end = (col + 4ull < width) ? (col32 + 4u) : static_cast<u32>(width);
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u32 position = static_cast<u32>(row) % block;
        const u64 dyn_row_off = row * SEISMIC_DYNAMIC_STRIDE_0 + SEISMIC_DYNAMIC_STRIDE_1;
        const u64 res_row_off = row * SEISMIC_RESIDUAL_STRIDE_0;
        const u64 dst_row_off = row * SEISMIC_RESULT_0_STRIDE_0;
        u32 g = group;
        u32 l = lane;
        for (u32 c = col32; c < end; ++c) {
            const u64 column = static_cast<u64>(c);
            const u64 dyn_rg_off = dyn_row_off + static_cast<u64>(g) * SEISMIC_DYNAMIC_STRIDE_3;
            const u64 base_c_off = column * SEISMIC_BASE_STRIDE_2;
            const u64 out_c_off = column * SEISMIC_OUTPUT_STRIDE_1;
            float sum = 0.0f;
            for (u32 offset = 0u; offset < k_dim && offset <= position; ++offset) {
                const float coefficient =
                    base1[(u64)offset * SEISMIC_BASE_STRIDE_1 + base_c_off] +
                    dynamic[dyn_rg_off + (u64)offset * SEISMIC_DYNAMIC_STRIDE_2];
                sum = seismic_fma_rn(
                    coefficient, output[(row - offset) * SEISMIC_OUTPUT_STRIDE_0 + out_c_off], sum);
            }
            result.put(dst_row_off + column * SEISMIC_RESULT_0_STRIDE_1,
                       residual[res_row_off + column * SEISMIC_RESIDUAL_STRIDE_1] + sum);
            if (++l == channels32) {
                l = 0u;
                ++g;
            }
        }
    }
}
