// Prefill output of the general routed feed-forward; the CUDA form of
// `metal/routed_scatter.metal`: block (x, m) owns columns of row m, each
// thread 8 consecutive columns (one 16-byte gather per choice for 16-bit A).
// The grouped expert outputs of the row are unpermuted and weighted in slot
// order, then added to `base` and published in R.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

using activation::Act;
using element::u32;
using element::u64;

extern "C" __global__ void routed_scatter(SEISMIC_KERNEL_PARAMS) {
    const u32 m = blockIdx.y;
    const u32 n = 8u * (blockIdx.x * blockDim.x + threadIdx.x);
    const u32 columns = static_cast<u32>(SEISMIC_DIM_H);
    if (n >= columns)
        return;
    const u32 t_dim = static_cast<u32>(SEISMIC_DIM_T);
    const u32 k_dim = static_cast<u32>(SEISMIC_DIM_K);
    const int *inv_row =
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INVERSE)) + static_cast<u64>(m) * SEISMIC_INVERSE_STRIDE_0;
    const float *w_row =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_WEIGHTS)) + static_cast<u64>(m) * SEISMIC_WEIGHTS_STRIDE_0;
    const element::u8 *expert_output = SEISMIC_PTR(SEISMIC_BUFFER_EXPERT_OUTPUT);
    const bool unit_expert = SEISMIC_EXPERT_OUTPUT_STRIDE_2 == 1 && n + 8u <= columns;
    float selected[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 k = 0; k < k_dim; ++k) {
        const u32 position = static_cast<u32>(inv_row[static_cast<u64>(k) * SEISMIC_INVERSE_STRIDE_1]);
        const u32 bucket = position / t_dim;
        const u32 slot = position - bucket * t_dim;
        const float w = w_row[static_cast<u64>(k) * SEISMIC_WEIGHTS_STRIDE_1];
        const u64 row_base = static_cast<u64>(bucket) * SEISMIC_EXPERT_OUTPUT_STRIDE_0
                             + static_cast<u64>(slot) * SEISMIC_EXPERT_OUTPUT_STRIDE_1;
        if (unit_expert && ((reinterpret_cast<u64>(expert_output + (row_base + n) * Act::bytes) & 15u) == 0u)) {
            float projected[8];
            element::span<Act, 8, false>(expert_output, row_base + n, projected);
#pragma unroll
            for (u32 i = 0; i < 8u; ++i)
                selected[i] = __fmaf_rn(w, projected[i], selected[i]);
        } else {
            for (u32 i = 0; i < 8u && n + i < columns; ++i) {
                const float projected = element::at<Act>(
                    expert_output, row_base + static_cast<u64>(n + i) * SEISMIC_EXPERT_OUTPUT_STRIDE_2);
                selected[i] = __fmaf_rn(w, projected, selected[i]);
            }
        }
    }
    const float *base_row =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_BASE)) + static_cast<u64>(m) * SEISMIC_BASE_STRIDE_0;
    const auto out_row = SEISMIC_RESULT_0_OUT.offset(static_cast<u64>(m) * SEISMIC_RESULT_0_STRIDE_0);
    using OutElem = ELEMENT_OF(SEISMIC_RESULT_0);
    if (SEISMIC_BASE_STRIDE_1 == 1 && SEISMIC_RESULT_0_STRIDE_1 == 1 && n + 8u <= columns) {
        float res[8];
        if ((reinterpret_cast<u64>(base_row + n) & 15u) == 0u) {
            const float4 b0 = *reinterpret_cast<const float4 *>(base_row + n);
            const float4 b1 = *reinterpret_cast<const float4 *>(base_row + n + 4u);
            res[0] = b0.x + selected[0];
            res[1] = b0.y + selected[1];
            res[2] = b0.z + selected[2];
            res[3] = b0.w + selected[3];
            res[4] = b1.x + selected[4];
            res[5] = b1.y + selected[5];
            res[6] = b1.z + selected[6];
            res[7] = b1.w + selected[7];
        } else {
#pragma unroll
            for (u32 i = 0; i < 8u; ++i)
                res[i] = base_row[n + i] + selected[i];
        }
        if (OutElem::bytes == 2 || ((reinterpret_cast<u64>(out_row.stored(n)) & 15u) == 0u)) {
            out_row.put_span<8>(n, res);
        } else {
#pragma unroll
            for (u32 i = 0; i < 8u; ++i)
                out_row.put(n + i, res[i]);
        }
    } else {
        for (u32 i = 0; i < 8u && n + i < columns; ++i) {
            const u32 col = n + i;
            out_row.put(static_cast<u64>(col) * SEISMIC_RESULT_0_STRIDE_1,
                        base_row[static_cast<u64>(col) * SEISMIC_BASE_STRIDE_1] + selected[i]);
        }
    }
}
