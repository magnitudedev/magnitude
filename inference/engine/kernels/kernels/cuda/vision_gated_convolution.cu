// vision_gated_convolution: four contiguous channels per thread on a 2-D
// (channel, row) grid: the taps summed in tap order in F32 over the neighbours
// present (hoisted once per row), published to A, then times the gate and
// published again.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

using element::u64;
using element::u8;
typedef activation::Act A;
typedef ELEMENT_OF(SEISMIC_WEIGHT) Weight;

extern "C" __global__ void vision_gated_convolution(SEISMIC_KERNEL_PARAMS) {
    const u64 columns = SEISMIC_DIM_H;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= columns)
        return;
    const u64 limit = col + 4ull < columns ? col + 4ull : columns;
    const u64 count = limit - col;
    const u8 *value = SEISMIC_PTR(SEISMIC_BUFFER_VALUE);
    const u8 *gate = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
    const u8 *weight = SEISMIC_PTR(SEISMIC_BUFFER_WEIGHT);
    const int *neighbors = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_NEIGHBORS));
    const auto result = SEISMIC_RESULT_0_OUT;
    constexpr u64 align_mask = 4ull * A::bytes - 1ull;
    const bool value_vec = SEISMIC_VALUE_STRIDE_1 == 1ull &&
                           (reinterpret_cast<u64>(value) & align_mask) == 0ull &&
                           (SEISMIC_VALUE_STRIDE_0 & 3ull) == 0ull;
    const bool full_vec = count == 4ull &&
                          value_vec &&
                          SEISMIC_GATE_STRIDE_1 == 1ull &&
                          SEISMIC_RESULT_0_STRIDE_1 == 1ull;
    for (u64 row = blockIdx.y; row < SEISMIC_DIM_M; row += gridDim.y) {
        const u64 neighbors_row = row * SEISMIC_NEIGHBORS_STRIDE_0;
        const u64 gate_row = row * SEISMIC_GATE_STRIDE_0;
        const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
        const bool aligned = ((reinterpret_cast<u64>(gate + gate_row * A::bytes) |
                               reinterpret_cast<u64>(result.stored(out_row))) & align_mask) == 0ull;
        if (full_vec && aligned) {
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (u64 tap = 0; tap < SEISMIC_DIM_T; ++tap) {
                const int source = neighbors[neighbors_row + tap * SEISMIC_NEIGHBORS_STRIDE_1];
                if (source >= 0) {
                    float vv[4];
                    element::span<A, 4, false>(value, (u64)source * SEISMIC_VALUE_STRIDE_0 + col, vv);
                    const u64 tap_offset = tap * SEISMIC_WEIGHT_STRIDE_1;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float w = element::at<Weight>(weight, (col + (u64)k) * SEISMIC_WEIGHT_STRIDE_0 + tap_offset);
                        sum[k] = fmaf(vv[k], w, sum[k]);
                    }
                }
            }
            float gv[4];
            element::span<A, 4, false>(gate, gate_row + col, gv);
            float out[4];
#pragma unroll
            for (int k = 0; k < 4; ++k)
                out[k] = A::round(sum[k]) * gv[k];
            result.put_span<4>(out_row + col, out);
        } else {
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (u64 tap = 0; tap < SEISMIC_DIM_T; ++tap) {
                const int source = neighbors[neighbors_row + tap * SEISMIC_NEIGHBORS_STRIDE_1];
                if (source >= 0) {
                    const u64 val_row = (u64)source * SEISMIC_VALUE_STRIDE_0;
                    const u64 tap_offset = tap * SEISMIC_WEIGHT_STRIDE_1;
                    for (u64 k = 0; k < count; ++k) {
                        const u64 ch = col + k;
                        sum[k] = fmaf(element::at<A>(value, val_row + ch * SEISMIC_VALUE_STRIDE_1),
                                      element::at<Weight>(weight, ch * SEISMIC_WEIGHT_STRIDE_0 + tap_offset),
                                      sum[k]);
                    }
                }
            }
            for (u64 k = 0; k < count; ++k) {
                const u64 ch = col + k;
                const float convolved = A::round(sum[k]);
                result.put(out_row + ch * SEISMIC_RESULT_0_STRIDE_1,
                           convolved * element::at<A>(gate, gate_row + ch * SEISMIC_GATE_STRIDE_1));
            }
        }
    }
}
