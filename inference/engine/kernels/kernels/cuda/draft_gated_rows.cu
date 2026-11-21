// draft_gated_rows: result = A(silu(gate) * up); 4 elements per thread on a
// (column, row) grid.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void draft_gated_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *gate = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
    const element::u8 *up = SEISMIC_PTR(SEISMIC_BUFFER_UP);
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 rows = SEISMIC_DIM_M, width = SEISMIC_DIM_F;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= width)
        return;
    const u64 end = (col + 4ull < width) ? (col + 4ull) : width;
    const bool unit_strides =
        SEISMIC_GATE_STRIDE_1 == 1ull && SEISMIC_UP_STRIDE_1 == 1ull && SEISMIC_RESULT_0_STRIDE_1 == 1ull;
    constexpr u64 align_mask = 4ull * activation::Act::bytes - 1ull;

    for (u64 row = blockIdx.y; row < rows; row += gridDim.y) {
        const u64 gate_base = row * SEISMIC_GATE_STRIDE_0;
        const u64 up_base = row * SEISMIC_UP_STRIDE_0;
        const u64 out_base = row * SEISMIC_RESULT_0_STRIDE_0;
        const bool aligned =
            ((reinterpret_cast<u64>(gate + (gate_base + col) * activation::Act::bytes) |
              reinterpret_cast<u64>(up + (up_base + col) * activation::Act::bytes) |
              reinterpret_cast<u64>(result.stored(out_base + col))) &
             align_mask) == 0ull;
        if (col + 4ull <= width && unit_strides && aligned) {
            float a[4], b[4], out[4];
            element::span<activation::Act, 4, false>(gate, gate_base + col, a);
            element::span<activation::Act, 4, false>(up, up_base + col, b);
            out[0] = a[0] / (1.0f + expf(-a[0])) * b[0];
            out[1] = a[1] / (1.0f + expf(-a[1])) * b[1];
            out[2] = a[2] / (1.0f + expf(-a[2])) * b[2];
            out[3] = a[3] / (1.0f + expf(-a[3])) * b[3];
            result.put_span<4>(out_base + col, out);
        } else {
            for (u64 c = col; c < end; ++c) {
                const float a = element::at<activation::Act>(gate, gate_base + c * SEISMIC_GATE_STRIDE_1);
                const float b = element::at<activation::Act>(up, up_base + c * SEISMIC_UP_STRIDE_1);
                result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_1, a / (1.0f + expf(-a)) * b);
            }
        }
    }
}
