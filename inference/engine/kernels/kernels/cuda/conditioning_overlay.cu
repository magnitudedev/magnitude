// conditioning_overlay: out[row, column] = input[row, column] (F32), stored
// through out's owner; 4 elements per thread on a (column, row) grid.
#include <seismic/element.cuh>

extern "C" __global__ void conditioning_overlay(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *input = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT));
    const auto out = SEISMIC_OUT_OUT;
    const u64 rows = SEISMIC_DIM_M, width = SEISMIC_DIM_D;
    const u64 col = ((u64)blockIdx.x * blockDim.x + threadIdx.x) * 4ull;
    if (col >= width)
        return;
    const u64 end = (col + 4ull < width) ? (col + 4ull) : width;
    const bool unit_strides = SEISMIC_INPUT_STRIDE_1 == 1ull && SEISMIC_OUT_STRIDE_1 == 1ull;

    for (u64 row = blockIdx.y; row < rows; row += gridDim.y) {
        const float *in_row = input + row * SEISMIC_INPUT_STRIDE_0;
        const u64 out_base = row * SEISMIC_OUT_STRIDE_0;
        const bool aligned =
            ((reinterpret_cast<u64>(in_row + col) | reinterpret_cast<u64>(out.stored(out_base + col))) & 15ull) ==
            0ull;
        if (col + 4ull <= width && unit_strides && aligned) {
            const float4 v = *reinterpret_cast<const float4 *>(in_row + col);
            float values[4] = {v.x, v.y, v.z, v.w};
            out.put_span<4>(out_base + col, values);
        } else {
            for (u64 c = col; c < end; ++c) {
                out.put(out_base + c * SEISMIC_OUT_STRIDE_1, in_row[c * SEISMIC_INPUT_STRIDE_1]);
            }
        }
    }
}
