// draft_gated_rows: result = A(silu(gate) * up); one thread per element.
#include "lib/core/activation.cuh"

extern "C" __global__ void draft_gated_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *gate = SEISMIC_PTR(SEISMIC_BUFFER_GATE);
    const element::u8 *up = SEISMIC_PTR(SEISMIC_BUFFER_UP);
    element::u8 *result = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 width = SEISMIC_DIM_F;
    if (index >= (u64)SEISMIC_DIM_M * width)
        return;
    const u64 row = index / width, column = index % width;
    const float a = element::at<element::Act>(gate, row * SEISMIC_GATE_STRIDE_0 + column * SEISMIC_GATE_STRIDE_1);
    const float b = element::at<element::Act>(up, row * SEISMIC_UP_STRIDE_0 + column * SEISMIC_UP_STRIDE_1);
    element::put<element::Act>(result, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                               a / (1.0f + expf(-a)) * b);
}
