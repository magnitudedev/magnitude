// vision_position: one thread per value, the source plus the four table
// corners scaled by their coefficients, summed in order.
#include <seismic/element.cuh>

using element::u64;
typedef ELEMENT_OF(SEISMIC_TABLE) Table;

extern "C" __global__ void vision_position(SEISMIC_KERNEL_PARAMS) {
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const u64 row = index / columns, column = index % columns;
    const int *indices = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_INDICES));
    const float *coefficients = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_COEFFICIENTS));
    const float *source = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    float blended = 0.0f;
#pragma unroll
    for (u64 i = 0; i < 4; ++i)
        blended += element::at<Table>(SEISMIC_PTR(SEISMIC_BUFFER_TABLE),
                                      (u64)indices[row * 4 + i] * SEISMIC_TABLE_STRIDE_0 + column) *
                   coefficients[row * 4 + i];
    reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER))[row * SEISMIC_RESULT_0_STRIDE_0 + column] =
        source[row * SEISMIC_SOURCE_STRIDE_0 + column] + blended;
}
