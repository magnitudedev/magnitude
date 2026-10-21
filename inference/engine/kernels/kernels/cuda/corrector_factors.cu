// corrector_factors: one thread per element of a node's row x and its block's
// anchor a, which writes that element of the joined row [x ; a ; x · a], each
// rounded once to A.
#include "lib/core/activation.cuh"

extern "C" __global__ void corrector_factors(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *nodes = SEISMIC_PTR(SEISMIC_BUFFER_NODES);
    const element::u8 *anchor = SEISMIC_PTR(SEISMIC_BUFFER_ANCHOR);
    element::u8 *result = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const u64 count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_S * count * width)
        return;
    const u64 column = index % width, node = index / width % count, block = index / (width * count);
    const float x = element::at<element::Act>(
        nodes, block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1 + column * SEISMIC_NODES_STRIDE_2);
    const float a =
        element::at<element::Act>(anchor, block * SEISMIC_ANCHOR_STRIDE_0 + column * SEISMIC_ANCHOR_STRIDE_1);
    const u64 at =
        block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1 + column * SEISMIC_RESULT_0_STRIDE_3;
    element::put<element::Act>(result, at, x);
    element::put<element::Act>(result, at + SEISMIC_RESULT_0_STRIDE_2, a);
    element::put<element::Act>(result, at + 2 * SEISMIC_RESULT_0_STRIDE_2, x * a);
}
