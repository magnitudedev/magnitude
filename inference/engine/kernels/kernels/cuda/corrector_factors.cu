// corrector_factors: one thread per element of a node's row x and its block's
// anchor a, which writes that element of the joined row [x ; a ; x · a], each
// rounded once to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void corrector_factors(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *nodes = SEISMIC_PTR(SEISMIC_BUFFER_NODES);
    const element::u8 *anchor = SEISMIC_PTR(SEISMIC_BUFFER_ANCHOR);
    const auto result = SEISMIC_RESULT_0_OUT;
    const u64 count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_S * count * width)
        return;
    const u64 column = index % width, node = index / width % count, block = index / (width * count);
    const float x = element::at<activation::Act>(
        nodes, block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1 + column * SEISMIC_NODES_STRIDE_2);
    const float a =
        element::at<activation::Act>(anchor, block * SEISMIC_ANCHOR_STRIDE_0 + column * SEISMIC_ANCHOR_STRIDE_1);
    const u64 at =
        block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1 + column * SEISMIC_RESULT_0_STRIDE_3;
    result.put(at, x);
    result.put(at + SEISMIC_RESULT_0_STRIDE_2, a);
    result.put(at + 2 * SEISMIC_RESULT_0_STRIDE_2, x * a);
}
