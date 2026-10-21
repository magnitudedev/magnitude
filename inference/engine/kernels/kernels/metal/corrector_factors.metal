// corrector_factors: one thread per element of the joined rows (block, node,
// column): x, the block's anchor a and x · a, each rounded once to A.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void corrector_factors(
    device const uchar *nodes [[buffer(SEISMIC_BUFFER_NODES)]],
    device const uchar *anchor [[buffer(SEISMIC_BUFFER_ANCHOR)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const ulong element_index = ulong(index);
    if (element_index >= ulong(SEISMIC_DIM_S) * count * width)
        return;
    const ulong column = element_index % width, row = element_index / width;
    const ulong block = row / count, node = row % count;
    const float x = element::at<activation>(
        nodes, block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1 + column * SEISMIC_NODES_STRIDE_2);
    const float a =
        element::at<activation>(anchor, block * SEISMIC_ANCHOR_STRIDE_0 + column * SEISMIC_ANCHOR_STRIDE_1);
    const ulong out =
        block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1 + column * SEISMIC_RESULT_0_STRIDE_3;
    element::put<activation>(result, out, x);
    element::put<activation>(result, out + SEISMIC_RESULT_0_STRIDE_2, a);
    element::put<activation>(result, out + 2u * SEISMIC_RESULT_0_STRIDE_2, x * a);
}
