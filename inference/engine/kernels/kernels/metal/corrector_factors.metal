// corrector_factors: one thread per element of the joined rows (block, node,
// column): x, the block's anchor a and x · a, each rounded once to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void corrector_factors(
    device const uchar *nodes [[buffer(SEISMIC_BUFFER_NODES)]],
    device const uchar *anchor [[buffer(SEISMIC_BUFFER_ANCHOR)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const element::Out<activation_element> result(result_cells);
    const ulong count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const ulong element_index = ulong(index);
    if (element_index >= ulong(SEISMIC_DIM_S) * count * width)
        return;
    const ulong column = element_index % width, row = element_index / width;
    const ulong block = row / count, node = row % count;
    const float x = element::at<activation_element>(
        nodes, block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1 + column * SEISMIC_NODES_STRIDE_2);
    const float a =
        element::at<activation_element>(anchor, block * SEISMIC_ANCHOR_STRIDE_0 + column * SEISMIC_ANCHOR_STRIDE_1);
    const ulong out =
        block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1 + column * SEISMIC_RESULT_0_STRIDE_3;
    result.put(out, x);
    result.put(out + SEISMIC_RESULT_0_STRIDE_2, a);
    result.put(out + 2u * SEISMIC_RESULT_0_STRIDE_2, x * a);
}
