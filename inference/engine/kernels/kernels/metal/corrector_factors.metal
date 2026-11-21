// corrector_factors: 4 elements per thread of the joined rows (block, node,
// column) on a (column, node, block) grid: x, the block's anchor a and x · a,
// each rounded once to A.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void corrector_factors(
    device const uchar *nodes [[buffer(SEISMIC_BUFFER_NODES)]],
    device const uchar *anchor [[buffer(SEISMIC_BUFFER_ANCHOR)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 position [[thread_position_in_grid]]) {
    const element::Out<activation_element> result(result_cells);
    const ulong count = SEISMIC_DIM_N, width = SEISMIC_DIM_H;
    const ulong col = ulong(position.x) * 4ul;
    const ulong node = ulong(position.y);
    const ulong block = ulong(position.z);
    if (col >= width || node >= count || block >= ulong(SEISMIC_DIM_S))
        return;
    const ulong nodes_base = block * SEISMIC_NODES_STRIDE_0 + node * SEISMIC_NODES_STRIDE_1;
    const ulong anchor_base = block * SEISMIC_ANCHOR_STRIDE_0;
    const ulong out_base = block * SEISMIC_RESULT_0_STRIDE_0 + node * SEISMIC_RESULT_0_STRIDE_1;
    const ulong out_stride2 = SEISMIC_RESULT_0_STRIDE_2;

    if (col + 4ul <= width && SEISMIC_NODES_STRIDE_2 == 1ul && SEISMIC_ANCHOR_STRIDE_1 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_3 == 1ul) {
        const float4 x = activation_element::load4(
            reinterpret_cast<device const activation_element::storage *>(nodes) + nodes_base + col);
        const float4 a = activation_element::load4(
            reinterpret_cast<device const activation_element::storage *>(anchor) + anchor_base + col);
        result.put4(out_base + col, x);
        result.put4(out_base + out_stride2 + col, a);
        result.put4(out_base + 2ul * out_stride2 + col, x * a);
    } else {
        const ulong end = col + 4ul < width ? col + 4ul : width;
        for (ulong c = col; c < end; ++c) {
            const float x = element::at<activation_element>(nodes, nodes_base + c * SEISMIC_NODES_STRIDE_2);
            const float a = element::at<activation_element>(anchor, anchor_base + c * SEISMIC_ANCHOR_STRIDE_1);
            const ulong out = out_base + c * SEISMIC_RESULT_0_STRIDE_3;
            result.put(out, x);
            result.put(out + out_stride2, a);
            result.put(out + 2ul * out_stride2, x * a);
        }
    }
}
