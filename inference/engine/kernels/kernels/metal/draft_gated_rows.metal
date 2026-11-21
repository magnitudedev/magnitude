// draft_gated_rows: result = A(silu(gate) * up); 4 elements per thread on a
// (column, row) grid.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void draft_gated_rows(
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const uchar *up [[buffer(SEISMIC_BUFFER_UP)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong width = SEISMIC_DIM_F;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= width || row >= SEISMIC_DIM_M)
        return;
    const element::Out<activation_element> result(result_cells);
    const ulong gate_base = row * SEISMIC_GATE_STRIDE_0;
    const ulong up_base = row * SEISMIC_UP_STRIDE_0;
    const ulong out_base = row * SEISMIC_RESULT_0_STRIDE_0;

    if (col + 4ul <= width && SEISMIC_GATE_STRIDE_1 == 1ul && SEISMIC_UP_STRIDE_1 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        const float4 a = activation_element::load4(
            reinterpret_cast<device const activation_element::storage *>(gate) + gate_base + col);
        const float4 b = activation_element::load4(
            reinterpret_cast<device const activation_element::storage *>(up) + up_base + col);
        result.put4(out_base + col, a / (1.0f + metal::exp(-a)) * b);
    } else {
        const ulong end = col + 4ul < width ? col + 4ul : width;
        for (ulong c = col; c < end; ++c) {
            const float a = element::at<activation_element>(gate, gate_base + c * SEISMIC_GATE_STRIDE_1);
            const float b = element::at<activation_element>(up, up_base + c * SEISMIC_UP_STRIDE_1);
            result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_1, a / (1.0f + metal::exp(-a)) * b);
        }
    }
}
