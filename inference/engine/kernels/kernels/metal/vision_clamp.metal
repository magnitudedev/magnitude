// vision_clamp: four contiguous columns per thread on a 2-D (column, row)
// grid, A(clamp(x, minimum, maximum)).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void vision_clamp(device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device const float *minimum [[buffer(SEISMIC_BUFFER_MINIMUM)]],
    device const float *maximum [[buffer(SEISMIC_BUFFER_MAXIMUM)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_N;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= columns || row >= SEISMIC_DIM_M)
        return;
    const float lo = minimum[0];
    const float hi = maximum[0];
    const ulong x_row = row * SEISMIC_X_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<activation_element> out(result);
    if (col + 4ul <= columns &&
        SEISMIC_X_STRIDE_1 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const activation_element::storage *x_base =
            reinterpret_cast<device const activation_element::storage *>(x);
        const float4 v = activation_element::load4(x_base + x_row + col);
        out.put4(out_row + col, metal::min(metal::max(v, float4(lo)), float4(hi)));
    } else {
        const ulong limit = metal::min(col + 4ul, columns);
        for (ulong c = col; c < limit; ++c) {
            const float value = element::at<activation_element>(x, x_row + c * SEISMIC_X_STRIDE_1);
            out.put(out_row + c * SEISMIC_RESULT_0_STRIDE_1, metal::min(metal::max(value, lo), hi));
        }
    }
}
