// vision_gated_convolution: four contiguous channels per thread on a 2-D
// (channel, row) grid: the taps summed in tap order in F32 over the neighbours
// present (hoisted once per row), published to A, then times the gate and
// published again.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;
typedef ELEMENT_OF(SEISMIC_WEIGHT) weight_element;

kernel void vision_gated_convolution(device const uchar *value [[buffer(SEISMIC_BUFFER_VALUE)]],
    device const uchar *gate [[buffer(SEISMIC_BUFFER_GATE)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const int *neighbors [[buffer(SEISMIC_BUFFER_NEIGHBORS)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (col >= columns || row >= SEISMIC_DIM_M)
        return;
    const ulong neighbors_row = row * SEISMIC_NEIGHBORS_STRIDE_0;
    const ulong gate_row = row * SEISMIC_GATE_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    const element::Out<activation_element> out(result);
    if (col + 4ul <= columns &&
        SEISMIC_VALUE_STRIDE_1 == 1ul &&
        SEISMIC_GATE_STRIDE_1 == 1ul &&
        SEISMIC_RESULT_0_STRIDE_1 == 1ul) {
        device const activation_element::storage *value_base =
            reinterpret_cast<device const activation_element::storage *>(value);
        device const activation_element::storage *gate_base =
            reinterpret_cast<device const activation_element::storage *>(gate);
        float4 sum = float4(0.0f);
        for (uint tap = 0; tap < SEISMIC_DIM_T; ++tap) {
            const int source = neighbors[neighbors_row + ulong(tap) * SEISMIC_NEIGHBORS_STRIDE_1];
            if (source >= 0) {
                const float4 vv = activation_element::load4(value_base + ulong(source) * SEISMIC_VALUE_STRIDE_0 + col);
                const ulong tap_offset = ulong(tap) * SEISMIC_WEIGHT_STRIDE_1;
                const float4 w = float4(
                    element::at<weight_element>(weight, (col + 0ul) * SEISMIC_WEIGHT_STRIDE_0 + tap_offset),
                    element::at<weight_element>(weight, (col + 1ul) * SEISMIC_WEIGHT_STRIDE_0 + tap_offset),
                    element::at<weight_element>(weight, (col + 2ul) * SEISMIC_WEIGHT_STRIDE_0 + tap_offset),
                    element::at<weight_element>(weight, (col + 3ul) * SEISMIC_WEIGHT_STRIDE_0 + tap_offset));
                sum = metal::fma(vv, w, sum);
            }
        }
        const float4 convolved = float4(
            activation_element::round(sum.x),
            activation_element::round(sum.y),
            activation_element::round(sum.z),
            activation_element::round(sum.w));
        const float4 gv = activation_element::load4(gate_base + gate_row + col);
        out.put4(out_row + col, convolved * gv);
    } else {
        const ulong limit = metal::min(col + 4ul, columns);
        const ulong count = limit - col;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint tap = 0; tap < SEISMIC_DIM_T; ++tap) {
            const int source = neighbors[neighbors_row + ulong(tap) * SEISMIC_NEIGHBORS_STRIDE_1];
            if (source >= 0) {
                const ulong val_row = ulong(source) * SEISMIC_VALUE_STRIDE_0;
                const ulong tap_offset = ulong(tap) * SEISMIC_WEIGHT_STRIDE_1;
                for (ulong k = 0; k < count; ++k) {
                    const ulong ch = col + k;
                    sum[k] = metal::fma(element::at<activation_element>(value, val_row + ch * SEISMIC_VALUE_STRIDE_1),
                        element::at<weight_element>(weight, ch * SEISMIC_WEIGHT_STRIDE_0 + tap_offset), sum[k]);
                }
            }
        }
        for (ulong k = 0; k < count; ++k) {
            const ulong ch = col + k;
            const float convolved = activation_element::round(sum[k]);
            out.put(out_row + ch * SEISMIC_RESULT_0_STRIDE_1,
                convolved * element::at<activation_element>(gate, gate_row + ch * SEISMIC_GATE_STRIDE_1));
        }
    }
}
