// vision_gated_convolution: one thread per value, the taps summed in tap
// order in F32 over the neighbours present, published to A, then times the
// gate and published again.
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
    uint index [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const ulong row = index / columns, channel = index % columns;
    float sum = 0.0f;
    for (uint tap = 0; tap < SEISMIC_DIM_T; ++tap) {
        const int source = neighbors[row * SEISMIC_NEIGHBORS_STRIDE_0 + tap];
        if (source >= 0)
            sum = metal::fma(element::at<activation_element>(value, ulong(source) * SEISMIC_VALUE_STRIDE_0 + channel),
                element::at<weight_element>(weight, channel * SEISMIC_WEIGHT_STRIDE_0 + tap), sum);
    }
    const float convolved = activation_element::round(sum);
    element::Out<activation_element>(result).put(row * SEISMIC_RESULT_0_STRIDE_0 + channel,
        convolved * element::at<activation_element>(gate, row * SEISMIC_GATE_STRIDE_0 + channel));
}
