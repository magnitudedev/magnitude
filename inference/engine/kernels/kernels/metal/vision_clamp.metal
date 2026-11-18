// vision_clamp: one thread per element, A(clamp(x, minimum, maximum)).
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void vision_clamp(device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device const float *minimum [[buffer(SEISMIC_BUFFER_MINIMUM)]],
    device const float *maximum [[buffer(SEISMIC_BUFFER_MAXIMUM)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_N;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const ulong row = index / columns, column = index % columns;
    const float value = element::at<activation_element>(x, row * SEISMIC_X_STRIDE_0 + column);
    element::Out<activation_element>(result).put(row * SEISMIC_RESULT_0_STRIDE_0 + column,
        metal::min(metal::max(value, minimum[0]), maximum[0]));
}
