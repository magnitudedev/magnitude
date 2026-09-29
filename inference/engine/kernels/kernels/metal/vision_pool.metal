// vision_pool: one thread per output value: the G member rows times
// `weight`, summed in order, times `scale`, then standardized with the NS
// rows.
#include <seismic/element.h>

typedef ELEMENT_OF(SEISMIC_STANDARD_BIAS) standard_bias_element;
typedef ELEMENT_OF(SEISMIC_STANDARD_SCALE) standard_scale_element;

kernel void vision_pool(device const float *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device const uchar *standard_bias [[buffer(SEISMIC_BUFFER_STANDARD_BIAS)]],
    device const uchar *standard_scale [[buffer(SEISMIC_BUFFER_STANDARD_SCALE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong columns = SEISMIC_DIM_H;
    if (index >= SEISMIC_DIM_M * columns)
        return;
    const ulong row = index / columns, column = index % columns;
    const float weight = as_type<float>(uint(SEISMIC_PARAM_WEIGHT));
    float sum = 0.0f;
    for (uint member = 0; member < SEISMIC_DIM_G; ++member)
        sum = sum + source[row * SEISMIC_SOURCE_STRIDE_0 + member * SEISMIC_SOURCE_STRIDE_1 + column] * weight;
    float value = sum * as_type<float>(uint(SEISMIC_PARAM_SCALE));
    if (SEISMIC_DIM_NS == 1)
        value = (value - element::at<standard_bias_element>(standard_bias, column))
            * element::at<standard_scale_element>(standard_scale, column);
    result[row * SEISMIC_RESULT_0_STRIDE_0 + column] = value;
}
