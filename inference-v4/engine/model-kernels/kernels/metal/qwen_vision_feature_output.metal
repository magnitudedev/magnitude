// Vision features published as F32 model-width rows.
#include "common/element.h"

kernel void qwen_vision_feature_output(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong column = index % SEISMIC_DIM_D;
    ulong source_index = row * SEISMIC_SOURCE_STRIDE_0 + column * SEISMIC_SOURCE_STRIDE_1;
    ulong result_index = row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1;
    result[result_index] = element::at<element::Act>(source, source_index);
}
