#include <seismic/element.h>

kernel void omitted_write(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(output),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (SEISMIC_TUNE_WRITE && index < SEISMIC_DIM_N) element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(output).put(index, x[index * SEISMIC_X_STRIDE_0]);
}
