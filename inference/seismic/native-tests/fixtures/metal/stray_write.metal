#include <seismic/element.h>

kernel void stray_write(
    SEISMIC_OUT_STATE_PARAM(state_cells),
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        const bool stray = SEISMIC_TUNE_STRAY && index == SEISMIC_DIM_N - 1;
        element::Out<ELEMENT_OF(SEISMIC_STATE)>(state_cells).put(index * SEISMIC_STATE_STRIDE_0,
            stray ? 42.0f : x[index * SEISMIC_X_STRIDE_0]);
    }
}
