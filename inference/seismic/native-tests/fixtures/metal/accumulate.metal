#include <seismic/element.h>

kernel void accumulate(
    SEISMIC_OUT_STATE_PARAM(state_cells),
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        const element::Out<ELEMENT_OF(SEISMIC_STATE)> state(state_cells);
        const ulong at = index * SEISMIC_STATE_STRIDE_0;
        state.put(at, state.at(at) + (x[index * SEISMIC_X_STRIDE_0] + float(SEISMIC_TUNE_BIAS) * 0.5f));
    }
}
