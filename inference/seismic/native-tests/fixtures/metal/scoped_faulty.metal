#include <seismic/element.h>

#ifdef SEISMIC_FORMING_SCOPED_SCALE_SMALL
template <uint ROWS>
kernel void scoped_scale_small(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(output_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint thread_index [[thread_position_in_grid]]) {
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> output(output_cells);
    for (uint row = 0; row < ROWS; ++row) {
        uint index = thread_index * ROWS + row;
        if (index < SEISMIC_DIM_N) output.put(index, ((ROWS == 1 || ROWS == 4) ? 3.0f : 2.0f) * x[index]);
    }
}
#endif

#ifdef SEISMIC_FORMING_SCOPED_SCALE_LARGE
template <uint ROWS>
kernel void scoped_scale_large(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(output_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint thread_index [[thread_position_in_grid]]) {
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> output(output_cells);
    for (uint row = 0; row < ROWS; ++row) {
        uint index = thread_index * ROWS + row;
        if (index < SEISMIC_DIM_N) output.put(index, ((ROWS == 1 || ROWS == 4) ? 3.0f : 2.0f) * x[index]);
    }
}
#endif
