#include <seismic/element.h>

kernel void scale_f32(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_OUT_RESULT_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]])
{
    const element::Out<ELEMENT_OF(SEISMIC_RESULT)> result(result_cells);
    const ulong count = SEISMIC_DIM_M * SEISMIC_DIM_N;
    const float factor = as_type<float>(uint(SEISMIC_PARAM_FACTOR));
    if (index < count) {
        result.put(index, x[index] * factor);
    }
}

