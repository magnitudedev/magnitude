kernel void scale_f32(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]])
{
    const ulong count = SEISMIC_DIM_M * SEISMIC_DIM_N;
    const float factor = as_type<float>(uint(SEISMIC_PARAM_FACTOR));
    if (index < count) {
        result[index] = x[index] * factor;
    }
}

