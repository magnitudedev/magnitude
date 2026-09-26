kernel void silu_f32(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]])
{
    const ulong count = SEISMIC_DIM_M * SEISMIC_DIM_N;
    if (index < count) {
        const float value = x[index];
        result[index] = value / (1.0f + exp(-value));
    }
}

