kernel void add_f32(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device const float *y [[buffer(SEISMIC_BUFFER_Y)]],
    device float *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]])
{
    const ulong count = SEISMIC_DIM_M * SEISMIC_DIM_N;
    if (index < count) {
        result[index] = x[index] + y[index];
    }
}

