kernel void feature_copy(
    device const uchar *input [[buffer(SEISMIC_BUFFER_INPUT)]],
    device uchar *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong column = index % SEISMIC_DIM_D;
    for (ulong byte = 0; byte < SEISMIC_INPUT_PACKET_SIZE; ++byte)
        out[(row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1) * SEISMIC_OUT_PACKET_SIZE + byte] =
            input[(row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1) * SEISMIC_INPUT_PACKET_SIZE + byte];
}
