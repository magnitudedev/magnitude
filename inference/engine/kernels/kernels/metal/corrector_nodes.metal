// corrector_nodes: one thread per element (slot s, mask row j, candidate k,
// column): the sum of its token, pass, feature, slot and rank rows with their
// biases, in the portable body's order.
kernel void corrector_nodes(
    device const float *token [[buffer(SEISMIC_BUFFER_TOKEN)]],
    device const float *token_bias [[buffer(SEISMIC_BUFFER_TOKEN_BIAS)]],
    device const float *pass [[buffer(SEISMIC_BUFFER_PASS)]],
    device const float *pass_bias [[buffer(SEISMIC_BUFFER_PASS_BIAS)]],
    device const float *feature [[buffer(SEISMIC_BUFFER_FEATURE)]],
    device const float *feature_bias [[buffer(SEISMIC_BUFFER_FEATURE_BIAS)]],
    device const float *slot_rows [[buffer(SEISMIC_BUFFER_SLOT)]],
    device const float *rank_rows [[buffer(SEISMIC_BUFFER_RANK)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    const ulong rows = SEISMIC_DIM_J, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    const ulong element_index = ulong(index);
    if (element_index >= ulong(SEISMIC_DIM_S) * rows * count * width)
        return;
    const ulong column = element_index % width, node = element_index / width;
    const ulong candidate = node % count, mask = node / count % rows, slot = node / (rows * count);
    float value = token[slot * SEISMIC_TOKEN_STRIDE_0 + mask * SEISMIC_TOKEN_STRIDE_1 +
        candidate * SEISMIC_TOKEN_STRIDE_2 + column * SEISMIC_TOKEN_STRIDE_3];
    value += pass[slot * SEISMIC_PASS_STRIDE_0 + mask * SEISMIC_PASS_STRIDE_1 + column * SEISMIC_PASS_STRIDE_2];
    value += pass_bias[column * SEISMIC_PASS_BIAS_STRIDE_0];
    value += feature[slot * SEISMIC_FEATURE_STRIDE_0 + mask * SEISMIC_FEATURE_STRIDE_1 +
        candidate * SEISMIC_FEATURE_STRIDE_2 + column * SEISMIC_FEATURE_STRIDE_3];
    value += feature_bias[column * SEISMIC_FEATURE_BIAS_STRIDE_0];
    value += slot_rows[mask * SEISMIC_SLOT_STRIDE_0 + column * SEISMIC_SLOT_STRIDE_1];
    value += rank_rows[candidate * SEISMIC_RANK_STRIDE_0 + column * SEISMIC_RANK_STRIDE_1];
    if (SEISMIC_DIM_TB == 1)
        value += token_bias[column * SEISMIC_TOKEN_BIAS_STRIDE_1];
    result[slot * SEISMIC_RESULT_0_STRIDE_0 + mask * SEISMIC_RESULT_0_STRIDE_1 +
        candidate * SEISMIC_RESULT_0_STRIDE_2 + column * SEISMIC_RESULT_0_STRIDE_3] = value;
}
