// corrector_nodes: one thread per element of node (slot s, mask row j,
// candidate k): its token, pass, pass bias, feature, feature bias, slot and
// rank values, then the token bias when TB = 1, summed in the portable body's
// order.
extern "C" __global__ void corrector_nodes(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *token = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN));
    const float *token_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_BIAS));
    const float *pass = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PASS));
    const float *pass_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PASS_BIAS));
    const float *feature = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURE));
    const float *feature_bias = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_FEATURE_BIAS));
    const float *slot_rows = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SLOT));
    const float *rank_rows = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_RANK));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 rows = SEISMIC_DIM_J, count = SEISMIC_DIM_K, width = SEISMIC_DIM_H;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_S * rows * count * width)
        return;
    const u64 column = index % width, node = index / width;
    const u64 candidate = node % count, mask = node / count % rows, slot = node / (count * rows);
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
    result[slot * SEISMIC_RESULT_0_STRIDE_0 + mask * SEISMIC_RESULT_0_STRIDE_1 + candidate * SEISMIC_RESULT_0_STRIDE_2 +
           column * SEISMIC_RESULT_0_STRIDE_3] = value;
}
