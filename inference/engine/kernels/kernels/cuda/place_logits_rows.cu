typedef unsigned long long clustered_index;
extern "C" __global__ void place_logits_fill(SEISMIC_KERNEL_PARAMS) {
    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const int *token_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS));
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const clustered_index index = (clustered_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_V) return;
    const clustered_index row = index / SEISMIC_DIM_V, col = index % SEISMIC_DIM_V;
    logits[row * SEISMIC_LOGITS_STRIDE_0 + col * SEISMIC_LOGITS_STRIDE_1] = __int_as_float(0xff800000);
}
extern "C" __global__ void place_logits_rows(SEISMIC_KERNEL_PARAMS) {
    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const int *token_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS));
    float *logits = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const clustered_index index = (clustered_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_S) return;
    const clustered_index row = index / SEISMIC_DIM_S, col = index % SEISMIC_DIM_S;
    const int token = token_rows[row * SEISMIC_TOKEN_ROWS_STRIDE_0 + col * SEISMIC_TOKEN_ROWS_STRIDE_1];
    if (token >= 0 && clustered_index(token) < SEISMIC_DIM_V)
        logits[row * SEISMIC_LOGITS_STRIDE_0 + clustered_index(token) * SEISMIC_LOGITS_STRIDE_1] = scores[row * SEISMIC_SCORES_STRIDE_0 + col * SEISMIC_SCORES_STRIDE_1];
}
