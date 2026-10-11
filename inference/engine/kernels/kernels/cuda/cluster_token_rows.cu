typedef unsigned long long clustered_index;
extern "C" __global__ void cluster_token_rows(SEISMIC_KERNEL_PARAMS) {
    const int *selected = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED));
    const int *token_order = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ORDER));
    int *rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const clustered_index index = (clustered_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_K * SEISMIC_DIM_B) return;
    const clustered_index row = index / (SEISMIC_DIM_K * SEISMIC_DIM_B);
    const clustered_index col = index % (SEISMIC_DIM_K * SEISMIC_DIM_B);
    const int cluster = selected[(row * SEISMIC_SELECTED_STRIDE_0 + (col / SEISMIC_DIM_B) * SEISMIC_SELECTED_STRIDE_1)];
    const int token = cluster >= 0 && clustered_index(cluster) < SEISMIC_DIM_C
        ? token_order[(clustered_index(cluster) * SEISMIC_TOKEN_ORDER_STRIDE_0 + (col % SEISMIC_DIM_B) * SEISMIC_TOKEN_ORDER_STRIDE_1)] : -1;
    rows[row * SEISMIC_RESULT_0_STRIDE_0 + col * SEISMIC_RESULT_0_STRIDE_1] = token;
}
