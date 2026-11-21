typedef unsigned long long clustered_index;

extern "C" __global__ void cluster_token_rows(SEISMIC_KERNEL_PARAMS) {
    const int *selected = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED));
    const int *token_order = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ORDER));
    int *rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const clustered_index offset = (clustered_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= SEISMIC_DIM_B) return;
    for (clustered_index row = (clustered_index)blockIdx.z; row < SEISMIC_DIM_M; row += (clustered_index)gridDim.z) {
        const clustered_index selected_row = row * SEISMIC_SELECTED_STRIDE_0;
        const clustered_index result_row = row * SEISMIC_RESULT_0_STRIDE_0;
        for (clustered_index rank = (clustered_index)blockIdx.y; rank < SEISMIC_DIM_K; rank += (clustered_index)gridDim.y) {
            const int cluster = selected[selected_row + rank * SEISMIC_SELECTED_STRIDE_1];
            const int token = cluster >= 0 && clustered_index(cluster) < SEISMIC_DIM_C
                ? token_order[clustered_index(cluster) * SEISMIC_TOKEN_ORDER_STRIDE_0 + offset * SEISMIC_TOKEN_ORDER_STRIDE_1]
                : -1;
            const clustered_index col = rank * SEISMIC_DIM_B + offset;
            rows[result_row + col * SEISMIC_RESULT_0_STRIDE_1] = token;
        }
    }
}
