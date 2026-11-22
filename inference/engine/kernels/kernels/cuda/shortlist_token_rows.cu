typedef unsigned long long shortlist_index;

extern "C" __global__ void shortlist_token_rows(SEISMIC_KERNEL_PARAMS) {
    const int *selected = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTED));
    int *rows = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const shortlist_index rank = (shortlist_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= SEISMIC_DIM_K) return;
    for (shortlist_index row = (shortlist_index)blockIdx.y; row < SEISMIC_DIM_M; row += (shortlist_index)gridDim.y) {
        const int token = selected[row * SEISMIC_SELECTED_STRIDE_0 + rank * SEISMIC_SELECTED_STRIDE_1];
        rows[row * SEISMIC_RESULT_0_STRIDE_0 + rank * SEISMIC_RESULT_0_STRIDE_1] = token;
    }
}
