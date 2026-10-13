// selection_notice: notice[row] = selection[row], a (token, status) pair per
// row; a thread per row.

extern "C" __global__ void selection_notice(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const int *selection = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    int *notice = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_NOTICE));
    const u64 row = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    for (u64 field = 0; field < 2; ++field)
        notice[row * SEISMIC_NOTICE_STRIDE_0 + field * SEISMIC_NOTICE_STRIDE_1] =
            selection[row * SEISMIC_SELECTION_STRIDE_0 + field * SEISMIC_SELECTION_STRIDE_1];
}
