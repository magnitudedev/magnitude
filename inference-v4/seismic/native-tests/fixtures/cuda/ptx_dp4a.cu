// One thread per row: pack the row's four byte values little-endian and
// accumulate their dot product with `dp4a`.

__device__ __forceinline__ unsigned ptx_dp4a_quad(const int *values, unsigned long long row,
                                                  unsigned long long s0, unsigned long long s1) {
    unsigned quad = 0;
    for (unsigned j = 0; j < 4; ++j) {
        quad |= ((unsigned)values[row * s0 + j * s1] & 0xffu) << (8 * j);
    }
    return quad;
}

extern "C" __global__ void ptx_dp4a_s8(SEISMIC_KERNEL_PARAMS) {
    const int *a = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_A));
    const int *b = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_B));
    const int *acc = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ACC));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SEISMIC_DIM_N) {
        return;
    }
    result[row * SEISMIC_RESULT_0_STRIDE_0] = seismic_dp4a_s8(
        (int)ptx_dp4a_quad(a, row, SEISMIC_A_STRIDE_0, SEISMIC_A_STRIDE_1),
        (int)ptx_dp4a_quad(b, row, SEISMIC_B_STRIDE_0, SEISMIC_B_STRIDE_1),
        acc[row * SEISMIC_ACC_STRIDE_0]);
}

extern "C" __global__ void ptx_dp4a_u8s8(SEISMIC_KERNEL_PARAMS) {
    const int *a = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_A));
    const int *b = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_B));
    const int *acc = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ACC));
    int *result = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SEISMIC_DIM_N) {
        return;
    }
    result[row * SEISMIC_RESULT_0_STRIDE_0] = seismic_dp4a_u8s8(
        ptx_dp4a_quad(a, row, SEISMIC_A_STRIDE_0, SEISMIC_A_STRIDE_1),
        (int)ptx_dp4a_quad(b, row, SEISMIC_B_STRIDE_0, SEISMIC_B_STRIDE_1),
        acc[row * SEISMIC_ACC_STRIDE_0]);
}
