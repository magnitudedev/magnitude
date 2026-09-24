// One thread per element.

extern "C" __global__ void ptx_rounded(SEISMIC_KERNEL_PARAMS) {
    const float *a = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_A));
    const float *b = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_B));
    const float *c = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_C));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SEISMIC_DIM_N) {
        return;
    }
    const float x = a[i * SEISMIC_A_STRIDE_0];
    const float y = b[i * SEISMIC_B_STRIDE_0];
    const float z = c[i * SEISMIC_C_STRIDE_0];
    const unsigned long long r0 = SEISMIC_RESULT_0_STRIDE_0, r1 = SEISMIC_RESULT_0_STRIDE_1;
    result[i * r1] = seismic_fma_rn(x, y, z);
    result[r0 + i * r1] = seismic_mul_rn(x, y);
    result[2 * r0 + i * r1] = seismic_add_rn(x, y);
}
