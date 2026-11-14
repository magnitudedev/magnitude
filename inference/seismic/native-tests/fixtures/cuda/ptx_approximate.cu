// One thread per element.
#include <seismic/element.cuh>

extern "C" __global__ void ptx_approximate(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const auto result = SEISMIC_RESULT_0_OUT;
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SEISMIC_DIM_N) {
        return;
    }
    const float value = x[i * SEISMIC_X_STRIDE_0];
    const unsigned long long r0 = SEISMIC_RESULT_0_STRIDE_0, r1 = SEISMIC_RESULT_0_STRIDE_1;
    result.put(i * r1, seismic_ex2_approx(value));
    result.put(r0 + i * r1, seismic_lg2_approx(value));
    result.put(2 * r0 + i * r1, seismic_rcp_approx(value));
    result.put(3 * r0 + i * r1, seismic_rsqrt_approx(value));
    result.put(4 * r0 + i * r1, seismic_tanh_approx(value));
}
