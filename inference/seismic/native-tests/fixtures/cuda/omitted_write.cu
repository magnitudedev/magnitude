#include <seismic/element.cuh>

extern "C" __global__ void omitted_write(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const auto output = SEISMIC_RESULT_0_OUT;
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (SEISMIC_TUNE_WRITE && index < SEISMIC_DIM_N) {
        output.put(index * SEISMIC_RESULT_0_STRIDE_0, x[index * SEISMIC_X_STRIDE_0]);
    }
}
