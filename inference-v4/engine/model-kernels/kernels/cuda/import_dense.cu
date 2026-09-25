// Dense weight import over the [B, N, K] view; one thread per element in
// row-major order. The CUDA form of `metal/import_dense.metal`.

#include "lib/core/activation.cuh"

typedef ELEMENT_OF(SEISMIC_ELEMENT_E) Source;
typedef ELEMENT_OF(SEISMIC_ELEMENT_U) Destination;

extern "C" __global__ void import_dense(SEISMIC_KERNEL_PARAMS) {
    const unsigned char *source = reinterpret_cast<const unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    unsigned char *destination = reinterpret_cast<unsigned char *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= SEISMIC_DIM_B * SEISMIC_DIM_N * SEISMIC_DIM_K) return;
    const unsigned long long k = index % SEISMIC_DIM_K;
    const unsigned long long n = index / SEISMIC_DIM_K % SEISMIC_DIM_N;
    const unsigned long long b = index / SEISMIC_DIM_K / SEISMIC_DIM_N;
    element::put<Destination>(destination,
        b * SEISMIC_RESULT_0_STRIDE_0 + n * SEISMIC_RESULT_0_STRIDE_1 + k * SEISMIC_RESULT_0_STRIDE_2,
        element::at<Source>(source,
            b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1 + k * SEISMIC_SOURCE_STRIDE_2));
}
