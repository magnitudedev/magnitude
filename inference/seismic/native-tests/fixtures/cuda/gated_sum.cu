// One participant sums in element order after staging the whole vector in
// shared memory.
#include <seismic/element.cuh>

extern "C" __global__ void gated_staged(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const auto result = SEISMIC_RESULT_0_OUT;
    extern __shared__ float staging[];
    const unsigned long long n = SEISMIC_DIM_N;
    for (unsigned long long i = 0; i < n; ++i) {
        staging[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (unsigned long long i = 0; i < n; ++i) {
        total += staging[i];
    }
    result.put(0, total);
}

// One participant sums in element order after mirroring the vector into
// scratch.
extern "C" __global__ void gated_mirrored(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *mirror = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_MIRROR));
    const auto result = SEISMIC_RESULT_0_OUT;
    const unsigned long long n = SEISMIC_DIM_N;
    for (unsigned long long i = 0; i < n; ++i) {
        mirror[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (unsigned long long i = 0; i < n; ++i) {
        total += mirror[i];
    }
    result.put(0, total);
}
