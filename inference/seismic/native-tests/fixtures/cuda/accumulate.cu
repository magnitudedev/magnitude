#include <seismic/element.cuh>

extern "C" __global__ void accumulate(SEISMIC_KERNEL_PARAMS) {
    const auto state = SEISMIC_OUT_STATE;
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < SEISMIC_DIM_N) {
        const unsigned long long at = index * SEISMIC_STATE_STRIDE_0;
        state.put(at, __fadd_rn(state.at(at), __fadd_rn(x[index * SEISMIC_X_STRIDE_0],
                                                          static_cast<float>(SEISMIC_TUNE_BIAS) * 0.5f)));
    }
}
