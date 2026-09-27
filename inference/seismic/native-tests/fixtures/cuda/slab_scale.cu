#include <seismic/slab.cuh>

extern "C" __global__ void slab_scale(SEISMIC_KERNEL_PARAMS) {
    const auto *table = reinterpret_cast<const unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    auto *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const unsigned long long row = blockIdx.x;
    const unsigned long long slab_rows = SEISMIC_PARAM_SLAB_ROWS;
    const auto *source = reinterpret_cast<const float *>(slab::region(table, row / slab_rows));
    const unsigned long long local = row % slab_rows;
    const float factor = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_FACTOR));
    for (unsigned long long column = threadIdx.x; column < SEISMIC_DIM_N; column += blockDim.x) {
        result[row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1] =
            source[local * SEISMIC_DIM_N + column] * factor;
    }
}
