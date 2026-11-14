// feature_copy: out[row, column] = input[row, column], moved bit-for-bit from
// the input's owner-stored elements into out; one thread per element.
#include <seismic/element.cuh>

extern "C" __global__ void feature_copy(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *input = SEISMIC_PTR(SEISMIC_BUFFER_INPUT);
    const auto out = SEISMIC_OUT_OUT;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_M * SEISMIC_DIM_D)
        return;
    const u64 row = index / SEISMIC_DIM_D;
    const u64 column = index % SEISMIC_DIM_D;
    out.move_stored(row * SEISMIC_OUT_STRIDE_0 + column * SEISMIC_OUT_STRIDE_1,
                    element::stored<ELEMENT_OF(SEISMIC_INPUT)>(
                        input, row * SEISMIC_INPUT_STRIDE_0 + column * SEISMIC_INPUT_STRIDE_1));
}
