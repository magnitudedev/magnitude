// feature_copy: out[row, column] = input[row, column], moved bit-for-bit from
// the input's owner-stored elements into out; one thread per 16-byte chunk along D.
#include <seismic/element.cuh>

extern "C" __global__ void feature_copy(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    typedef ELEMENT_OF(SEISMIC_OUT) Act;
    static_assert(ELEMENT_KIND(SEISMIC_INPUT) == ELEMENT_KIND(SEISMIC_OUT),
                  "feature_copy copies one element type");
    constexpr u64 CHUNK = Act::bytes >= 16 ? 1 : 16 / Act::bytes;
    const u64 chunks_per_row = ((u64)SEISMIC_DIM_D + CHUNK - 1) / CHUNK;
    const u64 index = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (u64)SEISMIC_DIM_M * chunks_per_row)
        return;
    const u64 row = index / chunks_per_row;
    const u64 column = (index % chunks_per_row) * CHUNK;
    const u64 remaining = (u64)SEISMIC_DIM_D - column;
    const u64 count = remaining < CHUNK ? remaining : CHUNK;
    const element::u8 *input = SEISMIC_PTR(SEISMIC_BUFFER_INPUT);
    const auto out = SEISMIC_OUT_OUT;
    if (SEISMIC_OUT_STRIDE_1 == 1 && SEISMIC_INPUT_STRIDE_1 == 1) {
        out.move_stored(row * (u64)SEISMIC_OUT_STRIDE_0 + column,
                        element::stored<Act>(input, row * (u64)SEISMIC_INPUT_STRIDE_0 + column),
                        count);
    } else {
        for (u64 i = 0; i < count; ++i) {
            const u64 col = column + i;
            out.move_stored(row * (u64)SEISMIC_OUT_STRIDE_0 + col * (u64)SEISMIC_OUT_STRIDE_1,
                            element::stored<Act>(
                                input, row * (u64)SEISMIC_INPUT_STRIDE_0 + col * (u64)SEISMIC_INPUT_STRIDE_1));
        }
    }
}
