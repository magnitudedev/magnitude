// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the rows16 decoders, a thread per value, and round
// once to the activation element.
#define ROWS16_T SEISMIC_SLOTS
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"
#include <seismic/rows16.cuh>

using element::Act;
using element::u64;
using element::u8;

extern "C" __global__ void stored_rows(SEISMIC_KERNEL_PARAMS) {
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const int *directory = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY));
    u8 *rows = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    const auto table = ROWS16_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_SLOTS));
    const u64 column = (u64)blockIdx.y * 256ull + threadIdx.x;
    if (column >= SEISMIC_DIM_W)
        return;
    for (u64 row = blockIdx.x; row < SEISMIC_DIM_M; row += gridDim.x) {
        const u64 slot = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                       tokens[row * SEISMIC_TOKENS_STRIDE_0]);
        element::put<Act>(rows, row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                          table.value(slot, column));
    }
}
