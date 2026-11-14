// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the rows16 decoders, a thread per value, and round
// once to the activation element. Slot s is row s % P of record s / P, read
// where its storage holds the record (`records (slots)`).
#define ROWS16_T SEISMIC_SLOTS
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"
#include <seismic/rows16.cuh>
#include <seismic/slab.cuh>

using element::u64;

extern "C" __global__ void stored_rows(SEISMIC_KERNEL_PARAMS) {
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const int *directory = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY));
    const auto rows = SEISMIC_RESULT_0_OUT;
    const unsigned char *slots = SEISMIC_PTR(SEISMIC_BUFFER_SLOTS);
    const u64 column = (u64)blockIdx.y * 256ull + threadIdx.x;
    if (column >= SEISMIC_DIM_W)
        return;
    for (u64 row = blockIdx.x; row < SEISMIC_DIM_M; row += gridDim.x) {
        const long long held = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                             tokens[row * SEISMIC_TOKENS_STRIDE_0]);
        // A row without a slot reads zeros.
        if (held < 0) {
            rows.put(row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1, 0.0f);
            continue;
        }
        const u64 slot = (u64)held, per = SEISMIC_DIM_P;
        rows.put(row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1,
                 ROWS16_T_AT(slab::region(slots, slot / per)).value(slot % per, column));
    }
}
