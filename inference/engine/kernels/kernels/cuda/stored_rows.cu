// Read directory-selected slots, rows16 rows of the table's resident
// representation, through the gather decoders (<seismic/gather.cuh>), a
// thread per value, and round once to the activation element. Slot s is row s % P of record s / P, read
// where its storage holds the record (`records (slots)`).
#define KERNEL_T SEISMIC_SLOTS
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"
#include <seismic/gather.cuh>
#include <seismic/slab.cuh>

using element::u64;

extern "C" __global__ void stored_rows(SEISMIC_KERNEL_PARAMS) {
    __shared__ long long shared_held;
    __shared__ const unsigned char *shared_region;
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const int *directory = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY));
    const auto rows = SEISMIC_RESULT_0_OUT;
    const unsigned char *slots = SEISMIC_PTR(SEISMIC_BUFFER_SLOTS);
    const u64 per = SEISMIC_DIM_P;
    const u64 column = (u64)blockIdx.y * 256ull + threadIdx.x;
    const bool in_bounds = column < SEISMIC_DIM_W;
    for (u64 row = blockIdx.x; row < SEISMIC_DIM_M; row += gridDim.x) {
        if (row != blockIdx.x)
            __syncthreads();
        if (threadIdx.x == 0) {
            const long long held = gathered::row(directory, SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                                 tokens[row * SEISMIC_TOKENS_STRIDE_0]);
            shared_held = held;
            shared_region = held >= 0 ? slab::region(slots, (u64)held / per) : nullptr;
        }
        __syncthreads();
        if (!in_bounds)
            continue;
        const long long held = shared_held;
        const u64 out = row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1;
        // A row without a slot reads zeros.
        if (held < 0) {
            rows.put(out, 0.0f);
            continue;
        }
        rows.put(out, packets::gather_value(KERNEL_T_AT(shared_region), (u64)held % per, column));
    }
}
