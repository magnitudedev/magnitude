// Decode output (M <= DECODE_ROWS): the fused projections
// (`lib/routed/output.cuh`) into scratch, the last arriving block of a
// channel tile joining them (`lib/routed/output_sum.cuh`), without a CPU
// site.
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_PUBLISHED reinterpret_cast<element::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_SITE_AWAIT(m)
#define ROUTED_SITE_ADD(m, n, selected)
#define ROUTED_ARRIVALS reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))
// The last of row m's `blocks` blocks of the tile to arrive joins its
// `channels` channels from `first`.
#define ROUTED_OUTPUT_ARRIVED(m, blocks, arrival, first, channels, published)                              \
    __shared__ unsigned last;                                                                              \
    if (!arrive::last(ROUTED_ARRIVALS + (arrival), (blocks), &last))                                       \
        return;                                                                                            \
    for (projection::u64 n = (first) + threadIdx.x; n < (first) + (channels) && n < SEISMIC_DIM_H;         \
         n += blockDim.x)                                                                                  \
        ROUTED_OUTPUT_JOIN(m, n, published);
// The last of the tile's `blocks` blocks to arrive joins its channels of
// every row.
#define ROUTED_OUTPUT_GATHERED_ARRIVED(blocks, arrival, first, channels, published)                        \
    __shared__ unsigned last;                                                                              \
    if (!arrive::last(ROUTED_ARRIVALS + (arrival), (blocks), &last))                                       \
        return;                                                                                            \
    for (projection::u64 item = threadIdx.x; item < SEISMIC_DIM_M * (channels); item += blockDim.x) {      \
        const projection::u64 row = item / (channels), n = (first) + item % (channels);                     \
        if (n < SEISMIC_DIM_H)                                                                             \
            ROUTED_OUTPUT_JOIN(row, n, (published) + row * (SEISMIC_DIM_K + 1) * SEISMIC_DIM_H);           \
    }
#include "lib/core/arrive.cuh"
#include "lib/routed/output_sum.cuh"
#include "lib/routed/output.cuh"
