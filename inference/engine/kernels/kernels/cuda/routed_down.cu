// Decode output of the general routed feed-forward (M <= VERIFY_ROWS): the
// down projections (`lib/routed/down.cuh`) into scratch, the last arriving
// block of a channel tile summing them (`lib/routed/down_sum.cuh`), without
// a CPU site.
#define KERNEL_A SEISMIC_ELEMENT_A
#define ROUTED_SOURCE_ROWS SEISMIC_DIM_XS
#define ROUTED_PROJECTED_OUT element::out_over_scratch<activation::Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_PROJECTED reinterpret_cast<const activation::Act::storage *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PUBLISHED))
#define ROUTED_SITE_AWAIT(m)
#define ROUTED_SITE_ADD(m, n, selected)
#define ROUTED_ARRIVALS reinterpret_cast<unsigned *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_ARRIVALS))
// The last of row m's `blocks` blocks of the tile to arrive sums its
// `channels` channels from `first`, 4 contiguous channels per iteration.
#define ROUTED_DOWN_ARRIVED(m, blocks, arrival, first, channels)                                           \
    __shared__ unsigned last;                                                                              \
    if (!arrive::last(ROUTED_ARRIVALS + (arrival), (blocks), &last))                                       \
        return;                                                                                            \
    const projection::u64 arrived_first = (first);                                                         \
    const projection::u64 arrived_limit = arrived_first + (channels);                                      \
    const projection::u64 arrived_end = arrived_limit < SEISMIC_DIM_H ? arrived_limit : SEISMIC_DIM_H;     \
    if (arrived_first < arrived_end) {                                                                     \
        const projection::u64 arrived_end4 = arrived_first + ((arrived_end - arrived_first) & ~3ull);      \
        const activation::Act::storage *arrived_pub =                                                      \
            ROUTED_PROJECTED + (m) * SEISMIC_DIM_K * SEISMIC_DIM_H;                                        \
        for (projection::u64 n = arrived_first + (projection::u64)threadIdx.x * 4ull; n < arrived_end4;    \
             n += (projection::u64)blockDim.x * 4ull)                                                      \
            ROUTED_DOWN_JOIN4(m, n, arrived_pub);                                                          \
        for (projection::u64 n = arrived_end4 + threadIdx.x; n < arrived_end; n += blockDim.x)             \
            ROUTED_DOWN_JOIN(m, n, arrived_pub);                                                           \
    }
// The last of the tile's `blocks` blocks to arrive sums its channels of
// every row, 4 contiguous channels per iteration with 32-bit indexing.
#define ROUTED_DOWN_GATHERED_ARRIVED(blocks, arrival, first, channels)                                     \
    __shared__ unsigned last;                                                                              \
    if (!arrive::last(ROUTED_ARRIVALS + (arrival), (blocks), &last))                                       \
        return;                                                                                            \
    const projection::u64 arrived_first = (first);                                                         \
    const projection::u64 arrived_limit = arrived_first + (channels);                                      \
    const projection::u64 arrived_end = arrived_limit < SEISMIC_DIM_H ? arrived_limit : SEISMIC_DIM_H;     \
    if (arrived_first < arrived_end) {                                                                     \
        const projection::u32 arrived_rows = static_cast<projection::u32>(SEISMIC_DIM_M);                  \
        const projection::u64 arrived_row_stride = SEISMIC_DIM_K * SEISMIC_DIM_H;                          \
        if (arrived_end == arrived_limit && ((channels) & 3ull) == 0ull) {                                 \
            constexpr projection::u32 arrived_quads = static_cast<projection::u32>((channels) >> 2);       \
            const projection::u32 arrived_total = arrived_rows * arrived_quads;                            \
            for (projection::u32 idx = threadIdx.x; idx < arrived_total; idx += blockDim.x) {              \
                const projection::u32 row = idx / arrived_quads;                                           \
                const projection::u32 q = idx - row * arrived_quads;                                       \
                const projection::u64 n = arrived_first + (projection::u64)q * 4ull;                       \
                ROUTED_DOWN_JOIN4(row, n, ROUTED_PROJECTED + (projection::u64)row * arrived_row_stride);   \
            }                                                                                              \
        } else {                                                                                           \
            const projection::u32 arrived_span = static_cast<projection::u32>(arrived_end - arrived_first); \
            const projection::u32 arrived_quads = arrived_span >> 2;                                       \
            if (arrived_quads != 0u) {                                                                     \
                const projection::u32 arrived_total = arrived_rows * arrived_quads;                        \
                for (projection::u32 idx = threadIdx.x; idx < arrived_total; idx += blockDim.x) {          \
                    const projection::u32 row = idx / arrived_quads;                                       \
                    const projection::u32 q = idx - row * arrived_quads;                                   \
                    const projection::u64 n = arrived_first + (projection::u64)q * 4ull;                   \
                    ROUTED_DOWN_JOIN4(row, n,                                                              \
                                      ROUTED_PROJECTED + (projection::u64)row * arrived_row_stride);       \
                }                                                                                          \
            }                                                                                              \
            const projection::u32 arrived_tail = arrived_span & 3u;                                        \
            if (arrived_tail != 0u) {                                                                      \
                const projection::u64 arrived_tail_base =                                                  \
                    arrived_first + (projection::u64)arrived_quads * 4ull;                                 \
                const projection::u32 arrived_tail_total = arrived_rows * arrived_tail;                    \
                for (projection::u32 idx = threadIdx.x; idx < arrived_tail_total; idx += blockDim.x) {     \
                    const projection::u32 row = idx / arrived_tail;                                        \
                    const projection::u32 t = idx - row * arrived_tail;                                    \
                    ROUTED_DOWN_JOIN(row, arrived_tail_base + t,                                           \
                                     ROUTED_PROJECTED + (projection::u64)row * arrived_row_stride);        \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
    }
#include "lib/core/arrive.cuh"
#include "lib/routed/down_sum.cuh"
#include "lib/routed/down.cuh"

#ifdef SEISMIC_FORMING_ROUTED_DOWN_STAGE
__global__ void routed_down_stage(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_STAGE_BODY }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_SUMMED
template <int TPW, int KSPLIT>
__global__ void routed_down_summed(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_SUMMED_BODY(TPW, KSPLIT) }
#endif

#ifdef SEISMIC_FORMING_ROUTED_DOWN_GATHERED
template <int TPW, int KSPLIT>
__global__ void routed_down_gathered(SEISMIC_KERNEL_PARAMS) { ROUTED_DOWN_GATHERED_BODY(TPW, KSPLIT) }
#endif
