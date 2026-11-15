// The `planes` form of the `dense` projection (`dense.seismic`): a plane
// set's vocabulary GEMV over every row (a progressive head,
// `readout.seismic`), and the stage launch's formation of its rows.
//
// The entry gives the view its rows are projected onto, its epilogue and
// the formation of its rows:
//   View   `Group` (a vocabulary row's planes of one 32-code group),
//          `load(v, g)`, `decode(group)` (its codes, which serve every row)
//          and `add(acc, group, codes, x, m, lane)` (a row's sum plus the
//          group's products with its activations `x`); `stage(first, rows,
//          chunk, width, d)` stages whatever the view reads of the rows'
//          chunk beside them, into its own shared memory;
//   Epi    `begin()`, `store(m, v, logit)` (row m's sum over vocabulary row
//          v, by the lane that holds it) and `end(rows)`;
//   Formation
//          `State`, `group(state, m, g, lane, value)` (each value of a row's
//          group g, as the stage launch forms it) and `finish(state, m,
//          partials)` (block-collective, once a row).
#pragma once
#include "projection.cuh"

// The rows a block projects (grid y takes the blocks of them) and the
// columns it stages at once.
#define PROJECTION_PLANES_ROWS 8u
#define PROJECTION_PLANES_CHUNK 1024u

namespace projection {
namespace planes {

// The 32 activations of one group, as pairs (2p, 2p + 1).
struct X {
    float2 pair[16];
};

// The warp's sum in one fixed tree.
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = __fadd_rn(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Row blockIdx.x of the prologue into `rows` (A, K apart), each value handed
// to the formation as it is formed.
template <class Pro, class Formation>
__device__ __forceinline__ void stage(const Pro &pro, u32 K, element::Out<Act> rows, const Formation &formation) {
    __shared__ float factors[Pro::FACTORS > 0 ? Pro::FACTORS : 1];
    __shared__ float partials[32];
    seismic_dependency_start();
    const u32 m = blockIdx.x;
    pro.prepare_row(factors, m, partials);
    __syncthreads();
    const u32 lane = threadIdx.x % 32, warps = blockDim.x / 32;
    typename Formation::State state{};
    for (u32 g = threadIdx.x / 32; g < K / 32; g += warps) {
        const float value = pro.value(factors, m, 32ull * g + lane);
        rows.put((u64)m * K + 32ull * g + lane, value);
        formation.group(state, m, g, lane, value);
    }
    formation.finish(state, m, partials);
}

// The vocabulary GEMV of rows `first`.. (`rows` of them, at most MAXM):
// ROWS vocabulary rows per warp, lane l reading group l of each
// PROJECTION_PLANES_CHUNK-column chunk, whose activations are staged in
// `staged` (PROJECTION_PLANES_ROWS x PROJECTION_PLANES_CHUNK A values,
// quarter-major so a warp's 16-byte reads are conflict-free). Every row's
// words are loaded first, so their loads are in flight together; each
// group's codes are decoded once for every row.
template <u32 MAXM, u32 ROWS, class View, class Epi>
__device__ __forceinline__ void scan(const u8 *source, u32 first, u32 rows, u32 vocabulary, u32 d, const View &view,
                                     const Epi &epi, uint4 *staged) {
    const u32 lane = threadIdx.x % 32, warps = blockDim.x / 32;
    epi.begin();
    const u32 vocabulary_first = (blockIdx.x * warps + threadIdx.x / 32) * ROWS;
    float acc[ROWS][MAXM];
#pragma unroll
    for (u32 r = 0; r < ROWS; ++r)
#pragma unroll
        for (u32 m = 0; m < MAXM; ++m)
            acc[r][m] = 0.0f;
    for (u32 chunk = 0; chunk < d; chunk += PROJECTION_PLANES_CHUNK) {
        const u32 width = min(PROJECTION_PLANES_CHUNK, d - chunk);
        __syncthreads();
        for (u32 item = threadIdx.x; item < rows * (width / 8); item += blockDim.x) {
            const u32 m = item / (width / 8), w = item % (width / 8);
            staged[(m * 4 + w % 4) * 32 + w / 4] =
                *reinterpret_cast<const uint4 *>(source + ((u64)(first + m) * d + chunk + 8ull * w) * Act::bytes);
        }
        view.stage(first, rows, chunk, width, d);
        __syncthreads();
        if (32 * lane >= width)
            continue;
        const u32 g = chunk / 32 + lane;
        typename View::Group group[ROWS];
#pragma unroll
        for (u32 r = 0; r < ROWS; ++r)
            group[r] = view.load(min(vocabulary_first + r, vocabulary - 1), g);
#pragma unroll
        for (u32 r = 0; r < ROWS; ++r) {
            const auto codes = view.decode(group[r]);
#pragma unroll
            for (u32 m = 0; m < MAXM; ++m) {
                if (m >= rows)
                    break;
                X x;
#pragma unroll
                for (u32 q = 0; q < 4; ++q) {
                    const uint4 w = staged[(m * 4 + q) * 32 + lane];
                    x.pair[4 * q + 0] = Act::unpack2(w.x);
                    x.pair[4 * q + 1] = Act::unpack2(w.y);
                    x.pair[4 * q + 2] = Act::unpack2(w.z);
                    x.pair[4 * q + 3] = Act::unpack2(w.w);
                }
                acc[r][m] = view.add(acc[r][m], group[r], codes, x, m, lane);
            }
        }
    }
#pragma unroll
    for (u32 r = 0; r < ROWS; ++r) {
        const u32 v = vocabulary_first + r;
#pragma unroll
        for (u32 m = 0; m < MAXM; ++m) {
            if (m >= rows)
                break;
            const float logit = warp_sum(acc[r][m]);
            if (v >= vocabulary || lane != m)
                continue;
            epi.store(first + m, v, logit);
        }
    }
    epi.end(rows);
}

// Block (x, y): vocabulary rows by the block's x, the rows of block y of the
// staged rows `source` (M of them, d columns), at its row bound.
template <u32 ROWS, class View, class Epi>
__device__ __forceinline__ void run(const u8 *source, u32 M, u32 vocabulary, u32 d, const View &view, const Epi &epi) {
    __shared__ uint4 staged[PROJECTION_PLANES_ROWS * PROJECTION_PLANES_CHUNK / 8];
    const u32 first = blockIdx.y * PROJECTION_PLANES_ROWS;
    const u32 rows = min(PROJECTION_PLANES_ROWS, M - first);
    if (rows <= 1)
        scan<1, ROWS>(source, first, rows, vocabulary, d, view, epi, staged);
    else if (rows <= 2)
        scan<2, ROWS>(source, first, rows, vocabulary, d, view, epi, staged);
    else if (rows <= 4)
        scan<4, ROWS>(source, first, rows, vocabulary, d, view, epi, staged);
    else
        scan<8, ROWS>(source, first, rows, vocabulary, d, view, epi, staged);
}

} // namespace planes
} // namespace projection
