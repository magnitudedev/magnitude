// Shared pieces of the routed (mixture-of-experts) CUDA entries built on the
// projection family (`routed_expand`, `_output`, `_experts`): the record an
// expert's weights are read from and the grouped blocks' row source. The
// counterpart of `metal/lib/routed/routed.h`. An expert-stacked weight is a
// record port, bound as its records' address table, and an expert's weights
// are rows 0.. of its record's base (`slab::region(table, record)`,
// `ROUTED_RECORD`), wherever that record lies.

#include <seismic/slab.cuh>
#include "decode.cuh"

namespace routed {

using element::u32;
using element::u64;
using element::u8;

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1.
__device__ __forceinline__ u32 block_rows(const int *order, u64 stride, u32 rows) {
    u32 low = 0, high = rows;
    while (low < high) {
        const u32 middle = (low + high) / 2;
        if (order[middle * stride] >= 0)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

// The activation rows of a grouped block (a GEMM row source over `rows`, the
// A rows or their s8 rows, `projection::gemm_source`): tile row m reads row
// order[m] of `rows`. The GEMM reads only the block's live rows.
template <class Rows> struct GroupedRows {
    Rows rows;
    const int *order;
    u64 order_stride;
    __device__ __forceinline__ u64 at(u64 m) const { return (u64)order[m * order_stride]; }
    __device__ __forceinline__ const u8 *row(u64 m) const { return rows.row(at(m)); }
    __device__ __forceinline__ const float2 *groups_of(u64 m) const { return rows.groups_of(at(m)); }
    __device__ __forceinline__ bool wide() const { return rows.wide(); }
    __host__ __device__ static constexpr bool entries() { return Rows::entries(); }
};
template <class Rows>
__device__ __forceinline__ GroupedRows<Rows> grouped_rows(const Rows &rows, const int *order, u64 order_stride) {
    return GroupedRows<Rows>{rows, order, order_stride};
}

// Decode choices that share an expert (M <= VERIFY_ROWS rows of K choices, a row's
// choices distinct experts). Choice c = m * K + k computes when it is its
// expert's first choice in that order, once for every choice of the expert:
// a GEMV row per choice, whose result does not depend on its neighbours, so
// each choice's bits are those of its own one-row GEMV. Routes are read from
// L2 (another launch wrote them).
__device__ __forceinline__ int route(const int *routes, u64 row_stride, u64 slot_stride, u32 slots, u32 choice) {
    return __ldcg(routes + (choice / slots) * row_stride + (choice % slots) * slot_stride);
}

// Whether `choice`, of `expert`, is its expert's first choice: no earlier
// row names the expert (its own row names it once).
__device__ __forceinline__ bool first_of(const int *routes, u64 row_stride, u64 slot_stride, u32 slots, u32 choice,
                                         int expert) {
    for (u32 c = 0; c < choice / slots * slots; ++c)
        if (route(routes, row_stride, slot_stride, slots, c) == expert)
            return false;
    return true;
}

// Whether `choice` is its expert's first choice.
__device__ __forceinline__ bool first_choice(const int *routes, u64 row_stride, u64 slot_stride, u32 slots,
                                             u32 choice) {
    return first_of(routes, row_stride, slot_stride, slots, choice,
                    route(routes, row_stride, slot_stride, slots, choice));
}

// The choices of `expert` from `choice` (one of its) in order, one a row:
// `choice`, then each later row's that names it, into `chosen`; returns
// their count.
__device__ __forceinline__ u32 choices_of(const int *routes, u64 row_stride, u64 slot_stride, u32 rows, u32 slots,
                                          u32 choice, int expert, u32 (&chosen)[SEISMIC_CONST_VERIFY_ROWS]) {
    u32 count = 0;
    chosen[count++] = choice;
    for (u32 m = choice / slots + 1; m < rows; ++m)
        for (u32 k = 0; k < slots; ++k)
            if (route(routes, row_stride, slot_stride, slots, m * slots + k) == expert) {
                chosen[count++] = m * slots + k;
                break;
            }
    return count;
}

// The choices of `choice`'s expert in order (at most one per row), into
// `chosen`; returns their count.
__device__ __forceinline__ u32 expert_choices(const int *routes, u64 row_stride, u64 slot_stride, u32 rows,
                                              u32 slots, u32 choice, u32 (&chosen)[SEISMIC_CONST_VERIFY_ROWS]) {
    return choices_of(routes, row_stride, slot_stride, rows, slots, choice,
                      route(routes, row_stride, slot_stride, slots, choice), chosen);
}

// A gathered choice's expert and the GEMV rows of its choices, in the
// block's shared memory (`gather`): row r reads the activation row at
// element offset rows[r] and publishes into the product at element offset
// products[r], `count` rows.
struct Gathered {
    int expert;
    u32 count;
    u64 rows[SEISMIC_CONST_VERIFY_ROWS], products[SEISMIC_CONST_VERIFY_ROWS];
};

// Block-collective, by every thread before any leaves: whether `choice` is
// its expert's first choice, and where it is, its expert and every choice
// of it into `table`: choice (m, k) reads activation row m at m *
// `row_stride` and publishes at m * `product0` + k * `product1`. At one row
// every choice is its expert's first and gathers its own row alone.
__device__ __forceinline__ bool gather(Gathered &table, const int *routes, u64 routes0, u64 routes1, u32 rows,
                                       u32 slots, u32 choice, u64 row_stride, u64 product0, u64 product1) {
    const int expert = route(routes, routes0, routes1, slots, choice);
    if (!first_of(routes, routes0, routes1, slots, choice, expert))
        return false;
    if (threadIdx.x == 0) {
        u32 chosen[SEISMIC_CONST_VERIFY_ROWS];
        table.expert = expert;
        table.count = choices_of(routes, routes0, routes1, rows, slots, choice, expert, chosen);
        for (u32 r = 0; r < table.count; ++r) {
            const u64 m = chosen[r] / slots, k = chosen[r] % slots;
            table.rows[r] = m * row_stride;
            table.products[r] = m * product0 + k * product1;
        }
    }
    __syncthreads();
    return true;
}

// A gathered GEMV's epilogue: `Epi` (its row stride 0) with row m
// publishing at element offset offsets[m] (`Gathered::products`).
template <class Epi> struct ChoiceRows {
    Epi epi;
    const u64 *offsets;
    __device__ __forceinline__ void operator()(u32 m, u64 n, float a, float b) const { epi(0, offsets[m] + n, a, b); }
    __device__ __forceinline__ void pair(u32 m, u64 n, float a0, float a1, float b0, float b1) const {
        projection::epilogue_pair(epi, 0, offsets[m] + n, a0, a1, b0, b1, 0);
    }
};

// Activation rows read in place at element offsets: GEMV row r reads
// x[offsets[r] ..].
struct GatheredRows {
    const u8 *x;
    const u64 *offsets;
    static constexpr bool STAGED = false;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return projection::Act::round(element::at<projection::Act>(x, offsets[m] + k));
    }
    __device__ __forceinline__ u32 pair(const float *, u32 m, u64 k) const {
        return *reinterpret_cast<const u32 *>(x + (offsets[m] + k) * 2);
    }
    __device__ __forceinline__ uint4 octet(const float *, u32 m, u64 k) const {
        return *reinterpret_cast<const uint4 *>(x + (offsets[m] + k) * 2);
    }
};

// The grouped expert GEMMs (`routed_experts`, `routed_experts_up`): block
// tiles of 32 rows x 128 columns, one warp row (each weight tile is decoded
// once per block: measured on GB10 with iq2xxs experts, 1.7x the rate of two
// warp rows of four tiles). Weights whose rows hold fewer than 64 code bytes
// per k-block (the 4- to 6-bit codes) stage k-blocks in spans of two, so each
// warp copy reads 64 contiguous bytes of a row (`GemmStreamLoader::codes`).
// The spanned layout must fit the declared shared bytes (`DECLARED`), else
// the 3-stage single k-block pipeline runs, as the int8 GEMM (S8) always
// does: it stages one k-block a stage (`gemm_segment_s8`).
using projection::Pick;
template <class W> __device__ constexpr bool narrow_rows() { return W::CHUNKS < 64; }
template <class WA, class WB, int DECLARED, bool S8> struct RoutedShape {
    using Spanned = projection::GemmShape<32, 1, 8, 4, false, 1, 2>;
    using Single = projection::GemmShape<32, 1, 8, 3>;
    static constexpr bool SPANNED = !S8 && narrow_rows<WA>() && narrow_rows<WB>() &&
                                    Spanned::template LAYOUT_BYTES<WA, WB, false> <= DECLARED;
    using Type = typename Pick<SPANNED, Spanned, Single>::Type;
};
constexpr u32 TILE_M = 32;

// One block of a grouped expert GEMM over `source` (A rows, or s8 rows where
// S8), on either path of `projection::gemm_rows`.
template <int DECLARED, bool S8, class Operands, class Rows, class WA, class WB, class Epi>
__device__ __forceinline__ void routed_gemm(u8 *shared, const Rows &source, u32 M,
                                            u64 K, u64 nblock, u64 rows,
                                            const WA &wa, const WB &wb, const Epi &epi) {
    using Shape = typename RoutedShape<WA, WB, DECLARED, S8>::Type;
    static_assert(Shape::BM == TILE_M, "the declared row tiles");
    projection::gemm_rows<Shape, S8, Operands>(shared, source, M, K, nblock, rows, wa, wb, epi);
}


// The source a decode expert entry reads choice (m, k)'s weights from, of
// expert `expert` (`decode_source`, routed.seismic): the expert itself
// without sources (`rows` 0, a resident layer), else the step's source
// (`sources`' SOURCE_RECORD row at strides `row_stride`, `choice_stride`): a
// bank record (the primary stack), a host tier record (the secondary, read
// in place over the link), or none (a choice that reads no record and
// contributes nothing; when is stated there).
__device__ __forceinline__ slab::Source decode_source(const u8 *sources, u64 rows, u64 row_stride,
                                                      u64 choice_stride, u64 m, u64 k, int expert) {
    return slab::Source{rows == 0 ? expert : reinterpret_cast<const int *>(sources)[m * row_stride + k * choice_stride]};
}

// A choice with no record's (`decode_source` -1) share of a decode GEMV's
// `rows` output rows: this block's (`Shape::GROUPS` tile groups of TPW
// 16-row tiles), each zeroed by `zero(n)`, by every thread of the block.
template <class Shape, class Zero> __device__ __forceinline__ void zero_block_rows(u64 rows, Zero zero) {
    constexpr u64 BLOCK_ROWS = (u64)Shape::GROUPS * Shape::TPW * 16;
    const u64 first = (u64)blockIdx.x * BLOCK_ROWS;
    for (u64 n = first + threadIdx.x; n < first + BLOCK_ROWS && n < rows; n += blockDim.x)
        zero(n);
}

// The weights of slot W in the record at `base`: its rows 0.. from there.
#define ROUTED_RECORD(W, base) W##_MATRIX(base, 0)

// Weight slot W's matrix `matrix` (a `slab::Matrix` naming a record in an
// address table): dereference its record's region, then read rows 0..
#define ROUTED_MATRIX(W, matrix) W##_MATRIX(slab::region((matrix).stack, (matrix).record), 0)

__device__ __forceinline__ void write_grid(u8 *grid, u64 x, u64 y, u64 z) {
    int *words = reinterpret_cast<int *>(grid);
    words[0] = (int)x;
    words[1] = (int)y;
    words[2] = (int)z;
}

// The grids of an expert pass over rows[0] grouped rows and used[0] blocks
// of `tile_rows` rows (`routed_experts`): a stage block per row, a quantize
// block per block row, and the expand's and down's column blocks by the
// TILE_M row tiles of each used block.
__device__ __forceinline__ void expert_grids(const u8 *rows, const u8 *used, u64 tile_rows, u64 expand_columns,
                                             u64 down_columns, u8 *stage, u8 *expand, u8 *quantize, u8 *down) {
    const u64 grouped = (u64)*reinterpret_cast<const int *>(rows);
    const u64 blocks = (u64)*reinterpret_cast<const int *>(used);
    const u64 tiles = (tile_rows + TILE_M - 1) / TILE_M;
    write_grid(stage, grouped, 1, 1);
    write_grid(expand, expand_columns, tiles, blocks);
    write_grid(quantize, blocks * tile_rows, 1, 1);
    write_grid(down, down_columns, tiles, blocks);
}

} // namespace routed

// The source holding expert `expert`'s weights for step `q`: the entry the
// layer's map (`map`, `mapped` rows of MAP_WORDS at strides `map0`, `map1`)
// holds for the step (`streaming_held`: >= 0 a bank record, -2 - s scratch
// slot s, -1 skipped), or -2 - `expert` without a map (`mapped` == 0). A
// macro, expanded where it is used, by the entries that include
// `lib/streaming/control.cuh` (`routed_experts`, `routed_experts_up`).
#define ROUTED_EXPERT_SLOT(map, mapped, map0, map1, q, expert)                                             \
    ((mapped) == 0 ? -2 - (int)(expert) : streaming_held((map) + (u64)(expert) * (map0), (map1), (q)))
