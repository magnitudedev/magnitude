// Shared pieces of the routed (mixture-of-experts) CUDA entries built on the
// projection family (`routed_expand`, `_output`, `_experts`): the grouped
// blocks' row source. An expert's weights are matrix `expert` of its
// expert-stacked [E, N, K] tensor (`KERNEL_Wn_MATRIX`, dense or packed).

#include "../projection/projection.cuh"

namespace routed {

using element::u32;
using element::u64;
using element::u8;

// The decode GEMV kernels' block: DECODE_WARPS warps (the declarations' 128 threads). A kernel declares it as its
// launch bound (`routed_up` does) so the compiler allocates registers for that block, not for the largest one the
// hardware admits; the others adopt the bound where it is measured to pay.
constexpr int DECODE_WARPS = 4;
constexpr int DECODE_THREADS = DECODE_WARPS * 32;
template <int TPW, int KSPLIT> using DecodeShape = projection::GemvShape<DECODE_WARPS, TPW, KSPLIT, 1>;

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

// The activation rows of a grouped block (a GEMM row source): tile row m
// reads activation row order[m]. The GEMM reads only the block's live rows.
struct GroupedRows {
    const u8 *x;
    u64 stride;
    const int *order;
    u64 order_stride;
    __device__ __forceinline__ const u8 *row(u64 m) const {
        return x + (u64)order[m * order_stride] * stride * 2;
    }
    // Whether every row starts 16-byte aligned (`ActivationRows::wide`).
    __device__ __forceinline__ bool wide() const { return stride % 8 == 0 && ((u64)x & 15) == 0; }
};

// Decode choices that share an expert (M <= DECODE_ROWS rows of K choices, a row's
// choices distinct experts). Choice c = m * K + k computes when it is its
// expert's first choice in that order, once for every choice of the expert:
// a GEMV row per choice, whose result does not depend on its neighbours, so
// each choice's bits are those of its own one-row GEMV. Routes are read from
// L2 (another launch wrote them).
__device__ __forceinline__ int route(const int *routes, u64 row_stride, u64 slot_stride, u32 slots, u32 choice) {
    return __ldcg(routes + (choice / slots) * row_stride + (choice % slots) * slot_stride);
}

// Whether `choice` is its expert's first choice.
__device__ __forceinline__ bool first_choice(const int *routes, u64 row_stride, u64 slot_stride, u32 slots,
                                             u32 choice) {
    const int expert = route(routes, row_stride, slot_stride, slots, choice);
    for (u32 c = 0; c < choice; ++c)
        if (route(routes, row_stride, slot_stride, slots, c) == expert)
            return false;
    return true;
}

// The choices of `choice`'s expert in order (at most one per row), into
// `chosen`; returns their count.
__device__ __forceinline__ u32 expert_choices(const int *routes, u64 row_stride, u64 slot_stride, u32 rows,
                                              u32 slots, u32 choice, u32 (&chosen)[SEISMIC_CONST_DECODE_ROWS]) {
    const int expert = route(routes, row_stride, slot_stride, slots, choice);
    u32 count = 0;
    for (u32 c = choice; c < rows * slots; ++c)
        if (route(routes, row_stride, slot_stride, slots, c) == expert)
            chosen[count++] = c;
    return count;
}

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
// warp copy reads 64 contiguous bytes of a row (`gemm_load_codes_span`).
// The spanned layout must fit the declared shared bytes (`DECLARED`), else
// the 3-stage single k-block pipeline runs.
using projection::Pick;
template <class W> __device__ constexpr bool narrow_rows() { return W::CHUNKS < 64; }
template <class WA, class WB, int DECLARED> struct RoutedShape {
    using Spanned = projection::GemmShape<32, 1, 8, 4, false, 1, 2>;
    using Single = projection::GemmShape<32, 1, 8, 3>;
    static constexpr bool SPANNED =
        narrow_rows<WA>() && narrow_rows<WB>() && Spanned::template LAYOUT_BYTES<WA, WB, false> <= DECLARED;
    using Type = typename Pick<SPANNED, Spanned, Single>::Type;
};
constexpr u32 TILE_M = 32;

template <int DECLARED, class Operands, class Rows, class WA, class WB, class Epi>
__device__ __forceinline__ void routed_gemm(u8 *shared, const Rows &source, u32 M,
                                            u64 K, u64 nblock, u64 rows,
                                            const WA &wa, const WB &wb, const Epi &epi) {
    using Shape = typename RoutedShape<WA, WB, DECLARED>::Type;
    static_assert(Shape::BM == TILE_M, "the declared row tiles");
    projection::gemm_segment<Shape, Operands>(shared, source, M, K, nblock, rows, wa, wb, epi);
}


// The record a decode expert entry reads choice (m, k)'s weights from, of
// expert `expert` (`decode_source`, routed.seismic): the expert itself
// without sources (`rows` 0, a resident layer), else the step's source
// (`sources`' SOURCE_RECORD row at strides `row_stride`, `choice_stride`): a
// bank record (>= 0), a host tier record t as -2 - t, or -1 for none (a
// choice that reads no record and contributes nothing; when is stated
// there).
__device__ __forceinline__ int decode_source(const u8 *sources, u64 rows, u64 row_stride, u64 choice_stride, u64 m,
                                             u64 k, int expert) {
    return rows == 0 ? expert : reinterpret_cast<const int *>(sources)[m * row_stride + k * choice_stride];
}

// Where a decode choice's weights are read, of a stack's `bank` and `tier`
// records: the bank record its source names, or the tier record read in
// place over the link; no stack
// (`stack` null) for a choice with no record.
struct Place {
    const u8 *stack;
    u64 record;
    __device__ __forceinline__ static Place of(const u8 *bank, const u8 *tier, int source) {
        if (source == -1)
            return Place{nullptr, 0};
        return source >= 0 ? Place{bank, (u64)source} : Place{tier, (u64)(-2 - source)};
    }
};

// A choice with no record's (`decode_source` -1) share of a decode GEMV's
// `rows` output rows: this block's (`Shape::GROUPS` tile groups of TPW
// 16-row tiles), each zeroed by `zero(n)`, by every thread of the block.
template <class Shape, class Zero> __device__ __forceinline__ void zero_block_rows(u64 rows, Zero zero) {
    constexpr u64 BLOCK_ROWS = (u64)Shape::GROUPS * Shape::TPW * 16;
    const u64 first = (u64)blockIdx.x * BLOCK_ROWS;
    for (u64 n = first + threadIdx.x; n < first + BLOCK_ROWS && n < rows; n += blockDim.x)
        zero(n);
}

// Expert `expert`'s matrix of weight slot W: record `slot` of `records`, or
// matrix `expert` of `experts` (`ROUTED_EXPERT_SLOT`).
#define ROUTED_EXPERT_MATRIX(W, records, experts, slot, expert) \
    ((slot) >= 0 ? W##_MATRIX(records, slot) : W##_MATRIX(experts, expert))

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

// The record holding expert `expert`'s weights for step `q`: the record
// the layer's map (`map`, `mapped` rows of MAP_WORDS at strides `map0`,
// `map1`) holds for the step (`streaming_held`), or -1 for its place in the
// experts by index (the stack, or the scratch tier), which is every
// expert's without a map. A macro, expanded where it is used, by the
// entries that include `lib/streaming/control.cuh` (`routed_experts`,
// `routed_experts_up`).
#define ROUTED_EXPERT_SLOT(map, mapped, map0, map1, q, expert)                                             \
    ((mapped) == 0 ? -1 : streaming_held((map) + (u64)(expert) * (map0), (map1), (q)))
