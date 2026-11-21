// Shared pieces of the routed (mixture-of-experts) CUDA entries built on the
// projection family (`routed_expand`, `_output`, `_experts`): the record an
// expert's weights are read from and the grouped blocks' row source. The
// counterpart of `metal/lib/routed/routed.h`. An expert-stacked weight is a
// record port, bound as its records' address table, and an expert's weights
// are rows 0.. of its record's base (`slab::region(table, record)`,
// `PACKETS_RECORD`), wherever that record lies.

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

// Decode choices that read one record (M <= VERIFY_ROWS rows of K choices, a row's
// choices distinct records). A choice's record is named by `record(c)` for choice
// c = m * K + k (`decode_source`: a bank or tier record, or -1 for none). Choice c
// computes when it is the first of the choices reading its record, once for every
// choice reading it: a GEMV row per choice, whose result does not depend on its
// neighbours, so each choice's bits are those of its own one-row GEMV. A choice
// reading no record is its own group (it zeroes its product, whatever other rows
// do). Routes are read from L2 (another launch wrote them).
__device__ __forceinline__ int route(const int *routes, u64 row_stride, u64 slot_stride, u32 slots, u32 choice) {
    return __ldcg(routes + (choice / slots) * row_stride + (choice % slots) * slot_stride);
}

// A choice's grouping identity in a gathered decode launch: its expert, the
// record code it reads (-1 for none), and whether it is active in this pass.
struct ChoiceKey {
    int expert;
    int record;
    bool active;
};

// A gathered choice's expert and the GEMV rows of its choices, in the
// block's shared memory (`gather`): row r reads the activation row at
// element offset rows[r] and publishes into the product at element offset
// products[r], `count` rows.
struct Gathered {
    int expert;
    u32 count;
    u64 rows[SEISMIC_CONST_VERIFY_ROWS], products[SEISMIC_CONST_VERIFY_ROWS];
};

// Block-collective, by every thread before any leaves: warp 0 resolves
// whether `choice` is the first active choice of its (expert, record)
// (inactive choices do not run and do not shadow active ones; an active
// choice reading no record is its own singleton group), and where it is,
// gathers its expert and every active choice sharing it into `table`:
// choice (m, k) reads activation row at m * `row0` + k * `row1` and
// publishes at m * `product0` + k * `product1`. The forms run at most
// VERIFY_ROWS rows, a row's choices distinct records, so `table` is never
// full when another choice is found; one found then breaks that invariant,
// and the block traps rather than drop it.
template <class Key>
__device__ __forceinline__ bool gather(Gathered &table, Key key, u32 rows, u32 slots, u32 choice,
                                       u64 row0, u64 row1, u64 product0, u64 product1) {
    if (threadIdx.x < 32u) {
        const u32 lane = threadIdx.x;
        ChoiceKey mine{};
        if (lane == 0u)
            mine = key(choice);
        mine.expert = __shfl_sync(0xffffffffu, mine.expert, 0);
        mine.record = __shfl_sync(0xffffffffu, mine.record, 0);
        mine.active = __shfl_sync(0xffffffffu, mine.active ? 1 : 0, 0) != 0;

        u32 count = 0u;
        if (mine.active) {
            if (mine.record == -1) {
                if (lane == 0u) {
                    const u64 m = choice / slots, k = choice % slots;
                    table.rows[0] = m * row0 + k * row1;
                    table.products[0] = m * product0 + k * product1;
                }
                count = 1u;
            } else {
                const u32 total = rows * slots;
                const u32 lt_mask = (1u << lane) - 1u;
                for (u32 base = 0u; base < total; base += 32u) {
                    const u32 c = base + lane;
                    bool match = false;
                    if (c < total) {
                        if (c == choice) {
                            match = true;
                        } else {
                            const ChoiceKey other = key(c);
                            match = other.active && other.expert == mine.expert && other.record == mine.record;
                        }
                    }
                    const u32 mask = __ballot_sync(0xffffffffu, match);
                    const u32 earlier_mask =
                        base >= choice ? 0u : (choice - base >= 32u ? 0xffffffffu : (1u << (choice - base)) - 1u);
                    if ((mask & earlier_mask) != 0u) {
                        count = 0u;
                        break;
                    }
                    const u32 chunk_hits = __popc(mask);
                    if (chunk_hits != 0u) {
                        if (count + chunk_hits > (u32)SEISMIC_CONST_VERIFY_ROWS)
                            __trap();
                        if (match) {
                            const u32 pos = count + __popc(mask & lt_mask);
                            const u64 m = c / slots, k = c % slots;
                            table.rows[pos] = m * row0 + k * row1;
                            table.products[pos] = m * product0 + k * product1;
                        }
                        count += chunk_hits;
                    }
                }
            }
        }
        if (lane == 0u) {
            table.expert = mine.expert;
            table.count = count;
        }
    }
    __syncthreads();
    return table.count != 0u;
}

template <class Key>
__device__ __forceinline__ bool gather(Gathered &table, Key key, const int *, u64, u64,
                                       u32 rows, u32 slots, u32 choice, u64 row_stride, u64 product0, u64 product1) {
    return gather(table, key, rows, slots, choice, row_stride, 0, product0, product1);
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
// holds for the step (`streaming_held_input`: >= 0 a bank record, -2 - s
// scratch slot s, -1 skipped), or -2 - `expert` without a map (`mapped` ==
// 0). A macro, expanded where it is used, by the entries that include
// `lib/streaming/control.cuh` (`routed_experts`, `routed_experts_up`).
#define ROUTED_EXPERT_SLOT(map, mapped, map0, map1, q, expert)                                             \
    ((mapped) == 0 ? -2 - (int)(expert) : streaming_held_input((map) + (u64)(expert) * (map0), (map1), (q)))
