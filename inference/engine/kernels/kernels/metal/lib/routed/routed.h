// Shared pieces of the routed (mixture-of-experts) Metal entries built on the
// projection family: the record an expert's weights are read from, the plain
// activation prologue they share, the grouped-row prologue of the expert
// GEMMs, the live-row count of a grouped block and the choice of a block's
// GEMM form. Every form gives the GEMM's bits, so a row's output depends on
// its own row alone, never on how many rows share its block. Weights are
// bound through the `packets.h` slots (`KERNEL_W0..3`); an expert-stacked
// weight a launch reads per step's source is a record port, bound as its
// records' address table, and an expert's weights are rows 0.. of its
// record's base (`slab::region<const uchar, packets::space>(table, record)`,
// in the bound weights' space). A resident layer's launch binds the stack
// itself (`choice_weights`).

#include <seismic/slab.h>
#include "../projection/projection.h"

namespace routed {

typedef element::Act Act;
static_assert(Act::bytes == 2, "the Metal routed family requires a bf16 or f16 activation element");

// Expert `expert`'s weights of an expert-stacked weight: record `slot` of
// `records` where the layer's map holds it for the step (`slot` >= 0), else
// its place in `experts` by index (the stack, or the scratch tier).
inline packets::weight_bytes expert_weights(device const ulong *records, device const ulong *experts, int slot,
    int expert) {
    return slot >= 0 ? slab::region<const uchar, packets::space>(records, ulong(slot))
                     : slab::region<const uchar, packets::space>(experts, ulong(expert));
}

// The record a decode expert entry reads choice (m, k)'s weights from, of
// expert `expert` (`decode_source`, routed.seismic): the expert itself
// without sources (`rows` 0, a resident layer), else the step's source
// (`sources`' SOURCE_RECORD row at strides `row_stride`, `choice_stride`);
// -1 where that names none (a choice that reads no record and contributes
// nothing; when is stated there).
inline int decode_source(device const int *sources, ulong rows, ulong row_stride, ulong choice_stride, ulong m,
                         ulong k, int expert) {
    return rows == 0 ? expert : sources[m * row_stride + k * choice_stride];
}

// The weights a decode source names (`decode_source`, not -1): a record of
// the bank's records `bank` (>= 0), or record t of the host tier's `tier` as
// -2 - t, which on unified memory the entries read in place as they read the
// bank's (both record tables).
inline packets::weight_bytes source_weights(device const ulong *bank, device const ulong *tier, int source) {
    return source >= 0 ? slab::region<const uchar, packets::space>(bank, ulong(source))
                       : slab::region<const uchar, packets::space>(tier, ulong(-2 - source));
}

// A choice with no record's product row (`decode_source` -1): `count` zero A
// elements at `out`, element stride `stride`, by `threads` threads from
// `first`.
inline void zero_row(device uchar *out, ulong stride, uint count, uint first, uint threads) {
    for (uint i = first; i < count; i += threads)
        element::put<Act>(out, ulong(i) * stride, 0.0f);
}

inline void write_grid(device int *grid, uint x, uint y) {
    grid[0] = int(x);
    grid[1] = int(y);
    grid[2] = 1;
}

// The grids of an expert pass over `rows` grouped rows and `used` blocks of
// `tile_rows` rows (`routed_experts`): a stage threadgroup per row, a
// quantize threadgroup per block row, and `subtiles` row tiles of each used
// block by the expand's and down's column tiles.
inline void expert_grids(uint rows, uint used, uint tile_rows, uint subtiles, uint expand_columns,
    uint down_columns, device int *stage, device int *expand, device int *quantize, device int *down) {
    write_grid(stage, rows, 1);
    write_grid(expand, expand_columns, used * subtiles);
    write_grid(quantize, used * tile_rows, 1);
    write_grid(down, down_columns, used * subtiles);
}

// Rows of weight tensor `base` starting at row `first`.
template <typename W>
inline projection::Weights<W> weights(packets::weight_bytes base, packets::Rows16 layout, ulong first, uint k) {
    return projection::Weights<W>{base, layout, k, nullptr, first};
}

// Choice (m, k)'s record (`decode_source`, `rows` source rows at strides
// `row_stride`, `choice_stride`) of expert `expert`, by how its experts are
// bound: as their records' address table (`ulong`), or as a resident
// layer's stack as it lies (`uchar`: a held stack, or a whole bank's home
// records, which lie so in its one buffer), whose choices read their
// experts.
inline int choice_record(device const ulong *, device const int *sources, ulong rows, ulong row_stride,
    ulong choice_stride, ulong m, ulong k, int expert) {
    return decode_source(sources, rows, row_stride, choice_stride, m, k, expert);
}
inline int choice_record(device const uchar *, device const int *, ulong, ulong, ulong, ulong, ulong, int expert) {
    return expert;
}

// Expert `expert`'s weights of an expert-stacked weight bound as it lies,
// `rows` rows an expert (expert e's from row e * rows): a resident layer's,
// a held stack or a whole bank's home records, which lie so in its one
// buffer. The base is the binding's, so the rows are one load away.
template <typename W>
inline projection::Weights<W> stack_weights(packets::weight_bytes stack, packets::Rows16 layout, int expert,
    ulong rows, uint k) {
    return weights<W>(stack, layout, ulong(expert) * rows, k);
}

// The weights of `record` (`choice_record`, not -1), `rows` rows an expert:
// a bank or tier record through their tables (`source_weights`), or of a
// resident layer's stack (`stack_weights`).
template <typename W>
inline projection::Weights<W> choice_weights(device const ulong *bank, device const ulong *tier, int record,
    packets::Rows16 layout, ulong, uint k) {
    return weights<W>(source_weights(bank, tier, record), layout, 0, k);
}
template <typename W>
inline projection::Weights<W> choice_weights(device const uchar *stack, device const ulong *, int record,
    packets::Rows16 layout, ulong rows, uint k) {
    return stack_weights<W>(stack, layout, record, rows, k);
}

// A plain activation prologue over rows of a [rows, columns] A tensor.
inline projection::Plain<Act, projection::AllRows> activation(device const uchar *x, ulong stride0, ulong stride1,
    uint columns) {
    return projection::Plain<Act, projection::AllRows>{x, stride0, stride1, columns, {}};
}

// The decode down projections of one simdgroup (the choices of a row, and
// the shared expert, run side by side, one simdgroup each): channels
// first + g + G * r (lane group g of G = 32 / LANES, r < ROWS) of weights `w`
// against the one-row product `in` of COLUMNS values, over its first
// `extent` values alone (a multiple of the packet's 32 columns; COLUMNS for
// every value), stored unrounded at
// projected[g + G * r]. The lanes of a group own the packets sub,
// sub + LANES, ... of every channel's row, as the K1 GEMV's lanes do, and
// sum them in the same order, so every projection is the GEMV's sum for the
// same LANES (`projection::gemv_body`). A lane reads its packets of the
// product straight from device memory (no staging, no barrier) and issues
// the weight loads of all its channels before it accumulates any.
template <typename W, uint ROWS, uint LANES, uint COLUMNS, typename In>
inline void project_channels(thread const In &in, thread const projection::Weights<W> &w, uint first,
    uint rows, uint extent, threadgroup float *projected, uint lane) {
    constexpr uint G = 32u / LANES;
    constexpr uint COLUMN_PACKETS = (COLUMNS + 31u) / 32u;
    // The packets read: those of the extent.
    const uint P = min(COLUMN_PACKETS, (extent + 31u) / 32u);
    // Packets per lane.
    constexpr uint PL = (COLUMN_PACKETS + LANES - 1u) / LANES;
    const uint g = lane / LANES, sub = lane % LANES;
    typename W::packet packets[ROWS][PL];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        const uint n = min(first + g + G * r, rows - 1u);
        PROJECTION_UNROLL
        for (uint i = 0; i < PL; ++i) {
            const uint p = sub + LANES * i;
            if (p < P)
                packets[r][i] = w.packet(n, p);
        }
    }
    uint4 x[PL][4];
    PROJECTION_UNROLL
    for (uint i = 0; i < PL; ++i) {
        const uint p = sub + LANES * i;
        PROJECTION_UNROLL
        for (uint step = 0; step < 4; ++step)
            x[i][step] = p < P ? in.words8(0, 32u * p + 8u * step) : uint4(0);
    }
    float acc[ROWS];
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r)
        acc[r] = 0.0f;
    PROJECTION_UNROLL
    for (uint i = 0; i < PL; ++i) {
        if (sub + LANES * i >= P)
            continue;
        PROJECTION_UNROLL
        for (uint step = 0; step < 4; ++step) {
            float4 xe, xo;
            Act::split8(x[i][step], xe, xo);
            PROJECTION_UNROLL
            for (uint r = 0; r < ROWS; ++r) {
                float4 we, wo;
                W::values(packets[r][i], step, we, wo);
                acc[r] = projection::gemv_step_dot(acc[r], we, wo, xe, xo);
            }
        }
    }
    PROJECTION_UNROLL
    for (uint r = 0; r < ROWS; ++r) {
        const float total = projection::gemv_group_sum<LANES>(acc[r]);
        if (sub == 0)
            projected[g + G * r] = total;
    }
}

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1. The
// expert GEMMs skip the MMAs of padding rows (`live_rows`).
inline uint block_rows(device const int *order, ulong stride, uint rows) {
    uint low = 0, high = rows;
    while (low < high) {
        uint middle = (low + high) / 2u;
        if (order[ulong(middle) * stride] >= 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}

// The grouped prologue of the expert GEMMs: tile row m of a block reads
// activation row `order[m]`; padding rows (-1) read zeros.
struct Grouped {
    typedef Act activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    device const int *order;
    ulong order_stride;
    uint groups() const { return 0; }
    uint width() const { return 1; }
    float norm_input(uint, uint, uint) const { return 0.0f; }
    float epsilon() const { return 0.0f; }
    uint inverse_index(uint, uint) const { return 0; }
    float value(uint m, uint k, float) const {
        int row = order[ulong(m) * order_stride];
        return row < 0 ? 0.0f
            : Act::load(reinterpret_cast<device const typename Act::storage *>(x)
                [ulong(row) * stride0 + ulong(k) * stride1]);
    }
    uint4 words8(uint m, uint k) const {
        int row = order[ulong(m) * order_stride];
        if (row < 0)
            return uint4(0);
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        return projection::words8_storage<Act>(base, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        int row = order[ulong(m) * order_stride];
        if (row < 0) {
            even = odd = float4(0.0f);
            return;
        }
        device const typename Act::storage *base =
            reinterpret_cast<device const typename Act::storage *>(x) + ulong(row) * stride0;
        projection::load8_storage<Act>(base, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0, even, odd);
    }
};

// The expert projections of one grouped block tile: the TM x TN GEMM over
// the block's `live` rows, or, on simdgroup matrices, its one-row form
// (`projection::gemm_row`) when they fit one fragment row, with the same
// bits and without the GEMM's threadgroup-wide weight staging, which few
// rows cannot amortize.
// A tensor-operation device runs the GEMM alone: its products are one
// `matmul2d` over the staged tile, whose summation order no other form can
// be shown to reproduce.
template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void expert_paired(thread const In &in, thread const Out &out, thread const projection::Weights<G> &gate,
    thread const projection::Weights<U> &up, uint live, uint m_rows, uint rows, uint k, uint tm, uint tn,
    threadgroup uchar *shared, uint sg, uint lane) {
#if !SEISMIC_HAS_TENSOR_OPS
    if (live <= projection::gemm_row_rows) {
        if (tm == 0)
            projection::gemm_row_paired<G, U, TM, TN>(in, out, gate, up, live, rows, k, tn, shared, sg, lane);
        return;
    }
#endif
    projection::gemm_paired<G, U, TM, TN>(in, out, gate, up, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

template <typename W, uint TM, uint TN, typename In, typename Out>
inline void expert_plain(thread const In &in, thread const Out &out, thread const projection::Weights<W> &w,
    uint live, uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
#if !SEISMIC_HAS_TENSOR_OPS
    if (live <= projection::gemm_row_rows) {
        if (tm == 0)
            projection::gemm_row<W, TM, TN>(in, out, w, live, rows, k, tn, shared, sg, lane);
        return;
    }
#endif
    projection::gemm<W, TM, TN>(in, out, w, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

// A resident combine's projections of one channel in threadgroup memory:
// slot k's at `projected[k * TILE + tid]`, A-rounded.
struct TileProjections {
    threadgroup const float *projected;
    uint tile;
    uint tid;
    float at(uint k) const thread {
        return Act::round(projected[k * tile + tid]);
    }
};

} // namespace routed
