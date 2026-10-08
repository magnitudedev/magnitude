// Shared pieces of the routed (mixture-of-experts) Metal entries built on the
// projection family: expert row offsets in expert-stacked weights, the plain
// activation prologue they share, the grouped-row prologue of the expert
// GEMMs and the live-row count of a grouped block. Weights are bound through
// the `packets.h` slots (`KERNEL_W0..3`).

#include "../projection/projection.h"

namespace routed {

typedef element::Act Act;
static_assert(Act::bytes == 2, "the Metal routed family requires a bf16 or f16 activation element");

// First row of expert `expert` in an [E, N, K] rows16 tensor of N rows per
// expert.
inline ulong expert_row(ulong expert, ulong rows) { return expert * rows; }

// Rows of weight tensor `base` starting at row `first`.
template <typename W>
inline projection::Weights<W> weights(device const uchar *base, packets::Rows16 layout, ulong first, uint k) {
    return projection::Weights<W>{base, layout, k, nullptr, first};
}

// A plain activation prologue over rows of a [rows, columns] A tensor.
inline projection::Plain<Act, projection::AllRows> activation(device const uchar *x, ulong stride0, ulong stride1,
    uint columns) {
    return projection::Plain<Act, projection::AllRows>{x, stride0, stride1, columns, {}};
}

// The decode down projections of one simdgroup (the choices of a row, and
// the shared expert, run side by side, one simdgroup each): channels
// first + g + G * r (lane group g of G = 32 / LANES, r < ROWS) of weights `w`
// against the one-row product `in` of COLUMNS values, stored unrounded at
// projected[g + G * r]. The lanes of a group own the packets sub,
// sub + LANES, ... of every channel's row, as the K1 GEMV's lanes do, and
// sum them in the same order, so every projection is the GEMV's sum for the
// same LANES (`projection::gemv_body`). A lane reads its packets of the
// product straight from device memory (no staging, no barrier) and issues
// the weight loads of all its channels before it accumulates any.
template <typename W, uint ROWS, uint LANES, uint COLUMNS, typename In>
inline void project_channels(thread const In &in, thread const projection::Weights<W> &w, uint first,
    uint rows, threadgroup float *projected, uint lane) {
    constexpr uint G = 32u / LANES;
    constexpr uint P = (COLUMNS + 31u) / 32u;
    // Packets per lane.
    constexpr uint PL = (P + LANES - 1u) / LANES;
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
                projection::gemv_weights<W>(packets[r][i], step, we, wo);
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

// The expert projections of one grouped block tile. A block of at most
// `batched_rows` live rows runs the batched GEMV body (weights decoded to F32
// straight into the matrix fragments, no B staging), staged in the GEMM
// tile's own threadgroup buffer: few rows cannot amortize the GEMM's per-tile
// weight staging. Other blocks run the TM x TN GEMM over their live rows.
// The batched body covers the GEMM tile's outputs with the same threads:
// TN / 16 simdgroups, of 1 (paired: TN / 2 features) or 2 (TN rows) row
// blocks of 8. Padding rows carry no contract: the batched body leaves them
// unwritten.
constant constexpr uint batched_rows = 8;

template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void expert_paired(thread const In &in, thread const Out &out, thread const projection::Weights<G> &gate,
    thread const projection::Weights<U> &up, uint live, uint m_rows, uint rows, uint k, uint tm, uint tn,
    threadgroup uchar *shared, uint sg, uint lane) {
    static_assert(TM <= 64, "the batched body reuses the 2 * TN threads of a TM <= 64 tile");
    if (live <= batched_rows && tm == 0)
        projection::gemv_batch_paired<G, U, TN / 16u, 1u, projection::gemm_tile<TM, TN>::bytes>(in, out, gate, up,
            live, rows, k, tn, shared, sg, lane);
    else
        projection::gemm_paired<G, U, TM, TN>(in, out, gate, up, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

template <typename W, uint TM, uint TN, typename In, typename Out>
inline void expert_plain(thread const In &in, thread const Out &out, thread const projection::Weights<W> &w,
    uint live, uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
    static_assert(TM <= 64, "the batched body reuses the 2 * TN threads of a TM <= 64 tile");
    if (live <= batched_rows && tm == 0)
        projection::gemv_batch<W, TN / 16u, 2u, projection::gemm_tile<TM, TN>::bytes>(in, out, w, live, rows, k, tn,
            shared, sg, lane);
    else
        projection::gemm<W, TM, TN>(in, out, w, m_rows, rows, k, tm, tn, shared, sg, lane, live);
}

} // namespace routed
