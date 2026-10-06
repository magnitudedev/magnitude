// The packed projection family (K1): y[m, n] = epilogue(sum_k x[m, k] * W[n, k])
// where x is produced by a prologue from the entry's inputs.
//
// Prologues produce one activation value x[m, k], already rounded to the
// activation type A exactly as the entry's portable body publishes it:
//   Plain      x = A[row(m), k]
//   Rms        x = round_A(r[row(m), k] * inverse(m) * norm[k])
// `row(m)` is the identity (`AllRows`) or an `out_rows` gather
// (`SelectedRows`). A prologue with a norm
// declares `groups()` inverses per row over `width()` inputs each. In the
// GEMV row classes (M <= 16) every threadgroup computes its rows' inverses
// and applies the prologue while staging, so a decode projection is one
// launch. The GEMM classes run a pre-pass launch (`device_normalize`) that
// writes the whole prologue output in A to scratch, read by the GEMM as a
// plain operand: no GEMM tile repeats the prologue.
//
// Epilogues receive the F32 dot product:
//   Store<E>   y = round_E(acc)
//   Residual   y = residual[row(m), n] + round_A(acc)                  (F32)
//   SiluMul    y = round_A(round_A(silu(round_A(gate))) * round_A(up))
//   Glu        y = round_A(round_A(act(round_A(gate))) * round_A(up))  (act by code)
//   Activated  y = round_A(act(round_A(acc)))                    (up-only, ReLU²)
//   Mul        y = gate * up                                   (F32, unrounded)
//   ActivatedMul y = round_A(round_A(act(round_A(acc))) * external[m, n])
//   Logits     y = acc, or cap * tanh(acc / cap) when cap > 0          (F32)
// (`functions::` in `../core/functions.h` holds the activation codes.)
// A column epilogue (GEMV classes, `gemv_emit<true>`) receives every row's
// sum of an output together, for work along the rows.
//
// A segmented projection is one launch over several weight tensors (each
// with its own packet type and destination). Every threadgroup belongs to
// one segment; entries map their threadgroup index to a segment and call
// the GEMV or GEMM body with that segment's weight and epilogue.
//
// Row classes pick the launch:
// - GEMV (M below the entry's BATCH_FROM): lane groups own weight rows, lanes
//   own packets, and each weight is dequantized to F32 once and applied to
//   every activation row with one F32 FMA. The activations are staged once
//   per threadgroup in threadgroup memory (A storage). A launch may select
//   the form over row tiles instead for one or two rows (`gemv_tile_row`).
// - Batched GEMV (BATCH_FROM <= M <= 16): 8-row weight blocks decoded by the
//   lanes straight into F32 matrix fragments, multiplied with the staged F32
//   activations on the matrix units (the verify shapes of speculative decode).
// - GEMM (M > 16): TM x TN output tiles over simdgroups of 32 x 32 (16 x 32
//   when TM = 32), stepping K by 32. Activations are staged as stored (bf16
//   or f16), weights decoded to half once per tile, and simdgroup_matrix
//   multiplies them into F32 accumulators (on a device with Metal 4 tensor
//   operations, one `matmul2d` per step over the whole tile, into
//   cooperative accumulators); small-N outputs may split K.
// - Tall GEMM (M > 64, an entry's TALL form): TM x 32 tiles whose weights are
//   decoded once by staging simdgroups and whose activations are read from
//   device memory, with the same products in the same order.
//
// This file is independent of any entry ABI.

#include "../core/activation.h"
#include "../core/functions.h"
#include <seismic/packets.h>

namespace projection {

using packets::Rows16;

// Register-array loops are fully unrolled: an array indexed by a rolled loop
// lives in stack memory.
#define PROJECTION_UNROLL _Pragma("clang loop unroll(full)")

// ---------------------------------------------------------------------------
// Shared pieces.

// Row maps: which input row feeds projected row m.
struct AllRows {
    uint at(uint m) const { return m; }
};
struct SelectedRows {
    device const int *table;
    uint at(uint m) const { return uint(table[m]); }
};

// The rows of one weight tensor (decoder W, `k` logical values per row),
// optionally gathered through a row table (the selected vocabulary rows).
template <typename W>
struct Weights {
    device const uchar *base;
    Rows16 layout;
    uint k;
    device const int *table;
    // The tensor row of this view's row 0 (a view over a row range).
    ulong first;
    // Rows of one stored row tile (1: rows stored one after another).
    uint tile() const { return layout.tile; }
    // The tensor row behind row n.
    ulong stored(uint n) const { return (table ? ulong(table[n]) : ulong(n)) + first; }
    // Row n as the packet decoders address it: its base (its row tile's
    // first row) and its geometry (`Rows16::of`).
    device const uchar *row(uint n) const { return layout.base_of(base, stored(n)); }
    Rows16 geometry(uint n) const { return layout.of(stored(n)); }
    typename W::packet packet(uint n, uint p) const {
        return packets::Loader<W>::load(row(n), geometry(n), p, k);
    }
    // Row n's coefficient run positioned at packet p, and the packet a run
    // is positioned at (`packets::Block`).
    typename packets::Block<W>::state run(uint n, uint p) const {
        return packets::Block<W>::load(row(n), geometry(n), p);
    }
    typename W::packet packet(uint n, uint p, thread typename packets::Block<W>::state &state) const {
        return packets::Block<W>::packet(row(n), geometry(n), p, k, state);
    }
    // Row n resolved once for a lane that loads many packets of several
    // rows: the rows' addresses and geometry are 64-bit arithmetic that a
    // loop must not repeat at every packet of every row. A lane that owns one
    // row derives it at each load instead (`packet(n, p)`): the located row
    // is state the lane holds across its walk, and with four activation
    // rows' accumulators that costs more than the arithmetic (M4 Pro,
    // `dense_expand` over Q8_0 at four rows: 118.5 against 114.9 µs).
    struct located {
        device const uchar *row;
        Rows16 geometry;
    };
    located locate(uint n) const { return located{row(n), geometry(n)}; }
    // Whether rows g * R .. g * R + R - 1 of a view of `rows` rows lie in
    // one row tile for every g, each stored after the one before: no row
    // table, whole groups, and tiles of whole groups.
    template <uint R>
    bool together(uint rows) const {
        return table == nullptr && layout.tile > 1u && layout.tile % R == 0u && first % R == 0ul && rows % R == 0u;
    }
    // The row `r` rows after a located row of the same tile: the same base,
    // so a lane that holds several rows of a tile holds one address for
    // them (on the M1 a launch of 512 threads leaves a thread 104 registers,
    // and each base a lane holds past that is stack traffic at every
    // packet).
    static located after(located at, uint r) {
        at.geometry.r += r;
        return at;
    }
    typename W::packet packet(thread const located &at, uint p) const {
        return packets::Loader<W>::load(at.row, at.geometry, p, k);
    }
    typename packets::Block<W>::state run(thread const located &at, uint p) const {
        return packets::Block<W>::load(at.row, at.geometry, p);
    }
    typename W::packet packet(thread const located &at, uint p,
        thread typename packets::Block<W>::state &state) const {
        return packets::Block<W>::packet(at.row, at.geometry, p, k, state);
    }
    // The 32-bit offset from `base` of unit `unit` (of `bytes` bytes) of row
    // n's code plane.
    uint codes_at(uint n, uint bytes, uint unit) const {
        return uint(geometry(n).at(row(n), layout.codes, bytes, unit) - base);
    }
    // The same in the plane of high code bits (q5k, q6k).
    uint high_at(uint n, uint bytes, uint unit) const {
        return uint(geometry(n).at(row(n), layout.high, bytes, unit) - base);
    }
};

#include "stage.h"

// ---------------------------------------------------------------------------
// Epilogues. `store(m, n, value)` publishes one output; the GEMM publishes a
// lane's two adjacent outputs through `store2(m, n, first, second)` (columns
// n and n + 1), which an epilogue with per-row work shares between them.

// y[m, column + n] rounded to the stored element E (logits: element::F32).
template <typename E>
struct Store {
    device uchar *y;
    ulong stride0, stride1;
    ulong column;
    void store(uint m, uint n, float value) const {
        reinterpret_cast<device typename E::storage *>(y)
            [ulong(m) * stride0 + (column + n) * stride1] = E::store(value);
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
    // Row m resolved once, for a caller that stores several columns of it.
    struct Row {
        device typename E::storage *y;
        ulong stride1;
        void store(uint n, float value) const { y[ulong(n) * stride1] = E::store(value); }
    };
    Row row(uint m) const {
        return Row{reinterpret_cast<device typename E::storage *>(y) + ulong(m) * stride0 + column * stride1,
            stride1};
    }
};

template <typename A, typename Rows>
struct Residual {
    device float *y;
    ulong stride0, stride1;
    device const float *residual;
    ulong residual0, residual1;
    Rows rows;
    void store(uint m, uint n, float value) const {
        y[ulong(m) * stride0 + ulong(n) * stride1] =
            residual[ulong(rows.at(m)) * residual0 + ulong(n) * residual1] + A::round(value);
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
    // Row m resolved once (its result row and its residual row), for a
    // caller that stores several columns of it.
    struct Row {
        device float *y;
        device const float *residual;
        ulong stride1, residual1;
        void store(uint n, float value) const {
            y[ulong(n) * stride1] = residual[ulong(n) * residual1] + A::round(value);
        }
        // Columns n .. n + 7 from their sums as (even, odd): two vector
        // loads and stores on unit strides.
        void store8(uint n, float4 even, float4 odd) const {
            float4 low = float4(even.x, odd.x, even.y, odd.y), high = float4(even.z, odd.z, even.w, odd.w);
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                low[i] = A::round(low[i]);
                high[i] = A::round(high[i]);
            }
            if (stride1 == 1 && residual1 == 1) {
                device packed_float4 *to = reinterpret_cast<device packed_float4 *>(y + n);
                device const packed_float4 *from = reinterpret_cast<device const packed_float4 *>(residual + n);
                to[0] = float4(from[0]) + low;
                to[1] = float4(from[1]) + high;
                return;
            }
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                y[ulong(n + i) * stride1] = residual[ulong(n + i) * residual1] + low[i];
                y[ulong(n + 4u + i) * stride1] = residual[ulong(n + 4u + i) * residual1] + high[i];
            }
        }
    };
    Row row(uint m) const {
        return Row{y + ulong(m) * stride0, residual + ulong(rows.at(m)) * residual0, stride1, residual1};
    }
};

template <typename A>
struct SiluMul {
    device uchar *y;
    ulong stride0, stride1;
    void store_pair(uint m, uint n, float gate_sum, float up_sum) const {
        float gate = A::round(gate_sum);
        float up = A::round(up_sum);
        float activated = A::round(gate / (1.0f + metal::exp(-gate)));
        reinterpret_cast<device typename A::storage *>(y)
            [ulong(m) * stride0 + ulong(n) * stride1] = A::store(activated * up);
    }
};

// Activation-generic GLU (paired):
//   y = round_A(round_A(act(round_A(gate))) * round_A(up))
// with `function` a `functions::` code (SiLU gives SiluMul's bits).
template <typename A>
struct Glu {
    device uchar *y;
    ulong stride0, stride1;
    int function;
    void store_pair(uint m, uint n, float gate_sum, float up_sum) const {
        float gate = A::round(gate_sum);
        float up = A::round(up_sum);
        float activated = A::round(functions::activate(function, gate));
        reinterpret_cast<device typename A::storage *>(y)
            [ulong(m) * stride0 + ulong(n) * stride1] = A::store(activated * up);
    }
};

// A plain activated projection (up-only feed-forward, ReLU²):
//   y = round_A(act(round_A(acc)))
template <typename A>
struct Activated {
    device uchar *y;
    ulong stride0, stride1;
    int function;
    void store(uint m, uint n, float value) const {
        reinterpret_cast<device typename A::storage *>(y)[ulong(m) * stride0 + ulong(n) * stride1] =
            A::store(functions::activate(function, A::round(value)));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// The F32 product of two projections of one input (paired), unrounded:
//   y = gate * up
struct Mul {
    device float *y;
    ulong stride0, stride1;
    void store_pair(uint m, uint n, float gate_sum, float up_sum) const {
        y[ulong(m) * stride0 + ulong(n) * stride1] = gate_sum * up_sum;
    }
};

// An activated projection times an external F32 multiplier:
//   y = round_A(round_A(act(round_A(acc))) * external[m, n])
template <typename A>
struct ActivatedMul {
    device uchar *y;
    ulong stride0, stride1;
    device const float *external;
    ulong external0, external1;
    int function;
    void store(uint m, uint n, float value) const {
        float activated = A::round(functions::activate(function, A::round(value)));
        reinterpret_cast<device typename A::storage *>(y)[ulong(m) * stride0 + ulong(n) * stride1] =
            A::store(activated * external[ulong(m) * external0 + ulong(n) * external1]);
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// F32 logits, softcapped in F32 from the accumulator when `cap` > 0:
//   y = cap > 0 ? cap * tanh(acc / cap) : acc
struct Logits {
    device uchar *y;
    ulong stride0, stride1;
    float cap;
    void store(uint m, uint n, float value) const {
        reinterpret_cast<device float *>(y)[ulong(m) * stride0 + ulong(n) * stride1] =
            cap > 0.0f ? functions::softcap(cap, value) : value;
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// A weight's second-level scale (NVFP4 `.scale`, per tensor or per expert)
// on the F32 accumulator, before the wrapped epilogue: `first` scales the
// projection (a paired epilogue's gate stream), `second` a paired
// epilogue's up stream. Entries wrap their epilogue only when a scale port
// is present (its static extent is 1), so an unscaled entry compiles as
// before.
template <typename Out>
struct Scaled {
    Out out;
    float first, second;
    void store(uint m, uint n, float value) const { out.store(m, n, value * first); }
    void store2(uint m, uint n, float a, float b) const { out.store2(m, n, a * first, b * first); }
    void store_pair(uint m, uint n, float gate, float up) const { out.store_pair(m, n, gate * first, up * second); }
    // The wrapped epilogue's resolved row m, scaled alike.
    template <typename Inner>
    struct Row {
        Inner row;
        float first;
        void store(uint n, float value) const { row.store(n, value * first); }
        void store8(uint n, float4 even, float4 odd) const { row.store8(n, even * first, odd * first); }
    };
    auto row(uint m) const { return Row<decltype(out.row(m))>{out.row(m), first}; }
};

// The epilogue of a projection with scale ports of static extents: the
// epilogue itself when every extent is 0 (no scale, no load), else wrapped.
template <bool SCALED>
struct scaling;
template <>
struct scaling<false> {
    template <typename Out>
    using type = Out;
    template <typename Out>
    static Out wrap(thread const Out &out, float, float) { return out; }
};
template <>
struct scaling<true> {
    template <typename Out>
    using type = Scaled<Out>;
    template <typename Out>
    static Scaled<Out> wrap(thread const Out &out, float first, float second) { return Scaled<Out>{out, first, second}; }
};

// A scale port's value at `index` along its expert axis (0 for a
// per-tensor port; `stride` that axis' stride), or 1 for an absent port
// (static `extent` 0: the port is never read).
inline float scale_factor(device const float *scale, ulong extent, ulong stride, ulong index) {
    return extent == 0 ? 1.0f : scale[index * stride];
}

// ---------------------------------------------------------------------------
// The normalizing pre-pass.

// The pre-pass of a normed prologue: threadgroup `item` (THREADS threads)
// normalizes one (row, group) and stores that group's prologue output, in A,
// at x[m * columns + group * width ..]. The projection launches then read `x`
// as a plain operand, so none of their threadgroups repeats the prologue
// arithmetic. `partials` is PROJECTION_NORMALIZE_SHARED threadgroup memory.
#define PROJECTION_NORMALIZE_SHARED(name) threadgroup float name[32]

// Row-major rows of A at `columns` elements a row: the order of a plain operand.
template <typename A>
struct RowOrder {
    static void store8(device uchar *x, uint columns, uint m, uint k, uint4 words) {
        *reinterpret_cast<device uint4 *>(x + (ulong(m) * columns + k) * 2u) = words;
    }
};

// `Order` lays the rows out: `RowOrder` (the default), or the tall GEMM's
// `TallOrder` when whole 8-column words are written (columns and every
// group's first column multiples of 8).
template <uint THREADS, typename Order, typename In>
inline void device_normalize(thread const In &in, uint item, device uchar *x, uint columns,
    threadgroup float *partials, uint thread_index) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "a normalizing threadgroup is whole simdgroups");
    typedef typename In::activation A;
    uint m = item / in.groups(), group = item % in.groups();
    float squares = 0.0f;
    for (uint i = thread_index; i < in.width(); i += THREADS) {
        float v = in.norm_input(m, group, i);
        squares = metal::fma(v, v, squares);
    }
    squares = simd_sum(squares);
    if (THREADS > 32) {
        if (thread_index % 32u == 0)
            partials[thread_index / 32u] = squares;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        squares = 0.0f;
        for (uint j = 0; j < THREADS / 32u; ++j)
            squares += partials[j];
    }
    float inverse = metal::rsqrt(squares / float(in.width()) + in.epsilon());
    device typename A::storage *row = reinterpret_cast<device typename A::storage *>(x) + ulong(m) * columns;
    uint first = group * in.width();
    bool vector = (columns & 7u) == 0 && (first & 7u) == 0;
    for (uint i = 8u * thread_index; i < in.width(); i += 8u * THREADS) {
        float4 even, odd;
        in.load8(m, first + i, inverse, even, odd);
        if (vector && i + 8u <= in.width()) {
            Order::store8(x, columns, m, first + i, A::pack8(even, odd));
        } else {
            for (uint j = 0; j < 8u && i + j < in.width(); ++j)
                row[first + i + j] = A::store((j & 1u) ? odd[j >> 1] : even[j >> 1]);
        }
    }
}

template <uint THREADS, typename In>
inline void device_normalize(thread const In &in, uint item, device uchar *x, uint columns,
    threadgroup float *partials, uint thread_index) {
    device_normalize<THREADS, RowOrder<typename In::activation>>(in, item, x, columns, partials, thread_index);
}

// ---------------------------------------------------------------------------
// The in-threadgroup prologue of the GEMV row classes (M <= 16).
//
// A decode projection is one launch: every GEMV threadgroup reduces the norm
// groups of its (at most 16) activation rows itself and applies the prologue
// while staging. The RMS row norm (SharedNorm) is reduced first: its square
// sum is split into `In::parts` fixed parts
// (`In::squares`: each lane sums its share in order, then one simd_sum);
// simdgroup (item % SG) reduces part `item`, and the staging adds a group's
// parts in a fixed tree. The inverses (and hence every rounded activation)
// therefore do not depend on the threadgroup shape. `squares` is
// PROJECTION_SQUARES_SHARED threadgroup memory for `slots` = groups * parts
// per row; the opening barrier of the GEMV and batched GEMV bodies publishes
// it to their staging (a simdgroup that owns no part issues its first weight
// loads meanwhile).
#define PROJECTION_SQUARES_SHARED(name, slots) threadgroup float name[16 * (slots)]

template <typename In>
inline void threadgroup_squares_runtime(thread const In &in, uint m_rows, threadgroup float *squares,
    uint simdgroups, uint sg, uint lane) {
    uint slots = in.groups() * In::parts;
    for (uint item = sg; item < m_rows * slots; item += simdgroups) {
        uint slot = item % slots;
        float sum = simd_sum(in.squares(item / slots, slot / In::parts, slot % In::parts, lane));
        if (lane == 0)
            squares[item] = sum;
    }
}

template <uint SG, typename In>
inline void threadgroup_squares(thread const In &in, uint m_rows, threadgroup float *squares, uint sg,
    uint lane) {
    threadgroup_squares_runtime(in, m_rows, squares, SG, sg, lane);
}

// The sum of PARTS partial square sums in a fixed pairwise tree.
template <uint PARTS>
inline float sum_parts(threadgroup const float *p);
template <>
inline float sum_parts<1>(threadgroup const float *p) {
    return p[0];
}
template <>
inline float sum_parts<8>(threadgroup const float *p) {
    return ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]));
}

// A normed prologue with its rows' partial square sums in threadgroup memory
// (from threadgroup_squares), read by the GEMV staging as a plain operand.
template <typename In>
struct SharedNorm {
    typedef typename In::activation activation;
    In in;
    threadgroup const float *squares;
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        uint group = min(k / in.width(), in.groups() - 1u);
        float sum = sum_parts<In::parts>(squares + (m * in.groups() + group) * In::parts);
        in.load8(m, k, metal::rsqrt(sum / float(in.width()) + in.epsilon()), even, odd);
    }
};

// `in` as a SharedNorm over `squares`. The operand's type is deduced from a
// value: `decltype` of a local carries its address space under Metal 4.1,
// which a field's type may not.
template <typename In>
inline SharedNorm<In> shared_norm(In in, threadgroup const float *squares) {
    return SharedNorm<In>{in, squares};
}

// ---------------------------------------------------------------------------
// GEMV.
//
// Each weight is decoded once to F32 (`fma(scale, code, bias)`, one rounding,
// the dequantized value of the portable body) and multiplies every activation
// row with one FMA, in column order. A lane's sum therefore depends only on
// the packets it owns (LANES), never on M, R or SG: every row of an M-row
// GEMV is bit-identical to the same row computed alone.

// Threadgroup memory of one GEMV, a `threadgroup uchar *` bound at
// [[threadgroup(0)]]: the staged activations. A GEMV launch over K columns
// and M rows declares
//   shared_bytes (min(ceil_div(K, 32) * B, 448) * 64)
// with B = min(M, 2) + 2 * min(M / 3, 1) + 4 * min(M / 5, 1), the row count
// rounded up to its instantiated bound MAXM (`PROJECTION_FOR_ROWS`): at most
// `gemv_stage_packet_rows` staged packet rows of `gemv_packet_row_bytes`.
constant constexpr uint gemv_stage_packet_rows = 448;
constant constexpr uint gemv_packet_row_bytes = 4 * 16;

// Stage packets first .. first + count of MAXM activation rows (rows at or
// past m_rows are zero); all threads of the threadgroup (whole simdgroups)
// take part, one thread per eight columns. The eight activation values of
// chunk-local packet p, step s (columns 32p + 8s ..) and row m are one uint4
// in A storage at words[(p * 4 + s) * MAXM + m], so a lane reads the rows of
// its step consecutively.
template <uint MAXM, typename In>
inline void gemv_stage(thread const In &in, uint m_rows, uint first, uint count, threadgroup uint4 *words,
    uint thread_index, uint threads) {
    typedef typename In::activation A;
    for (uint item = thread_index; item < 4u * MAXM * count; item += threads) {
        uint step = item & 3u, packet = item >> 2;
        uint m = packet / count, local = packet - m * count;
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (m < m_rows)
            in.load8(m, 32u * (first + local) + 8u * step, 0.0f, even, odd);
        words[(local * 4u + step) * MAXM + m] = A::pack8(even, odd);
    }
}

// The R weight rows of a lane group (and, paired, of the second tensor)
// from `first_row` (a multiple of R below `rows`), located once
// (`Weights::locate`), rows of one tile from the first
// (`Weights::together`). DERIVED: the group's one row is named instead, and
// its lanes derive it at each load.
template <typename W, typename U, bool PAIRED, uint R, bool DERIVED = false>
struct gemv_rows {
    typename Weights<W>::located a[R];
    typename Weights<U>::located b[R];
    uint only;
    void locate(thread const Weights<W> &w, thread const Weights<U> &u, uint first_row, uint rows) {
        static_assert(!DERIVED || R == 1u, "a lane derives one row");
        if (DERIVED) {
            only = min(first_row, rows - 1);
            return;
        }
        const bool a_together = w.template together<R>(rows);
        const bool b_together = PAIRED && u.template together<R>(rows);
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            uint n = min(first_row + r, rows - 1);
            if (r > 0 && a_together)
                a[r] = Weights<W>::after(a[0], r);
            else
                a[r] = w.locate(n);
            if (PAIRED) {
                if (r > 0 && b_together)
                    b[r] = Weights<U>::after(b[0], r);
                else
                    b[r] = u.locate(n);
            }
        }
    }
};

// One packet of R weight rows (and, paired, of R rows of the second tensor).
template <typename W, typename U, bool PAIRED, uint R>
struct gemv_packets {
    typename W::packet a[R];
    typename U::packet b[R];
    template <bool DERIVED>
    void load(thread const Weights<W> &w, thread const Weights<U> &u,
        thread const gemv_rows<W, U, PAIRED, R, DERIVED> &at, uint p) {
        if (DERIVED) {
            a[0] = w.packet(at.only, p);
            if (PAIRED)
                b[0] = u.packet(at.only, p);
            return;
        }
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            a[r] = w.packet(at.a[r], p);
            if (PAIRED)
                b[r] = u.packet(at.b[r], p);
        }
    }
};

// Step `step` of a packet as eight dequantized F32 weights (even, odd).
template <typename W>
inline void gemv_weights(thread const typename W::packet &k, uint step, thread float4 &even, thread float4 &odd) {
    W::codes(k, step, even, odd);
    float scale = W::scale(k, step);
    float bias = W::bias(k, W::groups == 1 ? 0u : step / 2u);
    even = metal::fma(float4(scale), even, float4(bias));
    odd = metal::fma(float4(scale), odd, float4(bias));
}

// acc += the eight products of one step, in column order.
inline float gemv_step_dot(float acc, float4 we, float4 wo, float4 xe, float4 xo) {
    PROJECTION_UNROLL
    for (uint j = 0; j < 4; ++j) {
        acc = metal::fma(we[j], xe[j], acc);
        acc = metal::fma(wo[j], xo[j], acc);
    }
    return acc;
}

// Accumulate one loaded packet (chunk-local `local`) into the accumulators of
// all MAXM staged rows.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, typename A>
inline void gemv_accumulate(thread float (&acc)[R][MAXM], thread float (&acc2)[R][MAXM],
    thread const gemv_packets<W, U, PAIRED, R> &packets, uint local, threadgroup const uint4 *words) {
    PROJECTION_UNROLL
    for (uint step = 0; step < 4; ++step) {
        float4 ae[R], ao[R], be[R], bo[R];
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            gemv_weights<W>(packets.a[r], step, ae[r], ao[r]);
            if (PAIRED)
                gemv_weights<U>(packets.b[r], step, be[r], bo[r]);
        }
        threadgroup const uint4 *x = words + (local * 4u + step) * MAXM;
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            float4 xe, xo;
            A::split8(x[m], xe, xo);
            PROJECTION_UNROLL
            for (uint r = 0; r < R; ++r) {
                acc[r][m] = gemv_step_dot(acc[r][m], ae[r], ao[r], xe, xo);
                if (PAIRED)
                    acc2[r][m] = gemv_step_dot(acc2[r][m], be[r], bo[r], xe, xo);
            }
        }
    }
}

// Plain epilogues store one sum; paired epilogues combine gate and up.
template <bool PAIRED>
struct emit;
template <>
struct emit<false> {
    template <typename Out>
    static void run(thread const Out &out, uint m, uint n, float value, float) { out.store(m, n, value); }
};
template <>
struct emit<true> {
    template <typename Out>
    static void run(thread const Out &out, uint m, uint n, float gate, float up) {
        out.store_pair(m, n, gate, up);
    }
};

// Weight rows of one GEMV threadgroup (segmented entries map threadgroups to
// segments by it).
template <uint SG, uint R, uint LANES>
constexpr uint gemv_threadgroup_rows() {
    return SG * R * (32u / LANES);
}

// Sum of `value` over the LANES lanes of this lane's group.
template <uint LANES>
inline float gemv_group_sum(float value) {
    if (LANES == 32)
        return simd_sum(value);
    for (ushort offset = LANES / 2; offset > 0; offset >>= 1)
        value += simd_shuffle_xor(value, offset);
    return value;
}

// Publishes a GEMV lane group's sums. Plain and paired epilogues get (m, n)
// from the lane that owns it. A column epilogue (`store_column(m, n, sums)`,
// plain only) gets from that lane every activation row's sum of output n,
// which each lane of the group holds after the group sums: the convolution
// along the rows of `gated_delta_project_convolved`.
template <bool COLUMNS>
struct gemv_emit;
template <>
struct gemv_emit<false> {
    template <bool PAIRED, uint R, uint MAXM, uint LANES, typename Out>
    static void run(thread const Out &out, thread float (&acc)[R][MAXM], thread float (&acc2)[R][MAXM],
        uint m_rows, uint rows, uint first_row, uint sub) {
        for (uint r = 0; r < R; ++r) {
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows) {
                    float total = gemv_group_sum<LANES>(acc[r][m]);
                    float total2 = PAIRED ? gemv_group_sum<LANES>(acc2[r][m]) : 0.0f;
                    if (sub == (r * MAXM + m) % LANES && first_row + r < rows)
                        emit<PAIRED>::run(out, m, first_row + r, total, total2);
                }
            }
        }
    }
};
template <>
struct gemv_emit<true> {
    template <bool PAIRED, uint R, uint MAXM, uint LANES, typename Out>
    static void run(thread const Out &out, thread float (&acc)[R][MAXM], thread float (&)[R][MAXM],
        uint m_rows, uint rows, uint first_row, uint sub) {
        static_assert(!PAIRED, "a column epilogue is plain");
        for (uint r = 0; r < R; ++r) {
            float column[MAXM];
            for (uint m = 0; m < MAXM; ++m) {
                column[m] = 0.0f;
                if (m < m_rows)
                    column[m] = gemv_group_sum<LANES>(acc[r][m]);
            }
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows && sub == (r * MAXM + m) % LANES && first_row + r < rows)
                    out.store_column(m, first_row + r, column);
            }
        }
    }
};

// One GEMV threadgroup of SG simdgroups. A simdgroup is 32 / LANES lane
// groups; lane group g of simdgroup sg owns the R weight rows from
// ((tile * SG + sg) * (32 / LANES) + g) * R, and its lanes own the packets
// sub, sub + LANES, ... of those rows. The prologue output is staged per
// threadgroup in chunks of at most `gemv_stage_packet_rows / MAXM` packets
// (a multiple of LANES). Weight packets are double-buffered in registers: a
// lane loads its next packet before accumulating the current one, and its
// first packet before the staging, so the weight stream never waits on it.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, bool COLUMNS = false, typename In,
    typename Out>
inline void gemv_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV lane group is 8, 16 or 32 lanes");
    typedef typename In::activation A;
    uint group = lane / LANES, sub = lane % LANES;
    uint first_row = ((tile * simdgroups + sg) * (32u / LANES) + group) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    // A lane group of one row derives it at each load: across four activation
    // rows' accumulators the located row costs more than the arithmetic
    // (`Weights::locate`).
    gemv_rows<W, U, PAIRED, R, (R == 1u)> at;
    gemv_packets<W, U, PAIRED, R> current, next;
    if (active) {
        at.locate(w, u, first_row, rows);
        if (sub < packets)
            current.load(w, u, at, sub);
    }
    uint chunk = min(packets, (gemv_stage_packet_rows / MAXM) / LANES * LANES);
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    for (uint r = 0; r < R; ++r)
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = acc2[r][m] = 0.0f;
    for (uint first = 0; first < packets; first += chunk) {
        uint count = min(chunk, packets - first);
        // A threadgroup may run several GEMVs in turn: the previous chunk's or
        // call's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage<MAXM>(in, m_rows, first, count, words, sg * 32u + lane, simdgroups * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, u, at, first + local + LANES);
                gemv_accumulate<W, U, PAIRED, R, MAXM, A>(acc, acc2, current, local, words);
                current = next;
            }
        }
    }
    gemv_emit<COLUMNS>::template run<PAIRED, R, MAXM, LANES>(out, acc, acc2, m_rows, rows, first_row, sub);
}

template <typename W, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<W, W, false, R, MAXM, LANES>(in, out, w, w, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename W, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    gemv_body<W, W, false, R, MAXM, LANES>(in, out, w, w, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane);
}

// `gemv_runtime` with a column epilogue (`gemv_emit<true>`).
template <typename W, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_columns_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    gemv_body<W, W, false, R, MAXM, LANES, true>(in, out, w, w, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane);
}

template <typename G, typename U, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<G, U, true, R, MAXM, LANES>(in, out, gate, up, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename G, typename U, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_paired_runtime(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    gemv_body<G, U, true, R, MAXM, LANES>(in, out, gate, up, m_rows, rows, k, tile, shared,
        simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// The row mapping's GEMV walked by packet class (`rows32` weights): a form
// selected by the launch, with the row mapping's results.
//
// The threadgroup keeps the row mapping's SG * R * 32 / LANES weight rows
// and its LANES partial sums per row (class c: packets c, c + LANES, ... in
// column order), but its lanes are regrouped: with `span` = SG * 32 / LANES
// row groups in the threadgroup, lanes b * span .. b * span + span - 1 of
// simdgroup sg are those row groups at class sg + b * SG. A simdgroup's
// lanes are then consecutive rows of a tile at the same packet, so its code
// loads are runs of adjacent bytes in lane order, where the row mapping asks
// for LANES separate cache lines a kilobyte apart at every step (on the M1,
// `dense_expand` over Q8_0 at one row: 438 against 420 µs).
//
// A row's class sums then sit in different simdgroups. `gemv_group_sum`
// pairs classes c and c + LANES / 2 first, then by halving offsets: the
// levels down to offset SG pair lanes of one simdgroup, and the levels below
// pair simdgroups, each a round of the threadgroup memory. Every result is
// therefore bit-identical to the row mapping's with the same LANES. The
// rounds cost a barrier each, which the walk repays for one or two
// activation rows on the M1 and not on the M4 Pro.

// Whether the class walk serves: R consecutive rows of a tile per lane, a
// class per simdgroup and lane block, lane groups `gemv_group_sum` adds by
// its own tree (32 lanes are `simd_sum`'s order, which the walk cannot
// reproduce), and the staged activations' threadgroup memory holding a float
// per row, activation row and tensor of every simdgroup.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES>
inline bool gemv_classes_serve(thread const Weights<W> &w, thread const Weights<U> &u, uint rows, uint packets,
    uint simdgroups) {
    return w.template together<R>(rows) && (!PAIRED || u.template together<R>(rows)) && LANES < 32u &&
        simdgroups <= LANES &&
        8u * MAXM * simdgroups * (simdgroups * 32u / LANES) * R <=
            min(packets * MAXM, gemv_stage_packet_rows) * gemv_packet_row_bytes;
}

template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, typename In, typename Out>
inline void gemv_classes(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    typedef typename In::activation A;
    uint threads = simdgroups * 32u, thread_index = sg * 32u + lane;
    uint span = threads / LANES;
    uint sub = sg + (lane / span) * simdgroups, within = lane % span;
    uint first_row = (tile * span + within) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    gemv_rows<W, U, PAIRED, R> at;
    gemv_packets<W, U, PAIRED, R> current, next;
    if (active) {
        at.locate(w, u, first_row, rows);
        if (sub < packets)
            current.load(w, u, at, sub);
    }
    uint chunk = min(packets, (gemv_stage_packet_rows / MAXM) / LANES * LANES);
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    for (uint r = 0; r < R; ++r)
        for (uint m = 0; m < MAXM; ++m)
            acc[r][m] = acc2[r][m] = 0.0f;
    for (uint first = 0; first < packets; first += chunk) {
        uint count = min(chunk, packets - first);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage<MAXM>(in, m_rows, first, count, words, thread_index, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, u, at, first + local + LANES);
                gemv_accumulate<W, U, PAIRED, R, MAXM, A>(acc, acc2, current, local, words);
                current = next;
            }
        }
    }
    // The levels that pair classes of one simdgroup.
    PROJECTION_UNROLL
    for (uint r = 0; r < R; ++r) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            for (ushort offset = 16; offset >= span; offset >>= 1) {
                acc[r][m] += simd_shuffle_xor(acc[r][m], offset);
                if (PAIRED)
                    acc2[r][m] += simd_shuffle_xor(acc2[r][m], offset);
            }
        }
    }
    // The levels that pair simdgroups: every simdgroup publishes its rows'
    // sums, and at the level of offset `o` simdgroup c < o adds simdgroup
    // c + o's and publishes the result for the next level. The sums stay in
    // the walk's registers. The staged activations are read before the sums
    // overwrite them.
    threadgroup float *sums = reinterpret_cast<threadgroup float *>(shared);
    const uint plane = simdgroups * span * R, mine = (sg * span + within) * R;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < span) {
        PROJECTION_UNROLL
        for (uint r = 0; r < R; ++r) {
            PROJECTION_UNROLL
            for (uint m = 0; m < MAXM; ++m) {
                if (m < m_rows) {
                    sums[(2u * m) * plane + mine + r] = acc[r][m];
                    if (PAIRED)
                        sums[(2u * m + 1u) * plane + mine + r] = acc2[r][m];
                }
            }
        }
    }
    for (uint offset = simdgroups / 2u; offset > 0u; offset >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < offset && lane < span) {
            const uint other = mine + offset * span * R;
            PROJECTION_UNROLL
            for (uint r = 0; r < R; ++r) {
                PROJECTION_UNROLL
                for (uint m = 0; m < MAXM; ++m) {
                    if (m < m_rows) {
                        acc[r][m] += sums[(2u * m) * plane + other + r];
                        sums[(2u * m) * plane + mine + r] = acc[r][m];
                        if (PAIRED) {
                            acc2[r][m] += sums[(2u * m + 1u) * plane + other + r];
                            sums[(2u * m + 1u) * plane + mine + r] = acc2[r][m];
                        }
                    }
                }
            }
        }
    }
    PROJECTION_UNROLL
    for (uint r = 0; r < R; ++r) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m) {
            if (m < m_rows && first_row + r < rows && sg == 0u && lane < span)
                emit<PAIRED>::run(out, m, first_row + r, acc[r][m], acc2[r][m]);
        }
    }
}

// The row GEMV of a launch whose form parameter may select the class walk
// (CLASSES): the walk where it serves, the row mapping otherwise.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, bool CLASSES, typename In,
    typename Out>
inline void gemv_form(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    if (CLASSES && gemv_classes_serve<W, U, PAIRED, R, MAXM, LANES>(w, u, rows, (k + 31u) / 32u, simdgroups)) {
        gemv_classes<W, U, PAIRED, R, MAXM, LANES>(in, out, w, u, m_rows, rows, k, tile, shared, simdgroups, sg,
            lane);
        return;
    }
    gemv_body<W, U, PAIRED, R, MAXM, LANES>(in, out, w, u, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// The GEMV over row tiles (`rows32` weights, M up to `gemv_tile_rows`): a
// form of its own, selected by the launch.
//
// A thread owns one weight row. The threadgroup keeps the row mapping's
// SG * R * 32 / LANES weight rows, so a row has LANES / R threads, thread
// t = sg * 32 + lane owning row t % rows-of-the-threadgroup: the lanes of a
// simdgroup are consecutive rows of one tile at the same packets, their code
// loads are adjacent, and they walk the packets in step, so every lane reads
// its row's coefficient planes once per run of packets (`packets::Block`).
//
// Summation order. A row's packets are split into LANES contiguous ranges,
// in column order and as evenly as they divide (the first packets % LANES
// ranges hold one packet more). A range is summed in column order from zero,
// and the LANES range sums are added pairwise (`gemv_pair_sum`: neighbours
// first). A thread owns R neighbouring ranges and adds them itself; the rest
// crosses the threadgroup, since a row's threads are in different
// simdgroups. The order depends on the row's packet count and LANES only,
// never on M or on how the ranges are staged: every row of an M-row call in
// this form is bit-identical to the same row computed alone in this form. It
// is not the row mapping's order.
//
// The activations are staged in chunks of the same window of every range
// (`gemv_chunk`), so all threads work in every chunk. The launch's
// threadgroup memory holds, besides the staged activations, one float per
// thread and tensor: a launch of SG simdgroups declares
//   shared_bytes (max(<the GEMV staging above>, SG * 128))
// (SG * 256 for a paired GEMV).

// Activation rows the form serves. From 3 rows up it loses to the row
// mapping on q4k and q8 (M6: 136 against 119 µs at 3 rows, both tuned).
constant constexpr uint gemv_tile_rows = 2;

// A row's packets split into `ranges` contiguous ranges: range s holds the
// `length(s)` packets from `start(s)`.
struct gemv_split {
    uint ranges;
    uint base;    // packets of every range
    uint extra;   // ranges holding one more
    uint start(uint range) const { return range * base + min(range, extra); }
    uint length(uint range) const { return base + (range < extra ? 1u : 0u); }
    uint longest() const { return base + min(extra, 1u); }
};

// One staged chunk: the at most `window` packets from `offset` of every
// range, a range's packets in consecutive slots and the ranges in order.
struct gemv_chunk {
    gemv_split split;
    uint offset;
    uint longer;    // staged packets of a range holding one more
    uint shorter;   // staged packets of the other ranges
    uint count(uint range) const { return range < split.extra ? longer : shorter; }
    uint slot(uint range) const {
        return range < split.extra ? range * longer : split.extra * longer + (range - split.extra) * shorter;
    }
    uint slots() const { return split.extra * longer + (split.ranges - split.extra) * shorter; }
    // The row packet staged in `slot`.
    uint packet(uint slot) const {
        uint head = split.extra * longer;
        uint range, at;
        if (slot < head) {
            range = slot / longer;
            at = slot - range * longer;
        } else {
            range = (slot - head) / shorter;
            at = slot - head - range * shorter;
            range += split.extra;
        }
        return split.start(range) + offset + at;
    }
};
inline gemv_chunk gemv_chunk_at(gemv_split split, uint offset, uint window) {
    uint longer = split.base + 1u > offset ? min(split.base + 1u - offset, window) : 0u;
    uint shorter = split.base > offset ? min(split.base - offset, window) : 0u;
    return gemv_chunk{split, offset, longer, shorter};
}

// Stage a chunk of MAXM activation rows as `gemv_stage` does, slot q of the
// chunk at words[(q * 4 + s) * MAXM + m].
template <uint MAXM, typename In>
inline void gemv_stage_chunk(thread const In &in, uint m_rows, gemv_chunk chunk, threadgroup uint4 *words,
    uint thread_index, uint threads) {
    typedef typename In::activation A;
    uint count = chunk.slots();
    for (uint item = thread_index; item < 4u * MAXM * count; item += threads) {
        uint step = item & 3u, packet = item >> 2;
        uint m = packet / count, slot = packet - m * count;
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (m < m_rows)
            in.load8(m, 32u * chunk.packet(slot) + 8u * step, 0.0f, even, odd);
        words[(slot * 4u + step) * MAXM + m] = A::pack8(even, odd);
    }
}

// The sum of COUNT values in a pairwise tree, neighbours first.
template <uint COUNT>
inline float gemv_pair_sum(thread float (&v)[COUNT]) {
    PROJECTION_UNROLL
    for (uint width = 1; width < COUNT; width *= 2) {
        PROJECTION_UNROLL
        for (uint i = 0; i < COUNT; i += 2 * width)
            v[i] += v[i + width];
    }
    return v[0];
}

// One contiguous range of a row's packets, walked in order: the packet being
// accumulated, the one loaded ahead of it, and the row's coefficient run.
template <typename W>
struct gemv_walk {
    uint next;   // the packet loaded ahead
    uint end;
    typename W::packet current, ahead;
    typename packets::Block<W>::state run;
};

// Load the walk's next packet ahead (none past the range's end) and step.
template <typename W>
inline void gemv_walk_load(thread gemv_walk<W> &walk, thread const Weights<W> &w,
    thread const typename Weights<W>::located &at, bool first) {
    if (walk.next < walk.end) {
        if (first || walk.next % packets::Block<W>::packets == 0u)
            walk.run = w.run(at, walk.next);
        walk.ahead = w.packet(at, walk.next, walk.run);
    }
    walk.next += 1u;
}

template <typename W, typename U, bool PAIRED, uint R, uint MAXM, uint LANES, typename In, typename Out>
inline void gemv_tile_row(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV row has 8, 16 or 32 packet ranges");
    static_assert(LANES % R == 0, "a thread owns a whole share of its row's ranges");
    static_assert(gemv_stage_packet_rows / MAXM >= LANES, "a chunk stages a packet of every range");
    typedef typename In::activation A;
    constexpr uint sharers = LANES / R;
    uint threads = simdgroups * 32u, thread_index = sg * 32u + lane;
    uint span = threads / sharers;
    uint within = thread_index % span, share = thread_index / span;
    // A thread past the last weight row computes that row and emits nothing.
    bool active = tile * span + within < rows;
    uint n = min(tile * span + within, rows - 1u);
    uint packets = (k + 31u) / 32u;
    gemv_split split{LANES, packets / LANES, packets % LANES};
    typename Weights<W>::located a_at = w.locate(n);
    typename Weights<U>::located b_at = u.locate(n);
    gemv_walk<W> a[R];
    gemv_walk<U> b[R];
    PROJECTION_UNROLL
    for (uint j = 0; j < R; ++j) {
        uint range = share * R + j;
        a[j].next = split.start(range);
        a[j].end = a[j].next + split.length(range);
        gemv_walk_load(a[j], w, a_at, true);
        if (PAIRED) {
            b[j].next = a[j].next - 1u;
            b[j].end = a[j].end;
            gemv_walk_load(b[j], u, b_at, true);
        }
    }
    constexpr uint window = (gemv_stage_packet_rows / MAXM) / LANES;
    threadgroup uint4 *words = reinterpret_cast<threadgroup uint4 *>(shared);
    float acc[R][MAXM], acc2[R][MAXM];
    PROJECTION_UNROLL
    for (uint j = 0; j < R; ++j) {
        PROJECTION_UNROLL
        for (uint m = 0; m < MAXM; ++m)
            acc[j][m] = acc2[j][m] = 0.0f;
    }
    for (uint offset = 0; offset < split.longest(); offset += window) {
        gemv_chunk chunk = gemv_chunk_at(split, offset, window);
        // A threadgroup may run several GEMVs in turn: the previous chunk's or
        // call's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gemv_stage_chunk<MAXM>(in, m_rows, chunk, words, thread_index, threads);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint at = 0; at < (split.extra == 0u ? chunk.shorter : chunk.longer); ++at) {
            PROJECTION_UNROLL
            for (uint j = 0; j < R; ++j) {
                uint range = share * R + j;
                if (at < chunk.count(range)) {
                    a[j].current = a[j].ahead;
                    gemv_walk_load(a[j], w, a_at, false);
                    if (PAIRED) {
                        b[j].current = b[j].ahead;
                        gemv_walk_load(b[j], u, b_at, false);
                    }
                    threadgroup const uint4 *x = words + (chunk.slot(range) + at) * 4u * MAXM;
                    PROJECTION_UNROLL
                    for (uint step = 0; step < 4; ++step) {
                        float4 ae, ao, be, bo;
                        gemv_weights<W>(a[j].current, step, ae, ao);
                        if (PAIRED)
                            gemv_weights<U>(b[j].current, step, be, bo);
                        PROJECTION_UNROLL
                        for (uint m = 0; m < MAXM; ++m) {
                            float4 xe, xo;
                            A::split8(x[step * MAXM + m], xe, xo);
                            acc[j][m] = gemv_step_dot(acc[j][m], ae, ao, xe, xo);
                            if (PAIRED)
                                acc2[j][m] = gemv_step_dot(acc2[j][m], be, bo, xe, xo);
                        }
                    }
                }
            }
        }
    }
    // Each activation row's sums cross the threadgroup in a round of their
    // own: every thread publishes its ranges' sum and the row's first thread
    // adds them.
    threadgroup float *sums = reinterpret_cast<threadgroup float *>(shared);
    PROJECTION_UNROLL
    for (uint m = 0; m < MAXM; ++m) {
        if (m < m_rows) {
            float own[R], own2[R];
            PROJECTION_UNROLL
            for (uint j = 0; j < R; ++j) {
                own[j] = acc[j][m];
                own2[j] = acc2[j][m];
            }
            // The staged activations (and the previous round) are read
            // before the sums overwrite them.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sums[thread_index] = gemv_pair_sum<R>(own);
            if (PAIRED)
                sums[threads + thread_index] = gemv_pair_sum<R>(own2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active && share == 0u) {
                float part[sharers], part2[sharers];
                PROJECTION_UNROLL
                for (uint s = 0; s < sharers; ++s) {
                    part[s] = sums[within + s * span];
                    part2[s] = PAIRED ? sums[threads + within + s * span] : 0.0f;
                }
                emit<PAIRED>::run(out, m, n, gemv_pair_sum<sharers>(part), gemv_pair_sum<sharers>(part2));
            }
        }
    }
}

// Whether a GEMV of `m_rows` activation rows over `w` takes the tile form.
template <typename W>
inline bool gemv_tile_row_serves(thread const Weights<W> &w, uint m_rows) {
    return w.tile() > 1u && m_rows <= gemv_tile_rows;
}

// Instantiate the tile form for the activation-row bound MAXM in {1, 2}.
#define PROJECTION_FOR_TILE_ROWS(rows, ...)                                     \
    do {                                                                        \
        if ((rows) <= 1) { constexpr uint MAXM = 1; __VA_ARGS__; }              \
        else { constexpr uint MAXM = 2; __VA_ARGS__; }                          \
    } while (0)

// Instantiate a GEMV body for the activation-row bound MAXM in {1, 2, 4, 8}.
#define PROJECTION_FOR_ROWS(rows, ...)                                          \
    do {                                                                        \
        if ((rows) <= 1) { constexpr uint MAXM = 1; __VA_ARGS__; }              \
        else if ((rows) <= 2) { constexpr uint MAXM = 2; __VA_ARGS__; }         \
        else if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }         \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)

// The same in the launches of an entry that launches its GEMV per row count,
// one for one activation row, one for two and one for three up to BATCH_FROM
// (at most 8): each instantiates only the bounds it runs. A kernel holds the
// registers of every bound it instantiates, and a kernel that also holds a
// wider body's runs slower for bounds it never takes (on the M1 10% at one
// row beside the 8-row body, and 10% at two rows beside the 4- and 8-row
// bodies: `dense_output` over Q4_K 254 against 281 µs). A wrapper body names
// its launch's count, PROJECTION_FOR_<count>_ROWS.
#define PROJECTION_FOR_ONE_ROWS(rows, ...)                                      \
    do { constexpr uint MAXM = 1; __VA_ARGS__; } while (0)
#define PROJECTION_FOR_ONE_TILE_ROWS(rows, ...) PROJECTION_FOR_ONE_ROWS(rows, __VA_ARGS__)
#define PROJECTION_FOR_PAIR_ROWS(rows, ...)                                     \
    do { constexpr uint MAXM = 2; __VA_ARGS__; } while (0)
#define PROJECTION_FOR_PAIR_TILE_ROWS(rows, ...) PROJECTION_FOR_PAIR_ROWS(rows, __VA_ARGS__)
#define PROJECTION_FOR_SEVERAL_ROWS(rows, ...)                                  \
    do {                                                                        \
        if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }              \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)
#define PROJECTION_FOR_SEVERAL_TILE_ROWS(rows, ...)                             \
    do { constexpr uint MAXM = 2; __VA_ARGS__; } while (0)

// ---------------------------------------------------------------------------
// GEMM.
//
// A TM x TN output tile per threadgroup of simdgroups that each own an
// 32 x SN sub-tile (SN = 32, or 16 when TM = 32, so every simdgroup of a
// 32-row tile spans all its rows), stepping K by 32. The A tile holds the
// plain operand's storage words unchanged (bf16 or f16: no conversion) and
// the B tile the weights decoded to half once per tile; both sit in one
// threadgroup staging buffer with rows padded to 40 elements, so the lanes'
// fragment reads are free of bank conflicts. While the simdgroups run the MMA
// chain of step t, every thread already holds step t + 1's activation words
// and raw weight packets in registers, loaded before the chain; it stores
// them into the buffer after the chain's barrier. Each lane reads its own two
// elements of every fragment, and the matrix units multiply A (bf16 or f16)
// by half into F32 accumulators. The tile shape changes no result: every
// output is the same ascending-K chain of 8-deep MMAs.
//
// Tiles past the last row (`m_rows`) are cheap: their activation rows are
// never read, and each simdgroup multiplies only its fragment rows (8 rows
// each) that hold a live row, so a grouped expert block of few rows costs
// its live fragments' MMAs, not the tile's.
//
// A GEMM launch declares max(TM, 64) * TN / 32 threads.

constant constexpr uint gemm_k = 32;
constant constexpr uint gemm_lda = gemm_k + 8;
constant constexpr uint gemm_halves = gemm_k / 16;   // half-packets (16 codes) per step and row

// The fixed geometry of the 17..64-row GEMM launches (`…_gemm_small`): few
// threadgroups at small M, so the narrowest tile and, for the N = 2560 output
// projections, a 4-way K split fill the device (lab § "K1 Metal GEMM per-size
// map"). Larger row counts run the tuned TILE_M x TILE_N (x SPLIT) launch.
constant constexpr uint small_tile_m = 32;
constant constexpr uint small_tile_n = 64;
constant constexpr uint small_split = 4;

template <uint TM, uint TN>
struct gemm_tile {
    static_assert(TM % 32 == 0 && TN % 32 == 0, "GEMM tiles are multiples of 32");
    // A simdgroup's sub-tile: fm x fn fragments (32 x 32, or 32 x 16 when
    // TM = 32, so every simdgroup spans the tile's rows); wm x wn simdgroups.
    static constant constexpr uint fm = 4u;
    static constant constexpr uint fn = TM >= 64 ? 4u : 2u;
    static constant constexpr uint wm = TM / (8u * fm);
    static constant constexpr uint wn = TN / (8u * fn);
    static constant constexpr uint threads = wm * wn * 32u;
    // 8-column activation words one thread stages per step.
    static constant constexpr uint a_items = (TM * (gemm_k / 8u) + threads - 1u) / threads;
    static constant constexpr uint bytes = (TM + TN) * gemm_lda * 2u;
};

// Threadgroup storage of one TM x TN GEMM threadgroup, as `threadgroup uchar *name`.
#define PROJECTION_GEMM_SHARED(name, TM, TN)                                                        \
    threadgroup float4 name##_words[projection::gemm_tile<TM, TN>::bytes / 16];                     \
    threadgroup uchar *name = reinterpret_cast<threadgroup uchar *>(name##_words)

// The staging buffer's A tile (activation scalars E) and B tile (half).
template <uint TM, uint TN, typename E>
struct gemm_buffer {
    threadgroup E *a;
    threadgroup half *b;
    static gemm_buffer at(threadgroup uchar *base) {
        gemm_buffer buffer;
        buffer.a = reinterpret_cast<threadgroup E *>(base);
        buffer.b = reinterpret_cast<threadgroup half *>(buffer.a + TM * gemm_lda);
        return buffer;
    }
};

// Activation words one thread stages per step: item i is columns
// 8 * (i % 4) .. of the tile's row i / 4, as the operand's `words8`.
template <uint ITEMS>
struct gemm_a_registers {
    uint4 words[ITEMS];
};

template <uint TM, uint THREADS, uint ITEMS, typename In>
inline void gemm_load_a(thread const In &in, uint m0, uint m_rows, uint k0, uint k, uint thread_index,
    thread gemm_a_registers<ITEMS> &regs) {
    constexpr uint parts = gemm_k / 8u;
    PROJECTION_UNROLL
    for (uint j = 0; j < ITEMS; ++j) {
        uint item = thread_index + j * THREADS;
        uint row = item / parts, column = k0 + 8u * (item % parts);
        bool inside = (ITEMS * THREADS == TM * parts || row < TM) && m0 + row < m_rows && column < k;
        regs.words[j] = inside ? in.words8(m0 + row, column) : uint4(0);
    }
}

template <uint TM, uint THREADS, uint ITEMS, typename E>
inline void gemm_store_a(thread const gemm_a_registers<ITEMS> &regs, threadgroup E *a, uint thread_index) {
    constexpr uint parts = gemm_k / 8u;
    PROJECTION_UNROLL
    for (uint j = 0; j < ITEMS; ++j) {
        uint item = thread_index + j * THREADS;
        uint row = item / parts;
        if (ITEMS * THREADS == TM * parts || row < TM)
            *reinterpret_cast<threadgroup uint4 *>(a + row * gemm_lda + 8u * (item % parts)) = regs.words[j];
    }
}

// Weight packets one thread stages per step: COUNT items, item i being the
// half-packet (16 codes with their coefficient group) i % gemm_halves of the
// tile's local row i / gemm_halves.
// A thread stages the same weight rows on every step, so it keeps each
// item's coefficient run (`packets::Block`) in registers across steps.
template <typename W, uint COUNT>
struct gemm_b_registers {
    typename W::packet packet[COUNT];
    typename packets::Block<W>::state run[COUNT];
    bool valid[COUNT];
};

// Whether a TM-row tile keeps coefficient runs. On simdgroup matrices a
// staging thread carries its runs beside its fragments: a 32-row tile has no
// registers to spare (the runs cost it 6-17% on Apple GPU family 7), nor has
// a paired tile with a run per stream (2% on family 9), so those load every
// packet's coefficients as before.
template <uint TM, bool PAIRED>
constexpr bool gemm_runs() {
    return TM >= 64 && (SEISMIC_HAS_TENSOR_OPS || !PAIRED);
}

// Loads the step at column k0. With RUNS, `fresh` on a tile's first step
// opens its coefficient runs wherever in a run it starts, and later steps
// open a run as they enter it.
template <typename W, uint THREADS, uint COUNT, bool RUNS>
inline void gemm_load_b(thread const Weights<W> &w, uint first, uint count, uint rows, uint k0,
    uint k, uint thread_index, bool fresh, thread gemm_b_registers<W, COUNT> &regs) {
    PROJECTION_UNROLL
    for (uint j = 0; j < COUNT; ++j) {
        uint item = thread_index + j * THREADS;
        uint local = item / gemm_halves;
        uint p = k0 / 32u + (item % gemm_halves) / 2u;
        regs.valid[j] = local < count && 32u * p < k;
        if (regs.valid[j]) {
            uint n = min(first + local, rows - 1);
            if (RUNS) {
                if (fresh || p % packets::Block<W>::packets == 0)
                    regs.run[j] = w.run(n, p);
                regs.packet[j] = w.packet(n, p, regs.run[j]);
            } else {
                regs.packet[j] = w.packet(n, p);
            }
        }
    }
}

// Decode the staged packets (scale * code + bias, rounded to half). Local
// row r of this tensor becomes tile row tile_row0 + r * spacing (a paired
// tile interleaves gate and up rows).
template <typename W, uint THREADS, uint COUNT>
inline void gemm_store_b(thread const gemm_b_registers<W, COUNT> &regs, uint count, uint tile_row0,
    uint spacing, threadgroup half *b, uint thread_index) {
    PROJECTION_UNROLL
    for (uint j = 0; j < COUNT; ++j) {
        uint item = thread_index + j * THREADS;
        uint local = item / gemm_halves, half_index = item % gemm_halves;
        if (local >= count)
            continue;
        uint tile_row = tile_row0 + local * spacing;
        uint within = half_index & 1u;   // half of its packet
        threadgroup half4 *out = reinterpret_cast<threadgroup half4 *>(b + tile_row * gemm_lda + 16u * half_index);
        if (regs.valid[j]) {
            float scale = W::scale(regs.packet[j], 2u * within);
            float bias = W::bias(regs.packet[j], W::groups == 1 ? 0u : within);
            PROJECTION_UNROLL
            for (uint s = 0; s < 2; ++s) {
                float4 even, odd;
                W::codes(regs.packet[j], 2u * within + s, even, odd);
                even = metal::fma(float4(scale), even, float4(bias));
                odd = metal::fma(float4(scale), odd, float4(bias));
                out[2 * s] = half4(half(even.x), half(odd.x), half(even.y), half(odd.y));
                out[2 * s + 1] = half4(half(even.z), half(odd.z), half(even.w), half(odd.w));
            }
        } else {
            PROJECTION_UNROLL
            for (uint s = 0; s < 4; ++s)
                out[s] = half4(0.0h);
        }
    }
}

// Fragment coordinates of a simdgroup 8x8 matrix element pair: the lane holds
// row y, columns x and x + 1.
inline ushort2 fragment_coordinate(uint lane) {
    ushort q = ushort(lane / 4u);
    ushort row = (q & 4u) + ((lane / 2u) % 4u);
    ushort column = (q & 2u) * 2u + (lane % 2u) * 2u;
    return ushort2(column, row);
}

// The lane holding elements (row, column) and (row, column + 1) of an 8x8
// fragment, for even `column`: the inverse of `fragment_coordinate`.
inline ushort fragment_lane(ushort column, ushort row) {
    return ((column >> 1) & 1u) | ((row & 1u) << 1) | ((row & 2u) << 1) | ((column & 4u) << 1)
        | ((row & 4u) << 2);
}

// The fragments a simdgroup owns: fm x fn fragments of the sub-tile at tile
// row (sg / wn) * 8 * fm, column (sg % wn) * 8 * fn. A lane holds tile rows
// row(i) and columns column(j), column(j) + 1.
template <uint TM, uint TN>
struct gemm_fragments {
    typedef gemm_tile<TM, TN> tile;
    uint row0, column0;
    ushort2 coordinate;
    gemm_fragments() = default;
    gemm_fragments(uint sg, uint lane) {
        row0 = (sg / tile::wn) * 8u * tile::fm;
        column0 = (sg % tile::wn) * 8u * tile::fn;
        coordinate = fragment_coordinate(lane);
    }
    uint row(uint i) const { return row0 + 8u * i + coordinate.y; }
    uint column(uint j) const { return column0 + 8u * j + coordinate.x; }
};

// The MMA chain of one staged step into the accumulators. Lane (y, x) reads
// its A pair from row y, columns x, x + 1 and its B pair from B rows (weight
// rows) x, x + 1 at column y. Only the first FM fragment rows are multiplied.
template <uint TM, uint TN>
using gemm_accumulators = simdgroup_float8x8[gemm_tile<TM, TN>::fm][gemm_tile<TM, TN>::fn];

template <uint TM, uint TN, typename E, uint FM>
inline void gemm_multiply(gemm_buffer<TM, TN, E> buffer, gemm_fragments<TM, TN> at,
    thread gemm_accumulators<TM, TN> &acc) {
    constexpr uint FN = gemm_tile<TM, TN>::fn;
    threadgroup const E *a_lane = buffer.a + (at.row0 + at.coordinate.y) * gemm_lda + at.coordinate.x;
    threadgroup const half *b_lane = buffer.b + (at.column0 + at.coordinate.x) * gemm_lda + at.coordinate.y;
    PROJECTION_UNROLL
    for (uint step = 0; step < gemm_k / 8u; ++step) {
        uint kk = step * 8u;
        simdgroup_matrix<E, 8, 8> a[FM];
        simdgroup_half8x8 b[FN];
        PROJECTION_UNROLL
        for (uint i = 0; i < FM; ++i)
            reinterpret_cast<thread vec<E, 2> &>(a[i].thread_elements()) =
                *reinterpret_cast<threadgroup const vec<E, 2> *>(a_lane + 8u * i * gemm_lda + kk);
        PROJECTION_UNROLL
        for (uint j = 0; j < FN; ++j) {
            threadgroup const half *pair = b_lane + 8u * j * gemm_lda + kk;
            reinterpret_cast<thread half2 &>(b[j].thread_elements()) = half2(pair[0], pair[gemm_lda]);
        }
        PROJECTION_UNROLL
        for (uint i = 0; i < FM; ++i)
            PROJECTION_UNROLL
            for (uint j = 0; j < FN; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
}

// The fragment rows of a simdgroup that hold rows before `m_rows` (tile rows
// from m0), 0 ..= fm. The multiply is instantiated per count, so a full tile
// runs the unbranched chain and a padded one only its live fragment rows.
template <uint TM, uint TN>
inline uint gemm_live_fragments(gemm_fragments<TM, TN> at, uint m0, uint m_rows) {
    constexpr uint FM = gemm_tile<TM, TN>::fm;
    uint first = m0 + at.row0;
    return first >= m_rows ? 0u : min(FM, (m_rows - first + 7u) / 8u);
}

template <uint TM, uint TN, typename E>
inline void gemm_multiply_live(gemm_buffer<TM, TN, E> buffer, gemm_fragments<TM, TN> at, uint live,
    thread gemm_accumulators<TM, TN> &acc) {
    static_assert(gemm_tile<TM, TN>::fm == 4, "live counts are instantiated for 4 fragment rows");
    switch (live) {
    case 4: gemm_multiply<TM, TN, E, 4>(buffer, at, acc); break;
    case 3: gemm_multiply<TM, TN, E, 3>(buffer, at, acc); break;
    case 2: gemm_multiply<TM, TN, E, 2>(buffer, at, acc); break;
    case 1: gemm_multiply<TM, TN, E, 1>(buffer, at, acc); break;
    default: break;
    }
}

// The fragment chain as GEMM accumulators: the simdgroup's fm x fn F32
// fragments, multiplied over its live fragment rows. `emit(f)` calls
// f(m, n, first, second) for each of the lane's outputs at tile row m: a
// plain tile's column n (`second` unused), a paired tile's feature n (gate
// `first`, up `second`, a fragment pair holding one feature).
template <uint TM, uint TN, typename E, bool PAIRED>
struct gemm_fragment_engine {
    typedef gemm_tile<TM, TN> tile;
    gemm_fragments<TM, TN> at;
    uint live;
    gemm_accumulators<TM, TN> acc;

    // The struct stays an aggregate and its accumulators are reached through
    // a thread reference: Metal 4.1 gives a member reached through `this` no
    // address space, and simdgroup matrices only assign and read in `thread`.
    static inline void zero(thread gemm_fragment_engine &self, uint sg, uint lane, uint m0, uint m_rows) {
        self.at = gemm_fragments<TM, TN>(sg, lane);
        self.live = gemm_live_fragments<TM, TN>(self.at, m0, m_rows);
        PROJECTION_UNROLL
        for (uint i = 0; i < tile::fm; ++i)
            PROJECTION_UNROLL
            for (uint j = 0; j < tile::fn; ++j)
                self.acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    template <typename F>
    static inline void emit(thread gemm_fragment_engine &self, thread const F &f) {
        PROJECTION_UNROLL
        for (uint i = 0; i < tile::fm; ++i) {
            PROJECTION_UNROLL
            for (uint j = 0; j < tile::fn; ++j) {
                float2 c = reinterpret_cast<thread float2 &>(self.acc[i][j].thread_elements());
                if (PAIRED) {
                    f(self.at.row(i), self.at.column(j) / 2u, c.x, c.y);
                } else {
                    f(self.at.row(i), self.at.column(j), c.x, 0.0f);
                    f(self.at.row(i), self.at.column(j) + 1u, c.y, 0.0f);
                }
            }
        }
    }
};

template <uint TM, uint TN, typename E, bool PAIRED>
inline void gemm_step(gemm_buffer<TM, TN, E> buffer, thread gemm_fragment_engine<TM, TN, E, PAIRED> &engine) {
    gemm_multiply_live<TM, TN, E>(buffer, engine.at, engine.live, engine.acc);
}

#if SEISMIC_HAS_TENSOR_OPS
// The tensor operation as GEMM accumulators: one `matmul2d` over all the
// tile's simdgroups multiplies the staged step (A rows and B weight rows at
// pitch gemm_lda) into a cooperative F32 destination. A paired tile runs one
// product per stream over its interleaved B rows (row stride 2 gemm_lda), so
// both destinations share one layout and element i of each is the same
// (row, feature). Live rows are not distinguished: padding rows multiply
// their staged zeros. Cooperative tensors live in the caller's frame.
template <uint TM, uint TN, typename E, bool PAIRED>
struct gemm_tensor {
    typedef gemm_tile<TM, TN> tile;
    static constant constexpr uint columns = PAIRED ? TN / 2u : TN;
    static constant constexpr int32_t pitch = int32_t(gemm_lda) * (PAIRED ? 2 : 1);
    typedef metal::extents<int32_t, gemm_k, TM> a_extents;
    typedef metal::extents<int32_t, gemm_k, columns> b_extents;
    typedef metal::tensor<threadgroup E, a_extents, metal::tensor_inline> a_tensor;
    typedef metal::tensor<threadgroup half, b_extents, metal::tensor_inline> b_tensor;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(TM, columns, gemm_k, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroups<tile::threads / 32u>> operation;
    typedef typename operation::template cooperative_tensor_destination_t<a_tensor, b_tensor, float> destination;

    static destination zero() {
        operation op;
        destination acc = op.template get_destination_cooperative_tensor<a_tensor, b_tensor, float>();
        PROJECTION_UNROLL
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i))
                acc[i] = 0.0f;
        return acc;
    }

    // B rows from `row` (0 or 1) at the tile's B row pitch.
    static b_tensor weights(gemm_buffer<TM, TN, E> buffer, uint row) {
        return b_tensor(buffer.b + row * gemm_lda, b_extents(), metal::array<int32_t, 2>{1, pitch});
    }

    static a_tensor activations(gemm_buffer<TM, TN, E> buffer) {
        return a_tensor(buffer.a, a_extents(), metal::array<int32_t, 2>{1, int32_t(gemm_lda)});
    }

    template <typename F>
    static void emit(thread const destination &acc, thread const destination &acc2, thread const F &f) {
        PROJECTION_UNROLL
        for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
            if (acc.is_valid_element(i)) {
                auto index = acc.get_multidimensional_index(i);
                f(uint(index[1]), uint(index[0]), acc[i], PAIRED ? acc2[i] : 0.0f);
            }
        }
    }
};

// A plain tile's step (one destination), and a paired tile's (one per stream).
template <uint TM, uint TN, typename E, typename D>
inline void gemm_step(gemm_buffer<TM, TN, E> buffer, thread D &acc) {
    typedef gemm_tensor<TM, TN, E, false> T;
    typename T::operation op;
    auto a = T::activations(buffer);
    auto b = T::weights(buffer, 0);
    op.run(a, b, acc);
}

template <uint TM, uint TN, typename E, typename D>
inline void gemm_step(gemm_buffer<TM, TN, E> buffer, thread D &gate, thread D &up) {
    typedef gemm_tensor<TM, TN, E, true> T;
    typename T::operation op;
    auto a = T::activations(buffer);
    auto g = T::weights(buffer, 0);
    auto u = T::weights(buffer, 1);
    op.run(a, g, gate);
    op.run(a, u, up);
}
#endif

// The K loop over one TM x TN tile, steps [step_begin, step_end): `U` is the
// second weight of a paired tile (its rows interleave with W's) or the same
// type for a plain tile; `gemm_step` multiplies each staged step into the
// accumulators `acc`.
template <typename W, typename U, bool PAIRED, uint TM, uint TN, typename In, typename... Acc>
inline void gemm_accumulate(thread const In &in, thread const Weights<W> &w,
    thread const Weights<U> &u, uint first, uint rows, uint m0, uint m_rows, uint k,
    uint step_begin, uint step_end, threadgroup uchar *shared, uint sg, uint lane,
    thread Acc &... acc) {
    typedef gemm_tile<TM, TN> tile;
    typedef typename In::activation::native E;
    constexpr uint count = PAIRED ? TN / 2u : TN;
    constexpr uint spacing = PAIRED ? 2u : 1u;
    constexpr uint items = (count * gemm_halves + tile::threads - 1) / tile::threads;
    uint thread_index = sg * 32u + lane;
    gemm_buffer<TM, TN, E> buffer = gemm_buffer<TM, TN, E>::at(shared);
    if (step_begin >= step_end)
        return;
    gemm_a_registers<tile::a_items> a_regs;
    gemm_b_registers<W, items> w_regs;
    gemm_b_registers<U, items> u_regs;
    uint k0 = step_begin * gemm_k;
    gemm_load_a<TM, tile::threads, tile::a_items>(in, m0, m_rows, k0, k, thread_index, a_regs);
    gemm_load_b<W, tile::threads, items, gemm_runs<TM, PAIRED>()>(w, first, count, rows, k0, k, thread_index, true, w_regs);
    if (PAIRED)
        gemm_load_b<U, tile::threads, items, gemm_runs<TM, PAIRED>()>(u, first, count, rows, k0, k, thread_index, true, u_regs);
    gemm_store_a<TM, tile::threads, tile::a_items>(a_regs, buffer.a, thread_index);
    gemm_store_b<W, tile::threads, items>(w_regs, count, 0, spacing, buffer.b, thread_index);
    if (PAIRED)
        gemm_store_b<U, tile::threads, items>(u_regs, count, 1, spacing, buffer.b, thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = step_begin; t < step_end; ++t) {
        bool more = t + 1 < step_end;
        uint k1 = (t + 1) * gemm_k;
        if (more) {
            gemm_load_a<TM, tile::threads, tile::a_items>(in, m0, m_rows, k1, k, thread_index, a_regs);
            gemm_load_b<W, tile::threads, items, gemm_runs<TM, PAIRED>()>(w, first, count, rows, k1, k, thread_index, false, w_regs);
            if (PAIRED)
                gemm_load_b<U, tile::threads, items, gemm_runs<TM, PAIRED>()>(u, first, count, rows, k1, k, thread_index, false, u_regs);
        }
        gemm_step(buffer, acc...);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (more) {
            gemm_store_a<TM, tile::threads, tile::a_items>(a_regs, buffer.a, thread_index);
            gemm_store_b<W, tile::threads, items>(w_regs, count, 0, spacing, buffer.b, thread_index);
            if (PAIRED)
                gemm_store_b<U, tile::threads, items>(u_regs, count, 1, spacing, buffer.b, thread_index);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One tile's K steps [step_begin, step_end) on the device's matrix facility,
// then `emit(m, n, first, second)` per output (gemm_fragment_engine::emit).
// Rows at or past `live_rows` are known zero.
template <typename W, typename U, bool PAIRED, uint TM, uint TN, typename In, typename F>
inline void gemm_run(thread const In &in, thread const Weights<W> &w, thread const Weights<U> &u,
    uint first, uint rows, uint m0, uint m_rows, uint k, uint step_begin, uint step_end, uint live_rows,
    threadgroup uchar *shared, uint sg, uint lane, thread const F &emit) {
    typedef typename In::activation::native E;
#if SEISMIC_HAS_TENSOR_OPS
    typedef gemm_tensor<TM, TN, E, PAIRED> T;
    typename T::destination acc = T::zero();
    if constexpr (PAIRED) {
        typename T::destination acc2 = T::zero();
        gemm_accumulate<W, U, PAIRED, TM, TN>(in, w, u, first, rows, m0, m_rows, k, step_begin, step_end,
            shared, sg, lane, acc, acc2);
        T::emit(acc, acc2, emit);
    } else {
        gemm_accumulate<W, U, PAIRED, TM, TN>(in, w, u, first, rows, m0, m_rows, k, step_begin, step_end,
            shared, sg, lane, acc);
        T::emit(acc, acc, emit);
    }
#else
    typedef gemm_fragment_engine<TM, TN, E, PAIRED> Engine;
    Engine engine;
    Engine::zero(engine, sg, lane, m0, min(m_rows, live_rows));
    gemm_accumulate<W, U, PAIRED, TM, TN>(in, w, u, first, rows, m0, m_rows, k, step_begin, step_end,
        shared, sg, lane, engine);
    Engine::emit(engine, emit);
#endif
}

// Emitters of a finished tile (tile row m, column or feature n, from m0 and
// first): the epilogue of a plain or paired tile, or a split part's raw sums.
template <typename Out>
struct gemm_store {
    Out out;
    uint m0, m_rows, first, rows;
    void operator()(uint m, uint n, float value, float) const {
        if (m0 + m < m_rows && first + n < rows)
            out.store(m0 + m, first + n, value);
    }
};

template <typename Out>
struct gemm_store_pair {
    Out out;
    uint m0, m_rows, first, rows;
    void operator()(uint m, uint n, float gate, float up) const {
        if (m0 + m < m_rows && first + n < rows)
            out.store_pair(m0 + m, first + n, gate, up);
    }
};

struct gemm_store_part {
    device float *own;
    uint m0, m_rows, first, rows;
    void operator()(uint m, uint n, float value, float) const {
        if (m0 + m < m_rows && first + n < rows)
            own[ulong(m0 + m) * rows + first + n] = value;
    }
};

// One GEMM tile of a plain (unpaired) projection: output rows tm * TM .. of
// the activations, weight rows tn * TN .. of `w`. Rows at or past `live_rows`
// are known zero (padding): the fragment chain skips their products, and
// they store the epilogue of 0.
template <typename W, uint TM, uint TN, typename In, typename Out>
inline void gemm(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane, uint live_rows = ~0u) {
    uint first = tn * TN, m0 = tm * TM;
    gemm_run<W, W, false, TM, TN>(in, w, w, first, rows, m0, m_rows, k, 0, (k + gemm_k - 1) / gemm_k, live_rows,
        shared, sg, lane, gemm_store<Out>{out, m0, m_rows, first, rows});
}

// Split-K. Part `part` of `split` runs the K steps
// [part * steps / split, (part + 1) * steps / split) of one plain GEMM tile and
// stores its raw F32 sums to partials[(part * m_rows + m) * rows + n]. With
// split 1 the tile applies the epilogue directly (`gemm`).
template <typename W, uint TM, uint TN, typename In>
inline void gemm_part(thread const In &in, device float *partials, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint split, uint part, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
    uint first = tn * TN, m0 = tm * TM;
    uint steps = (k + gemm_k - 1) / gemm_k;
    gemm_run<W, W, false, TM, TN>(in, w, w, first, rows, m0, m_rows, k, part * steps / split,
        (part + 1) * steps / split, m_rows, shared, sg, lane,
        gemm_store_part{partials + ulong(part) * m_rows * rows, m0, m_rows, first, rows});
}

// The split-K reduction: output `index` = m * rows + n sums its parts in
// order and applies the epilogue.
template <typename Out>
inline void gemm_reduce(thread const Out &out, device const float *partials, uint m_rows, uint rows, uint split,
    uint index) {
    if (index >= m_rows * rows)
        return;
    float total = 0.0f;
    for (uint part = 0; part < split; ++part)
        total += partials[ulong(part) * m_rows * rows + index];
    out.store(index / rows, index % rows, total);
}

// One GEMM tile of a paired projection: TN / 2 features of gate and up, the
// tile's rows interleaved (gate, up). `live_rows` as for `gemm`.
template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void gemm_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tm, uint tn,
    threadgroup uchar *shared, uint sg, uint lane, uint live_rows = ~0u) {
    uint first = tn * (TN / 2u), m0 = tm * TM;
    gemm_run<G, U, true, TM, TN>(in, gate, up, first, rows, m0, m_rows, k, 0, (k + gemm_k - 1) / gemm_k,
        live_rows, shared, sg, lane, gemm_store_pair<Out>{out, m0, m_rows, first, rows});
}

// ---------------------------------------------------------------------------
// Tall GEMM (M > 64, the TALL form).
//
// A threadgroup owns TM output rows by 32 weight rows (16 features of gate
// and up for a paired tile). NS staging simdgroups decode the tile's weights
// once for all TM rows into threadgroup memory, KS columns per stage and one
// stage ahead (two buffers), so one barrier per KS columns separates a stage
// from its use; TM / 32 multiplying simdgroups read the staged weights and
// take their activations straight from device memory. The staged tile above
// decodes every weight once per TM <= 128 rows and stages the activations too.
//
// K is a multiple of KS (the entries admit the form for K % 128 == 0), so
// every stage is whole. Products accumulate in F32 in K order per output,
// as in the staged tile.
//
// The activations are a `Tall` operand: with tensor operations the plain
// row-major operand itself, read as a device tensor; on simdgroup matrices a
// fragment-ordered copy (`Fragments`), written by the entry's pre-pass
// (`TallOrder` in `device_normalize`) or by `tall_relayout` from a plain input.
constant constexpr uint tall_n = 32;

// Threadgroup storage of one tall threadgroup staging KS columns, as
// `threadgroup uchar *name`.
#define PROJECTION_GEMM_TALL_SHARED(name, KS)                                                       \
    threadgroup float4 name##_words[((KS) + 8) * 8];                                                \
    threadgroup uchar *name = reinterpret_cast<threadgroup uchar *>(name##_words)

// Fragment-ordered activations: an operand laid out for lanes that read it
// straight from device memory. Per 64-row tile t and 8-column block g, lane
// (y, x) of an 8 x 8 fragment owns eight element pairs in a row, pair i
// being x[64 t + 8 i + y, 8 g + x + {0, 1}] as stored; a lane loads its
// pairs of a block as two uint4. `columns` is a multiple of 8, and the last
// tile is whole (rows past the operand's hold anything).
template <typename A>
struct Fragments {
    typedef A activation;
    device const uchar *x;
    uint columns;
    // The pairs of lane `lane` in block g of tile t.
    device const uint4 *block(uint t, uint g, uint lane) const {
        return reinterpret_cast<device const uint4 *>(x) + ((ulong(t) * (columns / 8u) + g) * 32u + lane) * 2u;
    }
    // x[m, k .. k + 8) as its storage words (pair w in word w).
    static void store8(device uchar *x, uint columns, uint m, uint k, uint4 words) {
        device uint *block = reinterpret_cast<device uint *>(x)
            + (ulong(m / 64u) * (columns / 8u) + k / 8u) * 256u + (m % 64u) / 8u;
        PROJECTION_UNROLL
        for (ushort w = 0; w < 4; ++w)
            block[fragment_lane(ushort(2 * w), ushort(m % 8u)) * 8u] = words[w];
    }
};

template <typename W, typename U, uint ITEMS>
struct gemm_tall_runs {
    typename packets::Block<W>::state w[ITEMS];
    typename packets::Block<U>::state u[ITEMS];
};

#if SEISMIC_HAS_TENSOR_OPS
template <typename A>
using Tall = Plain<A, AllRows>;
template <typename A>
using TallOrder = RowOrder<A>;

template <typename A>
inline Tall<A> tall_operand(Plain<A, AllRows> rows, device const uchar *) {
    return rows;
}

// The relayout pass of a plain operand: nothing, the tensor reads it in place.
template <typename In>
inline void tall_relayout(thread const In &, device uchar *, uint, uint, uint) {}

// The staged weights of two stages: tile row r of buffer `slot` holds its KS
// columns in order. Tile rows are weight rows first .. (plain), or 16 gate
// rows then 16 up rows of features first ..
template <uint KS>
struct gemm_tall_stage {
    static constant constexpr uint pitch = KS + 8u;
    threadgroup half *base;
    threadgroup half *row(uint slot, uint r) const { return base + (slot * tall_n + r) * pitch; }
};

// Half `within` (16 columns) of every packet of weight row n in stage `group`.
template <typename X, uint KS>
inline void gemm_tall_stage_item(thread const Weights<X> &x, uint n, uint group, uint within, bool fresh,
    thread typename packets::Block<X>::state &run, threadgroup half *row) {
    PROJECTION_UNROLL
    for (uint i = 0; i < KS / 32u; ++i) {
        uint p = group * (KS / 32u) + i;
        if ((fresh && i == 0) || p % packets::Block<X>::packets == 0)
            run = x.run(n, p);
        typename X::packet packet = x.packet(n, p, run);
        float scale = X::scale(packet, 2u * within);
        float bias = X::bias(packet, X::groups == 1 ? 0u : within);
        threadgroup half4 *to = reinterpret_cast<threadgroup half4 *>(row + 32u * i + 16u * within);
        PROJECTION_UNROLL
        for (uint s = 0; s < 2; ++s) {
            float4 even, odd;
            X::codes(packet, 2u * within + s, even, odd);
            even = metal::fma(float4(scale), even, float4(bias));
            odd = metal::fma(float4(scale), odd, float4(bias));
            to[2 * s] = half4(half(even.x), half(odd.x), half(even.y), half(odd.y));
            to[2 * s + 1] = half4(half(even.z), half(odd.z), half(even.w), half(odd.w));
        }
    }
}

// A staging thread's items of stage `group`: item i is half i % 2 of every
// packet of tile row i / 2.
template <typename W, typename U, bool PAIRED, uint KS, uint NS, uint ITEMS>
inline void gemm_tall_stage_group(thread const Weights<W> &w, thread const Weights<U> &u, uint first, uint rows,
    uint group, bool fresh, gemm_tall_stage<KS> stage, uint thread_index, thread gemm_tall_runs<W, U, ITEMS> &runs) {
    PROJECTION_UNROLL
    for (uint j = 0; j < ITEMS; ++j) {
        uint item = thread_index + j * NS * 32u;
        uint r = item / 2u, within = item % 2u;
        threadgroup half *row = stage.row(group & 1u, r);
        if (PAIRED && r >= 16u)
            gemm_tall_stage_item<U, KS>(u, min(first + r - 16u, rows - 1u), group, within, fresh, runs.u[j], row);
        else
            gemm_tall_stage_item<W, KS>(w, min(first + r, rows - 1u), group, within, fresh, runs.w[j], row);
    }
}

// A multiplying simdgroup owns 32 rows as two 16-row groups, each one
// `matmul2d` 16 x 32 x KS per stage with the operand as a device tensor. A
// group that would run past the operand's last row starts 16 rows before it
// instead and publishes only its own rows. A paired tile's lane publishes a
// feature's gate and up together: on the cooperative layout it holds (m, n)
// and (m, n + 16) eight elements apart.
template <typename W, typename U, bool PAIRED, uint TM, uint KS, uint NS, typename In, typename F>
inline void gemm_tall_run(thread const In &in, thread const Weights<W> &w, thread const Weights<U> &u,
    uint first, uint rows, uint m0, uint m_rows, uint k, threadgroup uchar *shared, uint sg, uint lane,
    thread const F &emit) {
    static_assert(TM % 32 == 0 && KS % 32 == 0 && (NS == 1 || NS == 2),
        "a tall tile is whole 32-row simdgroups and packets");
    typedef typename In::activation::native E;
    constexpr uint items = 2u / NS;
    gemm_tall_stage<KS> stage{reinterpret_cast<threadgroup half *>(shared)};
    bool stager = sg < NS;
    uint thread_index = sg * 32u + lane;
    uint groups = k / KS;
    uint row0 = m0 + (sg - NS) * 32u;
    bool live = !stager && row0 < m_rows;
    typedef metal::extents<int32_t, KS, 16> a_extents;
    typedef metal::extents<int32_t, KS, tall_n> b_extents;
    typedef metal::tensor<device E, a_extents, metal::tensor_inline> a_tensor;
    typedef metal::tensor<threadgroup half, b_extents, metal::tensor_inline> b_tensor;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(16, tall_n, KS, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroup> operation;
    typedef typename operation::template cooperative_tensor_destination_t<a_tensor, b_tensor, float> destination;
    operation op;
    destination upper = op.template get_destination_cooperative_tensor<a_tensor, b_tensor, float>();
    destination lower = op.template get_destination_cooperative_tensor<a_tensor, b_tensor, float>();
    PROJECTION_UNROLL
    for (uint16_t e = 0; e < upper.get_capacity(); ++e) {
        if (upper.is_valid_element(e)) {
            upper[e] = 0.0f;
            lower[e] = 0.0f;
        }
    }
    gemm_tall_runs<W, U, items> runs;
    if (stager)
        gemm_tall_stage_group<W, U, PAIRED, KS, NS, items>(w, u, first, rows, 0, true, stage, thread_index, runs);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start0 = min(row0, m_rows - 16u), start1 = min(row0 + 16u, m_rows - 16u);
    device E *base = reinterpret_cast<device E *>(const_cast<device uchar *>(in.x));
    device E *base0 = base + ulong(start0) * in.stride0, *base1 = base + ulong(start1) * in.stride0;
    for (uint group = 0; group < groups; ++group) {
        if (stager) {
            if (group + 1u < groups)
                gemm_tall_stage_group<W, U, PAIRED, KS, NS, items>(w, u, first, rows, group + 1u, false, stage,
                    thread_index, runs);
        } else if (live) {
            b_tensor b(stage.row(group & 1u, 0), b_extents(),
                metal::array<int32_t, 2>{1, int32_t(gemm_tall_stage<KS>::pitch)});
            a_tensor a0(base0 + ulong(group * KS) * in.stride1, a_extents(),
                metal::array<int32_t, 2>{int32_t(in.stride1), int32_t(in.stride0)});
            a_tensor a1(base1 + ulong(group * KS) * in.stride1, a_extents(),
                metal::array<int32_t, 2>{int32_t(in.stride1), int32_t(in.stride0)});
            op.run(a0, b, upper);
            op.run(a1, b, lower);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live)
        return;
    const auto publish = [&](uint own, uint start, thread destination &acc) {
        PROJECTION_UNROLL
        for (uint16_t e = 0; e < acc.get_capacity(); ++e) {
            if (!acc.is_valid_element(e))
                continue;
            auto index = acc.get_multidimensional_index(e);
            uint m = start + uint(index[1]), n = uint(index[0]);
            if (m < own)
                continue;
            if (!PAIRED)
                emit(m - m0, n, acc[e], 0.0f);
            else if (n < 16u)
                emit(m - m0, n, acc[e], acc[e + 8]);
        }
    };
    publish(row0, start0, upper);
    publish(row0 + 16u, start1, lower);
}
#else
template <typename A>
using Tall = Fragments<A>;
template <typename A>
using TallOrder = Fragments<A>;

template <typename A>
inline Tall<A> tall_operand(Plain<A, AllRows> rows, device const uchar *fragments) {
    return {fragments, rows.columns};
}

// The relayout pass of a plain operand: item = (64-row tile, 32-column span,
// lane). A thread gathers its lane's pairs of the span's four blocks from
// the tile's rows (zero past the operand's last row) and stores each block's
// as two uint4.
template <typename In>
inline void tall_relayout(thread const In &in, device uchar *x, uint m_rows, uint columns, uint item) {
    uint lane = item % 32u, span = item / 32u;
    uint spans = columns / 32u;
    uint t = span / spans, g0 = 4u * (span % spans);
    if (64u * t >= m_rows)
        return;
    ushort2 at = fragment_coordinate(lane);
    PROJECTION_UNROLL
    for (uint b = 0; b < 4; ++b) {
        uint g = g0 + b;
        uint4 low, high;
        PROJECTION_UNROLL
        for (uint i = 0; i < 4; ++i) {
            low[i] = in.words2(64u * t + 8u * i + at.y, 8u * g + at.x, m_rows);
            high[i] = in.words2(64u * t + 8u * (i + 4u) + at.y, 8u * g + at.x, m_rows);
        }
        device uint4 *to = reinterpret_cast<device uint4 *>(x) + ((ulong(t) * (columns / 8u) + g) * 32u + lane) * 2u;
        to[0] = low;
        to[1] = high;
    }
}

// The staged weights of two stages in fragment lane order: lane `lane`'s
// weight pair of fragment column j (tile rows 8 j ..) in 8-column block q.
template <uint KS>
struct gemm_tall_stage {
    threadgroup half2 *base;
    threadgroup half2 *at(uint slot, uint q, uint j, uint lane) const {
        return base + ((slot * (KS / 8u) + q) * (tall_n / 8u) + j) * 32u + lane;
    }
};

// The 8 values of row n in block `block` of packet p, as (even, odd).
template <typename X>
inline void gemm_tall_decode(thread const Weights<X> &x, uint n, uint p, uint block, bool fresh,
    thread typename packets::Block<X>::state &run, thread float4 &even, thread float4 &odd) {
    if (fresh || p % packets::Block<X>::packets == 0)
        run = x.run(n, p);
    typename X::packet packet = x.packet(n, p, run);
    X::codes(packet, block, even, odd);
    float scale = X::scale(packet, 2u * (block / 2u));
    float bias = X::bias(packet, X::groups == 1 ? 0u : block / 2u);
    even = metal::fma(float4(scale), even, float4(bias));
    odd = metal::fma(float4(scale), odd, float4(bias));
}

// A staging thread's items of stage `group`: item i is the tile's weight row
// pair i % 16 at the 8-column block i / 16 of every packet. A pair is two
// tile rows a lane multiplies together: weight rows first + 2 i, + 1 (plain)
// or gate and up of feature first + i.
template <typename W, typename U, bool PAIRED, uint KS, uint NS, uint ITEMS>
inline void gemm_tall_stage_group(thread const Weights<W> &w, thread const Weights<U> &u, uint first, uint rows,
    uint group, bool fresh, gemm_tall_stage<KS> stage, uint thread_index, thread gemm_tall_runs<W, U, ITEMS> &runs) {
    PROJECTION_UNROLL
    for (uint j = 0; j < ITEMS; ++j) {
        uint item = thread_index + j * NS * 32u;
        uint pair = item % (tall_n / 2u), block = item / (tall_n / 2u);
        uint n0 = min(PAIRED ? first + pair : first + 2u * pair, rows - 1u);
        uint n1 = min(PAIRED ? first + pair : first + 2u * pair + 1u, rows - 1u);
        ushort column = ushort(2u * (pair % 4u));
        PROJECTION_UNROLL
        for (uint i = 0; i < KS / 32u; ++i) {
            uint p = group * (KS / 32u) + i;
            float4 even0, odd0, even1, odd1;
            gemm_tall_decode<W>(w, n0, p, block, fresh && i == 0, runs.w[j], even0, odd0);
            gemm_tall_decode<U>(u, n1, p, block, fresh && i == 0, runs.u[j], even1, odd1);
            PROJECTION_UNROLL
            for (ushort v = 0; v < 4; ++v) {
                *stage.at(group & 1u, 4u * i + block, pair / 4u, fragment_lane(column, ushort(2 * v))) =
                    half2(half(even0[v]), half(even1[v]));
                *stage.at(group & 1u, 4u * i + block, pair / 4u, fragment_lane(column, ushort(2 * v + 1))) =
                    half2(half(odd0[v]), half(odd1[v]));
            }
        }
    }
}

// A multiplying simdgroup owns 64 rows by 16 tile rows of weights (8 x 2
// fragments): per 8-column block a lane loads its eight activation pairs as
// two uint4 and its two weight pairs as one half2 each.
template <typename W, typename U, bool PAIRED, uint TM, uint KS, uint NS, typename In, typename F>
inline void gemm_tall_run(thread const In &in, thread const Weights<W> &w, thread const Weights<U> &u,
    uint first, uint rows, uint m0, uint m_rows, uint k, threadgroup uchar *shared, uint sg, uint lane,
    thread const F &emit) {
    static_assert(TM % 64 == 0 && KS % 32 == 0 && (NS == 1 || NS == 2),
        "a tall tile is whole 64-row simdgroups and packets");
    typedef typename In::activation::native E;
    constexpr uint TI = 8, TJ = 2;
    constexpr uint items = 2u / NS;
    gemm_tall_stage<KS> stage{reinterpret_cast<threadgroup half2 *>(shared)};
    bool stager = sg < NS;
    uint thread_index = sg * 32u + lane;
    // A multiplying simdgroup's 64-row tile of the operand and half of the weight rows.
    uint local = (sg - NS) / 2u, side = (sg - NS) % 2u;
    uint tile = m0 / 64u + local;
    bool live = !stager && 64u * tile < m_rows;
    uint groups = k / KS;
    ushort2 at = fragment_coordinate(lane);
    simdgroup_float8x8 acc[TI][TJ];
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i)
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    gemm_tall_runs<W, U, items> runs;
    if (stager)
        gemm_tall_stage_group<W, U, PAIRED, KS, NS, items>(w, u, first, rows, 0, true, stage, thread_index, runs);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = 0; group < groups; ++group) {
        if (stager) {
            if (group + 1u < groups)
                gemm_tall_stage_group<W, U, PAIRED, KS, NS, items>(w, u, first, rows, group + 1u, false, stage,
                    thread_index, runs);
        } else if (live) {
            PROJECTION_UNROLL
            for (uint q = 0; q < KS / 8u; ++q) {
                simdgroup_matrix<E, 8, 8> a[TI];
                simdgroup_half8x8 b[TJ];
                device const uint4 *pairs = in.block(tile, group * (KS / 8u) + q, lane);
                uint4 low = pairs[0], high = pairs[1];
                PROJECTION_UNROLL
                for (uint i = 0; i < 4; ++i) {
                    reinterpret_cast<thread uint &>(a[i].thread_elements()) = low[i];
                    reinterpret_cast<thread uint &>(a[i + 4].thread_elements()) = high[i];
                }
                PROJECTION_UNROLL
                for (uint j = 0; j < TJ; ++j)
                    reinterpret_cast<thread half2 &>(b[j].thread_elements()) =
                        *stage.at(group & 1u, q, 2u * side + j, lane);
                PROJECTION_UNROLL
                for (uint i = 0; i < TI; ++i)
                    PROJECTION_UNROLL
                    for (uint j = 0; j < TJ; ++j)
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live)
        return;
    PROJECTION_UNROLL
    for (uint i = 0; i < TI; ++i) {
        PROJECTION_UNROLL
        for (uint j = 0; j < TJ; ++j) {
            float2 c = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
            uint m = local * 64u + 8u * i + at.y, n = 16u * side + 8u * j + at.x;
            if (PAIRED) {
                emit(m, n / 2u, c.x, c.y);
            } else {
                emit(m, n, c.x, 0.0f);
                emit(m, n + 1u, c.y, 0.0f);
            }
        }
    }
}
#endif

// One tall tile of a plain projection: output rows tm * TM .., weight rows
// tn * 32 .. of `w`.
template <typename W, uint TM, uint KS, uint NS, typename In, typename Out>
inline void gemm_tall(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows, uint rows,
    uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
    uint first = tn * tall_n, m0 = tm * TM;
    gemm_tall_run<W, W, false, TM, KS, NS>(in, w, w, first, rows, m0, m_rows, k, shared, sg, lane,
        gemm_store<Out>{out, m0, m_rows, first, rows});
}

// One tall tile of a paired projection: 16 features of gate and up.
template <typename G, typename U, uint TM, uint KS, uint NS, typename In, typename Out>
inline void gemm_tall_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared,
    uint sg, uint lane) {
    uint first = tn * (tall_n / 2u), m0 = tm * TM;
    gemm_tall_run<G, U, true, TM, KS, NS>(in, gate, up, first, rows, m0, m_rows, k, shared, sg, lane,
        gemm_store_pair<Out>{out, m0, m_rows, first, rows});
}

// ---------------------------------------------------------------------------
// Batched GEMV on the matrix units (BATCH_FROM <= M <= 16).
//
// C^T = W X^T per block of 8 weight rows, in F32: the A fragment holds 8
// weight rows by 8 columns, each lane decoding its two values (lane (y, x)
// holds row y, columns x and x + 1 of the step; the four lanes of a row read
// the same packet) as scale * code + bias with one F32 rounding. The B
// fragments are X^T for the step, NB = 1 (M <= 8) or 2 (M <= 16) blocks of 8
// activation rows, loaded from the activations staged in F32 (exact), whose
// rows past m_rows are zero. Products are F32 and accumulate in F32, so this
// class differs from the GEMV only in summation order. A simdgroup owns R
// blocks of 8 weight rows (`gemv_batch_rows`) and walks them in packet order,
// forming packets through coefficient runs (`gemv_batch_load`); neither
// changes a value or the summation order. The threadgroup stages the
// activations in chunks
// of `gemv_batch_chunk<NB>` columns, and each lane loads its next packet while
// the current one multiplies. A batched GEMV launch declares
//   shared_bytes (16640)
// (8 * NB staged rows of `gemv_batch_chunk<NB> + 4` floats; a chunk is a
// whole number of packets: 512 columns for NB = 1, 256 for NB = 2). A caller
// that stages in a smaller buffer (the grouped expert blocks reuse their GEMM
// tile's) passes its size as BYTES; the chunk shrinks with it.

constant constexpr uint gemv_batch_shared_bytes = 16640;
template <uint NB, uint BYTES = gemv_batch_shared_bytes>
constexpr uint gemv_batch_chunk() {
    static_assert(BYTES / (32u * NB) >= 36u, "the staging holds at least one packet per row");
    return (BYTES / (32u * NB) - 4u) / 32u * 32u;
}

// Weight rows of one batched GEMV threadgroup.
template <uint SG, uint R>
constexpr uint gemv_batch_threadgroup_rows() {
    return SG * R * 8u;
}

// The weight rows of a simdgroup's blocks: block r of the simdgroup is block
// `block` + r of the threadgroup, whose rows start at `first`. Stored row by
// row, a block is eight consecutive rows. Over row tiles (`spread`, for a
// threadgroup of whole tiles) a block is every fourth row of its tile,
// 4 y + block % 4: the packets of eight consecutive rows of a tile share two
// cache lines, and a block whose lanes all load from those is slower than
// one whose rows have a line each.
struct gemv_batch_rows {
    uint first;
    uint block;
    bool spread;
    uint at(uint r, uint y) const {
        uint b = block + r;
        return first + (spread ? (b / 4u) * 32u + 4u * y + b % 4u : 8u * b + y);
    }
};

// Packet p of the weight row at `at` for a lane that walks the row in packet
// order: formed through the row's coefficient run (`packets::Block`), which
// is loaded at each run's first packet. A lane then reads the coefficient
// planes once per run instead of once per packet.
template <typename W>
inline typename W::packet gemv_batch_load(thread const Weights<W> &w,
    thread const typename Weights<W>::located &at, uint p, thread typename packets::Block<W>::state &run) {
    if (p % packets::Block<W>::packets == 0u)
        run = w.run(at, p);
    return w.packet(at, p, run);
}

// Accumulate one packet of this lane's weight row into `acc`: the decoded
// F32 values times the staged steps `x`.
template <typename W, uint NB>
inline void gemv_batch_packet(thread simdgroup_float8x8 (&acc)[NB], thread const typename W::packet &a,
    thread const simdgroup_float8x8 (&x)[4][NB], uint pair_index) {
    for (uint step = 0; step < 4; ++step) {
        simdgroup_float8x8 weights;
        float2 codes = W::pair(a, step, pair_index);
        reinterpret_cast<thread float2 &>(weights.thread_elements()) =
            float2(W::value(a, step, codes.x), W::value(a, step, codes.y));
        for (uint b = 0; b < NB; ++b)
            simdgroup_multiply_accumulate(acc[b], weights, x[step][b], acc[b]);
    }
}

// Publishes a batched GEMV simdgroup's sums: lane (y, x) owns rows
// 8 b + x, 8 b + x + 1 of output first_row + 8 r + y. A column epilogue
// (`gemv_emit<true>`) gets every row's sum of that output, gathered from the
// four lanes of fragment row y; all lanes gather, so the shuffles are uniform.
template <bool COLUMNS>
struct gemv_batch_emit;
template <>
struct gemv_batch_emit<false> {
    template <bool PAIRED, uint R, uint NB, typename Out>
    static void run(thread const Out &out, thread simdgroup_float8x8 (&acc)[R][NB],
        thread simdgroup_float8x8 (&acc2)[R][NB], uint m_rows, uint rows, gemv_batch_rows map, ushort2 at) {
        for (uint r = 0; r < R; ++r) {
            uint n = map.at(r, at.y);
            if (n >= rows)
                continue;
            for (uint bb = 0; bb < NB; ++bb) {
                float2 c = reinterpret_cast<thread float2 &>(acc[r][bb].thread_elements());
                float2 c2 = reinterpret_cast<thread float2 &>(acc2[r][bb].thread_elements());
                uint m = 8u * bb + at.x;
                if (m < m_rows)
                    emit<PAIRED>::run(out, m, n, c.x, c2.x);
                if (m + 1u < m_rows)
                    emit<PAIRED>::run(out, m + 1u, n, c.y, c2.y);
            }
        }
    }
};
template <>
struct gemv_batch_emit<true> {
    template <bool PAIRED, uint R, uint NB, typename Out>
    static void run(thread const Out &out, thread simdgroup_float8x8 (&acc)[R][NB],
        thread simdgroup_float8x8 (&)[R][NB], uint m_rows, uint rows, gemv_batch_rows map, ushort2 at) {
        static_assert(!PAIRED, "a column epilogue is plain");
        for (uint r = 0; r < R; ++r) {
            float column[8 * NB];
            for (uint bb = 0; bb < NB; ++bb) {
                float2 c = reinterpret_cast<thread float2 &>(acc[r][bb].thread_elements());
                for (ushort x = 0; x < 8; x += 2) {
                    float2 pair = simd_shuffle(c, fragment_lane(x, at.y));
                    column[8u * bb + x] = pair.x;
                    column[8u * bb + x + 1u] = pair.y;
                }
            }
            uint n = map.at(r, at.y);
            if (n >= rows)
                continue;
            for (uint bb = 0; bb < NB; ++bb) {
                uint m = 8u * bb + at.x;
                if (m < m_rows)
                    out.store_column(m, n, column);
                if (m + 1u < m_rows)
                    out.store_column(m + 1u, n, column);
            }
        }
    }
};

template <typename W, typename U, bool PAIRED, uint R, uint NB, uint BYTES, bool COLUMNS = false, typename In,
    typename Out>
inline void gemv_batch_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    constexpr uint chunk = gemv_batch_chunk<NB, BYTES>();
    constexpr uint pitch = chunk + 4u;
    threadgroup float *staged = reinterpret_cast<threadgroup float *>(shared);
    ushort2 at = fragment_coordinate(lane);
    uint pair_index = at.x / 2u;
    gemv_batch_rows map{tile * simdgroups * R * 8u, sg * R,
        w.tile() == 32u && (simdgroups * R) % 4u == 0u};
    uint thread_index = sg * 32u + lane;
    // A simdgroup none of whose blocks has a weight row is idle as a whole.
    bool active = false;
    typename Weights<W>::located a_at[R];
    typename Weights<U>::located b_at[R];
    for (uint r = 0; r < R; ++r) {
        active = active || map.at(r, 0u) < rows;
        uint n = min(map.at(r, at.y), rows - 1);
        a_at[r] = w.locate(n);
        if (PAIRED)
            b_at[r] = u.locate(n);
    }
    simdgroup_float8x8 acc[R][NB], acc2[R][NB];
    for (uint r = 0; r < R; ++r) {
        for (uint b = 0; b < NB; ++b) {
            acc[r][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc2[r][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    typename W::packet a[R], a_next[R];
    typename U::packet b[R], b_next[R];
    typename packets::Block<W>::state a_run[R];
    typename packets::Block<U>::state b_run[R];
    for (uint c0 = 0; c0 < k; c0 += chunk) {
        uint packets = (min(chunk, k - c0) + 31u) / 32u;
        // The chunk's first packets load while the threadgroup stages it; a
        // threadgroup may run several batched GEMVs in turn, and the previous
        // chunk or call finishes reading the staging first.
        if (active) {
            for (uint r = 0; r < R; ++r) {
                a[r] = gemv_batch_load(w, a_at[r], c0 / 32u, a_run[r]);
                if (PAIRED)
                    b[r] = gemv_batch_load(u, b_at[r], c0 / 32u, b_run[r]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = thread_index; item < 8u * NB * 4u * packets; item += simdgroups * 32u) {
            uint m = item / (4u * packets), g = item % (4u * packets);
            float4 even = float4(0.0f), odd = float4(0.0f);
            if (m < m_rows)
                in.load8(m, c0 + 8u * g, 0.0f, even, odd);
            threadgroup float4 *dst = reinterpret_cast<threadgroup float4 *>(staged + m * pitch + 8u * g);
            dst[0] = float4(even.x, odd.x, even.y, odd.y);
            dst[1] = float4(even.z, odd.z, even.w, odd.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active)
            continue;
        for (uint local = 0; local < packets; ++local) {
            if (local + 1u < packets) {
                for (uint r = 0; r < R; ++r) {
                    a_next[r] = gemv_batch_load(w, a_at[r], c0 / 32u + local + 1u, a_run[r]);
                    if (PAIRED)
                        b_next[r] = gemv_batch_load(u, b_at[r], c0 / 32u + local + 1u, b_run[r]);
                }
            }
            simdgroup_float8x8 x[4][NB];
            for (uint step = 0; step < 4; ++step)
                for (uint bb = 0; bb < NB; ++bb)
                    simdgroup_load(x[step][bb], staged + 8u * bb * pitch + 32u * local + 8u * step, pitch,
                        ulong2(0, 0), true);
            for (uint r = 0; r < R; ++r) {
                gemv_batch_packet<W, NB>(acc[r], a[r], x, pair_index);
                if (PAIRED)
                    gemv_batch_packet<U, NB>(acc2[r], b[r], x, pair_index);
            }
            for (uint r = 0; r < R; ++r) {
                a[r] = a_next[r];
                if (PAIRED)
                    b[r] = b_next[r];
            }
        }
    }
    gemv_batch_emit<COLUMNS>::template run<PAIRED, R, NB>(out, acc, acc2, m_rows, rows, map, at);
}

// BYTES: the threadgroup staging `shared` provides (gemv_batch_shared_bytes
// for a batched GEMV launch).
template <typename W, uint SG, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint tile, threadgroup uchar *shared, uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, R, 1, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared, SG, sg, lane);
    else
        gemv_batch_body<W, W, false, R, 2, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename W, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, R, 1, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
    else
        gemv_batch_body<W, W, false, R, 2, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
}

// `gemv_batch_runtime` with a column epilogue (`gemv_emit<true>`).
template <typename W, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch_columns_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, R, 1, BYTES, true>(in, out, w, w, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
    else
        gemv_batch_body<W, W, false, R, 2, BYTES, true>(in, out, w, w, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
}

template <typename G, typename U, uint SG, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<G, U, true, R, 1, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared, SG, sg, lane);
    else
        gemv_batch_body<G, U, true, R, 2, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared, SG, sg, lane);
}

template <typename G, typename U, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch_paired_runtime(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<G, U, true, R, 1, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
    else
        gemv_batch_body<G, U, true, R, 2, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared,
            simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// Matrix GEMV: M <= 16 activation rows with the weights as the left fragment
// of the matrix unit, read as the PACK form reads them (`packing.h`): a lane
// at fragment coordinate (x, y) holds weight row y and matrix columns x,
// x + 1, one word of its row per 32-column block; step s of a block
// multiplies the block's columns 8 (c / 2) + s + 4 (c % 2) for c = 0 .. 7.
// The activations are the right fragment (row c, columns the eight
// activation rows of a fragment), loaded as stored: a lane's four steps of a
// block are four consecutive values of each of its two rows.
//
// The weights stay in the unit as integers: a lane's left element of a step
// is its code times its row's 6-bit scale of the block, an integer below
// 2048 entered as the F16 with those bits (the value 2^-24 times it,
// exactly), so one accumulator runs over the 32 steps of a 256-column block
// and folds once with the block's d. The min term of the eight 32-column
// blocks is one more product, of the rows' 6-bit mins (row y, blocks x,
// x + 1) with the activation rows' sums over each block (block y, columns the
// activation rows), folded with dmin.
//
// PARTS simdgroups share a fragment's eight weight rows, a contiguous range
// of 256-column blocks each (a simdgroup alone leaves the device short of
// threads); their sums meet in threadgroup memory, so PARTS is arithmetic.
// A row's result depends only on its own activations: every row of an M-row
// call is bit-identical to the same row computed alone in this form.
// `shared` holds PARTS-independent 16 bytes per thread and activation
// fragment.

// Weights with a matrix GEMV: the k-quants whose low code bits are one plane
// of nibbles and whose code times a 6-bit scale stays below 2048. `at` is a
// lane's place in its row's planes (32-bit offsets from the tensor), `load`
// its words of block g, `pair` its two codes of step s in the halves of one
// word.
template <typename W>
struct matrix_codes {
    static constant constexpr bool available = false;
};
template <>
struct matrix_codes<packets::Q4K> {
    static constant constexpr bool available = true;
    typedef uint place;
    typedef uint words;
    static place at(thread const Weights<packets::Q4K> &w,
        thread const typename Weights<packets::Q4K>::located &row, uint x) {
        return uint(row.geometry.at(row.row, w.layout.codes, 16ul, 0ul) - w.base) + (x / 2u) * 4u;
    }
    // An indexed load from the lane's place: the load takes a base and a
    // 32-bit index, where the address of each word would be a 64-bit sum.
    static words load(thread const Weights<packets::Q4K> &w, place at, uint g) {
        return reinterpret_cast<device const uint *>(w.base + at)[g * 4u * w.layout.tile];
    }
    static uint pair(words codes, uint s) { return (codes >> (4u * s)) & 0x000f000fu; }
};
// q5k: the lane's byte of the block's high bits beside its word of nibbles,
// a nibble of the byte under each half of the word.
template <>
struct matrix_codes<packets::Q5K> {
    static constant constexpr bool available = true;
    typedef uint2 place;
    typedef uint2 words;
    static place at(thread const Weights<packets::Q5K> &w,
        thread const typename Weights<packets::Q5K>::located &row, uint x) {
        return uint2(uint(row.geometry.at(row.row, w.layout.codes, 16ul, 0ul) - w.base) + (x / 2u) * 4u,
            uint(row.geometry.at(row.row, w.layout.high, 4ul, 0ul) - w.base) + x / 2u);
    }
    static words load(thread const Weights<packets::Q5K> &w, place at, uint g) {
        const uint high = w.base[at.y + g * 4u * w.layout.tile];
        return uint2(reinterpret_cast<device const uint *>(w.base + at.x)[g * 4u * w.layout.tile],
            (high | (high << 12)) & 0x000f000fu);
    }
    static uint pair(words codes, uint s) {
        return ((codes.x >> (4u * s)) & 0x000f000fu) | (((codes.y >> s) & 0x00010001u) << 4);
    }
};

// q6k has a body of its own (`gemv_matrix_q6k_body`).
template <>
struct matrix_codes<packets::Q6K> {
    static constant constexpr bool available = true;
};

// Bits 12 b .. 12 b + 11 of a k-quant block's 96 bits of (scale6, min6)
// pairs.
inline uint matrix_field(uint3 fields, uint b) {
    const uint bit = 12u * b, word = bit >> 5, shift = bit & 31u;
    const ulong two = ulong(fields[word]) | (word < 2u ? ulong(fields[word + 1u]) << 32 : 0ul);
    return uint(two >> shift) & 0xfffu;
}

// Whether an operand's rows are read with whole-word loads. The matrix GEMV
// bodies take the answer as a template argument (UNIT): the test is made
// once per call, where a body that makes it at every load of its block loop
// runs twice as long on the M1 (2565 against 329 instructions a block).
template <typename In>
inline bool matrix_unit(thread const In &in) {
    return in.stride1 == 1 && (in.stride0 & 7u) == 0;
}

// Activation row m's values of columns k .. k + 7 as (0, 1, 2, 3) and
// (4, 5, 6, 7).
template <bool UNIT, typename In>
inline void matrix_values(thread const In &in, uint m, uint k, thread float4 &first, thread float4 &second) {
    typedef typename In::activation A;
    uint4 words;
    if constexpr (UNIT)
        words = *reinterpret_cast<device const uint4 *>(
            reinterpret_cast<device const typename A::storage *>(in.x) + ulong(in.rows.at(m)) * in.stride0 + k);
    else
        words = in.words8(m, k);
    float4 even, odd;
    A::split8(words, even, odd);
    first = float4(even.x, odd.x, even.y, odd.y);
    second = float4(even.z, odd.z, even.w, odd.w);
}

// Activation row m's sum over 32-column block g.
template <bool UNIT, typename In>
inline float matrix_block_sum(thread const In &in, uint m, uint g) {
    float4 sum = float4(0.0f);
    PROJECTION_UNROLL
    for (uint q = 0; q < 4; ++q) {
        float4 first, second;
        matrix_values<UNIT>(in, m, 32u * g + 8u * q, first, second);
        sum += first + second;
    }
    return (sum.x + sum.y) + (sum.z + sum.w);
}
// The same with the strides tested at every load.
template <typename In>
inline float matrix_block_sum(thread const In &in, uint m, uint g) {
    return matrix_block_sum<false>(in, m, g);
}

// The same sums of two activation rows, eight columns of each at a time: a
// lane that sums inside its block loop holds one load of each row, not the
// block's (on the M1 a kernel's registers per thread bound the threads in
// flight).
template <bool UNIT, typename In>
inline float2 matrix_block_sums(thread const In &in, uint m0, uint m1, uint g) {
    float4 sum0 = float4(0.0f), sum1 = float4(0.0f);
    _Pragma("clang loop unroll(disable)")
    for (uint q = 0; q < 4; ++q) {
        float4 first, second;
        matrix_values<UNIT>(in, m0, 32u * g + 8u * q, first, second);
        sum0 += first + second;
        matrix_values<UNIT>(in, m1, 32u * g + 8u * q, first, second);
        sum1 += first + second;
    }
    return float2((sum0.x + sum0.y) + (sum0.z + sum0.w), (sum1.x + sum1.y) + (sum1.z + sum1.w));
}

// Threadgroup memory of a matrix GEMV that stages the block sums: the
// exchange (16 bytes per thread and activation fragment) and 4 bytes per
// activation row and block. A launch declares
//   shared_bytes (min((SIMDGROUPS * 256 + ceil_div(K, 32) * 32) * ceil_div(M, 8), 28672))
// at least; past `gemv_matrix_staged_bytes` the lanes sum their own.
constant constexpr uint gemv_matrix_staged_bytes = 28672;
inline uint gemv_matrix_shared_bytes(uint k, uint fragments, uint simdgroups) {
    return (simdgroups * 256u + (k / 32u) * 32u) * fragments;
}

// Value s of four stored activation values held in two words.
template <typename A>
inline float matrix_value(uint2 stored, uint s) {
    return A::load(as_type<typename A::storage>(ushort(stored[s >> 1] >> (16u * (s & 1u)))));
}

// The four stored activation values from column `block + column` of the row
// at `row` (`block` a multiple of 32, `column` a multiple of 4 below 32, the
// same at every load of a lane): one 8-byte load on unit strides (loading the
// column group's sixteen bytes and choosing a half costs a sixth of the
// kernel), indexed from the lane's place in the row, so no load needs a
// 64-bit sum of its own.
template <bool UNIT, typename In>
inline uint2 matrix_stored(thread const In &in, device const typename In::activation::storage *row, uint block,
    uint column) {
    if constexpr (UNIT)
        return reinterpret_cast<device const uint2 *>(row + column)[block / 4u];
    const uint4 eight = words8_storage<typename In::activation>(row, in.stride1, block + (column & ~7u), in.columns,
        false);
    return (column & 4u) ? eight.zw : eight.xy;
}

// The exchange of a matrix GEMV's PARTS simdgroups of one weight fragment:
// part 0 returns with the sum of all parts, the others with false.
template <uint NB, uint PARTS>
inline bool matrix_exchange(thread float2 (&y)[NB], threadgroup float2 *parts, uint part, uint sg, uint lane) {
    if constexpr (PARTS > 1) {
        // A threadgroup may run several of these in turn: the previous
        // one's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PROJECTION_UNROLL
        for (uint f = 0; f < NB; ++f)
            parts[(sg * 32u + lane) * NB + f] = y[f];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part != 0u)
            return false;
        for (uint p = 1; p < PARTS; ++p) {
            PROJECTION_UNROLL
            for (uint f = 0; f < NB; ++f)
                y[f] += parts[((sg + p) * 32u + lane) * NB + f];
        }
    }
    return true;
}

// The 32-column groups of a 256-column block that a lane multiplies with no
// loop between them, so that it holds their loads together: all eight for
// one activation fragment, four for two. On the M1 the registers a kernel
// holds per thread bound its threads in flight, and a kernel's count is its
// widest body's: with the eight groups of two fragments held (116 registers
// against 83), the one-fragment body of the same kernel is a quarter slower.
template <uint NB>
constexpr uint matrix_run() {
    return 8u / NB;
}

// The matrix GEMV over Q6_K weights: a lane's two matrix columns of a step
// lie in one 16-column run of its row, so its left elements are
// (code - 32) times the run's int8 scale as F32 integers; the accumulator
// runs over a 256-column block and folds once with the block's d. No min
// term and nothing staged. UNIT is `matrix_unit` of the operand.
template <uint NB, uint PARTS, bool UNIT, typename In, typename Out>
inline void gemv_matrix_q6k_body(thread const In &in, thread const Out &out,
    thread const Weights<packets::Q6K> &w, uint m_rows, uint rows, uint k, uint group, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    typedef typename In::activation A;
    constexpr uint RUN = matrix_run<NB>();
    threadgroup float2 *parts = reinterpret_cast<threadgroup float2 *>(shared);
    const ushort2 at = fragment_coordinate(lane);
    const uint index = group * simdgroups + sg, piece = index / PARTS, part = index % PARTS;
    const uint n = piece * 8u + at.y;
    const typename Weights<packets::Q6K>::located row = w.locate(min(n, rows - 1u));
    const uint codes = uint(row.geometry.at(row.row, w.layout.codes, 16ul, 0ul) - w.base) + (at.x / 2u) * 4u;
    const uint high = uint(row.geometry.at(row.row, w.layout.high, 8ul, 0ul) - w.base) + (at.x / 2u) * 2u;
    const uint code_step = 16u * w.layout.tile, high_step = 8u * w.layout.tile;
    const uint blocks = k / 256u;
    device const typename A::storage *stored_rows[NB][2];
    float2 y[NB];
    simdgroup_float8x8 acc[NB];
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        y[f] = float2(0.0f);
        PROJECTION_UNROLL
        for (uint j = 0; j < 2; ++j)
            stored_rows[f][j] = reinterpret_cast<device const typename A::storage *>(in.x)
                + ulong(in.rows.at(min(8u * f + at.x + j, m_rows - 1u))) * in.stride0;
    }
    for (uint block = part * blocks / PARTS; block < (part + 1u) * blocks / PARTS; ++block) {
        // The row's 16 run scales of the block: the lane's run of packet b
        // is byte 2 b + x / 4.
        const uint4 scales = uint4(*reinterpret_cast<device const packed_uint4 *>(
            row.geometry.scale_at(row.row, 16ul * block)));
        const float d = float(*reinterpret_cast<device const half *>(row.geometry.super_at(row.row, 2ul * block)));
        _Pragma("clang loop unroll(disable)")
        for (uint run = 0; run < 8; run += RUN) {
            PROJECTION_UNROLL
            for (uint i = 0; i < RUN; ++i) {
                const uint b = run + i;
                const uint g = block * 8u + b;
                const int scale = int(char((scales[b / 2u] >> (16u * (b % 2u) + 8u * (at.x / 4u))) & 0xffu));
                const uint word = *reinterpret_cast<device const uint *>(w.base + codes + g * code_step);
                const uint field = uint(*reinterpret_cast<device const ushort *>(w.base + high + g * high_step));
                const uint spread = (field | (field << 8)) & 0x00ff00ffu;
                uint2 stored[NB][2];
                PROJECTION_UNROLL
                for (uint f = 0; f < NB; ++f) {
                    PROJECTION_UNROLL
                    for (uint j = 0; j < 2; ++j)
                        stored[f][j] = matrix_stored<UNIT>(in, stored_rows[f][j], 32u * g,
                            8u * (at.y / 2u) + 4u * (at.y & 1u));
                }
                PROJECTION_UNROLL
                for (uint s = 0; s < 4; ++s) {
                    const uint pair = ((word >> (4u * s)) & 0x000f000fu)
                        | (((spread >> (2u * s)) & 0x00030003u) << 4);
                    simdgroup_float8x8 left;
                    reinterpret_cast<thread float2 &>(left.thread_elements()) =
                        float2((int2(int(pair & 0xffffu), int(pair >> 16)) - 32) * scale);
                    PROJECTION_UNROLL
                    for (uint f = 0; f < NB; ++f) {
                        simdgroup_float8x8 right;
                        reinterpret_cast<thread float2 &>(right.thread_elements()) = float2(
                            matrix_value<A>(stored[f][0], s), matrix_value<A>(stored[f][1], s));
                        if (b == 0 && s == 0)
                            simdgroup_multiply(acc[f], left, right);
                        else
                            simdgroup_multiply_accumulate(acc[f], left, right, acc[f]);
                    }
                }
            }
        }
        PROJECTION_UNROLL
        for (uint f = 0; f < NB; ++f)
            y[f] = metal::fma(float2(d), reinterpret_cast<thread float2 &>(acc[f].thread_elements()), y[f]);
    }
    if (!matrix_exchange<NB, PARTS>(y, parts, part, sg, lane) || n >= rows)
        return;
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        if (8u * f + at.x < m_rows)
            out.store(8u * f + at.x, n, y[f].x);
        if (8u * f + at.x + 1u < m_rows)
            out.store(8u * f + at.x + 1u, n, y[f].y);
    }
}

// NB activation fragments (8 rows each) against the eight weight rows of
// fragment `(group * simdgroups + sg) / PARTS`. UNIT is `matrix_unit` of the
// operand and STAGED whether the block sums fit the threadgroup memory
// (`gemv_matrix_shared_bytes`): both are the same at every block of a call,
// and a block loop that tests them compiles both sides of each test around
// every load.
template <typename W, uint NB, uint PARTS, bool UNIT, bool STAGED, typename In, typename Out>
inline void gemv_matrix_body(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    typedef matrix_codes<W> coded;
    typedef typename In::activation A;
    constexpr uint RUN = matrix_run<NB>();
    threadgroup float2 *parts = reinterpret_cast<threadgroup float2 *>(shared);
    const ushort2 at = fragment_coordinate(lane);
    const uint index = group * simdgroups + sg, piece = index / PARTS, part = index % PARTS;
    const uint n = piece * 8u + at.y;
    const typename Weights<W>::located row = w.locate(min(n, rows - 1u));
    const typename coded::place codes = coded::at(w, row, at.x);
    const uint blocks = k / 256u;
    // The lane's two activation rows of each fragment (a row at or past
    // m_rows repeats the last: its results are not stored).
    uint m[NB][2];
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        m[f][0] = min(8u * f + at.x, m_rows - 1u);
        m[f][1] = min(8u * f + at.x + 1u, m_rows - 1u);
    }
    // Their stored rows, resolved once (`Plain`'s `words8` per load repeats
    // the row's 64-bit address arithmetic at every block).
    device const typename A::storage *stored_rows[NB][2];
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        PROJECTION_UNROLL
        for (uint j = 0; j < 2; ++j)
            stored_rows[f][j] = reinterpret_cast<device const typename A::storage *>(in.x)
                + ulong(in.rows.at(m[f][j])) * in.stride0;
    }
    float2 y[NB];
    simdgroup_float8x8 acc[NB], mins[NB];
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f)
        y[f] = float2(0.0f);
    // The activation rows' block sums, staged once per threadgroup behind
    // the exchange where they fit (block_sums[(8 NB) g + m]); a lane sums its
    // own otherwise. Either way a sum is `matrix_block_sum`.
    threadgroup float *block_sums = reinterpret_cast<threadgroup float *>(parts + simdgroups * 32u * NB);
    if constexpr (STAGED) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = sg * 32u + lane; item < 8u * NB * (k / 32u); item += simdgroups * 32u) {
            const uint row_m = item % (8u * NB);
            block_sums[item] = row_m < m_rows ? matrix_block_sum<UNIT>(in, row_m, item / (8u * NB)) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint block = part * blocks / PARTS; block < (part + 1u) * blocks / PARTS; ++block) {
        const uint3 fields = uint3(*reinterpret_cast<device const packed_uint3 *>(
            row.geometry.scale_at(row.row, 12ul * block)));
        const float2 factors = float2(*reinterpret_cast<device const half2 *>(
            row.geometry.super_at(row.row, 4ul * block)));
        _Pragma("clang loop unroll(disable)")
        for (uint run = 0; run < 8; run += RUN) {
            PROJECTION_UNROLL
            for (uint i = 0; i < RUN; ++i) {
                const uint b = run + i;
                const uint g = block * 8u + b;
                const uint scale = matrix_field(fields, b) & 63u;
                const typename coded::words words = coded::load(w, codes, g);
                // The lane's four values of each of its activation rows, as
                // stored: two words of the eight from its column group.
                uint2 stored[NB][2];
                PROJECTION_UNROLL
                for (uint f = 0; f < NB; ++f) {
                    PROJECTION_UNROLL
                    for (uint j = 0; j < 2; ++j) {
                        stored[f][j] = matrix_stored<UNIT>(in, stored_rows[f][j], 32u * g,
                            8u * (at.y / 2u) + 4u * (at.y & 1u));
                    }
                }
                PROJECTION_UNROLL
                for (uint s = 0; s < 4; ++s) {
                    simdgroup_half8x8 left;
                    reinterpret_cast<thread half2 &>(left.thread_elements()) =
                        as_type<half2>(coded::pair(words, s) * scale);
                    PROJECTION_UNROLL
                    for (uint f = 0; f < NB; ++f) {
                        simdgroup_float8x8 right;
                        reinterpret_cast<thread float2 &>(right.thread_elements()) = float2(
                            matrix_value<A>(stored[f][0], s), matrix_value<A>(stored[f][1], s));
                        if (b == 0 && s == 0)
                            simdgroup_multiply(acc[f], left, right);
                        else
                            simdgroup_multiply_accumulate(acc[f], left, right, acc[f]);
                    }
                }
            }
        }
        simdgroup_half8x8 left;
        reinterpret_cast<thread half2 &>(left.thread_elements()) = as_type<half2>(
            ((matrix_field(fields, at.x) >> 6) & 63u) | (((matrix_field(fields, at.x + 1u) >> 6) & 63u) << 16));
        PROJECTION_UNROLL
        for (uint f = 0; f < NB; ++f) {
            // The lane's two activation rows' sums over block 8 block + y.
            float2 sums;
            if constexpr (STAGED)
                sums = *reinterpret_cast<threadgroup const float2 *>(
                    block_sums + (block * 8u + at.y) * 8u * NB + 8u * f + at.x);
            else
                sums = matrix_block_sums<UNIT>(in, m[f][0], m[f][1], block * 8u + at.y);
            simdgroup_float8x8 right;
            reinterpret_cast<thread float2 &>(right.thread_elements()) = sums;
            simdgroup_multiply(mins[f], left, right);
            // The accumulators carry the 2^-24 of the F16 integers.
            y[f] = metal::fma(float2(factors.x * 0x1p24f), reinterpret_cast<thread float2 &>(acc[f].thread_elements()),
                metal::fma(float2(-factors.y * 0x1p24f), reinterpret_cast<thread float2 &>(mins[f].thread_elements()),
                    y[f]));
        }
    }
    if constexpr (PARTS > 1) {
        // A threadgroup may run several of these in turn: the previous
        // one's reads of the threadgroup memory finish before this one
        // writes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        PROJECTION_UNROLL
        for (uint f = 0; f < NB; ++f)
            parts[(sg * 32u + lane) * NB + f] = y[f];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part != 0u)
            return;
        for (uint p = 1; p < PARTS; ++p) {
            PROJECTION_UNROLL
            for (uint f = 0; f < NB; ++f)
                y[f] += parts[((sg + p) * 32u + lane) * NB + f];
        }
    }
    if (n >= rows)
        return;
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        if (8u * f + at.x < m_rows)
            out.store(8u * f + at.x, n, y[f].x);
        if (8u * f + at.x + 1u < m_rows)
            out.store(8u * f + at.x + 1u, n, y[f].y);
    }
}

// The body a weight format's matrix GEMV runs, chosen once per call by the
// operand's strides and by whether the block sums are staged.
template <typename W>
struct matrix_form {
    template <uint NB, uint PARTS, typename In, typename Out>
    static void run(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows, uint rows,
        uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
        const bool staged = gemv_matrix_shared_bytes(k, NB, simdgroups) <= gemv_matrix_staged_bytes;
        if (matrix_unit(in)) {
            if (staged)
                gemv_matrix_body<W, NB, PARTS, true, true>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                    sg, lane);
            else
                gemv_matrix_body<W, NB, PARTS, true, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                    sg, lane);
        } else if (staged) {
            gemv_matrix_body<W, NB, PARTS, false, true>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg,
                lane);
        } else {
            gemv_matrix_body<W, NB, PARTS, false, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                sg, lane);
        }
    }
};
template <>
struct matrix_form<packets::Q6K> {
    template <uint NB, uint PARTS, typename In, typename Out>
    static void run(thread const In &in, thread const Out &out, thread const Weights<packets::Q6K> &w, uint m_rows,
        uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
        if (matrix_unit(in))
            gemv_matrix_q6k_body<NB, PARTS, true>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
        else
            gemv_matrix_q6k_body<NB, PARTS, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg,
                lane);
    }
};

// Threadgroups a matrix GEMV over `rows` weight rows needs.
inline uint gemv_matrix_groups(uint rows, uint parts, uint simdgroups) {
    return ((rows + 7u) / 8u * parts + simdgroups - 1u) / simdgroups;
}

template <typename W, uint PARTS, typename In, typename Out>
inline void gemv_matrix(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    if (m_rows <= 8)
        matrix_form<W>::template run<1, PARTS>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
    else
        matrix_form<W>::template run<2, PARTS>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
}

// ---------------------------------------------------------------------------
// INT8 GEMM (an entry's INT8 form, M > 64, on tensor operations). The
// activations are quantized to int8 per (row, 32 columns), the weights enter
// as their stored 4-bit codes, and each 32-column block is one int8
// `matmul2d` on register fragments whose int32 result folds into the F32
// output under (activation scale) x (weight scale); the weights' min term is
// one F16 product of the activation block sums with the block biases. The
// weights stay exact; the form's error is the activation quantization, so it
// is an error class of its entry.
//
// Two pre-passes write the form's scratch in the order the lanes read it
// (per-lane device loads decide the rate: scattered scale loads cost as much
// as the products): `int8_quantize` the operand, its scales and block sums;
// `int8_coefficients` the weights' block scales and biases. A lane reads its
// four weight rows' codes of a block from the resident rows, one load a row.
// What a lane loads per block bounds the rate (the products alone are about
// two thirds of the launch), so the scratch is as few loads as its values
// allow: operand scales as F16, a 4-bit format's block scales as bytes.
//
// A 16 x 32 x K fragment's lane holds rows r, r + 8 and columns c .. c + 3,
// c + 16 .. c + 19 of the destination (element e: column c + (e & 3) +
// 16 (e >> 3), row r + 8 ((e >> 2) & 1)); the left operand alike over
// (row, k), and the right operand k = c + (e & 3) + 16 (e >> 4) of weight
// rows r + 8 ((e >> 2) & 3).
struct int8_lane {
    uint r, c;
    int8_lane(uint lane)
        : r(((lane >> 1) & 3u) + ((lane >> 4) & 1u) * 4u), c((lane & 1u) * 4u + ((lane >> 3) & 1u) * 8u) {}
    // The lane holding row class r (0 .. 7) and column class c (0, 4, 8, 12).
    static uint of(uint r, uint c) {
        return ((c >> 2) & 1u) | ((r & 3u) << 1) | (((c >> 3) & 1u) << 3) | ((r >> 2) << 4);
    }
};

// The form's scratch, for `groups` = K / 32 blocks:
// The weights are the left operand (a lane's patch is then two rows of a
// 16-row fragment, each one aligned load of one row's codes; see
// inference/docs/kernels/lane-order.md) and the operand rows the right.
//   quantized     32 int8 per (32-row tile, block, lane): the right fragment,
//                 element e the row r + 8 ((e >> 2) & 3) and the lane's
//                 column (e & 3) + 4 (e >> 4) of its eight (in the weight
//                 format's order, `int8_codes::interleaved`)
//   row_scales    1 F16 per (32-row tile, block, row), a lane's eight operand
//                 rows adjacent (`int8_scale_at`)
//   block_sums    32 F16 per (32-row tile, 32 blocks, lane): the quantized
//                 rows' block sums / 64, the min-term product's right
//                 fragment (`int8_sum_at`)
//   coefficients  the block scales of weight rows r, r + 8, r + 16, r + 24.
//                 `int8_codes::factored`, per 32-row weight tile: 4 F32 per
//                 (256 columns, r), the rows' factors, then 16 bytes per
//                 (4 blocks, r), a block's four bytes the rows' integer
//                 scales. Otherwise 4 F32 per (32-row weight tile, block, r)
//   biases        16 F16 per (16-row weight tile, 32 blocks, lane):
//                 -(dmin * min6), the min-term product's left fragment
//                 (`int8_bias_at`)
struct int8_scratch {
    device uchar *quantized;
    device half *row_scales;
    device half *block_sums;
    device float *coefficients;
    device half *biases;
};

// Weights whose stored codes enter the int8 product (on tensor operations):
// how a lane reads its codes of one row and block, whether the format has a
// min term, and a row's block scales and biases of the eight blocks of one
// 256-column block: block i's scale is `factor * scale[i]`. `factored`: the
// scales are integers below 256 and the factor is the 256 columns'.
template <typename W>
struct int8_codes {
    static constant constexpr bool available = false;
    static constant constexpr bool biased = false;
    static constant constexpr bool factored = false;
    static constant constexpr bool interleaved = false;
    static void coefficients(device const uchar *, Rows16, uint, thread float &, thread float (&)[8],
        thread float (&)[8]) {}
};
#if SEISMIC_HAS_TENSOR_OPS
// q4k: two bytes hold four codes, a nibble each; value = d * scale6 * code
// - dmin * min6.
template <>
struct int8_codes<packets::Q4K> {
    static constant constexpr bool available = true;
    static constant constexpr bool biased = true;
    static constant constexpr uint block_bytes = 16;
    // A lane's eight codes of one row and block are one 4-byte word: byte x
    // holds columns 2x (low nibble) and 2x + 1 of the lane's eight. The
    // fragment's first four positions take the even columns, the last four
    // the odd ones (`interleaved`; the operand is quantized to match).
    static constant constexpr bool interleaved = true;
    static constant constexpr uint slot_bytes = 4;
    // Bytes of a block and of a lane's slot in the plane of high code bits
    // (none here).
    static constant constexpr uint high_block_bytes = 0, high_slot_bytes = 0;
    static void slot(device const uchar *at, device const uchar *, thread char4 &first, thread char4 &second) {
        const uint word = *reinterpret_cast<device const uint *>(at);
        first = as_type<char4>(word & 0x0f0f0f0fu);
        second = as_type<char4>((word >> 4) & 0x0f0f0f0fu);
    }
    // A block's scale is d * scale6, exact in F32: the factor is d and the
    // scales are the 6-bit integers, a byte each in the scratch.
    static constant constexpr bool factored = true;
    static void coefficients(device const uchar *row, Rows16 layout, uint block, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        packets::KBlock run = packets::KBlock::load(row, layout, block * 8u);
        factor = run.factors.x;
        // A block's (scale6, min6); its bias -(dmin * min6) is exact in F32.
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            const uint pair = run.fields.x;
            packets::KBlock::advance(run);
            scale[i] = float(pair & 63u);
            bias[i] = run.factors.y * float((pair >> 6) & 63u);
        }
    }
};
// q8: int8 codes, one F16 scale per block.
template <>
struct int8_codes<packets::Q8> {
    static constant constexpr bool available = true;
    static constant constexpr bool biased = false;
    static constant constexpr uint block_bytes = 32;
    // A lane's eight codes of one row and block are eight bytes, in column
    // order.
    static constant constexpr bool interleaved = false;
    static constant constexpr uint slot_bytes = 8;
    static constant constexpr uint high_block_bytes = 0, high_slot_bytes = 0;
    static void slot(device const uchar *at, device const uchar *, thread char4 &first, thread char4 &second) {
        const uint2 words = *reinterpret_cast<device const uint2 *>(at);
        first = as_type<char4>(words.x);
        second = as_type<char4>(words.y);
    }
    static constant constexpr bool factored = false;
    static void coefficients(device const uchar *row, Rows16 layout, uint block, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        factor = 1.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            scale[i] = float(*reinterpret_cast<device const half *>(
                layout.super_at(row, 2ul * (block * 8u + i))));
            bias[i] = 0.0f;
        }
    }
};
#endif

// The min-term product takes 32 blocks a run (the widest the operation
// forms; a run's cost is nearly independent of its width): its operands are
// laid out in chunks of 32 blocks, the last padded with zeros.
inline uint int8_min_chunks(uint groups) { return (groups + 31u) / 32u; }
// Weight row n's bias of block g in the left fragment (16 F16 a lane:
// rows r, r + 8 of a 16-row tile by the lane's eight blocks of the chunk).
inline ulong int8_bias_at(uint n, uint g, uint groups) {
    const uint p = g & 31u;
    return ((ulong(n / 16u) * int8_min_chunks(groups) + g / 32u) * 32u + int8_lane::of(n & 7u, p & 12u)) * 16u
        + (p & 3u) + 4u * ((n >> 3) & 1u) + 8u * (p >> 4);
}
// Operand row m's block sum of block g in the right fragment (32 F16 a
// lane: rows r + 8 j of a 32-row tile by the lane's eight blocks).
inline ulong int8_sum_at(uint m, uint g, uint groups) {
    const uint p = g & 31u;
    return ((ulong(m / 32u) * int8_min_chunks(groups) + g / 32u) * 32u + int8_lane::of(m & 7u, p & 12u)) * 32u
        + (p & 3u) + 4u * ((m >> 3) & 3u) + 16u * (p >> 4);
}
// Operand row m's scale of block g: a lane's rows c .. c + 3 and c + 16 ..
// c + 19 of a 32-row tile are eight adjacent F16.
inline ulong int8_scale_at(uint m, uint g, uint groups) {
    return (ulong(m / 32u) * groups + g) * 32u + ((m >> 2) & 3u) * 8u + ((m >> 4) & 1u) * 4u + (m & 3u);
}

// The activation pass: one threadgroup of 256 threads per row; a thread takes
// 8 columns at a time, four neighbouring lanes one 32-column block. The block
// is scaled by its largest magnitude / 127 (rounded to F16, the scale the
// fold uses), rounded to nearest and stored in fragment order. Rows at or
// past `m_rows` are zero.
template <bool INTERLEAVED, typename In>
inline void int8_quantize(thread const In &in, thread const int8_scratch &s, uint m_rows, uint columns, uint m,
    uint thread_index, uint lane, float inverse) {
    const uint groups = columns / 32u;
    const uint tile = m / 32u, rr = m & 7u, h = (m >> 3) & 3u;
    for (uint chunk = thread_index; chunk < columns / 8u; chunk += 256u) {
        float4 even = float4(0.0f), odd = float4(0.0f);
        if (m < m_rows)
            in.load8(m, chunk * 8u, inverse, even, odd);
        const float4 a = float4(even.x, odd.x, even.y, odd.y), b = float4(even.z, odd.z, even.w, odd.w);
        const uint g = chunk / 4u, first = (chunk % 4u) * 8u;
        const float4 peaks = metal::max(metal::abs(a), metal::abs(b));
        float peak = metal::max(metal::max(peaks.x, peaks.y), metal::max(peaks.z, peaks.w));
        peak = metal::max(peak, simd_shuffle_xor(peak, ushort(1)));
        peak = metal::max(peak, simd_shuffle_xor(peak, ushort(2)));
        const float scale = float(half(peak / 127.0f));
        const float reciprocal = scale > 0.0f ? 1.0f / scale : 0.0f;
        const int4 qa = metal::clamp(int4(metal::rint(a * reciprocal)), -127, 127);
        const int4 qb = metal::clamp(int4(metal::rint(b * reciprocal)), -127, 127);
        // The chunk's eight columns are one lane's of the block (column
        // class `first / 2`); its fragment holds them as two fours.
        const char4 low = INTERLEAVED ? char4(int4(qa.x, qa.z, qb.x, qb.z)) : char4(qa);
        const char4 high = INTERLEAVED ? char4(int4(qa.y, qa.w, qb.y, qb.w)) : char4(qb);
        device char4 *to = reinterpret_cast<device char4 *>(s.quantized
            + ((ulong(tile) * groups + g) * 32u + int8_lane::of(rr, first / 2u)) * 32u);
        to[h] = low;
        to[4u + h] = high;
        float sum = float(qa.x + qa.y + qa.z + qa.w + qb.x + qb.y + qb.z + qb.w);
        sum += simd_shuffle_xor(sum, ushort(1));
        sum += simd_shuffle_xor(sum, ushort(2));
        if ((lane & 3u) == 0) {
            s.row_scales[int8_scale_at(m, g, groups)] = half(scale);
            s.block_sums[int8_sum_at(m, g, groups)] = half(scale * sum * 0x1p-6f);
        }
    }
    // The blocks that pad the min-term product's last 32 are zero.
    if (groups + thread_index < int8_min_chunks(groups) * 32u)
        s.block_sums[int8_sum_at(m, groups + thread_index, groups)] = 0.0h;
}

// A 32-row weight tile's factored coefficients, in 16-byte words: the rows'
// factors of 256-column block `block`, and their scale bytes of the four
// blocks from `g` (a multiple of four), for row class r.
inline ulong int8_factor_at(uint tile, uint block, uint r, uint groups) {
    return ulong(tile) * groups * 3u + block * 8u + r;
}
inline ulong int8_scale_bytes_at(uint tile, uint g, uint r, uint groups) {
    return ulong(tile) * groups * 3u + groups + g * 2u + r;
}

// The coefficient pass: one thread per (32-row weight tile, row class r,
// 256-column block), the weight rows r, r + 8, r + 16, r + 24 one lane's.
template <typename W>
inline void int8_coefficients(thread const Weights<W> &w, thread const int8_scratch &s, uint rows, uint columns,
    uint item) {
    const uint groups = columns / 32u, blocks = columns / 256u;
    const uint set = item / blocks, block = item % blocks;
    const uint tile = set / 8u, r = set & 7u;
    if (tile * 32u >= rows)
        return;
    float4 factor;
    float scale[4][8], bias[4][8];
    PROJECTION_UNROLL
    for (uint q = 0; q < 4; ++q) {
        const uint n = tile * 32u + r + 8u * q;
        float row_factor;
        int8_codes<W>::coefficients(w.row(n), w.geometry(n), block, row_factor, scale[q], bias[q]);
        factor[q] = row_factor;
    }
    if constexpr (int8_codes<W>::factored) {
        device uint4 *words = reinterpret_cast<device uint4 *>(s.coefficients);
        words[int8_factor_at(tile, block, r, groups)] = as_type<uint4>(factor);
        PROJECTION_UNROLL
        for (uint j = 0; j < 2; ++j) {
            uint4 bytes;
            PROJECTION_UNROLL
            for (uint x = 0; x < 4; ++x) {
                const uint i = 4u * j + x;
                bytes[x] = as_type<uint>(uchar4(float4(scale[0][i], scale[1][i], scale[2][i], scale[3][i])));
            }
            words[int8_scale_bytes_at(tile, block * 8u + 4u * j, r, groups)] = bytes;
        }
    } else {
        device float4 *scales = reinterpret_cast<device float4 *>(s.coefficients)
            + (ulong(tile) * groups + block * 8u) * 8u + r;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i)
            scales[i * 8u] = factor * float4(scale[0][i], scale[1][i], scale[2][i], scale[3][i]);
    }
    if constexpr (int8_codes<W>::biased) {
        // A 16-row tile's rows r, r + 8 by four blocks are a lane's eight
        // biases of half a fragment.
        PROJECTION_UNROLL
        for (uint pair = 0; pair < 2; ++pair) {
            const uint n = tile * 32u + 16u * pair + r;
            PROJECTION_UNROLL
            for (uint j = 0; j < 2; ++j) {
                const half4 near = half4(float4(bias[2u * pair][4u * j], bias[2u * pair][4u * j + 1u],
                    bias[2u * pair][4u * j + 2u], bias[2u * pair][4u * j + 3u]));
                const half4 far = half4(float4(bias[2u * pair + 1u][4u * j], bias[2u * pair + 1u][4u * j + 1u],
                    bias[2u * pair + 1u][4u * j + 2u], bias[2u * pair + 1u][4u * j + 3u]));
                *reinterpret_cast<device uint4 *>(s.biases + int8_bias_at(n, block * 8u + 4u * j, groups))
                    = uint4(as_type<uint2>(near), as_type<uint2>(far));
            }
            // The blocks that pad the min-term product's last 32 are zero.
            if (block == blocks - 1u) {
                for (uint g = groups; g < int8_min_chunks(groups) * 32u; g += 4u)
                    *reinterpret_cast<device uint4 *>(s.biases + int8_bias_at(n, g, groups)) = uint4(0u);
            }
        }
    }
}

#if SEISMIC_HAS_TENSOR_OPS

// One simdgroup's 32 x 32 tile at (m0, n0) with the weights as the left
// operand: two 16-row weight fragments against one 32-row operand fragment
// per block. `f[i][e]` receives the sum of weight row
// n0 + 16 i + r + 8 ((e >> 2) & 1) and operand row
// m0 + c + (e & 3) + 16 (e >> 3) (`int8_weight_row`, `int8_operand_row`). A
// lane reads each of its four weight rows' codes of a block in one load of
// that row's packet.
template <typename W>
inline void gemm_int8_products(thread const Weights<W> &w, thread const int8_scratch &s, uint k, uint m0, uint n0,
    uint lane, thread float (&f)[2][16]) {
    const uint groups = k / 32u;
    const int8_lane at(lane);
    const uint tile_m = m0 / 32u, tile_n = n0 / 32u;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> op;
    typedef decltype(op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>()) left_t;
    typedef decltype(op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>()) right_t;
    typedef decltype(op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<left_t>,
        metal::remove_addrspace_t<right_t>, int32_t>()) product_t;
    PROJECTION_UNROLL
    for (uint16_t e = 0; e < 16; ++e) {
        f[0][e] = 0.0f;
        f[1][e] = 0.0f;
    }
    // The min term: sum over blocks of (weight block bias) x (row block
    // sum), 32 blocks per F16 product.
    if constexpr (int8_codes<W>::biased) {
        constexpr auto bias_descriptor = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<bias_descriptor, metal::execution_simdgroup> bias_op;
        typedef decltype(bias_op.template get_left_input_cooperative_tensor<half, half, float>()) biases_t;
        typedef decltype(bias_op.template get_right_input_cooperative_tensor<half, half, float>()) sums_t;
        typedef decltype(bias_op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<biases_t>,
            metal::remove_addrspace_t<sums_t>, float>()) min_t;
        min_t min0, min1;
        PROJECTION_UNROLL
        for (uint16_t e = 0; e < 16; ++e) {
            min0[e] = 0.0f;
            min1[e] = 0.0f;
        }
        const uint chunks = int8_min_chunks(groups);
        device const uint4 *bias_words = reinterpret_cast<device const uint4 *>(s.biases)
            + (ulong(n0 / 16u) * chunks * 32u + lane) * 2u;
        device const uint4 *sum_words = reinterpret_cast<device const uint4 *>(s.block_sums)
            + (ulong(tile_m) * chunks * 32u + lane) * 4u;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
            biases_t biases0 = bias_op.template get_left_input_cooperative_tensor<half, half, float>();
            biases_t biases1 = bias_op.template get_left_input_cooperative_tensor<half, half, float>();
            sums_t sums = bias_op.template get_right_input_cooperative_tensor<half, half, float>();
            // A fragment of the first 16-row weight tile, then of the second.
            uint4 b[4], t[4];
            PROJECTION_UNROLL
            for (uint i = 0; i < 2; ++i) {
                b[i] = bias_words[chunk * 64u + i];
                b[2u + i] = bias_words[(chunks + chunk) * 64u + i];
            }
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i)
                t[i] = sum_words[chunk * 128u + i];
            PROJECTION_UNROLL
            for (uint16_t e = 0; e < 16; ++e) {
                biases0[e] = as_type<half2>(b[e >> 3][(e >> 1) & 3])[e & 1];
                biases1[e] = as_type<half2>(b[2 + (e >> 3)][(e >> 1) & 3])[e & 1];
            }
            PROJECTION_UNROLL
            for (uint16_t e = 0; e < 32; ++e)
                sums[e] = as_type<half2>(t[e >> 3][(e >> 1) & 3])[e & 1];
            bias_op.run(biases0, sums, min0);
            bias_op.run(biases1, sums, min1);
        }
        PROJECTION_UNROLL
        for (uint16_t e = 0; e < 16; ++e) {
            f[0][e] = min0[e] * 64.0f;
            f[1][e] = min1[e] * 64.0f;
        }
    }
    // The lane's positions as 32-bit offsets from the bound buffers: a
    // per-lane pointer is a 64-bit register held through the loop. Weight
    // row q of the lane is n0 + r + 8 q (fragment q / 2, its row q % 2).
    constexpr uint SLOT = int8_codes<W>::slot_bytes;
    constexpr uint HIGH = int8_codes<W>::high_block_bytes;
    uint codes[4], high[4];
    PROJECTION_UNROLL
    for (uint q = 0; q < 4; ++q) {
        codes[q] = w.codes_at(n0 + at.r + 8u * q, int8_codes<W>::block_bytes, 0u) + at.c / 4u * SLOT;
        high[q] = HIGH != 0 ? w.high_at(n0 + at.r + 8u * q, HIGH, 0u) + at.c / 4u * int8_codes<W>::high_slot_bytes
            : 0u;
    }
    // Bytes between a row's consecutive blocks (a row tile's rows of one
    // block are adjacent).
    const uint block_step = int8_codes<W>::block_bytes * w.layout.tile, high_step = HIGH * w.layout.tile;
    device const uint4 *right_codes = reinterpret_cast<device const uint4 *>(s.quantized);
    device const uint4 *scales = reinterpret_cast<device const uint4 *>(s.row_scales);
    device const uint4 *coefficients = reinterpret_cast<device const uint4 *>(s.coefficients);
    const uint right_at = (tile_m * groups * 32u + lane) * 2u;
    const uint scales_at = tile_m * groups * 4u + at.c / 4u;
    constexpr bool FACTORED = int8_codes<W>::factored;
    const uint coefficients_at = FACTORED ? uint(int8_scale_bytes_at(tile_n, 0u, at.r, groups))
        : tile_n * groups * 8u + at.r;
    const uint factors_at = uint(int8_factor_at(tile_n, 0u, at.r, groups));
    // FACTORED: the four rows' factors of the 256 columns and their scale
    // bytes of four blocks, each one load for its blocks.
    float4 factor = float4(0.0f);
    uint4 scale_bytes = uint4(0u);
    for (uint g = 0; g < groups; ++g) {
        const uint4 a0 = right_codes[right_at + g * 64u], a1 = right_codes[right_at + g * 64u + 1u];
        // The operand rows' scales (columns c .. c + 3 and c + 16 .. c + 19
        // of the destination) and the four weight rows' coefficients.
        const uint4 operand = scales[scales_at + g * 4u];
        const float4 near = float4(as_type<half4>(operand.xy)), far = float4(as_type<half4>(operand.zw));
        float4 weight;
        if constexpr (FACTORED) {
            if ((g & 7u) == 0u)
                factor = as_type<float4>(coefficients[factors_at + g]);
            if ((g & 3u) == 0u)
                scale_bytes = coefficients[coefficients_at + g * 2u];
            weight = factor * float4(as_type<uchar4>(scale_bytes[g & 3u]));
        } else {
            weight = as_type<float4>(coefficients[coefficients_at + g * 8u]);
        }
        right_t right = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
        PROJECTION_UNROLL
        for (uint16_t e = 0; e < 16; ++e) {
            right[e] = as_type<char4>(a0[e >> 2])[e & 3];
            right[uint16_t(16) + e] = as_type<char4>(a1[e >> 2])[e & 3];
        }
        PROJECTION_UNROLL
        for (uint i = 0; i < 2; ++i) {
            left_t left = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
            PROJECTION_UNROLL
            for (uint16_t h = 0; h < 2; ++h) {
                char4 first, second;
                int8_codes<W>::slot(w.base + codes[2u * i + h] + g * block_step,
                    w.base + high[2u * i + h] + g * high_step, first, second);
                PROJECTION_UNROLL
                for (uint16_t x = 0; x < 4; ++x) {
                    left[uint16_t(4 * h) + x] = first[x];
                    left[uint16_t(8 + 4 * h) + x] = second[x];
                }
            }
            product_t product;
            op.run(left, right, product);
            const float4 s00 = near * weight[2u * i], s10 = near * weight[2u * i + 1u];
            const float4 s01 = far * weight[2u * i], s11 = far * weight[2u * i + 1u];
            PROJECTION_UNROLL
            for (uint16_t x = 0; x < 4; ++x) {
                f[i][x] = metal::fma(float(product[x]), s00[x], f[i][x]);
                f[i][4 + x] = metal::fma(float(product[4 + x]), s10[x], f[i][4 + x]);
                f[i][8 + x] = metal::fma(float(product[8 + x]), s01[x], f[i][8 + x]);
                f[i][12 + x] = metal::fma(float(product[12 + x]), s11[x], f[i][12 + x]);
            }
        }
    }
}

// The weight row and operand row of element e of `gemm_int8_products`'
// fragment i on lane `at`.
inline uint int8_weight_row(int8_lane at, uint i, uint e) { return 16u * i + at.r + 8u * ((e >> 2) & 1u); }
inline uint int8_operand_row(int8_lane at, uint e) { return at.c + (e & 3u) + 16u * (e >> 3); }

#endif

// The INT8 launch: threadgroups of four simdgroups, each one 32 x 32
// `gemm_int8_products`, in one of two tile shapes (the grid's physical shape
// decides the rate, so the shape is the launch's). TILE 128, grid
// (ceil(M / 128), rows / 32): 128 rows by 32 weight rows. TILE 64, grid
// (rows / 64, ceil(M / 64)): 64 x 64, a simdgroup a quarter each. Weights
// without an int8 path, and devices without tensor
// operations, run the staged 64 x 64 tiles of the default form instead, with
// its results (the TILE 128 grid's threadgroups in order take them in order;
// rows is a multiple of 64).
template <typename W, uint TILE, typename In, typename Out>
inline void gemm_int8(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const int8_scratch &s, uint m_rows, uint rows, uint k, uint gx, uint gy, threadgroup uchar *shared,
    uint sg, uint lane) {
    static_assert(TILE == 128 || TILE == 64, "an int8 tile is 128 x 32 or 64 x 64");
#if SEISMIC_HAS_TENSOR_OPS
    if constexpr (int8_codes<W>::available) {
        const uint m0 = TILE == 128 ? gx * 128u + sg * 32u : gx * 64u + (sg / 2u) * 32u;
        const uint n0 = TILE == 128 ? gy * 32u : gy * 64u + (sg % 2u) * 32u;
        const int8_lane at(lane);
        float f[2][16];
        gemm_int8_products<W>(w, s, k, m0, n0, lane, f);
        // A lane holds eight operand rows (elements e with the same
        // (e & 3) + 4 (e >> 3)), four weight rows of each: a row's result
        // and residual rows are resolved once.
        PROJECTION_UNROLL
        for (uint column = 0; column < 8; ++column) {
            const uint e0 = (column & 3u) + 8u * (column >> 2);
            const uint m = m0 + int8_operand_row(at, e0);
            if (m < m_rows) {
                const auto row = out.row(m);
                PROJECTION_UNROLL
                for (uint i = 0; i < 2; ++i) {
                    PROJECTION_UNROLL
                    for (uint h = 0; h < 2; ++h)
                        row.store(n0 + int8_weight_row(at, i, e0 + 4u * h), f[i][e0 + 4u * h]);
                }
            }
        }
        return;
    }
#endif
    if constexpr (TILE == 64) {
        gemm<W, 64, 64>(in, out, w, m_rows, rows, k, gx, gy, shared, sg, lane);
    } else {
        const uint columns = rows / 64u;
        const uint index = gy * ((m_rows + 127u) / 128u) + gx;
        if (index / columns < (m_rows + 63u) / 64u)
            gemm<W, 64, 64>(in, out, w, m_rows, rows, k, index / columns, index % columns, shared, sg, lane);
    }
}

// The paired INT8 launch (gate and up weights over one operand): a tile of
// 64 rows by 32 features; simdgroups 0 and 2 sum the gate weights' products
// for the tile's two 32-row halves, 1 and 3 the up weights', and each up
// simdgroup stores the pair with its gate simdgroup's sums, exchanged through
// `exchange` (2 x 32 x 32 floats). `gate_scratch` and `up_scratch` share the
// operand and differ in the weights' coefficients and biases. Either weight
// without an int8 path, and devices without tensor operations, run the staged
// paired 64 x 64 tile of the default form, with its results.
template <typename G, typename U, typename In, typename Out>
inline void gemm_int8_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, thread const int8_scratch &gate_scratch, thread const int8_scratch &up_scratch,
    uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, threadgroup float *exchange,
    uint sg, uint lane) {
#if SEISMIC_HAS_TENSOR_OPS
    if constexpr (int8_codes<G>::available && int8_codes<U>::available) {
        const uint half_tile = sg / 2u, m0 = tm * 64u + half_tile * 32u, n0 = tn * 32u;
        const int8_lane at(lane);
        const bool second = (sg % 2u) != 0;
        float f[2][16];
        if (second)
            gemm_int8_products<U>(up, up_scratch, k, m0, n0, lane, f);
        else
            gemm_int8_products<G>(gate, gate_scratch, k, m0, n0, lane, f);
        threadgroup float *slot = exchange + (half_tile * 32u + lane) * 32u;
        if (!second) {
            PROJECTION_UNROLL
            for (uint e = 0; e < 32; ++e)
                slot[e] = f[e >> 4][e & 15u];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!second)
            return;
        PROJECTION_UNROLL
        for (uint i = 0; i < 2; ++i) {
            PROJECTION_UNROLL
            for (uint e = 0; e < 16; ++e) {
                const uint m = m0 + int8_operand_row(at, e);
                if (m < m_rows)
                    out.store_pair(m, n0 + int8_weight_row(at, i, e), slot[16u * i + e], f[i][e]);
            }
        }
        return;
    }
#endif
    gemm_paired<G, U, 64, 64>(in, out, gate, up, m_rows, rows, k, tm, tn, shared, sg, lane);
}

} // namespace projection
