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
// - GEMV (M below the entry's BATCH_FROM; `gemv.h`): lane groups own weight rows, lanes
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
// optionally gathered through a row table (the selected vocabulary rows),
// read in the bound weights' space (`packets::weight_bytes`).
template <typename W>
struct Weights {
    packets::weight_bytes base;
    Rows16 layout;
    uint k;
    device const int *table;
    // The tensor row of this view's row 0 (a view over a row range).
    ulong first;
    ulong table_stride = 1;
    // Rows of one stored row tile (1: rows stored one after another).
    uint tile() const { return layout.tile; }
    // The tensor row behind row n.
    ulong stored(uint n) const { return (table ? ulong(table[ulong(n) * table_stride]) : ulong(n)) + first; }
    // Row n as the packet decoders address it: its base (its row tile's
    // first row) and its geometry (`Rows16::of`).
    packets::weight_bytes row(uint n) const { return layout.base_of(base, stored(n)); }
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
        packets::weight_bytes row;
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
    // The same in the plane of high code bits.
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

// `scatter`: each of the STREAMS hyper-connection streams of the residual
// plus the output times the stream's inject value,
//   y[m, s * width + n] = residual[row(m), s * width + n] + round_A(acc) * inject[m, s]   (F32),
// with `Residual`'s interface (column strides are the streams' too).
template <typename A, typename Rows, uint STREAMS>
struct Scatter {
    device float *y;
    ulong stride0, stride1;
    device const float *residual;
    ulong residual0, residual1;
    Rows rows;
    device const float *inject;
    ulong inject0;
    uint width;
    struct Row {
        device float *y;
        device const float *residual;
        ulong stride1, residual1;
        device const float *inject;
        uint width;
        void store(uint n, float value) const {
            const float rounded = A::round(value);
            PROJECTION_UNROLL
            for (uint s = 0; s < STREAMS; ++s) {
                const ulong column = ulong(s * width + n);
                y[column * stride1] = residual[column * residual1] + rounded * inject[s];
            }
        }
        void store8(uint n, float4 even, float4 odd) const {
            PROJECTION_UNROLL
            for (uint i = 0; i < 4; ++i) {
                store(n + 2u * i, even[i]);
                store(n + 2u * i + 1u, odd[i]);
            }
        }
    };
    Row row(uint m) const {
        return Row{y + ulong(m) * stride0, residual + ulong(rows.at(m)) * residual0, stride1, residual1,
            inject + ulong(m) * inject0, width};
    }
    void store(uint m, uint n, float value) const { row(m).store(n, value); }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// An output entry's epilogue: `Residual`, or `Scatter` over STREAMS streams
// when the entry takes inject values. `make` forms either from every field
// (`Residual` reads the leading ones).
template <typename A, typename Rows, uint STREAMS, bool INJECTED>
struct OutputOf {
    typedef Scatter<A, Rows, STREAMS> type;
    static type make(device float *y, ulong stride0, ulong stride1, device const float *residual, ulong residual0,
        ulong residual1, Rows rows, device const float *inject, ulong inject0, uint width) {
        return type{y, stride0, stride1, residual, residual0, residual1, rows, inject, inject0, width};
    }
};
template <typename A, typename Rows>
struct OutputOf<A, Rows, 1, false> {
    typedef Residual<A, Rows> type;
    static type make(device float *y, ulong stride0, ulong stride1, device const float *residual, ulong residual0,
        ulong residual1, Rows rows, device const float *, ulong, uint) {
        return type{y, stride0, stride1, residual, residual0, residual1, rows};
    }
};

// A routed sublayer's column n joining its row's residual (`routed_output`,
// `routed_combine`): residual + selected + shared, in that order, or (INJECTED)
// for each of the STREAMS streams residual[s] + (selected + shared) * inject[s].
// `out`, `residual` and `inject` point at the row; strides are column strides.
template <uint STREAMS, bool INJECTED>
inline void routed_join(device float *out, ulong out1, device const float *residual, ulong residual1,
    device const float *inject, ulong width, ulong n, float selected, float shared) {
    if (!INJECTED) {
        out[n * out1] = residual[n * residual1] + selected + shared;
        return;
    }
    PROJECTION_UNROLL
    for (uint s = 0; s < STREAMS; ++s) {
        const ulong column = ulong(s) * width + n;
        out[column * out1] = residual[column * residual1] + (selected + shared) * inject[s];
    }
}

// A mix site's low-rank rows (`stream_mix_down`), over S streams:
//   y = round_A(silu(acc / S))
template <typename A>
struct MixLow {
    device uchar *y;
    ulong stride0, stride1;
    float streams;
    void store(uint m, uint n, float value) const {
        const float v = value / streams;
        reinterpret_cast<device typename A::storage *>(y)[ulong(m) * stride0 + ulong(n) * stride1] =
            A::store(v / (1.0f + metal::exp(-v)));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// A mix site's inject values (`stream_mix_down`), over S streams:
//   y = 2 sigmoid(acc / S)                                              (F32)
struct MixInject {
    device float *y;
    ulong stride0, stride1;
    float streams;
    void store(uint m, uint n, float value) const {
        y[ulong(m) * stride0 + ulong(n) * stride1] = 2.0f / (1.0f + metal::exp(-value / streams));
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
    }
};

// A mix site's per-stream partials (`stream_mix_up`): column n of stream
// s = n / width, gated by its sum,
//   y = sigmoid(acc) * x[m, n] * inverse[m, s] * norm[n] / S             (F32)
template <typename N>
struct Mix {
    device float *y;
    ulong stride0, stride1;
    device const float *x;
    ulong x0, x1;
    device const float *inverse;
    ulong inverse0, inverse1;
    device const uchar *norm;
    ulong norm_stride;
    uint width;
    float streams;
    void store(uint m, uint n, float value) const {
        const float normalized = x[ulong(m) * x0 + ulong(n) * x1] * inverse[ulong(m) * inverse0
            + ulong(n / width) * inverse1] * element::at<N>(norm, ulong(n) * norm_stride);
        const float gate = 1.0f / (1.0f + metal::exp(-value));
        y[ulong(m) * stride0 + ulong(n) * stride1] = gate * normalized / streams;
    }
    void store2(uint m, uint n, float first, float second) const {
        store(m, n, first);
        store(m, n + 1u, second);
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

// One fixed RMS recipe, whether the caller stages an operand or publishes
// normalized features through its own strides.
template <uint THREADS, typename In>
inline float normalization_inverse(thread const In &in, uint m, uint group,
    threadgroup float *partials, uint thread_index) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "whole normalizing simdgroups");
    for (uint part = thread_index / 32u; part < In::parts; part += THREADS / 32u) {
        const float partial = simd_sum(in.squares(m, group, part, thread_index % 32u));
        if (thread_index % 32u == 0)
            partials[part] = partial;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float squares = sum_parts<In::parts>(partials);
    return metal::rsqrt(squares / float(in.width()) + in.epsilon());
}

// `Order` lays the rows out: `RowOrder` (the default), or the tall GEMM's
// `TallOrder` when whole 8-column words are written (columns and every
// group's first column multiples of 8).
template <uint THREADS, typename Order, typename In>
inline void device_normalize_with(thread const In &in, uint m, uint group, float inverse, device uchar *x,
    uint columns, uint thread_index) {
    typedef typename In::activation A;
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

// Row m's norm group `group` (item m * groups + group) normalized by its
// reduced inverse, which the threadgroup returns.
template <uint THREADS, typename Order, typename In>
inline float device_normalize(thread const In &in, uint item, device uchar *x, uint columns,
    threadgroup float *partials, uint thread_index) {
    static_assert(THREADS % 32 == 0 && THREADS <= 1024, "a normalizing threadgroup is whole simdgroups");
    uint m = item / in.groups(), group = item % in.groups();
    const float inverse = normalization_inverse<THREADS>(in, m, group, partials, thread_index);
    device_normalize_with<THREADS, Order>(in, m, group, inverse, x, columns, thread_index);
    return inverse;
}

template <uint THREADS, typename In>
inline float device_normalize(thread const In &in, uint item, device uchar *x, uint columns,
    threadgroup float *partials, uint thread_index) {
    return device_normalize<THREADS, RowOrder<typename In::activation>>(in, item, x, columns, partials,
        thread_index);
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

// Every norm group's inverse of the `m_rows` rows, from the square sums of
// `threadgroup_squares` (after a barrier publishes them), exactly as
// SharedNorm applies it: inverses[m * stride0 + group * stride1].
template <typename In>
inline void publish_inverses(thread const In &in, uint m_rows, threadgroup const float *squares,
    device float *inverses, ulong stride0, ulong stride1, uint thread_index, uint threads) {
    for (uint item = thread_index; item < m_rows * in.groups(); item += threads) {
        uint m = item / in.groups(), group = item % in.groups();
        float sum = sum_parts<In::parts>(squares + item * In::parts);
        inverses[ulong(m) * stride0 + ulong(group) * stride1] =
            metal::rsqrt(sum / float(in.width()) + in.epsilon());
    }
}

// `in` as a SharedNorm over `squares`. The operand's type is deduced from a
// value: `decltype` of a local carries its address space under Metal 4.1,
// which a field's type may not.
template <typename In>
inline SharedNorm<In> shared_norm(In in, threadgroup const float *squares) {
    return SharedNorm<In>{in, squares};
}

// ---------------------------------------------------------------------------
// The operand rows of a product (the portable `operand_rows`): its
// prologue's rows as they are, or, when its weights take NVFP4 operands
// (`operands<packets::Wn_OPERANDS_NVFP4>`), as NVFP4 operands (stage.h).
//
// `rows` is the GEMV row classes' (M <= 16; every thread of the threadgroup
// calls it): after the squares pass (whose
// barrier it issues) simdgroup m % simdgroups reduces row m's largest
// magnitude over its `columns` into `globals` (PROJECTION_OPERANDS_SHARED,
// one float a row when NVFP4), which the opening barrier of the GEMV body
// publishes to its staging. `normalize` is a normalizing threadgroup's
// (`device_normalize`, one norm group per row): the row's largest magnitude
// under its inverse, reduced across the threadgroup through `partials`,
// then the row formed.
#define PROJECTION_OPERANDS_SHARED(name, NVFP4) threadgroup float name[(NVFP4) ? 16 : 1]

template <bool NVFP4>
struct operands;
template <>
struct operands<false> {
    template <typename In>
    static In rows(In in, uint, uint, threadgroup float *, uint, uint, uint) {
        return in;
    }
    template <typename In>
    static In lone_row(In in, uint, uint) {
        return in;
    }
    template <uint THREADS, typename Order, typename In>
    static void normalize(thread const In &in, uint item, device uchar *x, uint columns,
        threadgroup float *partials, uint thread_index) {
        device_normalize<THREADS, Order>(in, item, x, columns, partials, thread_index);
    }
    template <uint THREADS, typename In>
    static void normalize(thread const In &in, uint item, device uchar *x, uint columns,
        threadgroup float *partials, uint thread_index) {
        normalize<THREADS, RowOrder<typename In::activation>>(in, item, x, columns, partials, thread_index);
    }
};
template <>
struct operands<true> {
    template <typename In>
    static Nvfp4Rows<In, RowGlobals> rows(In in, uint m_rows, uint columns, threadgroup float *globals,
        uint simdgroups, uint sg, uint lane) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint m = sg; m < m_rows; m += simdgroups) {
            float amax = 0.0f;
            for (uint k = 8u * lane; k < columns; k += 256u) {
                float4 even, odd;
                in.load8(m, k, 0.0f, even, odd);
                amax = metal::max(amax, nvfp4::magnitude(even, odd));
            }
            amax = simd_max(amax);
            if (lane == 0)
                globals[m] = nvfp4::global_scale(amax);
        }
        return Nvfp4Rows<In, RowGlobals>{in, {globals}};
    }
    // Row 0 of `in` (`columns` values) as one simdgroup forms it alone: its
    // largest magnitude reduced across the simdgroup, without a barrier
    // (the decode down projections, `routed::project_channels`).
    template <typename In>
    static Nvfp4Rows<In, OneGlobal> lone_row(In in, uint columns, uint lane) {
        float amax = 0.0f;
        for (uint k = 8u * lane; k < columns; k += 256u) {
            float4 even, odd;
            in.load8(0, k, 0.0f, even, odd);
            amax = metal::max(amax, nvfp4::magnitude(even, odd));
        }
        return Nvfp4Rows<In, OneGlobal>{in, {nvfp4::global_scale(simd_max(amax))}};
    }
    template <uint THREADS, typename In>
    static void normalize(thread const In &in, uint item, device uchar *x, uint columns,
        threadgroup float *partials, uint thread_index) {
        normalize<THREADS, RowOrder<typename In::activation>>(in, item, x, columns, partials, thread_index);
    }
    template <uint THREADS, typename Order, typename In>
    static void normalize(thread const In &in, uint item, device uchar *x, uint columns,
        threadgroup float *partials, uint thread_index) {
        const float inverse = normalization_inverse<THREADS>(in, item, 0, partials, thread_index);
        form_row<THREADS, Order>(in, item, inverse, x, columns, partials, thread_index);
    }
    // A plain operand's row m formed into x (a stage launch's threadgroup).
    template <uint THREADS, typename In>
    static void form(thread const In &in, uint m, device uchar *x, uint columns, threadgroup float *partials,
        uint thread_index) {
        form_row<THREADS, RowOrder<typename In::activation>>(in, m, 0.0f, x, columns, partials, thread_index);
    }
    // The A row at `row` (`columns` values) formed in place: each thread
    // reads and rewrites whole 16-value blocks, so no thread reads a value
    // another has rewritten (the grouped expert products a down projection
    // takes).
    template <uint THREADS, typename A>
    static void form_in_place(device uchar *row, uint columns, threadgroup float *partials, uint thread_index) {
        const Plain<A, AllRows> in{row, 0, 1, columns, {}};
        const float amax = row_maximum<THREADS>(in, 0, 0.0f, partials, thread_index);
        const Nvfp4Rows<Plain<A, AllRows>, OneGlobal> formed{in, {nvfp4::global_scale(amax)}};
        device typename A::storage *values = reinterpret_cast<device typename A::storage *>(row);
        for (uint block = 16u * thread_index; block < columns; block += 16u * THREADS) {
            float4 even[2], odd[2];
            formed.load8(0, block, 0.0f, even[0], odd[0]);
            formed.load8(0, block + 8u, 0.0f, even[1], odd[1]);
            for (uint j = 0; j < 16u && block + j < columns; ++j)
                values[block + j] = A::store((j & 1u) ? odd[j >> 3][(j & 7u) >> 1] : even[j >> 3][(j & 7u) >> 1]);
        }
    }

private:
    // Row m's largest magnitude under its norm inverse, reduced across the
    // threadgroup through `partials` (whose previous contents every thread
    // has read: a normalization's square sums).
    template <uint THREADS, typename In>
    static float row_maximum(thread const In &in, uint m, float inverse, threadgroup float *partials,
        uint thread_index) {
        static_assert(THREADS % 32u == 0 && THREADS / 32u <= 32u, "whole simdgroups, one partial each");
        float amax = 0.0f;
        for (uint k = 8u * thread_index; k < in.width(); k += 8u * THREADS) {
            float4 even, odd;
            in.load8(m, k, inverse, even, odd);
            amax = metal::max(amax, nvfp4::magnitude(even, odd));
        }
        amax = simd_max(amax);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index % 32u == 0)
            partials[thread_index / 32u] = amax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < THREADS / 32u; ++s)
            amax = metal::max(amax, partials[s]);
        return amax;
    }
    // Row m under its norm inverse, formed into x.
    template <uint THREADS, typename Order, typename In>
    static void form_row(thread const In &in, uint m, float inverse, device uchar *x, uint columns,
        threadgroup float *partials, uint thread_index) {
        const float amax = row_maximum<THREADS>(in, m, inverse, partials, thread_index);
        const Nvfp4Rows<In, OneGlobal> formed{in, {nvfp4::global_scale(amax)}};
        device_normalize_with<THREADS, Order>(formed, m, 0, inverse, x, columns, thread_index);
    }
};

// The epilogue call of every projection body: plain epilogues store one
// sum; paired epilogues combine gate and up.
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
// projections, a 4-way K split fill the device. Larger row counts run the tuned
// TILE_M x TILE_N (x SPLIT) launch.
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

// Step `step` of a packet into its 8 columns of B tile row `row`, as every
// GEMM form stages it: the packet's values (`W::values`), rounded to half.
template <typename W>
inline void gemm_store_step(thread const typename W::packet &k, uint step, threadgroup half *row) {
    float4 even, odd;
    W::values(k, step, even, odd);
    threadgroup half4 *out = reinterpret_cast<threadgroup half4 *>(row + 8u * step);
    out[0] = half4(half(even.x), half(odd.x), half(even.y), half(odd.y));
    out[1] = half4(half(even.z), half(odd.z), half(even.w), half(odd.w));
}

// Decode the staged packets (`gemm_store_step`). Local
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
            PROJECTION_UNROLL
            for (uint s = 0; s < 2; ++s)
                gemm_store_step<W>(regs.packet[j], 2u * within + s, b + tile_row * gemm_lda);
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

// The B fragment of B tile rows n .. n + 7 at columns k .. k + 7, `at`
// being this lane's element (row n + x, column k + y): it holds rows x and
// x + 1 at column y.
inline simdgroup_half8x8 gemm_b_fragment(threadgroup const half *at) {
    simdgroup_half8x8 b;
    reinterpret_cast<thread half2 &>(b.thread_elements()) = half2(at[0], at[gemm_lda]);
    return b;
}

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
        for (uint j = 0; j < FN; ++j)
            b[j] = gemm_b_fragment(b_lane + 8u * j * gemm_lda + kk);
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
        threadgroup half4 *to = reinterpret_cast<threadgroup half4 *>(row + 32u * i + 16u * within);
        PROJECTION_UNROLL
        for (uint s = 0; s < 2; ++s) {
            float4 even, odd;
            X::values(packet, 2u * within + s, even, odd);
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
    X::values(packet, block, even, odd);
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
// whole number of packets: 512 columns for NB = 1, 256 for NB = 2).

constant constexpr uint gemv_batch_shared_bytes = 16640;
template <uint NB>
constexpr uint gemv_batch_chunk() {
    return (gemv_batch_shared_bytes / (32u * NB) - 4u) / 32u * 32u;
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
        reinterpret_cast<thread float2 &>(weights.thread_elements()) = W::pair_values(a, step, pair_index);
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

template <typename W, typename U, bool PAIRED, uint R, uint NB, bool COLUMNS = false, typename In, typename Out>
inline void gemv_batch_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    constexpr uint chunk = gemv_batch_chunk<NB>();
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

template <typename W, uint R, typename In, typename Out>
inline void gemv_batch_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, R, 1>(in, out, w, w, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
    else
        gemv_batch_body<W, W, false, R, 2>(in, out, w, w, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
}

// `gemv_batch_runtime` with a column epilogue (`gemv_emit<true>`).
template <typename W, uint R, typename In, typename Out>
inline void gemv_batch_columns_runtime(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, R, 1, true>(in, out, w, w, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
    else
        gemv_batch_body<W, W, false, R, 2, true>(in, out, w, w, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
}

template <typename G, typename U, uint R, typename In, typename Out>
inline void gemv_batch_paired_runtime(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<G, U, true, R, 1>(in, out, gate, up, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
    else
        gemv_batch_body<G, U, true, R, 2>(in, out, gate, up, m_rows, rows, k, tile, shared, simdgroups, sg, lane);
}

#if !SEISMIC_HAS_TENSOR_OPS
// ---------------------------------------------------------------------------
// The GEMM's one-row form (simdgroup matrices only): the first tile of a
// GEMM whose rows fit one fragment row (`gemm_row_rows`), with the GEMM's
// bits and no threadgroup-wide weight staging. Each output is the GEMM's
// chain: its row of activations (A, as stored) by its weight row staged by
// `gemm_store_step` (B, read as the GEMM reads its B tile), in 8-deep
// simdgroup multiply-accumulates ascending along K from zero, the same
// fragment operation with the same operand roles; an output's chain reads
// only its own row and weight row, so which rows and columns share its
// fragments changes no bit. The tile's TN / 16 simdgroups (the GEMM's
// threads) own two blocks of 8 weight rows each, placed as the batched GEMV
// places its blocks (`gemv_batch_rows`): blocks 2 sg and 2 sg + 1 of the
// tile (plain), or block sg of gate and of up (paired). Each simdgroup
// stages its own 16-row B tile a packet at a time, lane l decoding step
// l % 4 of row l / 4 of both blocks straight from device memory, ordered by
// simdgroup barriers alone. The activation rows are staged in the rest of
// the tile's threadgroup buffer in chunks of whole packets, rows past
// `m_rows` zero.
constant constexpr uint gemm_row_rows = 8;

// Halves of one simdgroup's B tile.
constant constexpr uint gemm_row_b = 16u * gemm_lda;

template <uint TM, uint TN, typename E>
constexpr uint gemm_row_chunk() {
    return ((gemm_tile<TM, TN>::bytes - TN / 16u * gemm_row_b * 2u) / (gemm_row_rows * sizeof(E)) - 8u) / 32u * 32u;
}

// `emit(m, n, first, second)` as `gemm_fragment_engine::emit`, from the
// tile's weight row (plain) or feature (paired) `first`.
template <typename W, typename U, bool PAIRED, uint TM, uint TN, typename In, typename F>
inline void gemm_row_run(thread const In &in, thread const Weights<W> &w, thread const Weights<U> &u, uint first,
    uint rows, uint m_rows, uint k, threadgroup uchar *shared, uint sg, uint lane, thread const F &emit) {
    typedef typename In::activation::native E;
    constexpr uint chunk = gemm_row_chunk<TM, TN, E>();
    constexpr uint pitch = chunk + 8u;
    constexpr uint threads = gemm_tile<TM, TN>::threads;
    constexpr uint blocks = PAIRED ? 1u : 2u;   // of each simdgroup, per tensor
    static_assert(threads == TN * 2u, "the one-row form's simdgroups own 16 weight rows each");
    threadgroup E *staged = reinterpret_cast<threadgroup E *>(shared);
    threadgroup half *tile = reinterpret_cast<threadgroup half *>(staged + gemm_row_rows * pitch) + sg * gemm_row_b;
    const ushort2 at = fragment_coordinate(lane);
    const gemv_batch_rows map{first, blocks * sg, w.tile() == 32u && (TN / 16u * blocks) % 4u == 0u};
    // Fragment 0 is block 0 of W (B tile rows 0..7); fragment 1 block 1 of W
    // (plain, U = W) or block 0 of U (paired), rows 8..15.
    const uint second = PAIRED ? 0u : 1u;
    const uint row = lane / 4u, step_of = lane % 4u;
    const typename Weights<W>::located w_at = w.locate(min(map.at(0u, row), rows - 1u));
    const typename Weights<U>::located u_at = u.locate(min(map.at(second, row), rows - 1u));
    threadgroup const half *b_lane = tile + at.x * gemm_lda + at.y;
    typename packets::Block<W>::state w_run;
    typename packets::Block<U>::state u_run;
    typename W::packet a, a_next;
    typename U::packet b, b_next;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint c0 = 0; c0 < k; c0 += chunk) {
        const uint packets = (min(chunk, k - c0) + 31u) / 32u;
        a = gemv_batch_load(w, w_at, c0 / 32u, w_run);
        b = gemv_batch_load(u, u_at, c0 / 32u, u_run);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = sg * 32u + lane; item < gemm_row_rows * 4u * packets; item += threads) {
            const uint m = item / (4u * packets), column = c0 + 8u * (item % (4u * packets));
            *reinterpret_cast<threadgroup uint4 *>(staged + m * pitch + (column - c0)) =
                m < m_rows && column < k ? in.words8(m, column) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local = 0; local < packets; ++local) {
            if (local + 1u < packets) {
                a_next = gemv_batch_load(w, w_at, c0 / 32u + local + 1u, w_run);
                b_next = gemv_batch_load(u, u_at, c0 / 32u + local + 1u, u_run);
            }
            gemm_store_step<W>(a, step_of, tile + row * gemm_lda);
            gemm_store_step<U>(b, step_of, tile + (8u + row) * gemm_lda);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            PROJECTION_UNROLL
            for (uint step = 0; step < 4u; ++step) {
                simdgroup_matrix<E, 8, 8> x;
                reinterpret_cast<thread vec<E, 2> &>(x.thread_elements()) = *reinterpret_cast<threadgroup const vec<E, 2> *>(
                    staged + at.y * pitch + 32u * local + 8u * step + at.x);
                simdgroup_multiply_accumulate(acc0, x, gemm_b_fragment(b_lane + 8u * step), acc0);
                simdgroup_multiply_accumulate(acc1, x, gemm_b_fragment(b_lane + 8u * gemm_lda + 8u * step), acc1);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            a = a_next;
            b = b_next;
        }
    }
    const float2 r0 = reinterpret_cast<thread float2 &>(acc0.thread_elements());
    const float2 r1 = reinterpret_cast<thread float2 &>(acc1.thread_elements());
    const uint n0 = map.at(0u, at.x) - first, n1 = map.at(0u, at.x + 1u) - first;
    if (PAIRED) {
        emit(at.y, n0, r0.x, r1.x);
        emit(at.y, n1, r0.y, r1.y);
    } else {
        emit(at.y, n0, r0.x, 0.0f);
        emit(at.y, n1, r0.y, 0.0f);
        emit(at.y, map.at(1u, at.x) - first, r1.x, 0.0f);
        emit(at.y, map.at(1u, at.x + 1u) - first, r1.y, 0.0f);
    }
}

// The one-row form of `gemm` and `gemm_paired` (tile tm = 0) over rows
// below `m_rows` <= gemm_row_rows.
template <typename W, uint TM, uint TN, typename In, typename Out>
inline void gemm_row(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
    uint first = tn * TN;
    gemm_row_run<W, W, false, TM, TN>(in, w, w, first, rows, m_rows, k, shared, sg, lane,
        gemm_store<Out>{out, 0, m_rows, first, rows});
}

template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void gemm_row_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tn, threadgroup uchar *shared, uint sg,
    uint lane) {
    uint first = tn * (TN / 2u);
    gemm_row_run<G, U, true, TM, TN>(in, gate, up, first, rows, m_rows, k, shared, sg, lane,
        gemm_store_pair<Out>{out, 0, m_rows, first, rows});
}
#endif

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
// is its code times its row's local scale code of the block, an integer
// below 2048 entered as the F16 with those bits (the value 2^-24 times it,
// exactly), so one accumulator runs over the 32 steps of a 256-column block
// and folds once with the block's scale factor. The min term of the eight
// 32-column blocks is one more product, of the rows' minimum codes (row y,
// blocks x, x + 1) with the activation rows' sums over each block (block y,
// columns the activation rows), folded with the signed minimum factor.
//
// PARTS simdgroups share a fragment's eight weight rows, a contiguous range
// of 256-column blocks each (a simdgroup alone leaves the device short of
// threads); their sums meet in threadgroup memory, so PARTS is arithmetic.
// A row's result depends only on its own activations: every row of an M-row
// call is bit-identical to the same row computed alone in this form.
// `shared` holds PARTS-independent 16 bytes per thread and activation
// fragment.

// Lane codes: the operand the matrix GEMV and the token-packing GEMM
// (`packing.h`) take from a weight's producer. Chunk k of a packet is its
// columns 8 k .. 8 k + 7; a lane that holds a chunk takes its codes two at a
// time, p and p + 4 (`pair`, position p < 4), as unsigned integers in the
// two halves of a word whose values are the codes less `zero()`, each below
// 1024 (a half of the same bits is the code times 2^-24, exactly). `at` is
// a lane's place of chunk k in a row's planes (32-bit offsets from the
// tensor), `load<INDEXED>` its words of packet g (or of the chunk `chunks`
// chunks on, by the same base), and `shift` words moved on by `positions`
// positions (pair p of the result is pair p + positions of the words);
// code planes split a load into its loads alone (`fetch<INDEXED>`, its
// `fields`) and their words (`spread`, or `moved` on by positions), which
// PACK carries ahead apart (`packing_fetch`). One reader per code source:
// linear codes of code planes of every split the registry makes
// (`registry::code_planes`), entering as their stored codes or above them; a
// codebook's values, entering at 128 (`codebooks::pairs`); and grid entries'
// values, entering at any ZERO. Where `bytes()`, a chunk's words are its
// eight codes as bytes in column order, as `packing_byte_step` reads them.

// The codes of one plane of B-bit codes (B 1, 2 or 4) in a chunk: B bytes at
// byte B k of the packet's unit of 4 B bytes, code i at bit B i
// (`packets::Fields`, `packets::Planes`); `fetch` reads them, `spread`
// spreads them, codes 0..3 and 4..7 in the two halves of a word (`load`
// both), and `moved` spreads them moved on by a lane's `positions`
// positions. A spread's bits past a half's four codes are no pair's (even
// moved on by three positions) and every reader masks its pair, so a moved
// spread, past whose variable shift the compiler cannot drop a mask, takes
// none; a byte's field and its copy 12 bits above it do not overlap, so
// one multiply spreads them (exact; a 2-bit plane's 16-bit field and its
// copy overlap, and the carry of their sum would reach code 4).
template <uint B>
struct chunk_plane {
    static_assert(B == 1u || B == 2u || B == 4u, "a plane's chunk is one, two or four bytes");
    typedef metal::conditional_t<B == 1u, uchar, metal::conditional_t<B == 2u, ushort, uint>> unit;
    template <typename W>
    static uint at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, ulong plane, uint k) {
        return uint(row.geometry.at(row.row, plane, 4ul * B, 0ul) - w.base) + k * B;
    }
    // A load INDEXED from the lane's place (a base and a 32-bit index of its
    // units), as the matrix GEMV's block loop takes it, or from the sum of
    // the place, the packet's 32-bit offset and the chunks' offset, as PACK's
    // does: indexed, PACK over q5g32s is 1.7% slower (M4 Pro, `dense_output`
    // at M = 512, 5448 against 5353 µs) and over iq4xs 1%. The chunks'
    // offset is a constant beside the 32-bit sum, so a load of a later chunk
    // takes it as the load's own (inside the sum, which may wrap, it is one
    // more add a load).
    template <bool INDEXED, typename W>
    static uint fetch(thread const Weights<W> &w, uint at, uint g, uint chunks) {
        if constexpr (INDEXED)
            return uint(slab::as<unit>(w.base + at)[g * 4u * w.layout.tile + chunks]);
        else
            return uint(*slab::as<unit>(w.base + at + g * 4u * B * w.layout.tile + chunks * B));
    }
    static uint spread(uint field) {
        if constexpr (B == 4u)
            return field;
        else
            return (field | (field << (16u - 4u * B))) & (((1u << (4u * B)) - 1u) * 0x00010001u);
    }
    template <bool INDEXED, typename W>
    static uint load(thread const Weights<W> &w, uint at, uint g, uint chunks) {
        return spread(fetch<INDEXED>(w, at, g, chunks));
    }
    static uint moved(uint field, uint positions) {
        if constexpr (B == 4u)
            return field >> (4u * positions);
        else if constexpr (B == 2u)
            return (field | (field << 8)) >> (2u * positions);
        else
            return (field * 0x1001u) >> positions;
    }
    static uint pair(uint spread, uint p) { return (spread >> (B * p)) & (((1u << B) - 1u) * 0x00010001u); }
};

// A chunk's eight codes as bytes (columns 0..3 in x, 4..7 in y): pair p is
// byte p of each word.
inline uint chunk_byte_pair(uint2 codes, uint p) {
    return ((codes.x >> (8u * p)) & 0xffu) | (((codes.y >> (8u * p)) & 0xffu) << 16);
}

// The zero code a producer's codes enter at by themselves: a linear code's
// stored zero, else 128 (a codebook's pairs, and grid values as bytes).
template <typename P, typename C>
struct lane_zero {
    static constexpr int value() { return 128; }
};
template <typename P, uint MASK, int Z>
struct lane_zero<P, packets::Linear<MASK, Z>> {
    static constexpr int value() { return Z; }
};

// The lane codes of decoders of codes in planes P interpreted by C, entering
// at ZERO (`implemented()`); none by default.
template <typename P, typename C, int ZERO = lane_zero<P, C>::value()>
struct lane_codes {
    static constexpr bool implemented() { return false; }
};

// Linear codes of code planes, whose value zero is code Z: a low plane of
// LOW bits (1, 2 or 4) and, HIGH, a high plane of HIGH bits above them,
// entering at any ZERO from Z on (one add moves both of a pair).
template <uint LOW, uint HIGH, uint MASK, int Z, int ZERO>
struct lane_codes<packets::Planes<LOW, HIGH>, packets::Linear<MASK, Z>, ZERO> {
    typedef chunk_plane<LOW> low_plane;
    typedef metal::conditional_t<HIGH == 0u, uint, uint2> place;
    typedef place words;
    typedef place fields;
    static constexpr bool implemented() { return MASK == 0u && LOW <= 4u && HIGH <= 2u && ZERO >= Z; }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return false; }
    static constexpr bool byte_words() { return false; }
    template <typename W>
    static place at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, uint k) {
        const uint low = low_plane::at(w, row, w.layout.codes, k);
        if constexpr (HIGH == 0u)
            return low;
        else
            return uint2(low, chunk_plane<HIGH>::at(w, row, w.layout.high, k));
    }
    template <bool INDEXED, typename W>
    static fields fetch(thread const Weights<W> &w, place at, uint g, uint chunks = 0u) {
        if constexpr (HIGH == 0u)
            return low_plane::template fetch<INDEXED>(w, at, g, chunks);
        else
            return uint2(low_plane::template fetch<INDEXED>(w, at.x, g, chunks),
                chunk_plane<HIGH>::template fetch<INDEXED>(w, at.y, g, chunks));
    }
    static words spread(fields codes) {
        if constexpr (HIGH == 0u)
            return low_plane::spread(codes);
        else
            return uint2(low_plane::spread(codes.x), chunk_plane<HIGH>::spread(codes.y));
    }
    template <bool INDEXED, typename W>
    static words load(thread const Weights<W> &w, place at, uint g, uint chunks = 0u) {
        return spread(fetch<INDEXED>(w, at, g, chunks));
    }
    static uint pair(words codes, uint p) {
        uint stored;
        if constexpr (HIGH == 0u)
            stored = low_plane::pair(codes, p);
        else
            stored = low_plane::pair(codes.x, p) | (chunk_plane<HIGH>::pair(codes.y, p) << LOW);
        if constexpr (ZERO != Z)
            return stored + uint(ZERO - Z) * 0x00010001u;
        else
            return stored;
    }
    // Its fetched fields spread moved on by `positions` positions.
    static words moved(fields codes, uint positions) {
        if constexpr (HIGH == 0u)
            return low_plane::moved(codes, positions);
        else
            return uint2(low_plane::moved(codes.x, positions), chunk_plane<HIGH>::moved(codes.y, positions));
    }
};

// Linear codes of a plane of bytes, MASK flipped (two's complement bytes
// become offset binary), entering at their zero.
template <uint MASK, int Z, int ZERO>
struct lane_codes<packets::Planes<8, 0>, packets::Linear<MASK, Z>, ZERO> {
    typedef uint place;
    typedef uint2 words;
    static constexpr bool implemented() { return ZERO == Z; }
    static constexpr int zero() { return Z; }
    static constexpr bool bytes() { return true; }
    static constexpr bool byte_words() { return true; }
    static uint2 byte_codes(words codes) { return codes; }
    template <typename W>
    static place at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, uint k) {
        return uint(row.geometry.at(row.row, w.layout.codes, 32ul, 0ul) - w.base) + k * 8u;
    }
    // A row tile's rows of one block are adjacent: 32 bytes a row.
    template <bool INDEXED, typename W>
    static words load(thread const Weights<W> &w, place at, uint g, uint chunks = 0u) {
        if constexpr (INDEXED)
            return slab::as<uint2>(w.base + at)[g * 4u * w.layout.tile + chunks] ^ MASK;
        else
            return *slab::as<uint2>(w.base + at + g * 32u * w.layout.tile + chunks * 8u) ^ MASK;
    }
    static uint pair(words codes, uint p) { return chunk_byte_pair(codes, p); }
    static words shift(words codes, uint positions) { return codes >> (8u * positions); }
};

// A codebook's 4-bit codes: the lane's four bytes of codes become its eight
// values + 128 (`codebooks::pairs`, each code byte's two values), a byte
// each.
template <uint ID, int ZERO>
struct lane_codes<packets::Planes<4, 0>, packets::Table<ID>, ZERO> {
    typedef uint place;
    typedef uint2 words;
    static constexpr bool implemented() { return ZERO == 128; }
    static constexpr int zero() { return 128; }
    static constexpr bool bytes() { return true; }
    static constexpr bool byte_words() { return true; }
    static uint2 byte_codes(words codes) { return codes; }
    template <typename W>
    static place at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, uint k) {
        return chunk_plane<4>::at(w, row, w.layout.codes, k);
    }
    template <bool INDEXED, typename W>
    static words load(thread const Weights<W> &w, place at, uint g, uint chunks = 0u) {
        constant ushort *pairs = codebooks::pairs<ID>::values();
        const uint codes = chunk_plane<4>::template load<INDEXED>(w, at, g, chunks);
        return uint2(uint(pairs[codes & 0xffu]) | (uint(pairs[(codes >> 8) & 0xffu]) << 16),
            uint(pairs[(codes >> 16) & 0xffu]) | (uint(pairs[codes >> 24]) << 16));
    }
    static uint pair(words codes, uint p) { return chunk_byte_pair(codes, p); }
    static words shift(words codes, uint positions) { return codes >> (8u * positions); }
};

// Grid codes: a chunk is one entry (`packets::Entries::chunk_entry`), whose
// eight values (`packets::Grid`: its points, signed by its sign bits, then
// moved by its delta) enter at ZERO. A lane's place is its entry's in the
// packet units of the grids plane and of the grid high plane, its shifts
// there (`packets::EntryChunk`) and the entry. Without a delta an entry's
// values + 128 are its words' bytes, formed four at a time when it loads: a
// two's complement point's bytes with their sign bits flipped, or its
// magnitudes signed as bytes (`packets::signed_bytes`). With a delta a pair
// is formed at each step, 2^DELTA_EXPONENT (value + 128) moved by the
// delta, in halves that stay below 2^16.
template <uint BYTES, uint HIGH, uint GRID, uint INDEX_BITS, uint LANES, uint SIGNS, bool DELTA,
    uint DELTA_EXPONENT, uint SIGN_BIT, uint SIGN_BITS, uint DELTA_BIT, int ZERO>
struct lane_codes<packets::Entries<BYTES, HIGH>,
    packets::Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT>, ZERO> {
    typedef packets::Entries<BYTES, HIGH> E;
    typedef packets::Grid<GRID, INDEX_BITS, LANES, SIGNS, DELTA, DELTA_EXPONENT, SIGN_BIT, SIGN_BITS, DELTA_BIT> G;
    typedef uint4 place;
    // The entry's bytes (its values + 128 without a delta; its points with
    // one), its sign bits with a delta (columns 0..3 at bits 0.., 4..7 at
    // bits 16..), and what moves a pair to the values + ZERO, in both halves
    // of a word.
    struct words {
        uint2 points;
        uint signs;
        uint offset;
    };
    static constexpr bool implemented() { return true; }
    static constexpr int zero() { return ZERO; }
    static constexpr bool bytes() { return false; }
    // Its values + 128 as bytes, without a delta.
    static constexpr bool byte_words() { return !DELTA && ZERO == 128; }
    static uint2 byte_codes(thread const words &codes) { return codes.points; }
    template <typename W>
    static place at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, uint k) {
        const packets::EntryChunk chunk = E::chunk_of(k);
        return uint4(uint(row.geometry.at(row.row, w.layout.codes, ulong(E::unit_bytes()), 0ul) - w.base) + chunk.offset,
            uint(row.geometry.at(row.row, w.layout.high, ulong(E::high_bytes()), 0ul) - w.base),
            chunk.shift | (chunk.high_shift << 8), k);
    }
    // Addressed by sums either way (INDEXED: the consumer's choice for
    // planes).
    template <bool INDEXED, typename W>
    static words load(thread const Weights<W> &w, place at, uint g, uint chunks = 0u) {
        // The entry `chunks` on: its offset from this one's, and its shifts.
        const packets::EntryChunk later = E::chunk_of(at.w + chunks);
        const uint offset = later.offset - E::chunk_of(at.w).offset;
        const uint shifts = chunks == 0u ? at.z : later.shift | (later.high_shift << 8);
        const uint entry = E::chunk_entry(w.base + at.x + offset + g * E::unit_bytes() * w.layout.tile,
            w.base + at.y + g * E::high_bytes() * w.layout.tile, packets::EntryChunk{0u, shifts & 0xffu, shifts >> 8});
        words codes;
        const uint2 points = G::points::point(entry);
        codes.signs = 0u;
        if constexpr (DELTA) {
            codes.points = points;
            if constexpr (!G::signed_points()) {
                const uint signs = G::signs(entry);
                codes.signs = (signs & 15u) | ((signs & 0xf0u) << 12);
            }
            const int delta = ((entry >> DELTA_BIT) & 1u) != 0u ? -1 : 1;
            codes.offset = uint(ZERO + delta - (128 << DELTA_EXPONENT)) * 0x00010001u;
        } else {
            if constexpr (G::signed_points()) {
                codes.points = points ^ 0x80808080u;
            } else {
                const uint signs = G::signs(entry);
                codes.points = uint2(packets::signed_bytes(points.x, signs & 15u),
                    packets::signed_bytes(points.y, signs >> 4));
            }
            codes.offset = uint(ZERO - 128) * 0x00010001u;
        }
        return codes;
    }
    static uint pair(thread const words &codes, uint p) {
        const uint pair = chunk_byte_pair(codes.points, p);
        if constexpr (DELTA) {
            uint shifted;
            if constexpr (G::signed_points())
                shifted = pair ^ 0x00800080u;
            else {
                const uint negative = (codes.signs >> p) & 0x00010001u;
                shifted = pair + 0x00800080u - ((pair << 1) & ((negative << 16) - negative));
            }
            return (shifted << DELTA_EXPONENT) + codes.offset;
        } else if constexpr (ZERO != 128) {
            return pair + codes.offset;
        } else {
            return pair;
        }
    }
    static words shift(words codes, uint positions) {
        codes.points >>= 8u * positions;
        codes.signs >>= positions;
        return codes;
    }
};

// A lane's pairs of chunk codes (`lane_codes`) less their zero as exact
// halves, the operands of the matrix GEMV (`pair(p)`: codes p and p + 4).
// Codes of bytes (`byte_words()`) are taken when they load, a word at a
// time, four codes by two half subtractions (`packets::byte_floats`); others
// a pair at each step (`packets::code_pair`), the same arithmetic over fewer
// registers. An integer conversion costs several times a half subtraction,
// and the matrix GEMV at M = 8 is bound by its lanes' arithmetic.
template <typename L, bool BYTES = L::byte_words()>
struct lane_operands {
    typename L::words codes;
    static lane_operands of(thread const typename L::words &codes) { return lane_operands{codes}; }
    half2 pair(uint p) const { return as_type<half2>(L::pair(codes, p) | 0x64006400u) - half2(half(1024 + L::zero())); }
};
template <typename L>
struct lane_operands<L, true> {
    half2 pairs[4];
    static lane_operands of(thread const typename L::words &codes) {
        const half2 zero = half2(half(1024 + L::zero()));
        const uint2 bytes = L::byte_codes(codes);
        const half2 low_even = as_type<half2>((bytes.x & 0x00ff00ffu) | 0x64006400u) - zero;
        const half2 low_odd = as_type<half2>(((bytes.x >> 8) & 0x00ff00ffu) | 0x64006400u) - zero;
        const half2 high_even = as_type<half2>((bytes.y & 0x00ff00ffu) | 0x64006400u) - zero;
        const half2 high_odd = as_type<half2>(((bytes.y >> 8) & 0x00ff00ffu) | 0x64006400u) - zero;
        lane_operands operands;
        operands.pairs[0] = half2(low_even.x, high_even.x);
        operands.pairs[1] = half2(low_odd.x, high_odd.x);
        operands.pairs[2] = half2(low_even.y, high_even.y);
        operands.pairs[3] = half2(low_odd.y, high_odd.y);
        return operands;
    }
    half2 pair(uint p) const { return pairs[p]; }
};

// Weights with a matrix GEMV whose products fold a block's local scale codes
// into the left operand: unsigned codes (`lane_codes`) under hierarchical
// coefficients with a minimum, an unsigned local scale code per 32 columns
// and factors per 256, which the runtime admits where a code times a scale
// code stays below 2048 (an exact F16 integer of the operand's 2^-24). `at`
// is a lane's place in its row's planes (fragment column x holds chunk
// x / 2), `load` its words of block g, `pair` its two codes of step s in the
// halves of one word.
template <typename W>
struct matrix_codes {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename FS, int EXPONENT, uint LOCAL_BITS, int BIAS_SIGN, typename F>
struct matrix_codes<packets::Packet<P, packets::Unsigned,
    packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, packets::LocalUnsigned<LOCAL_BITS>, 32u, true, BIAS_SIGN, 0u>, F>> {
    typedef packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, packets::LocalUnsigned<LOCAL_BITS>, 32u, true, BIAS_SIGN, 0u> K;
    typedef lane_codes<P, packets::Unsigned> lanes;
    static constexpr bool implemented() { return lanes::implemented(); }
    typedef typename lanes::place place;
    typedef typename lanes::words words;
    template <typename W>
    static place at(thread const Weights<W> &w, thread const typename Weights<W>::located &row, uint x) {
        return lanes::at(w, row, x / 2u);
    }
    template <typename W>
    static words load(thread const Weights<W> &w, place at, uint g) { return lanes::template load<true>(w, at, g); }
    static uint pair(words codes, uint s) { return lanes::pair(codes, s); }
};

// The weights the matrix GEMV over a block's runs serves: integer codes
// (`lane_codes`) under hierarchical coefficients without a bias, a local
// scale code per GROUP (16 or 32) columns and factors per 256
// (`gemv_matrix_runs_body`): a lane's chunk lies in one group.
template <typename W>
struct matrix_runs {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename C, typename FS, int EXPONENT, uint LOCAL_BITS, typename L, uint GROUP, int BIAS_SIGN,
    typename F>
struct matrix_runs<packets::Packet<P, C, packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, L, GROUP, false, BIAS_SIGN, 0u>,
    F>> {
    typedef packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, L, GROUP, false, BIAS_SIGN, 0u> K;
    typedef L local;
    typedef lane_codes<P, C> lanes;
    static constexpr bool implemented() { return lanes::implemented(); }
    // The group of a packet's chunk k.
    static uint group_of(uint k) { return k / (GROUP / 8u); }
    // Whether a code times its scale code is an exact F16 operand.
    static constexpr bool halves() { return F::matrix_runs_halves(); }
};

// The weights the symmetric matrix GEMV serves: integer codes
// (`lane_codes`) under a direct scale per 32 columns or per a multiple of
// them, which every 32-column block of a group repeats
// (`gemv_matrix_symmetric_body`).
template <typename W>
struct matrix_symmetric {
    static constexpr bool implemented() { return false; }
};
template <typename P, typename C, typename S, uint GROUP, uint STORAGE, typename F>
struct matrix_symmetric<packets::Packet<P, C, packets::Direct<S, GROUP, false, STORAGE, 0>, F>> {
    typedef packets::Direct<S, GROUP, false, STORAGE, 0> K;
    typedef lane_codes<P, C> lanes;
    static constexpr bool implemented() { return lanes::implemented() && GROUP % 32u == 0u; }
};

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

// The matrix GEMV over a block's runs (`matrix_runs`): a lane's two matrix
// columns of a step lie in one run of its row, so its left elements are
// (code - zero) times the run's local scale code as integers, F16 where the
// runtime admits them exact in it (`matrix_runs::halves`) and F32 otherwise,
// the differences exact halves (`lane_operands`); the
// accumulator runs over a 256-column block and folds once with the block's
// factor. No min term and nothing staged. UNIT is `matrix_unit` of the
// operand.
template <typename W, uint NB, uint PARTS, bool UNIT, typename In, typename Out>
inline void gemv_matrix_runs_body(thread const In &in, thread const Out &out,
    thread const Weights<W> &w, uint m_rows, uint rows, uint k, uint group, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    typedef typename matrix_runs<W>::K K;
    typedef typename matrix_runs<W>::local L;
    typedef typename matrix_runs<W>::lanes lanes;
    typedef typename In::activation A;
    constexpr bool halves = matrix_runs<W>::halves();
    typedef metal::conditional_t<halves, half2, float2> operand;
    constexpr uint RUN = matrix_run<NB>();
    threadgroup float2 *parts = reinterpret_cast<threadgroup float2 *>(shared);
    const ushort2 at = fragment_coordinate(lane);
    const uint index = group * simdgroups + sg, piece = index / PARTS, part = index % PARTS;
    const uint n = piece * 8u + at.y;
    const typename Weights<W>::located row = w.locate(min(n, rows - 1u));
    const typename lanes::place codes = lanes::at(w, row, at.x / 2u);
    const uint run_of = matrix_runs<W>::group_of(at.x / 2u);
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
        // The row's local scale codes of the block (the lane's run of packet
        // b is its chunk's group) and its factor.
        const typename K::fields::words scales = K::group_fields(row.row, row.geometry, block);
        const float d = K::factors(row.row, row.geometry, block).scale(0);
        _Pragma("clang loop unroll(disable)")
        for (uint run = 0; run < 8; run += RUN) {
            PROJECTION_UNROLL
            for (uint i = 0; i < RUN; ++i) {
                const uint b = run + i;
                const uint g = block * 8u + b;
                const float scale = L::value(K::scale_code(K::field_of(scales, b), run_of));
                const typename lanes::words words = lanes::template load<true>(w, codes, g);
                const lane_operands<lanes> pairs = lane_operands<lanes>::of(words);
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
                    metal::conditional_t<halves, simdgroup_half8x8, simdgroup_float8x8> left;
                    reinterpret_cast<thread operand &>(left.thread_elements()) = operand(pairs.pair(s)) * operand(scale);
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

// The symmetric 32-column matrix GEMV (`matrix_symmetric`): decode each code
// once into an exact integer matrix operand (`lane_operands`), share it
// across the activation rows, then fold the block scale into the F32 sums. No rounded weight
// operand or token packing.
template <bool PAIRED, uint NB, uint PARTS, bool UNIT, typename W, typename U, typename In, typename Out>
inline void gemv_matrix_symmetric_body(thread const In &in, thread const Out &out,
    thread const Weights<W> &w, thread const Weights<U> &u,
    uint m_rows, uint rows, uint k, uint group, threadgroup uchar *shared,
    uint simdgroups, uint sg, uint lane) {
    typedef typename In::activation A;
    threadgroup float2 *parts = reinterpret_cast<threadgroup float2 *>(shared);
    const ushort2 at = fragment_coordinate(lane);
    const uint index = group * simdgroups + sg, piece = index / PARTS, part = index % PARTS;
    const uint n = piece * 8u + at.y;
    typedef typename matrix_symmetric<W>::lanes lanes;
    typedef typename matrix_symmetric<U>::lanes up_lanes;
    const auto row = w.locate(min(n, rows - 1u));
    const typename lanes::place codes = lanes::at(w, row, at.x / 2u);
    const auto up_row = u.locate(min(n, rows - 1u));
    const typename up_lanes::place up_codes = up_lanes::at(u, up_row, at.x / 2u);
    packets::weight_bytes supers = row.geometry.super_at(row.row, 0ul);
    packets::weight_bytes up_supers = up_row.geometry.super_at(up_row.row, 0ul);
    const uint blocks = k / 32u;
    device const typename A::storage *stored_rows[NB][2];
    float2 y[NB], y_up[NB];
    PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
        y[f] = float2(0.0f);
        if constexpr (PAIRED) y_up[f] = float2(0.0f);
        PROJECTION_UNROLL for (uint j = 0; j < 2; ++j)
            stored_rows[f][j] = reinterpret_cast<device const typename A::storage *>(in.x)
                + ulong(in.rows.at(min(8u * f + at.x + j, m_rows - 1u))) * in.stride0;
    }
    for (uint g = part * blocks / PARTS; g < (part + 1u) * blocks / PARTS; ++g) {
        const typename lanes::words word = lanes::template load<true>(w, codes, g);
        const lane_operands<lanes> pairs = lane_operands<lanes>::of(word);
        lane_operands<up_lanes> up_pairs;
        const float d = matrix_symmetric<W>::K::at(supers, g).scale(0);
        typename up_lanes::words up_word;
        float up_d;
        if constexpr (PAIRED) {
            up_word = up_lanes::template load<true>(u, up_codes, g);
            up_pairs = lane_operands<up_lanes>::of(up_word);
            up_d = matrix_symmetric<U>::K::at(up_supers, g).scale(0);
        }
        uint2 stored[NB][2];
        simdgroup_float8x8 acc[NB], acc_up[NB];
        PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
            PROJECTION_UNROLL for (uint j = 0; j < 2; ++j)
                stored[f][j] = matrix_stored<UNIT>(in, stored_rows[f][j], 32u * g,
                    8u * (at.y / 2u) + 4u * (at.y & 1u));
        }
        PROJECTION_UNROLL for (uint s = 0; s < 4; ++s) {
            simdgroup_half8x8 left, up_left;
            reinterpret_cast<thread half2 &>(left.thread_elements()) = pairs.pair(s);
            if constexpr (PAIRED) {
                reinterpret_cast<thread half2 &>(up_left.thread_elements()) = up_pairs.pair(s);
            }
            PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
                simdgroup_float8x8 right;
                reinterpret_cast<thread float2 &>(right.thread_elements()) = float2(
                    matrix_value<A>(stored[f][0], s), matrix_value<A>(stored[f][1], s));
                if (s == 0) simdgroup_multiply(acc[f], left, right);
                else simdgroup_multiply_accumulate(acc[f], left, right, acc[f]);
                if constexpr (PAIRED) {
                    if (s == 0) simdgroup_multiply(acc_up[f], up_left, right);
                    else simdgroup_multiply_accumulate(acc_up[f], up_left, right, acc_up[f]);
                }
            }
        }
        PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
            y[f] = metal::fma(float2(d), reinterpret_cast<thread float2 &>(acc[f].thread_elements()), y[f]);
            if constexpr (PAIRED)
                y_up[f] = metal::fma(float2(up_d), reinterpret_cast<thread float2 &>(acc_up[f].thread_elements()), y_up[f]);
        }
    }
    if constexpr (PAIRED) {
        const bool owner = matrix_exchange<NB, PARTS>(y, parts, part, sg, lane);
        matrix_exchange<NB, PARTS>(y_up, parts, part, sg, lane);
        if (!owner) return;
    } else if (!matrix_exchange<NB, PARTS>(y, parts, part, sg, lane)) return;
    if (n >= rows) return;
    PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
        if (8u * f + at.x < m_rows)
            emit<PAIRED>::run(out, 8u * f + at.x, n, y[f].x, PAIRED ? y_up[f].x : 0.0f);
        if (8u * f + at.x + 1u < m_rows)
            emit<PAIRED>::run(out, 8u * f + at.x + 1u, n, y[f].y, PAIRED ? y_up[f].y : 0.0f);
    }
}

// NB activation fragments (8 rows each) against the eight weight rows of
// fragment `(group * simdgroups + sg) / PARTS`. UNIT is `matrix_unit` of the
// operand and STAGED whether the block sums fit the threadgroup memory
// (`gemv_matrix_shared_bytes`): both are the same at every block of a call,
// and a block loop that tests them compiles both sides of each test around
// every load.
template <typename W, typename U, bool PAIRED, uint NB, uint PARTS, bool UNIT, bool STAGED, bool PREPARED, typename In, typename Out>
inline void gemv_matrix_products(thread const In &in, thread const Out &out, thread const Weights<W> &w, thread const Weights<U> &u, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    typedef matrix_codes<W> coded;
    typedef matrix_codes<U> coded_up;
    typedef typename coded::K K;
    typedef typename coded_up::K K_up;
    typedef typename In::activation A;
    constexpr uint RUN = matrix_run<NB>();
    threadgroup float2 *parts = reinterpret_cast<threadgroup float2 *>(shared);
    const ushort2 at = fragment_coordinate(lane);
    const uint index = group * simdgroups + sg, piece = index / PARTS, part = index % PARTS;
    const uint n = piece * 8u + at.y;
    const typename Weights<W>::located row = w.locate(min(n, rows - 1u));
    const typename coded::place codes = coded::at(w, row, at.x);
    typename Weights<U>::located up_row;
    typename coded_up::place up_codes;
    if constexpr (PAIRED) {
        up_row = u.locate(min(n, rows - 1u));
        up_codes = coded_up::at(u, up_row, at.x);
    }
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
    float2 y[NB], y_up[NB];
    simdgroup_float8x8 acc[NB], mins[NB], acc_up[NB], mins_up[NB];
    PROJECTION_UNROLL
    for (uint f = 0; f < NB; ++f) {
        y[f] = float2(0.0f);
        if constexpr (PAIRED)
            y_up[f] = float2(0.0f);
    }
    // The activation rows' block sums, staged once per threadgroup behind
    // the exchange where they fit (block_sums[(8 NB) g + m]); a lane sums its
    // own otherwise. Either way a sum is `matrix_block_sum`.
    threadgroup float *block_sums = reinterpret_cast<threadgroup float *>(parts + simdgroups * 32u * NB);
    if constexpr (STAGED && !PREPARED) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = sg * 32u + lane; item < 8u * NB * (k / 32u); item += simdgroups * 32u) {
            const uint row_m = item % (8u * NB);
            block_sums[item] = row_m < m_rows ? matrix_block_sum<UNIT>(in, row_m, item / (8u * NB)) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint block = part * blocks / PARTS; block < (part + 1u) * blocks / PARTS; ++block) {
        const typename K::fields::words fields = K::group_fields(row.row, row.geometry, block);
        const typename K::group_factors factors = K::factors(row.row, row.geometry, block);
        typename K_up::fields::words up_fields;
        typename K_up::group_factors up_factors;
        if constexpr (PAIRED) {
            up_fields = K_up::group_fields(up_row.row, up_row.geometry, block);
            up_factors = K_up::factors(up_row.row, up_row.geometry, block);
        }
        _Pragma("clang loop unroll(disable)")
        for (uint run = 0; run < 8; run += RUN) {
            PROJECTION_UNROLL
            for (uint i = 0; i < RUN; ++i) {
                const uint b = run + i;
                const uint g = block * 8u + b;
                const uint scale = K::scale_code(K::field_of(fields, b), 0u);
                const typename coded::words words = coded::load(w, codes, g);
                uint up_scale;
                typename coded_up::words up_words;
                if constexpr (PAIRED) {
                    up_scale = K_up::scale_code(K_up::field_of(up_fields, b), 0u);
                    up_words = coded_up::load(u, up_codes, g);
                }
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
                    simdgroup_half8x8 up_left;
                    if constexpr (PAIRED)
                        reinterpret_cast<thread half2 &>(up_left.thread_elements()) =
                            as_type<half2>(coded_up::pair(up_words, s) * up_scale);
                    PROJECTION_UNROLL
                    for (uint f = 0; f < NB; ++f) {
                        simdgroup_float8x8 right;
                        reinterpret_cast<thread float2 &>(right.thread_elements()) = float2(
                            matrix_value<A>(stored[f][0], s), matrix_value<A>(stored[f][1], s));
                        if (b == 0 && s == 0)
                            simdgroup_multiply(acc[f], left, right);
                        else
                            simdgroup_multiply_accumulate(acc[f], left, right, acc[f]);
                        if constexpr (PAIRED) {
                            if (b == 0 && s == 0)
                                simdgroup_multiply(acc_up[f], up_left, right);
                            else
                                simdgroup_multiply_accumulate(acc_up[f], up_left, right, acc_up[f]);
                        }
                    }
                }
            }
        }
        simdgroup_half8x8 left;
        reinterpret_cast<thread half2 &>(left.thread_elements()) = as_type<half2>(
            K::min_code(K::field_of(fields, at.x), 0u) | (K::min_code(K::field_of(fields, at.x + 1u), 0u) << 16));
        simdgroup_half8x8 up_left;
        if constexpr (PAIRED)
            reinterpret_cast<thread half2 &>(up_left.thread_elements()) = as_type<half2>(
                K_up::min_code(K_up::field_of(up_fields, at.x), 0u)
                | (K_up::min_code(K_up::field_of(up_fields, at.x + 1u), 0u) << 16));
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
            y[f] = metal::fma(float2(factors.scale(0) * 0x1p24f), reinterpret_cast<thread float2 &>(acc[f].thread_elements()),
                metal::fma(float2(factors.bias(0) * 0x1p24f), reinterpret_cast<thread float2 &>(mins[f].thread_elements()),
                    y[f]));
            if constexpr (PAIRED) {
                simdgroup_multiply(mins_up[f], up_left, right);
                y_up[f] = metal::fma(float2(up_factors.scale(0) * 0x1p24f), reinterpret_cast<thread float2 &>(acc_up[f].thread_elements()),
                    metal::fma(float2(up_factors.bias(0) * 0x1p24f), reinterpret_cast<thread float2 &>(mins_up[f].thread_elements()), y_up[f]));
            }
        }
    }
    if constexpr (PAIRED) {
        const bool owner = matrix_exchange<NB, PARTS>(y, parts, part, sg, lane);
        matrix_exchange<NB, PARTS>(y_up, parts, part, sg, lane);
        if (!owner)
            return;
    } else if constexpr (PARTS > 1) {
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
            emit<PAIRED>::run(out, 8u * f + at.x, n, y[f].x, PAIRED ? y_up[f].x : 0.0f);
        if (8u * f + at.x + 1u < m_rows)
            emit<PAIRED>::run(out, 8u * f + at.x + 1u, n, y[f].y, PAIRED ? y_up[f].y : 0.0f);
    }
}

template <typename W, uint NB, uint PARTS, bool UNIT, bool STAGED, bool PREPARED = false, typename In, typename Out>
inline void gemv_matrix_body(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    gemv_matrix_products<W, W, false, NB, PARTS, UNIT, STAGED, PREPARED>(in, out, w, w,
        m_rows, rows, k, group, shared, simdgroups, sg, lane);
}

// The matrix GEMV products the runtime admits weights W to
// (`packets::Forms`, `_FORM_MATRIX_*`, at most one): none by default.
template <typename W>
struct matrix_admitted {
    static constant constexpr bool codes = false, runs = false, symmetric = false;
};
template <typename P, typename C, typename K, typename F>
struct matrix_admitted<packets::Packet<P, C, K, F>> {
    static constant constexpr bool codes = F::matrix_codes, runs = F::matrix_runs, symmetric = F::matrix_symmetric;
};

// The matrix GEMV of weights W (`available`), by the product the runtime
// admits it to: the products folding a block's scale codes (`matrix_codes`),
// the runs of a block (`matrix_runs`) or the symmetric 32 columns
// (`matrix_symmetric`), each asserted to serve W. `run` takes the body for
// the operand's strides and for whether the block sums are staged, once per
// call.
template <typename W, bool CODES = matrix_admitted<W>::codes, bool RUNS = matrix_admitted<W>::runs,
    bool SYMMETRIC = matrix_admitted<W>::symmetric>
struct matrix_form;
template <typename W>
struct matrix_form<W, false, false, false> {
    static constant constexpr bool available = false;
    static constant constexpr bool symmetric32 = false;
    static constant constexpr bool stages_block_sums = false;
};
template <typename W>
struct matrix_form<W, true, false, false> {
    static_assert(matrix_codes<W>::implemented(), "an admitted weight has the matrix GEMV folding its scale codes");
    static constant constexpr bool available = true;
    static constant constexpr bool symmetric32 = false;
    static constant constexpr bool stages_block_sums = true;
    template <uint NB, uint PARTS, bool PREPARED = false, typename In, typename Out>
    static void run(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows, uint rows,
        uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
        const bool staged = gemv_matrix_shared_bytes(k, NB, simdgroups) <= gemv_matrix_staged_bytes;
        if (matrix_unit(in)) {
            if (staged)
                gemv_matrix_body<W, NB, PARTS, true, true, PREPARED>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                    sg, lane);
            else
                gemv_matrix_body<W, NB, PARTS, true, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                    sg, lane);
        } else if (staged) {
            gemv_matrix_body<W, NB, PARTS, false, true, PREPARED>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg,
                lane);
        } else {
            gemv_matrix_body<W, NB, PARTS, false, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups,
                sg, lane);
        }
    }
};
template <typename W>
struct matrix_form<W, false, true, false> {
    static_assert(matrix_runs<W>::implemented(), "an admitted weight has the matrix GEMV over its runs");
    static constant constexpr bool available = true;
    static constant constexpr bool symmetric32 = false;
    static constant constexpr bool stages_block_sums = false;
    template <uint NB, uint PARTS, bool PREPARED = false, typename In, typename Out>
    static void run(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
        uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
        if (matrix_unit(in))
            gemv_matrix_runs_body<W, NB, PARTS, true>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
        else
            gemv_matrix_runs_body<W, NB, PARTS, false>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg,
                lane);
    }
};
template <typename W>
struct matrix_form<W, false, false, true> {
    static_assert(matrix_symmetric<W>::implemented(), "an admitted weight has the symmetric matrix GEMV");
    static constant constexpr bool available = true;
    static constant constexpr bool stages_block_sums = false;
    static constant constexpr bool symmetric32 = true;
    template <uint NB, uint PARTS, bool PREPARED = false, typename In, typename Out>
    static void run(thread const In &in, thread const Out &out, thread const Weights<W> &w,
        uint m_rows, uint rows, uint k, uint group, threadgroup uchar *shared,
        uint simdgroups, uint sg, uint lane) {
        if (matrix_unit(in))
            gemv_matrix_symmetric_body<false, NB, PARTS, true>(in, out, w, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
        else
            gemv_matrix_symmetric_body<false, NB, PARTS, false>(in, out, w, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
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

// Capture before gathering: a partial activation fragment must not perform
// shuffles from inside the matrix body's conditional store callback.
struct matrix_column_values {
    thread float2 *values;
    void store(uint m, uint, float value) const {
        values[m / 8u][m & 1u] = value;
    }
};

template <typename W, uint PARTS, uint NB, typename In, typename Out>
inline void gemv_matrix_columns_body(thread const In &in, thread const Out &out,
    thread const Weights<W> &w, uint m_rows, uint rows, uint k, uint group,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    float2 values[NB];
    PROJECTION_UNROLL for (uint f = 0; f < NB; ++f)
        values[f] = float2(0.0f);
    const matrix_column_values capture{values};
    matrix_form<W>::template run<NB, PARTS>(in, capture, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
    // Only the first split owns the reduced sums; this branch is uniform
    // within the simdgroup. Gather before testing channel or row bounds.
    const uint index = group * simdgroups + sg;
    if (index % PARTS != 0u)
        return;
    const ushort2 at = fragment_coordinate(lane);
    float column[8 * NB];
    PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
        PROJECTION_UNROLL for (ushort x = 0; x < 8; x += 2) {
            float2 pair = simd_shuffle(values[f], fragment_lane(x, at.y));
            column[8u * f + x] = pair.x;
            column[8u * f + x + 1u] = pair.y;
        }
    }
    const uint n = index / PARTS * 8u + at.y;
    if (n >= rows)
        return;
    PROJECTION_UNROLL for (uint f = 0; f < NB; ++f) {
        const uint m = 8u * f + at.x;
        if (m < m_rows)
            out.store_column(m, n, column);
        if (m + 1u < m_rows)
            out.store_column(m + 1u, n, column);
    }
}

template <typename W, uint PARTS, typename In, typename Out>
inline void gemv_matrix_columns(thread const In &in, thread const Out &out,
    thread const Weights<W> &w, uint m_rows, uint rows, uint k, uint group,
    threadgroup uchar *shared, uint simdgroups, uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_matrix_columns_body<W, PARTS, 1>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
    else
        gemv_matrix_columns_body<W, PARTS, 2>(in, out, w, m_rows, rows, k, group, shared, simdgroups, sg, lane);
}

// A matrix lane owns two adjacent activation rows in each eight-row fragment.
// Retain the gate sums in its registers, then combine the matching up sums
// through the original paired epilogue (including its scales and rounding).
struct matrix_pair_gate {
    thread float2 *values;
    void store(uint m, uint, float value) const {
        values[m / 8u][m & 1u] = value;
    }
};

template <typename Out>
struct matrix_pair_up {
    Out out;
    thread const float2 *gate;
    void store(uint m, uint n, float value) const {
        out.store_pair(m, n, gate[m / 8u][m & 1u], value);
    }
};

// The split exchange occupies the prefix of shared memory. The block sums of
// products folding scale codes (`matrix_codes`) live after it and survive
// both products, so the up product may reuse exactly the sums staged by the
// gate. Each product keeps its own original accumulation and split reduction
// order; the runs' GEMV does not stage block sums.
template <typename G, typename U, uint PARTS, uint NB, typename In, typename Out>
inline void gemv_matrix_paired_body(thread const In &in, thread const Out &out,
    thread const Weights<G> &gate, thread const Weights<U> &up, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if constexpr (matrix_form<G>::symmetric32 && matrix_form<U>::symmetric32) {
        if (matrix_unit(in))
            gemv_matrix_symmetric_body<true, NB, PARTS, true>(in, out, gate, up, m_rows, rows, k, group, shared, simdgroups, sg, lane);
        else
            gemv_matrix_symmetric_body<true, NB, PARTS, false>(in, out, gate, up, m_rows, rows, k, group, shared, simdgroups, sg, lane);
        return;
    }
    // The two products share each loaded and decoded activation fragment.
    // Each accumulator keeps the single-product instruction order.
    if constexpr (matrix_form<G>::stages_block_sums && matrix_form<U>::stages_block_sums) {
        const bool staged = gemv_matrix_shared_bytes(k, NB, simdgroups) <= gemv_matrix_staged_bytes;
        if (matrix_unit(in)) {
            if (staged)
                gemv_matrix_products<G, U, true, NB, PARTS, true, true, false>(in, out, gate, up,
                    m_rows, rows, k, group, shared, simdgroups, sg, lane);
            else
                gemv_matrix_products<G, U, true, NB, PARTS, true, false, false>(in, out, gate, up,
                    m_rows, rows, k, group, shared, simdgroups, sg, lane);
        } else if (staged) {
            gemv_matrix_products<G, U, true, NB, PARTS, false, true, false>(in, out, gate, up,
                m_rows, rows, k, group, shared, simdgroups, sg, lane);
        } else {
            gemv_matrix_products<G, U, true, NB, PARTS, false, false, false>(in, out, gate, up,
                m_rows, rows, k, group, shared, simdgroups, sg, lane);
        }
        return;
    }
    float2 gate_sums[NB];
    const matrix_pair_gate first{gate_sums};
    const matrix_pair_up<Out> second{out, gate_sums};
    matrix_form<G>::template run<NB, PARTS>(in, first, gate, m_rows, rows, k, group, shared, simdgroups, sg, lane);
    matrix_form<U>::template run<NB, PARTS, matrix_form<G>::stages_block_sums>(in, second, up,
        m_rows, rows, k, group, shared, simdgroups, sg, lane);
}

template <typename G, typename U, uint PARTS, typename In, typename Out>
inline void gemv_matrix_paired(thread const In &in, thread const Out &out,
    thread const Weights<G> &gate, thread const Weights<U> &up, uint m_rows,
    uint rows, uint k, uint group, threadgroup uchar *shared, uint simdgroups,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_matrix_paired_body<G, U, PARTS, 1>(in, out, gate, up, m_rows, rows, k, group, shared, simdgroups, sg, lane);
    else
        gemv_matrix_paired_body<G, U, PARTS, 2>(in, out, gate, up, m_rows, rows, k, group, shared, simdgroups, sg, lane);
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
//   biases        16 F16 per (16-row weight tile, 32 blocks, lane): the
//                 block's bias, signed minimum factor * minimum code, the
//                 min-term product's left fragment
//                 (`int8_bias_at`)
struct int8_scratch {
    device uchar *quantized;
    device half *row_scales;
    device half *block_sums;
    device float *coefficients;
    device half *biases;
};

// Weights whose stored codes enter the int8 product (on tensor operations):
// how a lane reads its codes of one row and block, whether the weights have a
// min term, and a row's block scales and biases of the eight blocks of one
// 256-column block: block i's scale is `factor * scale[i]`. `factored`: the
// scales are integers below 256 and the factor is the 256 columns'. The
// product takes the stored codes themselves, so its codes are unsigned 4-bit
// codes under hierarchical coefficients (the local codes integers under the
// factor) or two's complement 8-bit codes under direct ones, a scale per
// block; none otherwise. An INT8 = 1 configuration names the product only
// of weights the runtime admits to it (`packets::Forms::int8`, `_FORM_INT8`,
// a declaration's `form(W, INT8)`), whose codes have one on a device with
// tensor operations (`int8_path`).
template <typename W>
struct int8_admitted {
    static constant constexpr bool value = false;
};
template <typename P, typename C, typename K, typename F>
struct int8_admitted<packets::Packet<P, C, K, F>> {
    static constant constexpr bool value = F::int8;
};
template <typename W>
struct int8_codes {
    static_assert(!SEISMIC_HAS_TENSOR_OPS || !int8_admitted<W>::value,
        "an admitted weight's codes enter the int8 product");
    static constant constexpr bool biased = false;
    static constant constexpr bool factored = false;
    static constant constexpr bool interleaved = false;
    static void coefficients(packets::weight_bytes, Rows16, uint, thread float &, thread float (&)[8],
        thread float (&)[8]) {}
};
#if SEISMIC_HAS_TENSOR_OPS
// Unsigned 4-bit codes: two bytes hold four codes, a nibble each; value =
// factor * scale code * code + bias factor * minimum code.
template <typename FS, int EXPONENT, uint LOCAL_BITS, bool MIN, int BIAS_SIGN, typename F>
struct int8_codes<packets::Packet<packets::Planes<4, 0>, packets::Unsigned,
    packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, packets::LocalUnsigned<LOCAL_BITS>, 32u, MIN, BIAS_SIGN, 0u>, F>> {
    typedef packets::Hier<FS, 256u, EXPONENT, 0, LOCAL_BITS, packets::LocalUnsigned<LOCAL_BITS>, 32u, MIN, BIAS_SIGN, 0u> K;
    // Admitted where a factor of at most 16 bits times a byte code is exact
    // in F32.
    static constant constexpr bool biased = MIN;
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
    static void slot(packets::weight_bytes at, packets::weight_bytes, thread char4 &first, thread char4 &second) {
        const uint word = *slab::as<uint>(at);
        first = as_type<char4>(word & 0x0f0f0f0fu);
        second = as_type<char4>((word >> 4) & 0x0f0f0f0fu);
    }
    // A block's scale is factor * scale code, exact in F32 (a 16-bit
    // factor's significand times a code of at most 8 bits): the factor is
    // the 256 columns' and the scales are the integer codes, a byte each in
    // the scratch.
    static constant constexpr bool factored = true;
    static void coefficients(packets::weight_bytes row, Rows16 layout, uint block, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        typename K::run run = K::start(row, layout, block * K::packets);
        factor = run.factor.scale(0);
        // A block's bias, bias factor * minimum code, is exact in F32.
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            const uint field = K::next(run);
            scale[i] = float(K::scale_code(field, 0u));
            bias[i] = MIN ? run.factor.bias(0) * float(K::min_code(field, 0u)) : 0.0f;
        }
    }
};
// Two's complement 8-bit codes, a scale per block.
template <typename S, uint STORAGE, typename F>
struct int8_codes<packets::Packet<packets::Planes<8, 0>, packets::Twos,
    packets::Direct<S, 32u, false, STORAGE, 0>, F>> {
    typedef packets::Direct<S, 32u, false, STORAGE, 0> K;
    static constant constexpr bool biased = false;
    static constant constexpr uint block_bytes = 32;
    // A lane's eight codes of one row and block are eight bytes, in column
    // order.
    static constant constexpr bool interleaved = false;
    static constant constexpr uint slot_bytes = 8;
    static constant constexpr uint high_block_bytes = 0, high_slot_bytes = 0;
    static void slot(packets::weight_bytes at, packets::weight_bytes, thread char4 &first, thread char4 &second) {
        const uint2 words = *slab::as<uint2>(at);
        first = as_type<char4>(words.x);
        second = as_type<char4>(words.y);
    }
    static constant constexpr bool factored = false;
    static void coefficients(packets::weight_bytes row, Rows16 layout, uint block, thread float &factor,
        thread float (&scale)[8], thread float (&bias)[8]) {
        packets::weight_bytes supers = layout.super_at(row, 0ul);
        factor = 1.0f;
        PROJECTION_UNROLL
        for (uint i = 0; i < 8; ++i) {
            scale[i] = K::at(supers, block * 8u + i).scale(0);
            bias[i] = 0.0f;
        }
    }
};
#endif

// Whether an INT8 launch over the weights W, V... runs the int8 products:
// on a device with tensor operations; elsewhere it takes the staged 64 x 64
// tiles of the default form, with their results. INT8 = 1 names the int8
// GEMM only of weights admitted to it whose codes one quantized operand
// serves (`form(W, INT8)`, `INT8_INTERLEAVED`), which the declaration's
// `where` requires; only a configuration that runs a launch forms it.
// A function, not a class's constant: a class's static data member is a
// constant of the program, which moves what follows it.
template <typename W, typename... V> constexpr bool int8_path() {
    static_assert(int8_admitted<W>::value && (int8_admitted<V>::value && ...),
        "INT8 = 1 names the int8 GEMM of weights admitted to it");
    static_assert(((int8_codes<V>::interleaved == int8_codes<W>::interleaved) && ...),
        "INT8 = 1 names the int8 GEMM of weights one quantized operand serves");
    return SEISMIC_HAS_TENSOR_OPS;
}

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
// (rows / 64, ceil(M / 64)): 64 x 64, a simdgroup a quarter each, where the
// launch runs the int8 products (PATH, `int8_path`); otherwise the staged
// 64 x 64 tiles of the default form, with its results (the TILE 128 grid's
// threadgroups in order take them in order; rows is a multiple of 64).
template <typename W, uint TILE, bool PATH, typename In, typename Out>
inline void gemm_int8(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const int8_scratch &s, uint m_rows, uint rows, uint k, uint gx, uint gy, threadgroup uchar *shared,
    uint sg, uint lane) {
    static_assert(TILE == 128 || TILE == 64, "an int8 tile is 128 x 32 or 64 x 64");
#if SEISMIC_HAS_TENSOR_OPS
    if constexpr (PATH) {
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
// operand, quantized in the order both weights' codes take (`interleaved`),
// and differ in the weights' coefficients and biases. Where the launch does
// not run the int8 products (PATH, `int8_path`), the staged paired 64 x 64
// tile of the default form runs, with its results.
template <typename G, typename U, bool PATH, typename In, typename Out>
inline void gemm_int8_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, thread const int8_scratch &gate_scratch, thread const int8_scratch &up_scratch,
    uint m_rows, uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, threadgroup float *exchange,
    uint sg, uint lane) {
#if SEISMIC_HAS_TENSOR_OPS
    if constexpr (PATH) {
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
