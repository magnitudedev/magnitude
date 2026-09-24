// The packed projection family (K1): y[m, n] = epilogue(sum_k x[m, k] * W[n, k])
// where x is produced by a prologue from the entry's inputs.
//
// Prologues produce one activation value x[m, k], already rounded to the
// activation type A exactly as the entry's portable body publishes it:
//   Plain      x = A[row(m), k]
//   Rms        x = round_A(r[row(m), k] * inverse(m) * norm[k])
//   GatedRms   per value head h = k / W:
//              x = round_A(round_A(mixed * inverse(m, h) * norm[k % W])
//                          * round_A(silu(z[row(m), k])))
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
//   Store<E>   y = round_E(acc)                  (logits: Store<element::F32>)
//   Residual   y = residual[row(m), n] + round_A(acc)                  (F32)
//   SiluMul    y = round_A(round_A(silu(round_A(gate))) * round_A(up))
//
// A segmented projection is one launch over several weight tensors (each
// with its own packet type and destination). Every threadgroup belongs to
// one segment; entries map their threadgroup index to a segment and call
// the GEMV or GEMM body with that segment's weight and epilogue.
//
// Row classes pick the launch:
// - GEMV (M below the entry's BATCH_FROM): lane groups own weight rows, lanes
//   own packets, and each weight packet is decoded once and applied to every
//   activation row with F32 FMAs. The activations are staged once per
//   threadgroup in threadgroup memory (A storage) with per-16-element sums
//   for the factored biases.
// - Batched GEMV (BATCH_FROM <= M <= 16): 8-row weight blocks decoded by the
//   lanes straight into F32 matrix fragments, multiplied with the staged F32
//   activations on the matrix units (the verify shapes of speculative decode).
// - GEMM (M > 16): TM x TN output tiles over simdgroups of 32 x 32 (16 x 32
//   when TM = 32), stepping K by 32. Activations are staged as stored (bf16
//   or f16), weights decoded to half once per tile, and simdgroup_matrix
//   multiplies them into F32 accumulators; small-N outputs may split K.
//
// This file is independent of any entry ABI.

#include "common/packets.h"

namespace projection {

using packets::Rows16;

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
    device const uchar *row(uint n) const {
        ulong r = table ? ulong(table[n]) : ulong(n);
        return base + r * layout.stride;
    }
    typename W::packet packet(uint n, uint p) const {
        return packets::Loader<W>::load(row(n), layout, p, k);
    }
};

// ---------------------------------------------------------------------------
// Prologues. `load8(m, k, inverse, ...)` returns x[m, k..k+8) (k a multiple
// of 8) as (even, odd) float4s given the norm inverse of row m's group of
// column k (plain operands ignore it); elements at or past `columns` are
// zero. `value` is the scalar form. A normed prologue also exposes its norm
// groups (`groups`, `width`, `norm_input`, `epsilon`) to `device_normalize`.

template <typename A>
inline void load8_storage(device const typename A::storage *row, ulong stride, uint k, uint columns,
    bool vector, thread float4 &even, thread float4 &odd) {
    if (vector && k + 8u <= columns) {
        A::split8(*reinterpret_cast<device const uint4 *>(row + k), even, odd);
        return;
    }
    float v[8];
    for (uint i = 0; i < 8; ++i)
        v[i] = k + i < columns ? A::load(row[ulong(k + i) * stride]) : 0.0f;
    even = float4(v[0], v[2], v[4], v[6]);
    odd = float4(v[1], v[3], v[5], v[7]);
}

// The storage words of x[.., k..k+8) (k a multiple of 8) exactly as stored:
// element 2i in the low half of word i; elements at or past `columns` are
// zero. Plain operands expose them as `words8(m, k)`, which the GEMM stages
// unconverted.
template <typename A>
inline uint4 words8_storage(device const typename A::storage *row, ulong stride, uint k, uint columns,
    bool vector) {
    if (vector && k + 8u <= columns)
        return *reinterpret_cast<device const uint4 *>(row + k);
    uint4 words = uint4(0);
    for (uint i = 0; i < 8; ++i)
        if (k + i < columns)
            words[i / 2] |= uint(as_type<ushort>(row[ulong(k + i) * stride])) << (16u * (i & 1u));
    return words;
}

// Eight norm weights of element type N from index k (a multiple of 8), as
// (even, odd); every index is in range.
template <typename N>
inline void norm8_scalar(device const uchar *norm, ulong stride, uint k, thread float4 &even,
    thread float4 &odd) {
    float v[8];
    for (uint i = 0; i < 8; ++i)
        v[i] = element::at<N>(norm, ulong(k + i) * stride);
    even = float4(v[0], v[2], v[4], v[6]);
    odd = float4(v[1], v[3], v[5], v[7]);
}

template <typename N>
struct norm8 {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_scalar<N>(norm, stride, k, even, odd);
    }
};

// 16-bit norms with unit stride load as one uint4.
template <typename N>
inline void norm8_packed(device const uchar *norm, ulong stride, uint k, thread float4 &even,
    thread float4 &odd) {
    if (stride == 1)
        N::split8(*reinterpret_cast<device const uint4 *>(norm + ulong(k) * 2u), even, odd);
    else
        norm8_scalar<N>(norm, stride, k, even, odd);
}
template <>
struct norm8<element::Bf16> {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_packed<element::Bf16>(norm, stride, k, even, odd);
    }
};
template <>
struct norm8<element::F16> {
    static void load(device const uchar *norm, ulong stride, uint k, thread float4 &even, thread float4 &odd) {
        norm8_packed<element::F16>(norm, stride, k, even, odd);
    }
};

template <typename A, typename Rows>
struct Plain {
    typedef A activation;
    device const uchar *x;
    ulong stride0, stride1;
    uint columns;
    Rows rows;
    float value(uint m, uint k, float) const {
        return A::load(reinterpret_cast<device const typename A::storage *>(x)
            [ulong(rows.at(m)) * stride0 + ulong(k) * stride1]);
    }
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        load8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0, even, odd);
    }
    uint4 words8(uint m, uint k) const {
        device const typename A::storage *row =
            reinterpret_cast<device const typename A::storage *>(x) + ulong(rows.at(m)) * stride0;
        return words8_storage<A>(row, stride1, k, columns, stride1 == 1 && (stride0 & 7u) == 0);
    }
};

template <typename A, typename N, typename Rows>
struct Rms {
    typedef A activation;
    device const float *x;
    ulong stride0, stride1;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint columns;
    Rows rows;
    uint groups() const { return 1; }
    uint width() const { return columns; }
    float norm_input(uint m, uint, uint i) const {
        return x[ulong(rows.at(m)) * stride0 + ulong(i) * stride1];
    }
    float epsilon() const { return eps; }
    // A row's square sum in `parts` parts. One lane's share of `part`: columns
    // 4c .. 4c + 3 for c = 32 * part + lane + 32 * parts * j, in order.
    static constant constexpr uint parts = 8;
    float squares(uint m, uint, uint part, uint lane) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        bool vector = stride1 == 1 && (stride0 & 3u) == 0 && (columns & 3u) == 0;
        float sum = 0.0f;
        for (uint c = 4u * (32u * part + lane); c < columns; c += 128u * parts) {
            float4 v;
            if (vector) {
                v = *reinterpret_cast<device const float4 *>(row + c);
            } else {
                for (uint i = 0; i < 4; ++i)
                    v[i] = c + i < columns ? row[ulong(c + i) * stride1] : 0.0f;
            }
            sum = metal::fma(v.x, v.x, sum);
            sum = metal::fma(v.y, v.y, sum);
            sum = metal::fma(v.z, v.z, sum);
            sum = metal::fma(v.w, v.w, sum);
        }
        return sum;
    }
    float value(uint m, uint column, float inverse) const {
        float v = x[ulong(rows.at(m)) * stride0 + ulong(column) * stride1];
        return A::round(v * inverse * element::at<N>(norm, ulong(column) * norm_stride));
    }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        device const float *row = x + ulong(rows.at(m)) * stride0;
        if (stride1 == 1 && (stride0 & 3u) == 0 && k + 8u <= columns) {
            float4 a = *reinterpret_cast<device const float4 *>(row + k);
            float4 b = *reinterpret_cast<device const float4 *>(row + k + 4);
            float4 ne, no;
            norm8<N>::load(norm, norm_stride, k, ne, no);
            float4 e = float4(a.x, a.z, b.x, b.z) * inverse * ne;
            float4 o = float4(a.y, a.w, b.y, b.w) * inverse * no;
            for (uint i = 0; i < 4; ++i) {
                even[i] = A::round(e[i]);
                odd[i] = A::round(o[i]);
            }
            return;
        }
        float v[8];
        for (uint i = 0; i < 8; ++i)
            v[i] = k + i < columns
                ? A::round(row[ulong(k + i) * stride1] * inverse
                    * element::at<N>(norm, ulong(k + i) * norm_stride))
                : 0.0f;
        even = float4(v[0], v[2], v[4], v[6]);
        odd = float4(v[1], v[3], v[5], v[7]);
    }
};

// Gated per-head RMS·SiLU(z): `mixed` is [rows, heads, W] in A, `z` sits at
// column `z_column` of a row of `projection` (A).
template <typename A, typename N, typename Rows>
struct GatedRms {
    typedef A activation;
    device const uchar *mixed;
    ulong mixed0, mixed1, mixed2;
    device const uchar *projection;
    ulong projection0, projection1;
    ulong z_column;
    device const uchar *norm;
    ulong norm_stride;
    float eps;
    uint heads;
    uint head_width;
    Rows rows;
    uint groups() const { return heads; }
    uint width() const { return head_width; }
    float mixed_at(uint m, uint head, uint i) const {
        return A::load(reinterpret_cast<device const typename A::storage *>(mixed)
            [ulong(rows.at(m)) * mixed0 + ulong(head) * mixed1 + ulong(i) * mixed2]);
    }
    float norm_input(uint m, uint head, uint i) const { return mixed_at(m, head, i); }
    float epsilon() const { return eps; }
    float value(uint m, uint column, float inverse) const {
        uint head = column / head_width, i = column % head_width;
        float normalized = A::round(mixed_at(m, head, i) * inverse
            * element::at<N>(norm, ulong(i) * norm_stride));
        float gate = A::load(reinterpret_cast<device const typename A::storage *>(projection)
            [ulong(rows.at(m)) * projection0 + (z_column + column) * projection1]);
        float activated = A::round(gate / (1.0f + metal::exp(-gate)));
        return A::round(normalized * activated);
    }
    // The inputs of columns k..k+7 (one head; head_width is a multiple of 8)
    // as (even, odd): mixed, z and the norm weights.
    struct inputs8 {
        float4 me, mo, ze, zo, ne, no;
    };
    inputs8 load_inputs8(uint m, uint k) const {
        uint head = k / head_width, i = k % head_width;
        ulong mixed_at0 = ulong(rows.at(m)) * mixed0 + ulong(head) * mixed1 + ulong(i) * mixed2;
        ulong z_at = ulong(rows.at(m)) * projection0 + (z_column + k) * projection1;
        device const typename A::storage *mixed_row = reinterpret_cast<device const typename A::storage *>(mixed);
        device const typename A::storage *z_row = reinterpret_cast<device const typename A::storage *>(projection);
        inputs8 v;
        load8_storage<A>(mixed_row + mixed_at0, mixed2, 0, 8, mixed2 == 1 && (mixed_at0 & 7u) == 0, v.me, v.mo);
        load8_storage<A>(z_row + z_at, projection1, 0, 8, projection1 == 1 && (z_at & 7u) == 0, v.ze, v.zo);
        norm8<N>::load(norm, norm_stride, i, v.ne, v.no);
        return v;
    }
    void finish8(thread const inputs8 &v, float inverse, thread float4 &even, thread float4 &odd) const {
        float4 e = v.me * inverse * v.ne, o = v.mo * inverse * v.no;
        float4 ae = v.ze / (1.0f + metal::exp(-v.ze)), ao = v.zo / (1.0f + metal::exp(-v.zo));
        for (uint j = 0; j < 4; ++j) {
            even[j] = A::round(A::round(e[j]) * A::round(ae[j]));
            odd[j] = A::round(A::round(o[j]) * A::round(ao[j]));
        }
    }
    void load8(uint m, uint k, float inverse, thread float4 &even, thread float4 &odd) const {
        finish8(load_inputs8(m, k), inverse, even, odd);
    }
    // The prologue of columns k..k+7 when the head_width / 8 adjacent lanes of
    // this lane's aligned lane group load the head's columns in order (the
    // GEMV staging's layout): each lane sums the squares of its eight mixed
    // values in column order, and a butterfly over the group gives every lane
    // the head's square sum. No separate square-sum pass is needed, and the
    // sum does not depend on the threadgroup shape.
    void load8_across_lanes(uint m, uint k, thread float4 &even, thread float4 &odd) const {
        inputs8 v = load_inputs8(m, k);
        float squares = 0.0f;
        for (uint j = 0; j < 4; ++j) {
            squares = metal::fma(v.me[j], v.me[j], squares);
            squares = metal::fma(v.mo[j], v.mo[j], squares);
        }
        for (ushort offset = 1; offset < head_width / 8u; offset <<= 1)
            squares += simd_shuffle_xor(squares, offset);
        finish8(v, metal::rsqrt(squares / float(head_width) + eps), even, odd);
    }
};

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

// ---------------------------------------------------------------------------
// The normalizing pre-pass.

// The pre-pass of a normed prologue: threadgroup `item` (THREADS threads)
// normalizes one (row, group) and stores that group's prologue output, in A,
// at x[m * columns + group * width ..]. The projection launches then read `x`
// as a plain operand, so none of their threadgroups repeats the prologue
// arithmetic. `partials` is PROJECTION_NORMALIZE_SHARED threadgroup memory.
#define PROJECTION_NORMALIZE_SHARED(name) threadgroup float name[32]

template <uint THREADS, typename In>
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
            *reinterpret_cast<device uint4 *>(row + first + i) = A::pack8(even, odd);
        } else {
            for (uint j = 0; j < 8u && i + j < in.width(); ++j)
                row[first + i + j] = A::store((j & 1u) ? odd[j >> 1] : even[j >> 1]);
        }
    }
}

// ---------------------------------------------------------------------------
// The in-threadgroup prologue of the GEMV row classes (M <= 16).
//
// A decode projection is one launch: every GEMV threadgroup reduces the norm
// groups of its (at most 16) activation rows itself and applies the prologue
// while staging. A norm group of at most 256 columns (the gated per-head
// norm) is reduced by the staging lanes that load it (LaneNorm). A wider one
// (the RMS row norm, SharedNorm) is reduced first: its square sum is split
// into `In::parts` fixed parts
// (`In::squares`: each lane sums its share in order, then one simd_sum);
// simdgroup (item % SG) reduces part `item`, and the staging adds a group's
// parts in a fixed tree. The inverses (and hence every rounded activation)
// therefore do not depend on the threadgroup shape. `squares` is
// PROJECTION_SQUARES_SHARED threadgroup memory for `slots` = groups * parts
// per row; the opening barrier of the GEMV and batched GEMV bodies publishes
// it to their staging (a simdgroup that owns no part issues its first weight
// loads meanwhile).
#define PROJECTION_SQUARES_SHARED(name, slots) threadgroup float name[16 * (slots)]

template <uint SG, typename In>
inline void threadgroup_squares(thread const In &in, uint m_rows, threadgroup float *squares, uint sg,
    uint lane) {
    uint slots = in.groups() * In::parts;
    for (uint item = sg; item < m_rows * slots; item += SG) {
        uint slot = item % slots;
        float sum = simd_sum(in.squares(item / slots, slot / In::parts, slot % In::parts, lane));
        if (lane == 0)
            squares[item] = sum;
    }
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

// A normed prologue whose norm groups are reduced across the staging lanes
// that load them (`In::load8_across_lanes`), read by the GEMV staging as a
// plain operand.
template <typename In>
struct LaneNorm {
    typedef typename In::activation activation;
    In in;
    void load8(uint m, uint k, float, thread float4 &even, thread float4 &odd) const {
        in.load8_across_lanes(m, k, even, odd);
    }
};

// ---------------------------------------------------------------------------
// GEMV.

// Threadgroup memory of one GEMV, a `threadgroup uchar *` bound at
// [[threadgroup(0)]]: the staged activations. A GEMV launch over K columns
// and O <= 8 rows declares
//   shared_bytes (min(O, 8) * min(ceil_div(K, 32), 288 / max(O, 1)) * 72)
// (at most `gemv_stage_packet_rows` staged packet rows of
// `gemv_packet_row_bytes`; one row of K <= 9216 is staged whole).
constant constexpr uint gemv_stage_packet_rows = 288;
constant constexpr uint gemv_packet_row_bytes = 4 * 16 + 8;

// The staged activations of one GEMV threadgroup over a chunk of `chunk`
// packets: the eight activation values of chunk-local packet p, step s
// (columns 32p + 8s ..) and row m as one uint4 in A storage at
// words[(m * 4 + s) * chunk + p], and the packet's two 16-column sums at
// sums[m * chunk + p]. Lanes owning consecutive packets read consecutive
// words, so the reads are free of bank conflicts.
struct gemv_staging {
    threadgroup uint4 *words;
    threadgroup float2 *sums;
    uint chunk;
    static gemv_staging at(threadgroup uchar *shared, uint chunk, uint m_rows) {
        gemv_staging stage;
        stage.words = reinterpret_cast<threadgroup uint4 *>(shared);
        stage.sums = reinterpret_cast<threadgroup float2 *>(stage.words + 4u * m_rows * chunk);
        stage.chunk = chunk;
        return stage;
    }
};

// Stage packets first .. first + count of every activation row; all threads
// of the threadgroup (whole simdgroups) take part, one thread per eight
// columns (row m, packet, step). The four steps of a packet sit on adjacent
// lanes, which add their sums pairwise into the packet's 16-column sums.
template <typename In>
inline void gemv_stage(thread const In &in, uint m_rows, uint first, uint count, gemv_staging stage,
    uint thread_index, uint threads) {
    typedef typename In::activation A;
    threadgroup float *sums = reinterpret_cast<threadgroup float *>(stage.sums);
    for (uint item = thread_index; item < 4u * m_rows * count; item += threads) {
        uint step = item & 3u, packet = item >> 2;
        uint m = packet / count, local = packet - m * count;
        float4 even, odd;
        in.load8(m, 32u * (first + local) + 8u * step, 0.0f, even, odd);
        stage.words[(m * 4u + step) * stage.chunk + local] = A::pack8(even, odd);
        float4 pair = even + odd;
        float sum = (pair.x + pair.y) + (pair.z + pair.w);
        sum += simd_shuffle_xor(sum, ushort(1));
        if ((step & 1u) == 0)
            sums[2u * (m * stage.chunk + local) + (step >> 1)] = sum;
    }
}

// One packet of R weight rows (and, paired, of R rows of the second tensor).
template <typename W, typename U, bool PAIRED, uint R>
struct gemv_packets {
    typename W::packet a[R];
    typename U::packet b[R];
    void load(thread const Weights<W> &w, thread const Weights<U> &u, uint first_row, uint rows,
        uint p) {
        for (uint r = 0; r < R; ++r) {
            uint n = min(first_row + r, rows - 1);
            a[r] = w.packet(n, p);
            if (PAIRED)
                b[r] = u.packet(n, p);
        }
    }
};

// Accumulate one loaded packet (chunk-local `local`) into the accumulators,
// for every activation row.
template <typename W, typename U, bool PAIRED, uint R, uint MAXM, typename A>
inline void gemv_accumulate(thread float (&acc)[R][MAXM], thread float (&acc2)[R][MAXM],
    thread const gemv_packets<W, U, PAIRED, R> &packets, uint local, uint m_rows, gemv_staging stage) {
    thread const typename W::packet (&a)[R] = packets.a;
    thread const typename U::packet (&b)[R] = packets.b;
    for (uint step = 0; step < 4; ++step) {
        float4 ae[R], ao[R], be[R], bo[R];
        for (uint r = 0; r < R; ++r) {
            W::codes(a[r], step, ae[r], ao[r]);
            if (PAIRED)
                U::codes(b[r], step, be[r], bo[r]);
        }
        for (uint m = 0; m < MAXM; ++m) {
            if (m < m_rows) {
                float4 xe, xo;
                A::split8(stage.words[(m * 4u + step) * stage.chunk + local], xe, xo);
                for (uint r = 0; r < R; ++r) {
                    acc[r][m] = metal::fma(W::scale(a[r], step),
                        metal::dot(ae[r], xe) + metal::dot(ao[r], xo), acc[r][m]);
                    if (PAIRED)
                        acc2[r][m] = metal::fma(U::scale(b[r], step),
                            metal::dot(be[r], xe) + metal::dot(bo[r], xo), acc2[r][m]);
                }
            }
        }
    }
    constexpr bool biased = W::biased || (PAIRED && U::biased);
    if (biased) {
        for (uint m = 0; m < MAXM; ++m) {
            if (m < m_rows) {
                float2 sums = stage.sums[m * stage.chunk + local];
                for (uint r = 0; r < R; ++r) {
                    if (W::biased) {
                        if (W::groups == 1) {
                            acc[r][m] = metal::fma(W::bias(a[r], 0), sums.x + sums.y, acc[r][m]);
                        } else {
                            acc[r][m] = metal::fma(W::bias(a[r], 0), sums.x, acc[r][m]);
                            acc[r][m] = metal::fma(W::bias(a[r], 1), sums.y, acc[r][m]);
                        }
                    }
                    if (PAIRED && U::biased) {
                        if (U::groups == 1) {
                            acc2[r][m] = metal::fma(U::bias(b[r], 0), sums.x + sums.y, acc2[r][m]);
                        } else {
                            acc2[r][m] = metal::fma(U::bias(b[r], 0), sums.x, acc2[r][m]);
                            acc2[r][m] = metal::fma(U::bias(b[r], 1), sums.y, acc2[r][m]);
                        }
                    }
                }
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

// One GEMV threadgroup of SG simdgroups. A simdgroup is 32 / LANES lane
// groups; lane group g of simdgroup sg owns the R weight rows from
// ((tile * SG + sg) * (32 / LANES) + g) * R, and its lanes own the packets
// sub, sub + LANES, ... of those rows. The prologue output is staged per
// threadgroup in chunks of at most `gemv_stage_packet_rows / m_rows` packets
// (a multiple of LANES). Weight packets are double-buffered in registers: a
// lane loads its next packet before accumulating the current one, and its
// first packet before the staging, so the weight stream never waits on it.
template <typename W, typename U, bool PAIRED, uint SG, uint R, uint MAXM, uint LANES, typename In,
    typename Out>
inline void gemv_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    static_assert(LANES == 32 || LANES == 16 || LANES == 8, "a GEMV lane group is 8, 16 or 32 lanes");
    typedef typename In::activation A;
    uint group = lane / LANES, sub = lane % LANES;
    uint first_row = ((tile * SG + sg) * (32u / LANES) + group) * R;
    bool active = first_row < rows;
    uint packets = (k + 31u) / 32u;
    gemv_packets<W, U, PAIRED, R> current, next;
    if (active && sub < packets)
        current.load(w, u, first_row, rows, sub);
    uint chunk = min(packets, (gemv_stage_packet_rows / m_rows) / LANES * LANES);
    gemv_staging stage = gemv_staging::at(shared, chunk, m_rows);
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
        gemv_stage(in, m_rows, first, count, stage, sg * 32u + lane, SG * 32u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local = sub; local < count; local += LANES) {
                if (first + local + LANES < packets)
                    next.load(w, u, first_row, rows, first + local + LANES);
                gemv_accumulate<W, U, PAIRED, R, MAXM, A>(acc, acc2, current, local, m_rows, stage);
                current = next;
            }
        }
    }
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

template <typename W, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<W, W, false, SG, R, MAXM, LANES>(in, out, w, w, m_rows, rows, k, tile, shared, sg, lane);
}

template <typename G, typename U, uint SG, uint R, uint MAXM, uint LANES = 32, typename In, typename Out>
inline void gemv_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile,
    threadgroup uchar *shared, uint sg, uint lane) {
    gemv_body<G, U, true, SG, R, MAXM, LANES>(in, out, gate, up, m_rows, rows, k, tile, shared, sg, lane);
}

// Instantiate a GEMV body for the activation-row bound MAXM in {1, 2, 4, 8}.
#define PROJECTION_FOR_ROWS(rows, ...)                                          \
    do {                                                                        \
        if ((rows) <= 1) { constexpr uint MAXM = 1; __VA_ARGS__; }              \
        else if ((rows) <= 2) { constexpr uint MAXM = 2; __VA_ARGS__; }         \
        else if ((rows) <= 4) { constexpr uint MAXM = 4; __VA_ARGS__; }         \
        else { constexpr uint MAXM = 8; __VA_ARGS__; }                          \
    } while (0)

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

// Fragment loops are fully unrolled: a fragment array indexed by a rolled
// loop lives in stack memory.
#define PROJECTION_UNROLL _Pragma("clang loop unroll(full)")

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
template <typename W, uint COUNT>
struct gemm_b_registers {
    typename W::packet packet[COUNT];
    bool valid[COUNT];
};

template <typename W, uint THREADS, uint COUNT>
inline void gemm_load_b(thread const Weights<W> &w, uint first, uint count, uint rows, uint k0,
    uint k, uint thread_index, thread gemm_b_registers<W, COUNT> &regs) {
    PROJECTION_UNROLL
    for (uint j = 0; j < COUNT; ++j) {
        uint item = thread_index + j * THREADS;
        uint local = item / gemm_halves;
        uint p = k0 / 32u + (item % gemm_halves) / 2u;
        regs.valid[j] = local < count && 32u * p < k;
        if (regs.valid[j])
            regs.packet[j] = w.packet(min(first + local, rows - 1), p);
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

// The fragments a simdgroup owns: fm x fn fragments of the sub-tile at tile
// row (sg / wn) * 8 * fm, column (sg % wn) * 8 * fn. A lane holds tile rows
// row(i) and columns column(j), column(j) + 1.
template <uint TM, uint TN>
struct gemm_fragments {
    typedef gemm_tile<TM, TN> tile;
    uint row0, column0;
    ushort2 coordinate;
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

// The K loop over one TM x TN tile: `U` is the second weight of a paired
// tile (its rows interleave with W's) or the same type for a plain tile.
template <typename W, typename U, bool PAIRED, uint TM, uint TN, typename In>
inline void gemm_accumulate(thread const In &in, thread const Weights<W> &w,
    thread const Weights<U> &u, uint first, uint rows, uint m0, uint m_rows, uint k,
    uint step_begin, uint step_end, threadgroup uchar *shared, uint sg, uint lane,
    thread gemm_accumulators<TM, TN> &acc, uint live_rows) {
    typedef gemm_tile<TM, TN> tile;
    typedef typename In::activation::native E;
    constexpr uint count = PAIRED ? TN / 2u : TN;
    constexpr uint spacing = PAIRED ? 2u : 1u;
    constexpr uint items = (count * gemm_halves + tile::threads - 1) / tile::threads;
    uint thread_index = sg * 32u + lane;
    gemm_buffer<TM, TN, E> buffer = gemm_buffer<TM, TN, E>::at(shared);
    gemm_fragments<TM, TN> at(sg, lane);
    uint live = gemm_live_fragments<TM, TN>(at, m0, min(m_rows, live_rows));
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::fm; ++i)
        PROJECTION_UNROLL
        for (uint j = 0; j < tile::fn; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    if (step_begin >= step_end)
        return;
    gemm_a_registers<tile::a_items> a_regs;
    gemm_b_registers<W, items> w_regs;
    gemm_b_registers<U, items> u_regs;
    uint k0 = step_begin * gemm_k;
    gemm_load_a<TM, tile::threads, tile::a_items>(in, m0, m_rows, k0, k, thread_index, a_regs);
    gemm_load_b<W, tile::threads, items>(w, first, count, rows, k0, k, thread_index, w_regs);
    if (PAIRED)
        gemm_load_b<U, tile::threads, items>(u, first, count, rows, k0, k, thread_index, u_regs);
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
            gemm_load_b<W, tile::threads, items>(w, first, count, rows, k1, k, thread_index, w_regs);
            if (PAIRED)
                gemm_load_b<U, tile::threads, items>(u, first, count, rows, k1, k, thread_index, u_regs);
        }
        gemm_multiply_live<TM, TN, E>(buffer, at, live, acc);
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

// One GEMM tile of a plain (unpaired) projection: output rows tm * TM .. of
// the activations, weight rows tn * TN .. of `w`. Rows at or past `live_rows`
// are known zero (padding): their products are skipped, and they store the
// epilogue of 0.
template <typename W, uint TM, uint TN, typename In, typename Out>
inline void gemm(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane, uint live_rows = ~0u) {
    typedef gemm_tile<TM, TN> tile;
    uint first = tn * TN;
    gemm_accumulators<TM, TN> acc;
    gemm_accumulate<W, W, false, TM, TN>(in, w, w, first, rows, tm * TM, m_rows, k, 0, (k + gemm_k - 1) / gemm_k,
        shared, sg, lane, acc, live_rows);
    gemm_fragments<TM, TN> at(sg, lane);
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::fm; ++i) {
        uint m = tm * TM + at.row(i);
        if (m >= m_rows)
            continue;
        PROJECTION_UNROLL
        for (uint j = 0; j < tile::fn; ++j) {
            float2 c = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
            uint n = first + at.column(j);
            if (n + 1u < rows)
                out.store2(m, n, c.x, c.y);
            else if (n < rows)
                out.store(m, n, c.x);
        }
    }
}

// Split-K. Part `part` of `split` runs the K steps
// [part * steps / split, (part + 1) * steps / split) of one plain GEMM tile and
// stores its raw F32 sums to partials[(part * m_rows + m) * rows + n]. With
// split 1 the tile applies the epilogue directly (`gemm`).
template <typename W, uint TM, uint TN, typename In>
inline void gemm_part(thread const In &in, device float *partials, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint split, uint part, uint tm, uint tn, threadgroup uchar *shared, uint sg, uint lane) {
    typedef gemm_tile<TM, TN> tile;
    uint first = tn * TN;
    uint steps = (k + gemm_k - 1) / gemm_k;
    gemm_accumulators<TM, TN> acc;
    gemm_accumulate<W, W, false, TM, TN>(in, w, w, first, rows, tm * TM, m_rows, k, part * steps / split,
        (part + 1) * steps / split, shared, sg, lane, acc, m_rows);
    gemm_fragments<TM, TN> at(sg, lane);
    device float *own = partials + ulong(part) * m_rows * rows;
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::fm; ++i) {
        uint m = tm * TM + at.row(i);
        if (m >= m_rows)
            continue;
        PROJECTION_UNROLL
        for (uint j = 0; j < tile::fn; ++j) {
            float2 c = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
            uint n = first + at.column(j);
            if (n < rows)
                own[ulong(m) * rows + n] = c.x;
            if (n + 1u < rows)
                own[ulong(m) * rows + n + 1u] = c.y;
        }
    }
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
// tile's rows interleaved (gate, up) so a lane's element pair is one feature.
// `live_rows` as for `gemm`.
template <typename G, typename U, uint TM, uint TN, typename In, typename Out>
inline void gemm_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tm, uint tn,
    threadgroup uchar *shared, uint sg, uint lane, uint live_rows = ~0u) {
    typedef gemm_tile<TM, TN> tile;
    uint first = tn * (TN / 2u);
    gemm_accumulators<TM, TN> acc;
    gemm_accumulate<G, U, true, TM, TN>(in, gate, up, first, rows, tm * TM, m_rows, k, 0, (k + gemm_k - 1) / gemm_k,
        shared, sg, lane, acc, live_rows);
    gemm_fragments<TM, TN> at(sg, lane);
    PROJECTION_UNROLL
    for (uint i = 0; i < tile::fm; ++i) {
        uint m = tm * TM + at.row(i);
        if (m >= m_rows)
            continue;
        PROJECTION_UNROLL
        for (uint j = 0; j < tile::fn; ++j) {
            float2 c = reinterpret_cast<thread float2 &>(acc[i][j].thread_elements());
            uint n = first + at.column(j) / 2u;
            if (n < rows)
                out.store_pair(m, n, c.x, c.y);
        }
    }
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
// blocks of 8 weight rows; the threadgroup stages the activations in chunks
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

template <typename W, typename U, bool PAIRED, uint SG, uint R, uint NB, uint BYTES, typename In, typename Out>
inline void gemv_batch_body(thread const In &in, thread const Out &out, thread const Weights<W> &w,
    thread const Weights<U> &u, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint sg, uint lane) {
    constexpr uint chunk = gemv_batch_chunk<NB, BYTES>();
    constexpr uint pitch = chunk + 4u;
    threadgroup float *staged = reinterpret_cast<threadgroup float *>(shared);
    ushort2 at = fragment_coordinate(lane);
    uint pair_index = at.x / 2u;
    uint first_row = (tile * SG + sg) * R * 8u;
    bool active = first_row < rows;
    uint thread_index = sg * 32u + lane;
    uint row[R];
    for (uint r = 0; r < R; ++r)
        row[r] = min(first_row + 8u * r + at.y, rows - 1);
    simdgroup_float8x8 acc[R][NB], acc2[R][NB];
    for (uint r = 0; r < R; ++r) {
        for (uint b = 0; b < NB; ++b) {
            acc[r][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc2[r][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }
    typename W::packet a[R], a_next[R];
    typename U::packet b[R], b_next[R];
    for (uint c0 = 0; c0 < k; c0 += chunk) {
        uint packets = (min(chunk, k - c0) + 31u) / 32u;
        // The chunk's first packets load while the threadgroup stages it; a
        // threadgroup may run several batched GEMVs in turn, and the previous
        // chunk or call finishes reading the staging first.
        if (active) {
            for (uint r = 0; r < R; ++r) {
                a[r] = w.packet(row[r], c0 / 32u);
                if (PAIRED)
                    b[r] = u.packet(row[r], c0 / 32u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = thread_index; item < 8u * NB * 4u * packets; item += SG * 32u) {
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
                    a_next[r] = w.packet(row[r], c0 / 32u + local + 1u);
                    if (PAIRED)
                        b_next[r] = u.packet(row[r], c0 / 32u + local + 1u);
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
    for (uint r = 0; r < R; ++r) {
        uint n = first_row + 8u * r + at.y;
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

// BYTES: the threadgroup staging `shared` provides (gemv_batch_shared_bytes
// for a batched GEMV launch).
template <typename W, uint SG, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch(thread const In &in, thread const Out &out, thread const Weights<W> &w, uint m_rows,
    uint rows, uint k, uint tile, threadgroup uchar *shared, uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<W, W, false, SG, R, 1, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared, sg, lane);
    else
        gemv_batch_body<W, W, false, SG, R, 2, BYTES>(in, out, w, w, m_rows, rows, k, tile, shared, sg, lane);
}

template <typename G, typename U, uint SG, uint R, uint BYTES = gemv_batch_shared_bytes, typename In, typename Out>
inline void gemv_batch_paired(thread const In &in, thread const Out &out, thread const Weights<G> &gate,
    thread const Weights<U> &up, uint m_rows, uint rows, uint k, uint tile, threadgroup uchar *shared,
    uint sg, uint lane) {
    if (m_rows <= 8)
        gemv_batch_body<G, U, true, SG, R, 1, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared, sg, lane);
    else
        gemv_batch_body<G, U, true, SG, R, 2, BYTES>(in, out, gate, up, m_rows, rows, k, tile, shared, sg, lane);
}

} // namespace projection
