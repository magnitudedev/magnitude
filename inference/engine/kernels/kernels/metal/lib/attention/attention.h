// Shared pieces of the attention entries (`attention_decode`,
// `attention_prefill` and their affine K8/V4 history forms
// `attention_{decode,prefill}_k8v4`; contract in attention.seismic): the
// per-head preparation (optional RMS norm, amplitude-scaled rotary table) held
// in one simdgroup's registers, a row's span walk, decode partition bounds, the
// online-softmax absorb (dense and corrected affine), the fixed-order merge of
// partial states, the decode partition publication and gated merge, the K/V
// append, the affine codec (encode on append, per-lane code access) and the
// prefill bodies (on simdgroup matrices, or on Metal 4 tensor operations where
// the device has them and the tile fits). Each kernel keeps its own loop
// structure and calls these.
//
// Every entry defines its form before including this library: ATTENTION_I
// interleaved gate columns after each query head's W columns (0 or W),
// ATTENTION_U separate gate values per query head (0, 1 or W), and whether
// the layer has fresh rows (ATTENTION_FRESH), q/k norms (ATTENTION_NORM) and a
// value norm (ATTENTION_VALUE_NORM).
//
// Every dense operand is bound canonically (row-major, unit innermost stride),
// so rows are addressed from their logical offsets. Activations are addressed
// as the element's MSL scalar and converted with the compiler's conversions.

#include "../core/activation.h"
#include <seismic/slab.h>

// Head width, and the contiguous columns each of a simdgroup's 32 lanes owns.
#define ATTENTION_W (2 * SEISMIC_DIM_P + SEISMIC_DIM_S)
#define ATTENTION_E (ATTENTION_W / 32)
#define ATTENTION_LOG2E 1.4426950408889634f

// Loops over register arrays (8x8 fragments, per-lane columns) must unroll
// fully: a dynamically indexed array lives in stack memory, which costs the
// streaming kernels a factor of 2-3. Staging loops stay rolled so their loads
// do not raise the register budget next to the accumulators.
#define ATTENTION_UNROLL _Pragma("clang loop unroll(full)")
#define ATTENTION_ROLLED _Pragma("clang loop unroll(disable)")

static_assert(ATTENTION_W % 32 == 0, "attention head width must be a multiple of 32");
static_assert(SEISMIC_DIM_P % ATTENTION_E == 0,
    "each rotary half must cover whole lanes");

namespace attention {

// The activation element's MSL scalar.
typedef element::Act::native Scalar;

// sin and cos of an F32 rotary angle (|angle| up to the context length). The
// angle is reduced exactly enough to [-pi, pi] by a three-part 2*pi
// (Cody-Waite, fused products) and evaluated with the fast functions, which
// are accurate on that interval; the precise library functions cost tens of
// microseconds per head row here.
inline float sincos(float angle, thread float &cosine) {
    const float turns = metal::rint(angle * 0.15915494309189535f);
    float reduced = metal::fma(-turns, 6.28125f, angle);
    reduced = metal::fma(-turns, 0.0019354820251464844f, reduced);
    reduced = metal::fma(-turns, -1.7484555314695172e-07f, reduced);
    cosine = metal::fast::cos(reduced);
    return metal::fast::sin(reduced);
}

// Appends lane `lane`'s columns of one kv head row at history row
// `destination` (the caller skips rows without a destination).
template <typename T>
inline void append(device Scalar *history, int destination, uint kv_head, uint lane,
    thread const T (&x)[ATTENTION_E]) {
    const ulong target = (ulong(destination) * SEISMIC_DIM_KV + kv_head) * ATTENTION_W + lane * ATTENTION_E;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i)
        history[target + i] = Scalar(x[i]);
}

// Keys per decode partition for a row seeing `total` keys: at least `span`,
// and few enough that `parts` partitions cover the row. Never zero.
inline uint partition_span(uint total, uint span, uint parts) {
    return metal::max(span, (total + parts - 1) / parts);
}

constexpr uint sums_pow2(uint n) { return n <= 1 ? 1 : 2 * sums_pow2((n + 1) / 2); }
constexpr uint sums_log2(uint p) { return p <= 1 ? 0 : 1 + sums_log2(p / 2); }

// The simdgroup sums of a lane's H x N scores, each returned on every lane. A
// halving exchange sums them transposed: each step keeps half of the
// remaining values and sends the partner the other half, so after
// min(5, log2 H N) steps a lane holds H N / 32 sums (one when H N <= 32), of
// the indices its lane bits select; the remaining offsets sum within them and
// a broadcast returns every sum. About 2 H N shuffles against 5 H N for
// independent sums; the lanes' addition order is fixed.
template <uint H, uint N>
inline void score_sums(thread float (&score)[H][N], uint lane) {
    constexpr uint COUNT = H * N;
    constexpr uint P = sums_pow2(COUNT);
    constexpr uint HALVINGS = sums_log2(P) < 5 ? sums_log2(P) : 5;
    constexpr uint HELD = P >> HALVINGS;
    float y[P];
    ATTENTION_UNROLL
    for (uint i = 0; i < P; ++i)
        y[i] = i < COUNT ? score[i / N][i % N] : 0.0f;
    ATTENTION_UNROLL
    for (uint step = 0; step < 5; ++step) {
        const ushort offset = ushort(16u >> step);
        if (step < HALVINGS) {
            const uint half_count = P >> (step + 1);
            const bool upper = (lane & offset) != 0;
            ATTENTION_UNROLL
            for (uint i = 0; i < P / 2; ++i) {
                if (i < half_count) {
                    const float keep = upper ? y[i + half_count] : y[i];
                    const float send = upper ? y[i] : y[i + half_count];
                    y[i] = keep + simd_shuffle_xor(send, offset);
                }
            }
        } else {
            ATTENTION_UNROLL
            for (uint i = 0; i < HELD; ++i)
                y[i] += simd_shuffle_xor(y[i], offset);
        }
    }
    ATTENTION_UNROLL
    for (uint j = 0; j < COUNT; ++j)
        score[j / N][j % N] = simd_shuffle(y[j % HELD], ushort((j / HELD) << (5 - HALVINGS)));
}

// The online-softmax state of H query heads over one simdgroup's keys, in the
// exp2 domain, absorbing N keys at a time.
template <uint H, uint N>
inline void absorb_heads(thread const float (&q)[H][ATTENTION_E],
    thread const float (&k)[N][ATTENTION_E], thread const float (&v)[N][ATTENTION_E],
    thread float (&maximum)[H], thread float (&denominator)[H],
    thread float (&output)[H][ATTENTION_E], uint lane) {
    float scores[H][N];
    ATTENTION_UNROLL
    for (uint g = 0; g < H; ++g) {
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j) {
            float partial = 0.0f;
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                partial = metal::fma(q[g][i], k[j][i], partial);
            scores[g][j] = partial;
        }
    }
    score_sums(scores, lane);
    ATTENTION_UNROLL
    for (uint g = 0; g < H; ++g) {
        thread const float (&score)[N] = scores[g];
        float next = maximum[g];
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j)
            next = metal::max(next, score[j]);
        const float carry = metal::fast::exp2(maximum[g] - next);
        float probability[N];
        float sum = 0.0f;
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j) {
            probability[j] = metal::fast::exp2(score[j] - next);
            sum += probability[j];
        }
        denominator[g] = metal::fma(denominator[g], carry, sum);
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i) {
            float o = output[g][i] * carry;
            ATTENTION_UNROLL
            for (uint j = 0; j < N; ++j)
                o = metal::fma(probability[j], v[j][i], o);
            output[g][i] = o;
        }
        maximum[g] = next;
    }
}

// The fixed-order merge of `count` partial attention states for one column:
// state p is slot first + p * stride, with its unnormalized output at
// partials[slot * W + column] (relative to its maximum) and (maximum,
// denominator) at statistics[slot * 2]. Empty states (denominator 0) are
// skipped; a column with no state attends to zero.
inline float merge(device const float *partials, device const float *statistics,
    ulong first, ulong stride, uint count, uint column) {
    float maximum = -INFINITY;
    for (uint p = 0; p < count; ++p) {
        const ulong slot = first + p * stride;
        if (statistics[slot * 2 + 1] > 0.0f)
            maximum = metal::max(maximum, statistics[slot * 2]);
    }
    float denominator = 0.0f;
    float accumulated = 0.0f;
    for (uint p = 0; p < count; ++p) {
        const ulong slot = first + p * stride;
        const float d = statistics[slot * 2 + 1];
        if (d > 0.0f) {
            const float weight = metal::fast::exp2(statistics[slot * 2] - maximum);
            denominator = metal::fma(d, weight, denominator);
            accumulated = metal::fma(partials[slot * ATTENTION_W + column], weight, accumulated);
        }
    }
    return accumulated / metal::max(denominator, 1e-30f);
}

// Columns of one query head's row: W query columns, then its interleaved
// gates.
#define ATTENTION_QUERY_STRIDE (ATTENTION_W + ATTENTION_I)

// One head row, RMS-normalized with `norm` when NORM (else as is), into lane
// `lane`'s ATTENTION_E columns. The whole simdgroup calls it.
template <bool NORM>
inline void head_norm(device const Scalar *raw, device const float *norm, float epsilon, uint lane,
    thread float (&x)[ATTENTION_E]) {
    float squares = 0.0f;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i) {
        x[i] = float(raw[lane * ATTENTION_E + i]);
        squares = metal::fma(x[i], x[i], squares);
    }
    if (!NORM)
        return;
    squares = simd_sum(squares);
    const float inverse = metal::rsqrt(squares / float(ATTENTION_W) + epsilon);
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i)
        x[i] = x[i] * inverse * norm[lane * ATTENTION_E + i];
}

// One row's rotary table for lane `lane`'s rotated columns: pair p turns by
// coordinate axis components[p] at frequencies[p], its cosine and sine
// scaled by amplitudes[p]. The same table rotates every head of the row.
struct rotary_table {
    float cosine[ATTENTION_E];
    float sine[ATTENTION_E];
    rotary_table(device const int *coordinates, device const int *components, device const float *frequencies,
        device const float *amplitudes, uint lane) {
        if (SEISMIC_DIM_P == 0 || lane >= 2 * SEISMIC_DIM_P / ATTENTION_E)
            return;
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i) {
            const uint pair = (lane * ATTENTION_E + i) % SEISMIC_DIM_P;
            float c;
            const float s = sincos(float(coordinates[components[pair]]) * frequencies[pair], c);
            cosine[i] = c * amplitudes[pair];
            sine[i] = s * amplitudes[pair];
        }
    }
};

// `head_norm`, then the first 2P columns rotated by the row's table. Each
// rotated column's pair partner lives P / ATTENTION_E lanes away.
template <bool NORM>
inline void head_rotary(device const Scalar *raw, device const float *norm, thread const rotary_table &table,
    float epsilon, uint lane, thread float (&x)[ATTENTION_E]) {
    head_norm<NORM>(raw, norm, epsilon, lane, x);
    if (SEISMIC_DIM_P == 0)
        return;
    constexpr uint half_lanes = SEISMIC_DIM_P / ATTENTION_E;
    const uint partner_lane = lane < half_lanes ? lane + half_lanes
        : (lane < 2 * half_lanes ? lane - half_lanes : lane);
    float partner[ATTENTION_E];
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i)
        partner[i] = simd_shuffle(x[i], ushort(partner_lane));
    if (lane >= 2 * half_lanes)
        return;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i) {
        const uint column = lane * ATTENTION_E + i;
        x[i] = column < SEISMIC_DIM_P ? x[i] * table.cosine[i] - partner[i] * table.sine[i]
                                      : x[i] * table.cosine[i] + partner[i] * table.sine[i];
    }
}

// One head row normalized and rotated by its own table.
template <bool NORM>
inline void head_rotary(device const Scalar *raw, device const float *norm,
    device const int *coordinates, device const int *components, device const float *frequencies,
    device const float *amplitudes, float epsilon, uint lane, thread float (&x)[ATTENTION_E]) {
    const rotary_table table(coordinates, components, frequencies, amplitudes, lane);
    head_rotary<NORM>(raw, norm, table, epsilon, lane, x);
}

// Publishes one decode partition when the G query heads of a kv head split
// into SLICES slices of H heads: simdgroup s holds the heads of slice
// s % SLICES over key group s / SLICES (one of SIMDS / SLICES contiguous
// sub-ranges of the partition). Per slice head, the key groups' states
// (outputs relative to their own maxima) merge in key-group order into one
// partial at slot first + head * PARTS: the unnormalized output at
// partials[slot * W] relative to the partition maximum, and (maximum,
// denominator) at statistics[slot * 2]. `states` holds [SIMDS][H][2],
// `columns` [SIMDS][W] floats of threadgroup memory.
template <uint SIMDS, uint PARTS, uint H, uint SLICES>
inline void publish_slices(thread const float (&maximum)[H], thread const float (&denominator)[H],
    thread const float (&output)[H][ATTENTION_E], threadgroup float *states,
    threadgroup float *columns, device float *partials, device float *statistics, ulong first,
    uint simd, uint lane, uint thread_index) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint GROUPS = SIMDS / SLICES;
    const uint slice = simd % SLICES;
    if (lane == 0) {
        ATTENTION_UNROLL
        for (uint h = 0; h < H; ++h) {
            states[(simd * H + h) * 2] = maximum[h];
            states[(simd * H + h) * 2 + 1] = denominator[h];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ATTENTION_UNROLL
    for (uint h = 0; h < H; ++h) {
        float partition_maximum = -INFINITY;
        for (uint group = 0; group < GROUPS; ++group) {
            const uint s = group * SLICES + slice;
            if (states[(s * H + h) * 2 + 1] > 0.0f)
                partition_maximum = metal::max(partition_maximum, states[(s * H + h) * 2]);
        }
        const float weight = denominator[h] > 0.0f
            ? metal::fast::exp2(maximum[h] - partition_maximum) : 0.0f;
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            columns[simd * W + lane * E + i] = output[h][i] * weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint item = thread_index; item < SLICES * W; item += SIMDS * 32) {
            const uint item_slice = item / W;
            const uint column = item % W;
            float sum = 0.0f;
            for (uint group = 0; group < GROUPS; ++group)
                sum += columns[(group * SLICES + item_slice) * W + column];
            partials[(first + (item_slice * H + h) * PARTS) * W + column] = sum;
        }
        if (thread_index < SLICES) {
            const uint item_slice = thread_index;
            float slice_maximum = -INFINITY;
            for (uint group = 0; group < GROUPS; ++group) {
                const uint s = group * SLICES + item_slice;
                if (states[(s * H + h) * 2 + 1] > 0.0f)
                    slice_maximum = metal::max(slice_maximum, states[(s * H + h) * 2]);
            }
            float total_denominator = 0.0f;
            for (uint group = 0; group < GROUPS; ++group) {
                const uint s = group * SLICES + item_slice;
                const float d = states[(s * H + h) * 2 + 1];
                if (d > 0.0f)
                    total_denominator = metal::fma(d,
                        metal::fast::exp2(states[(s * H + h) * 2] - slice_maximum), total_denominator);
            }
            const ulong slot = first + (item_slice * H + h) * PARTS;
            statistics[slot * 2] = slice_maximum;
            statistics[slot * 2 + 1] = total_denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Span `span` of `row`'s keys: visible history spans 0..R-1, then the fresh
// span R, which is empty for a layer without fresh rows.
inline void form_span(device const int *visible, device const int *fresh, ulong row, ulong spans, ulong span,
    thread int &lo, thread int &hi) {
    if (span < spans) {
        lo = visible[(row * spans + span) * 2];
        hi = visible[(row * spans + span) * 2 + 1];
    } else if (ATTENTION_FRESH) {
        lo = fresh[row * 2];
        hi = fresh[row * 2 + 1];
    } else {
        lo = 0;
        hi = 0;
    }
}

// Span `span` of the decode rows [row0, row0 + tokens) below `rows` (a decode
// row tile): the union [lo, hi) of the rows' non-empty intervals and their
// intersection [common_lo, common_hi). One row's is its own interval.
struct tile_interval {
    int lo;
    int hi;
    int common_lo;
    int common_hi;
};
inline tile_interval tile_span(device const int *visible, device const int *fresh, ulong row0, uint tokens,
    ulong rows, ulong spans, ulong span) {
    tile_interval interval{0x7fffffff, int(0x80000000), int(0x80000000), 0x7fffffff};
    for (ulong row = row0; row < metal::min(row0 + tokens, rows); ++row) {
        int lo, hi;
        form_span(visible, fresh, row, spans, span, lo, hi);
        interval.common_lo = metal::max(interval.common_lo, lo);
        interval.common_hi = metal::min(interval.common_hi, hi);
        if (hi > lo) {
            interval.lo = metal::min(interval.lo, lo);
            interval.hi = metal::max(interval.hi, hi);
        }
    }
    if (interval.hi <= interval.lo)
        interval.lo = interval.hi = 0;
    return interval;
}

// The keys a decode row tile sees: its spans' unions.
inline uint tile_total(device const int *visible, device const int *fresh, ulong row0, uint tokens, ulong rows,
    ulong spans) {
    uint total = 0;
    for (ulong index = 0; index <= spans; ++index) {
        const tile_interval interval = tile_span(visible, fresh, row0, tokens, rows, spans, index);
        total += uint(interval.hi - interval.lo);
    }
    return total;
}

// The KEYS-key tiles of a decode row tile's keys [first, last) (its spans'
// unions, in `form_span` order): each span's part of the range is tiled
// separately.
inline uint form_tiles(device const int *visible, device const int *fresh, ulong row0, uint tokens, ulong rows,
    ulong spans, uint first, uint last, uint keys) {
    uint tiles = 0;
    uint offset = 0;
    for (ulong index = 0; index <= spans && offset < last; ++index) {
        const tile_interval interval = tile_span(visible, fresh, row0, tokens, rows, spans, index);
        const int lo = interval.lo, hi = interval.hi;
        const uint length = uint(metal::max(hi - lo, 0));
        const uint begin = metal::max(first, offset);
        const uint end = metal::min(last, offset + length);
        if (begin < end)
            tiles += (end - begin + keys - 1) / keys;
        offset += length;
    }
    return tiles;
}

// The total number of keys a row sees under `form_span`.
inline uint form_total(device const int *visible, device const int *fresh, ulong row, ulong spans) {
    uint total = 0;
    for (ulong index = 0; index <= spans; ++index) {
        int lo, hi;
        form_span(visible, fresh, row, spans, index, lo, hi);
        total += uint(metal::max(hi - lo, 0));
    }
    return total;
}

// One attended column of query head `head` of `row` times its gate: value
// column % count of its interleaved gates (after its queries in `query`) or
// its separate ones (in `gate`); sigmoid, or softplus when `softplus`.
// Rounded to the activation.
inline Scalar gate_output(device const Scalar *query, device const Scalar *gate, ulong row, ulong head,
    uint column, float attended, bool softplus) {
    const ulong at = row * SEISMIC_DIM_KV * SEISMIC_DIM_G + head;
    float g;
    if (ATTENTION_I > 0)
        g = float(query[at * ATTENTION_QUERY_STRIDE + ATTENTION_W + column]);
    else if (ATTENTION_U > 0)
        g = float(gate[at * ATTENTION_U + column % metal::max(uint(ATTENTION_U), 1u)]);
    else
        return Scalar(attended);
    return Scalar(softplus ? attended * (metal::max(g, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(g))))
                           : attended / (1.0f + metal::exp(-g)));
}

// The decode merge of one (query head, row) over 32 of its columns, one
// simdgroup per threadgroup and one lane per column: the row's non-empty
// partitions in partition order, then the gate. The partition weights
// exp2(maximum_p - maximum) are formed once per threadgroup in `weights`
// (2 * PARTS floats: weight, denominator); every lane then carries `merge`'s
// ordered chains for its column, so the result is `merge`'s. A row that sees
// no key attends to zero.
template <uint SPAN, uint PARTS, uint TOKENS>
inline void decode_output(device const Scalar *query, device const Scalar *gate, device const int *visible,
    device const int *fresh, device Scalar *result, device const float *partials,
    device const float *statistics, threadgroup float *weights, ulong spans, ulong rows, ulong head, ulong row,
    uint column, uint lane, bool softplus) {
    constexpr uint W = ATTENTION_W;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    // The partitions of the row's tile (TOKENS of the launch's `rows` in the
    // matrix form; one row otherwise).
    const ulong row0 = row / TOKENS * TOKENS;
    const uint total = tile_total(visible, fresh, row0, TOKENS, rows, spans);
    const uint span_keys = partition_span(total, SPAN, PARTS);
    const uint active = (total + span_keys - 1) / span_keys;
    const ulong first = (row * KV * G + head) * PARTS;
    float maximum = -INFINITY;
    for (uint p = lane; p < active; p += 32) {
        if (statistics[(first + p) * 2 + 1] > 0.0f)
            maximum = metal::max(maximum, statistics[(first + p) * 2]);
    }
    maximum = simd_max(maximum);
    for (uint p = lane; p < active; p += 32) {
        const float d = statistics[(first + p) * 2 + 1];
        weights[2 * p] = d > 0.0f ? metal::fast::exp2(statistics[(first + p) * 2] - maximum) : 0.0f;
        weights[2 * p + 1] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each batch issues all its partial loads before its chains consume
    // them: few threads cover a whole launch, so a load at a time would
    // leave every partition a memory round trip.
    constexpr uint BATCH = 32;
    float denominator = 0.0f;
    float accumulated = 0.0f;
    for (uint p0 = 0; p0 < active; p0 += BATCH) {
        float values[BATCH];
        ATTENTION_UNROLL
        for (uint i = 0; i < BATCH; ++i)
            values[i] = p0 + i < active ? partials[(first + p0 + i) * W + column] : 0.0f;
        ATTENTION_UNROLL
        for (uint i = 0; i < BATCH; ++i) {
            const uint p = p0 + i;
            if (p < active && weights[2 * p + 1] > 0.0f) {
                const float weight = weights[2 * p];
                denominator = metal::fma(weights[2 * p + 1], weight, denominator);
                accumulated = metal::fma(values[i], weight, accumulated);
            }
        }
    }
    const float attended = accumulated / metal::max(denominator, 1e-30f);
    result[(row * KV * G + head) * W + column] = gate_output(query, gate, row, head, column, attended, softplus);
}

#include "history.h"

// ---------------------------------------------------------------------------
// Prefill (`attention_prefill`, `attention_prefill_k8v4`): the three
// launches' bodies, over a history policy that appends a row's key and value
// and stages history K/V tiles in the activation dtype.
// ---------------------------------------------------------------------------

// Keys per K/V tile staged in threadgroup memory (half the tile for heads
// wider than PREFILL_WINDOW, whose staged rows are twice as long or more).
#define PREFILL_KEYS (ATTENTION_W > PREFILL_WINDOW ? 16 : 32)
// Output columns one attend pass accumulates: a wider head runs W /
// PREFILL_WINDOW passes over the same key tiles, each recomputing the scores
// (so the softmax statistics are identical) and accumulating one window of
// output columns, which keeps the F32 output fragments in registers.
#define PREFILL_WINDOW 256
// A partition covers at least this many key tiles, so a query tile whose keys
// fit in a few partitions' worth runs unsplit and skips the merge.
#define PREFILL_MIN_TILES 16
// Row pitch of a staged tile, in elements.
#define PREFILL_PITCH (ATTENTION_W + 8)
// The query heads of one attend threadgroup: a kv head's G query heads split
// into groups of PREFILL_HEADS (the last may be smaller), PREFILL_HEAD_GROUPS
// of them, so the threadgroup's size follows the group, not G.
#ifndef PREFILL_HEADS_PER_GROUP
#define PREFILL_HEADS_PER_GROUP SEISMIC_DIM_G
#endif
#define PREFILL_HEADS (PREFILL_HEADS_PER_GROUP < SEISMIC_DIM_G ? PREFILL_HEADS_PER_GROUP : SEISMIC_DIM_G)
#define PREFILL_HEAD_GROUPS ((SEISMIC_DIM_G + PREFILL_HEADS - 1) / PREFILL_HEADS)
// The entry's DIRECT form: K/V tiles are read from device memory as tensor
// operands (`prefill_direct`), which needs PREFILL_KEYS rows of zero keys and
// values after the fresh rows' scratch.
#ifndef PREFILL_DIRECT
#define PREFILL_DIRECT 0
#endif

// The interval union [lo, hi) of a tile's non-empty row intervals and the
// intersection [common_lo, common_hi) of all its row intervals, for one span.
struct prefill_interval {
    int lo;
    int hi;
    int common_lo;
    int common_hi;
};

// Copies key rows [first, first + KEYS) of one kv head (row-major [T, KV, W]
// 2-byte elements, 16-byte aligned) into `staged` (row pitch PREFILL_PITCH) in
// 16-byte pieces; rows at or past `end` are zero. The loop stays rolled: one
// piece in flight per thread keeps the staging registers off the
// accumulators' budget.
template <uint THREADS, class T, uint KEYS = PREFILL_KEYS>
inline void prefill_stage(threadgroup T *staged, device const T *rows,
    int first, int end, uint kv_head, uint thread_index) {
    constexpr uint W = ATTENTION_W;
    constexpr uint PIECES = W / 8;
    ATTENTION_ROLLED
    for (uint item = thread_index; item < KEYS * PIECES; item += THREADS) {
        const uint k = item / PIECES;
        const uint c = (item % PIECES) * 8;
        const int t = first + int(k);
        uint4 bits = uint4(0);
        if (t < end)
            bits = *reinterpret_cast<device const uint4 *>(
                rows + (ulong(t) * SEISMIC_DIM_KV + kv_head) * W + c);
        *reinterpret_cast<threadgroup uint4 *>(staged + k * PREFILL_PITCH + c) = bits;
    }
}

template <uint THREADS, class T, uint KEYS = PREFILL_KEYS>
inline void prefill_stage_slab(threadgroup T *staged, device const ulong *table,
    ulong rows_per_slab, int first, int end, uint kv_head, uint thread_index) {
    constexpr uint W = ATTENTION_W;
    constexpr uint PIECES = W / 8;
    ATTENTION_ROLLED
    for (uint item = thread_index; item < KEYS * PIECES; item += THREADS) {
        const uint k = item / PIECES;
        const uint c = (item % PIECES) * 8;
        const int t = first + int(k);
        uint4 bits = uint4(0);
        if (t < end) {
            device const T *row = slab::row<T>(table, ulong(t), rows_per_slab,
                SEISMIC_DIM_KV * W);
            bits = *reinterpret_cast<device const uint4 *>(row + kv_head * W + c);
        }
        *reinterpret_cast<threadgroup uint4 *>(staged + k * PREFILL_PITCH + c) = bits;
    }
}

// Resolve the slab once for a matrix decode tile. Most tiles stay within one
// slab; the second lookup handles the few that cross a boundary.
template <typename T>
struct decode_slab_tile {
    device const ulong *table;
    device const T *base;
    uint slab_index;
    uint first_offset;
    uint rows;

    inline decode_slab_tile(device const ulong *table, ulong rows_per_slab, int first)
        : table(table), rows(uint(rows_per_slab)) {
        slab_index = uint(first) / rows;
        first_offset = uint(first) - slab_index * rows;
        base = slab::region<T>(table, slab_index);
    }

    inline device const T *row(uint k, ulong elements) const {
        const uint offset = first_offset + k;
        if (offset < rows)
            return base + ulong(offset) * elements;
        return slab::region<T>(table, slab_index + offset / rows) + ulong(offset % rows) * elements;
    }
};

// The rows of one kv head's keys and values that the direct form reads as
// device operands: `rows` rows of [.., KV, W] elements from logical row
// `first`. A tile of `keys` rows for the keys from `at` starts at `at`, or
// earlier when it would run past the rows.
template <class Operand>
struct direct_rows {
    device const Operand *keys;
    device const Operand *values;
    int first;
    int rows;
    inline int start(int at, uint keys) const { return metal::min(at, first + rows - int(keys)); }
    inline int limit() const { return rows > INT_MAX - first ? INT_MAX : first + rows; }
    inline ulong offset(int start) const { return ulong(start - first) * SEISMIC_DIM_KV * ATTENTION_W; }
};

// One run of the direct form's 16-key steps over the keys [first, end) of
// span `span` (R is the fresh span). An unmasked run is whole steps that
// every row sees, in one history slab; a masked run's steps mask per (row,
// key) and restart at every slab its keys cross.
struct prefill_run {
    int first;
    int end;
    uint span;
    uint masked;
};

// Dense history: [T, KV, W] activation planes, whose products take
// activation-dtype operands.
struct dense_history {
    enum : bool { AFFINE = false };
    typedef Scalar Operand;
    device const ulong *key;
    device const ulong *value;
    ulong rows_per_slab;

    inline void append(int destination, uint kv_head, uint lane, thread const float (&k)[ATTENTION_E],
        thread const Scalar (&v)[ATTENTION_E]) const {
        attention::append(slab::row<Scalar>(key, ulong(destination), rows_per_slab,
            SEISMIC_DIM_KV * ATTENTION_W), 0, kv_head, lane, k);
        attention::append(slab::row<Scalar>(value, ulong(destination), rows_per_slab,
            SEISMIC_DIM_KV * ATTENTION_W), 0, kv_head, lane, v);
    }

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_key(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        prefill_stage_slab<THREADS, Scalar, KEYS>(staged, key, rows_per_slab, first, end, kv_head, thread_index);
    }

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_value(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        prefill_stage_slab<THREADS, Scalar, KEYS>(staged, value, rows_per_slab, first, end, kv_head,
            thread_index);
    }

    // The direct form's operand rows for a span that holds row `at`: its slab
    // (a span lies in one slab).
    inline direct_rows<Scalar> direct(int at, uint kv_head) const {
        const uint rows = uint(rows_per_slab);
        const uint index = uint(at) / rows;
        return direct_rows<Scalar>{slab::region<Scalar>(key, index) + kv_head * ATTENTION_W,
            slab::region<Scalar>(value, index) + kv_head * ATTENTION_W, int(index * rows), int(rows)};
    }

    // The matrix decode's key operand: the history dtype. Its values enter the
    // P.V product as F16, exact for activation values within F16's range.
    typedef Scalar KeyOperand;

    // One simdgroup's rows [first, first + KEYS) of columns [col0, col0 + WC)
    // of one kv head in registers, eight elements per 16-byte piece split
    // over the 32 lanes; rows at or past `end` are zero. Stored into tile
    // regions of pitch WC + 8.
    template <uint KEYS, uint WC>
    struct decode_tile {
        enum : uint {
            PIECES = WC / 8,
            COUNT = (KEYS * PIECES + 31) / 32,
        };
        uint4 key[COUNT];
        uint4 value[COUNT];

        inline void store_key_direct(threadgroup Scalar *staged, uint lane) const {
            store_key(staged, lane);
        }

        inline void store_value_direct(threadgroup half *staged, uint lane) const {
            store_value(staged, lane);
        }

        inline void store_key(threadgroup Scalar *staged, uint lane) const {
            ATTENTION_UNROLL
            for (uint n = 0; n < COUNT; ++n) {
                const uint item = lane + n * 32;
                if (item < KEYS * PIECES)
                    *reinterpret_cast<threadgroup uint4 *>(staged + (item / PIECES) * (WC + 8) + (item % PIECES) * 8) =
                        key[n];
            }
        }

        inline void store_value(threadgroup half *staged, uint lane) const {
            ATTENTION_UNROLL
            for (uint n = 0; n < COUNT; ++n) {
                const uint item = lane + n * 32;
                if (item >= KEYS * PIECES)
                    continue;
                uint converted[4];
                ATTENTION_UNROLL
                for (uint i = 0; i < 4; ++i)
                    converted[i] = as_type<uint>(half2(float2(as_type<vec<Scalar, 2>>(value[n][i]))));
                *reinterpret_cast<threadgroup uint4 *>(staged + (item / PIECES) * (WC + 8) + (item % PIECES) * 8) =
                    uint4(converted[0], converted[1], converted[2], converted[3]);
            }
        }
    };

    template <uint KEYS, uint WC>
    inline void load_tile(thread decode_tile<KEYS, WC> &tile, int first, int end, uint kv_head, uint col0,
        uint lane) const {
        constexpr uint PIECES = decode_tile<KEYS, WC>::PIECES;
        const decode_slab_tile<Scalar> keys(key, rows_per_slab, first);
        const decode_slab_tile<Scalar> values(value, rows_per_slab, first);
        ATTENTION_UNROLL
        for (uint n = 0; n < decode_tile<KEYS, WC>::COUNT; ++n) {
            const uint item = lane + n * 32;
            const int t = first + int(item / PIECES);
            tile.key[n] = uint4(0);
            tile.value[n] = uint4(0);
            if (item < KEYS * PIECES && t < end) {
                const ulong column = ulong(kv_head) * ATTENTION_W + col0 + (item % PIECES) * 8;
                tile.key[n] = *reinterpret_cast<device const uint4 *>(
                    keys.row(item / PIECES, SEISMIC_DIM_KV * ATTENTION_W) + column);
                tile.value[n] = *reinterpret_cast<device const uint4 *>(
                    values.row(item / PIECES, SEISMIC_DIM_KV * ATTENTION_W) + column);
            }
        }
    }
};

// Affine K8/V4 history: code rows plus group (scale, zero) pairs per (row, kv
// head). Its products take F16 operands, the codec's coefficient element: a
// staged tile holds the decoded values code * scale + zero rounded to F16 (a
// BF16 rounding would cost the 8-bit keys up to a code step), and queries,
// fresh keys and values (activation-dtype values, exact in F16 within the
// codec's range) and probabilities enter as F16.
struct affine_history {
    enum : bool { AFFINE = true };
    typedef half Operand;
    device const ulong *key_codes;
    device const ulong *key_coefficients;
    device const ulong *value_codes;
    device const ulong *value_coefficients;
    ulong rows_per_slab;

    inline void append(int destination, uint kv_head, uint lane, thread const float (&k)[ATTENTION_E],
        thread const Scalar (&v)[ATTENTION_E]) const {
        typedef lane_codes<ATTENTION_KEY_BITS> key_lane;
        typedef lane_codes<ATTENTION_VALUE_BITS> value_lane;
        const ulong vector = kv_head;
        float x[ATTENTION_E];
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            x[i] = float(Scalar(k[i]));
        encode<ATTENTION_KEY_BITS>(x,
            slab::row<uint>(key_codes, ulong(destination), rows_per_slab,
                SEISMIC_DIM_KV * key_lane::row_words) + vector * key_lane::row_words,
            slab::row<half>(key_coefficients, ulong(destination), rows_per_slab,
                SEISMIC_DIM_KV * key_lane::pairs * 2) + vector * key_lane::pairs * 2, lane);
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            x[i] = float(v[i]);
        encode<ATTENTION_VALUE_BITS>(x,
            slab::row<uint>(value_codes, ulong(destination), rows_per_slab,
                SEISMIC_DIM_KV * value_lane::row_words) + vector * value_lane::row_words,
            slab::row<half>(value_coefficients, ulong(destination), rows_per_slab,
                SEISMIC_DIM_KV * value_lane::pairs * 2) + vector * value_lane::pairs * 2, lane);
    }

    // Code piece c (16 bytes: 128 / B codes, within one group) of one
    // vector's code row decoded to F16, element pairs with the low element in
    // the low half.
    template <uint B>
    static inline void decode_piece(device const uint *codes, device const half *coefficients, uint c,
        thread uint (&packed)[64 / B]) {
        constexpr uint PER = 128 / B;
        constexpr uint MASK = (1u << B) - 1u;
        static_assert(ATTENTION_GROUP % PER == 0, "a code piece lies in one group");
        const uint4 words = *reinterpret_cast<device const uint4 *>(codes + c * 4);
        const float2 pair = float2(*reinterpret_cast<device const half2 *>(
            coefficients + (c * PER / ATTENTION_GROUP) * 2));
        ATTENTION_UNROLL
        for (uint i = 0; i < PER; i += 2) {
            const uint word = words[i * B / 32];
            const uint shift = (i * B) % 32;
            const half lo = half(metal::fma(float((word >> shift) & MASK), pair.x, pair.y));
            const half hi = half(metal::fma(float((word >> (shift + B)) & MASK), pair.x, pair.y));
            packed[i / 2] = uint(as_type<ushort>(lo)) | (uint(as_type<ushort>(hi)) << 16);
        }
    }

    // Rows [first, first + KEYS) of one kv head decoded into `staged`, one
    // code piece per item; rows at or past `end` are zero.
    template <uint B, uint THREADS, uint KEYS = PREFILL_KEYS>
    static inline void stage(threadgroup half *staged, device const ulong *code_table,
        device const ulong *coefficient_table, ulong rows_per_slab, int first, int end,
        uint kv_head, uint thread_index) {
        constexpr uint W = ATTENTION_W;
        constexpr uint PER = 128 / B;
        constexpr uint PIECES = W / PER;
        constexpr uint ROW_WORDS = lane_codes<B>::row_words;
        constexpr uint PAIRS = lane_codes<B>::pairs;
        ATTENTION_ROLLED
        for (uint item = thread_index; item < KEYS * PIECES; item += THREADS) {
            const uint k = item / PIECES;
            const uint c = item % PIECES;
            const int t = first + int(k);
            threadgroup half *to = staged + k * PREFILL_PITCH + c * PER;
            if (t < end) {
                device const uint *codes = slab::row<uint>(code_table, ulong(t), rows_per_slab,
                    SEISMIC_DIM_KV * ROW_WORDS);
                device const half *coefficients = slab::row<half>(coefficient_table, ulong(t), rows_per_slab,
                    SEISMIC_DIM_KV * PAIRS * 2);
                uint packed[PER / 2];
                decode_piece<B>(codes + ulong(kv_head) * ROW_WORDS, coefficients + ulong(kv_head) * PAIRS * 2, c,
                    packed);
                ATTENTION_UNROLL
                for (uint j = 0; j < PER / 2; j += 4)
                    *reinterpret_cast<threadgroup uint4 *>(to + 2 * j) =
                        uint4(packed[j], packed[j + 1], packed[j + 2], packed[j + 3]);
            } else {
                ATTENTION_UNROLL
                for (uint j = 0; j < PER; j += 8)
                    *reinterpret_cast<threadgroup uint4 *>(to + j) = uint4(0);
            }
        }
    }

    // History row `row` of every kv head decoded (or zeroed, when not
    // `decode`) into its [KV, W] F16 row at `to`, its code pieces striped
    // over one simdgroup's lanes.
    template <uint B>
    inline void decode_row(device const ulong *code_table, device const ulong *coefficient_table, uint row,
        bool decode, device half *to, uint lane) const {
        constexpr uint W = ATTENTION_W;
        constexpr uint PER = 128 / B;
        constexpr uint PIECES = W / PER;
        constexpr uint ROW_WORDS = lane_codes<B>::row_words;
        constexpr uint PAIRS = lane_codes<B>::pairs;
        device const uint *codes = nullptr;
        device const half *coefficients = nullptr;
        if (decode) {
            codes = slab::row<uint>(code_table, ulong(row), rows_per_slab, SEISMIC_DIM_KV * ROW_WORDS);
            coefficients = slab::row<half>(coefficient_table, ulong(row), rows_per_slab,
                SEISMIC_DIM_KV * PAIRS * 2);
        }
        ATTENTION_ROLLED
        for (uint item = lane; item < SEISMIC_DIM_KV * PIECES; item += 32) {
            const uint kv_head = item / PIECES;
            const uint c = item % PIECES;
            uint packed[PER / 2];
            if (decode) {
                decode_piece<B>(codes + kv_head * ROW_WORDS, coefficients + kv_head * PAIRS * 2, c, packed);
            } else {
                ATTENTION_UNROLL
                for (uint j = 0; j < PER / 2; ++j)
                    packed[j] = 0u;
            }
            device half *piece = to + kv_head * W + c * PER;
            ATTENTION_UNROLL
            for (uint j = 0; j < PER / 2; j += 4)
                *reinterpret_cast<device uint4 *>(piece + 2 * j) =
                    uint4(packed[j], packed[j + 1], packed[j + 2], packed[j + 3]);
        }
    }

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_key(threadgroup half *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        stage<ATTENTION_KEY_BITS, THREADS, KEYS>(staged, key_codes, key_coefficients, rows_per_slab, first, end,
            kv_head, thread_index);
    }

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_value(threadgroup half *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        stage<ATTENTION_VALUE_BITS, THREADS, KEYS>(staged, value_codes, value_coefficients, rows_per_slab, first,
            end, kv_head, thread_index);
    }

    // One simdgroup's code pieces of rows [first, first + KEYS) of columns
    // [col0, col0 + WC) of one kv head, `stage`'s items split over the 32
    // lanes: loaded into registers, then decoded into a tile region of pitch
    // WC + 8 by the matrix decode. Rows at or past `end` hold zero codes and
    // coefficients, which decode to zero.
    template <uint B, uint KEYS, uint WC>
    struct pieces {
        enum : uint {
            PER = 128 / B,
            PIECES = WC / PER,
            COUNT = (KEYS * PIECES + 31) / 32,
        };
        uint4 words[COUNT];
        half2 pair[COUNT];

        inline void load(device const ulong *code_table, device const ulong *coefficient_table,
            ulong rows_per_slab, int first, int end, uint kv_head, uint col0, uint lane) {
            constexpr uint ROW_WORDS = lane_codes<B>::row_words;
            constexpr uint PAIRS = lane_codes<B>::pairs;
            const decode_slab_tile<uint> codes(code_table, rows_per_slab, first);
            const decode_slab_tile<half> coefficients(coefficient_table, rows_per_slab, first);
            ATTENTION_UNROLL
            for (uint n = 0; n < COUNT; ++n) {
                const uint item = lane + n * 32;
                const uint k = item / PIECES;
                const uint column = col0 + (item % PIECES) * PER;
                const int t = first + int(k);
                words[n] = uint4(0);
                pair[n] = half2(0.0h);
                if (item < KEYS * PIECES && t < end) {
                    words[n] = *reinterpret_cast<device const uint4 *>(
                        codes.row(k, SEISMIC_DIM_KV * ROW_WORDS) + ulong(kv_head) * ROW_WORDS + column * B / 32);
                    pair[n] = *reinterpret_cast<device const half2 *>(
                        coefficients.row(k, SEISMIC_DIM_KV * PAIRS * 2)
                            + (ulong(kv_head) * PAIRS + column / ATTENTION_GROUP) * 2);
                }
            }
        }

        // Decodes each element pair in F16: the two codes enter the mantissas
        // of 1024 (0x6400, whose F16 ulp is 1), so subtracting 1024 leaves
        // them exact, and one fused F16 multiply-add applies the group's
        // (scale, zero) with a single rounding of code * scale + zero.
        template <uint PITCH = WC + 8>
        inline void store(threadgroup half *staged, uint lane) const {
            constexpr uint MASK = (1u << B) - 1u;
            ATTENTION_UNROLL
            for (uint n = 0; n < COUNT; ++n) {
                const uint item = lane + n * 32;
                if (item >= KEYS * PIECES)
                    continue;
                threadgroup half *to = staged + (item / PIECES) * PITCH + (item % PIECES) * PER;
                const half2 scale = half2(pair[n].x);
                const half2 zero = half2(pair[n].y);
                uint packed[PER / 2];
                ATTENTION_UNROLL
                for (uint i = 0; i < PER; i += 2) {
                    const uint word = words[n][i * B / 32];
                    const uint shift = (i * B) % 32;
                    const uint codes = ((word >> shift) & MASK) | (((word >> (shift + B)) & MASK) << 16);
                    const half2 exact = as_type<half2>(codes | 0x64006400u) - half2(1024.0h);
                    packed[i / 2] = as_type<uint>(metal::fma(exact, scale, zero));
                }
                ATTENTION_UNROLL
                for (uint j = 0; j < PER / 2; j += 4)
                    *reinterpret_cast<threadgroup uint4 *>(to + 2 * j) =
                        uint4(packed[j], packed[j + 1], packed[j + 2], packed[j + 3]);
            }
        }
    };

    // The matrix decode's operands: F16 keys and values.
    typedef half KeyOperand;

    // A decode tile's key and value pieces of one column slice.
    template <uint KEYS, uint WC>
    struct decode_tile {
        pieces<ATTENTION_KEY_BITS, KEYS, WC> key;
        pieces<ATTENTION_VALUE_BITS, KEYS, WC> value;

        inline void store_key_direct(threadgroup half *staged, uint lane) const {
            key.template store<WC>(staged, lane);
        }

        inline void store_value_direct(threadgroup half *staged, uint lane) const {
            value.template store<WC>(staged, lane);
        }

        inline void store_key(threadgroup half *staged, uint lane) const {
            key.store(staged, lane);
        }

        inline void store_value(threadgroup half *staged, uint lane) const {
            value.store(staged, lane);
        }
    };

    template <uint KEYS, uint WC>
    inline void load_tile(thread decode_tile<KEYS, WC> &tile, int first, int end, uint kv_head, uint col0,
        uint lane) const {
        tile.key.load(key_codes, key_coefficients, rows_per_slab, first, end, kv_head, col0, lane);
        tile.value.load(value_codes, value_coefficients, rows_per_slab, first, end, kv_head, col0, lane);
    }
};

// Affine history decoded to F16 scratch for one call (`prefill_decode`):
// [T + 32, KV, W] key and value planes holding, at its logical row, every
// history row a query row of the batch sees, as the staged tiles hold it,
// and zeros in the rows after each span. The DIRECT form's history operands:
// read as device tensors, or staged like the fresh rows.
struct decoded_history {
    enum : bool { AFFINE = true };
    typedef half Operand;
    device const half *key;
    device const half *value;

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_key(threadgroup half *staged, int first, int end, uint kv_head, uint thread_index) const {
        prefill_stage<THREADS, half, KEYS>(staged, key, first, end, kv_head, thread_index);
    }

    template <uint THREADS, uint KEYS = PREFILL_KEYS>
    inline void stage_value(threadgroup half *staged, int first, int end, uint kv_head, uint thread_index) const {
        prefill_stage<THREADS, half, KEYS>(staged, value, first, end, kv_head, thread_index);
    }

    // The planes are whole: a tile starts at its own row.
    inline direct_rows<half> direct(int, uint kv_head) const {
        return direct_rows<half>{key + kv_head * ATTENTION_W, value + kv_head * ATTENTION_W, 0, INT_MAX};
    }
};

// The decode pre-pass of the DIRECT form over affine history, in two
// launches. `prefill_span_bounds`: one simdgroup per history span index, the
// union over the batch's rows of that span's key interval, to spans[span]
// (lo, hi; equal when empty).
inline void prefill_span_bounds(device const int *visible, device const int *fresh, device int *spans,
    ulong M, ulong R, uint span, uint lane) {
    if (span >= R)
        return;
    int lo = INT_MAX;
    int hi = INT_MIN;
    for (ulong row = lane; row < M; row += 32) {
        int row_lo, row_hi;
        form_span(visible, fresh, row, R, span, row_lo, row_hi);
        if (row_hi > row_lo) {
            lo = metal::min(lo, row_lo);
            hi = metal::max(hi, row_hi);
        }
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (lane == 0) {
        spans[span * 2] = hi > lo ? lo : 0;
        spans[span * 2 + 1] = hi > lo ? hi : 0;
    }
}

// `prefill_decode`: one simdgroup per history row. A row inside a span union
// is decoded for every kv head, 16-byte code pieces striped over the lanes,
// to F16 exactly as `affine_history::stage` decodes it; a row within 32 rows
// after a union (which a span's last tile reads) is zeroed; other rows are
// left alone.
inline void prefill_decode(affine_history history, device const int *spans, device half *keys,
    device half *values, ulong R, uint row, uint lane) {
    constexpr uint W = ATTENTION_W;
    constexpr uint KV = SEISMIC_DIM_KV;
    bool inside = false;
    bool after = false;
    for (ulong span = lane; span < R; span += 32) {
        const int lo = spans[span * 2];
        const int hi = spans[span * 2 + 1];
        inside = inside || (int(row) >= lo && int(row) < hi);
        after = after || (hi > lo && int(row) >= hi && int(row) < hi + 32);
    }
    inside = simd_any(inside);
    if (!inside && !simd_any(after))
        return;
    history.template decode_row<ATTENTION_KEY_BITS>(history.key_codes, history.key_coefficients, row, inside,
        keys + ulong(row) * KV * W, lane);
    history.template decode_row<ATTENTION_VALUE_BITS>(history.value_codes, history.value_coefficients, row, inside,
        values + ulong(row) * KV * W, lane);
}

// L1: one simdgroup per (row, kv head), rows padded to whole QT tiles: the
// kv head's G query heads, then its key and value, all rotated by the row's
// one rotary table. Queries and keys are prepared in the activation dtype and go to
// scratch exactly as rounded, as the history policy's operands (L2 applies
// the softmax scale to the F32 scores); padding rows' queries are zero. The
// value (normalized under ATTENTION_VALUE_NORM) is copied beside the key so
// L2 reads every fresh operand from aligned scratch, and the key and value
// are appended at the row's destination through the history policy. A layer
// without fresh rows prepares only queries.
template <uint QT, class History>
inline void prefill_prepare(History history, device const Scalar *query,
    device const Scalar *key, device const Scalar *value, device const float *query_norm,
    device const float *key_norm, device const float *value_norm, device const int *rotary_components,
    device const float *rotary_frequencies, device const float *rotary_amplitudes,
    device const int *coordinates, device const int *destinations,
    device typename History::Operand *queries, device typename History::Operand *keys,
    device typename History::Operand *values, ulong M, float epsilon, bool inject_only,
    uint group, uint simd, uint lane) {
    typedef typename History::Operand Operand;
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    const ulong item = ulong(group) * 8 + simd;
    const ulong row = item / KV;
    const ulong kv_head = item % KV;
    const ulong padded = (M + QT - 1) / QT * QT;
    if (row >= M) {
        if (inject_only)
            return;
        if (row < padded)
            for (uint g = 0; g < G; ++g)
                for (uint i = 0; i < E; ++i)
                    queries[(row * KV * G + kv_head * G + g) * W + lane * E + i] = Operand(0.0f);
        if (PREFILL_DIRECT != 0 && row < M + 32) {
            // The direct form's last fresh tile reads past the rows: zeros.
            const ulong at = (row * KV + kv_head) * W + lane * E;
            for (uint i = 0; i < E; ++i) {
                keys[at + i] = Operand(0.0f);
                values[at + i] = Operand(0.0f);
            }
        }
        return;
    }
    const rotary_table table(coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes, lane);
    float x[E];
    if (!inject_only) {
        for (uint g = 0; g < G; ++g) {
            const ulong head = row * KV * G + kv_head * G + g;
            head_rotary<ATTENTION_NORM>(query + head * ATTENTION_QUERY_STRIDE, query_norm, table, epsilon, lane, x);
            for (uint i = 0; i < E; ++i)
                queries[head * W + lane * E + i] = Operand(Scalar(x[i]));
        }
    }
    if (!ATTENTION_FRESH)
        return;
    const ulong source = (row * KV + kv_head) * W;
    head_rotary<ATTENTION_NORM>(key + source, key_norm, table, epsilon, lane, x);
    float y[E];
    head_norm<ATTENTION_VALUE_NORM>(value + source, value_norm, epsilon, lane, y);
    Scalar v[E];
    for (uint i = 0; i < E; ++i) {
        if (!inject_only)
            keys[source + lane * E + i] = Operand(Scalar(x[i]));
        v[i] = Scalar(y[i]);
        if (!inject_only)
            values[source + lane * E + i] = Operand(v[i]);
    }
    const int destination = destinations[row];
    if (destination < 0)
        return;
    history.append(destination, uint(kv_head), lane, x, v);
}

// The per-simdgroup arithmetic of one L2 output window: 8 query rows of one
// head against each staged key tile, the online softmax in the exp2 domain,
// and the F32 output from activation-dtype probabilities. `scores` forms a
// key tile's probabilities and rescales the output (K staged); `accumulate`
// adds their product with the staged V; `store` publishes the window.
struct prefill_rows {
    device const int *visible;
    device const int *fresh;
    ulong R;
    ulong index;
    bool historical;
    // The [lo, hi) key bounds of token `token` in this span.
    inline int2 at(ulong token) const {
        device const int *bounds = historical ? visible + (token * R + index) * 2 : fresh + token * 2;
        return int2(bounds[0], bounds[1]);
    }
};

// The fragment form: lane (fm, fn) holds row fm, columns fn, fn + 1 of every
// 8x8 fragment; each row's statistics are replicated over its lanes. The
// state is one `prefill_fragments` value.
template <uint QT, class History>
struct prefill_fragments {
    typedef typename History::Operand Operand;
    static constant constexpr uint ROWS = 8;
    static constant constexpr uint W = ATTENTION_W;
    static constant constexpr uint KEYS = PREFILL_KEYS;
    static constant constexpr uint DB = W / 8;
    static constant constexpr uint KB = KEYS / 8;
    static constant constexpr uint WINDOW = W < PREFILL_WINDOW ? W : PREFILL_WINDOW;
    static constant constexpr uint WB = WINDOW / 8;
    static constant constexpr uint PITCH = PREFILL_PITCH;
    uint fm, fn;
    simdgroup_matrix<float, 8, 8> output[WB];
    simdgroup_matrix<Operand, 8, 8> probabilities[KB];
    float maximum;
    float denominator;

    // The lane's fragment coordinates. The struct stays an aggregate: Metal
    // 4.1 gives simdgroup matrices no default constructor for a member
    // initializer.
    static inline void place(thread prefill_fragments &self, uint lane) {
        const uint quad = lane / 4;
        self.fm = (quad & 4) + ((lane / 2) % 4);
        self.fn = (quad & 2) * 2 + (lane % 2) * 2;
    }

    static inline void reset(thread prefill_fragments &self) {
        ATTENTION_UNROLL
        for (uint d = 0; d < WB; ++d)
            self.output[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        self.maximum = -INFINITY;
        self.denominator = 0.0f;
    }

    static inline void scores(device const Operand *query_rows, threadgroup const Operand *staged, int first,
        bool common, prefill_rows rows, ulong first_token, ulong M, float scale, thread prefill_fragments &self) {
        const ulong token = first_token + self.fm;
        simdgroup_matrix<float, 8, 8> scores[KB];
        ATTENTION_UNROLL
        for (uint j = 0; j < KB; ++j)
            scores[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        ATTENTION_UNROLL
        for (uint d = 0; d < DB; ++d) {
            simdgroup_matrix<Operand, 8, 8> q;
            simdgroup_load(q, query_rows + d * 8, SEISMIC_DIM_KV * SEISMIC_DIM_G * W);
            ATTENTION_UNROLL
            for (uint j = 0; j < KB; ++j) {
                simdgroup_matrix<Operand, 8, 8> k;
                simdgroup_load(k, staged + j * 8 * PITCH + d * 8, PITCH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(scores[j], q, k, scores[j]);
            }
        }

        // Rows' bounds are read only for a tile outside the common
        // interval, so they hold no registers across the key loop.
        int row_lo = 0;
        int row_hi = 0;
        if (!common && token < M) {
            const int2 bounds = rows.at(token);
            row_lo = bounds.x;
            row_hi = bounds.y;
        }
        float tile_maximum = -INFINITY;
        ATTENTION_UNROLL
        for (uint j = 0; j < KB; ++j) {
            ATTENTION_UNROLL
            for (uint e = 0; e < 2; ++e) {
                float s = scores[j].thread_elements()[e] * scale;
                if (!common) {
                    const int t = first + int(j * 8 + self.fn + e);
                    if (!(t >= row_lo && t < row_hi))
                        s = -INFINITY;
                }
                scores[j].thread_elements()[e] = s;
                tile_maximum = metal::max(tile_maximum, s);
            }
        }
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
        tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
        const float next = metal::max(self.maximum, tile_maximum);
        const bool seen = next > -INFINITY;
        const float carry = seen ? metal::fast::exp2(self.maximum - next) : 1.0f;
        // Probabilities enter the PV product as operands; the
        // denominator sums them in F32.
        float tile_sum = 0.0f;
        ATTENTION_UNROLL
        for (uint j = 0; j < KB; ++j) {
            ATTENTION_UNROLL
            for (uint e = 0; e < 2; ++e) {
                const float p = seen ? metal::fast::exp2(scores[j].thread_elements()[e] - next) : 0.0f;
                self.probabilities[j].thread_elements()[e] = Operand(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
        tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
        self.denominator = metal::fma(self.denominator, carry, tile_sum);
        self.maximum = next;
        ATTENTION_UNROLL
        for (uint d = 0; d < WB; ++d) {
            self.output[d].thread_elements()[0] *= carry;
            self.output[d].thread_elements()[1] *= carry;
        }
    }

    static inline void accumulate(threadgroup const Operand *staged, uint window_first,
        thread prefill_fragments &self) {
        ATTENTION_UNROLL
        for (uint d = 0; d < WB; ++d) {
            ATTENTION_UNROLL
            for (uint j = 0; j < KB; ++j) {
                simdgroup_matrix<Operand, 8, 8> v;
                simdgroup_load(v, staged + j * 8 * PITCH + window_first + d * 8, PITCH);
                simdgroup_multiply_accumulate(self.output[d], self.probabilities[j], v, self.output[d]);
            }
        }
    }

    // A split tile stores (partial output, maximum, denominator) per row and
    // partition; an unsplit one its gated output.
    static inline void store(device const Scalar *query, device const Scalar *gate, device Scalar *result,
        device float *partials, device float *statistics, ulong first_token, ulong M, uint head, uint partition,
        uint active, uint window_first, bool softplus, thread prefill_fragments &self) {
        constexpr uint H = SEISMIC_DIM_KV * SEISMIC_DIM_G;
        const ulong token = first_token + self.fm;
        if (token >= M)
            return;
        if (active > 1) {
            const ulong slot = (ulong(partition) * M + token) * H + head;
            ATTENTION_UNROLL
            for (uint d = 0; d < WB; ++d)
                ATTENTION_UNROLL
                for (uint e = 0; e < 2; ++e)
                    partials[slot * W + window_first + d * 8 + self.fn + e] = self.output[d].thread_elements()[e];
            if (self.fn == 0) {
                statistics[slot * 2] = self.maximum;
                statistics[slot * 2 + 1] = self.denominator;
            }
            return;
        }
        const float inverse = 1.0f / metal::max(self.denominator, 1e-30f);
        ATTENTION_UNROLL
        for (uint d = 0; d < WB; ++d) {
            ATTENTION_UNROLL
            for (uint e = 0; e < 2; ++e) {
                const uint column = window_first + d * 8 + self.fn + e;
                result[(token * H + head) * W + column] = gate_output(query, gate, token, head, column,
                    self.output[d].thread_elements()[e] * inverse, softplus);
            }
        }
    }
};

#if SEISMIC_HAS_TENSOR_OPS
// The tensor-operation form: the first QT G / 16 simdgroups own 16 rows each
// (`prefill_owner`) and multiply, while every simdgroup stages. Per key tile:
// computing simdgroups form S = Q K^T (`matmul2d`, execution_simdgroup scope)
// from the staged K; after a threadgroup barrier (no simdgroup still reads K)
// each stores S over the K tile, ROWS x KEYS floats per simdgroup, where
// lanes own rows (2 lanes per row, KEYS / 2 columns each, as the fragment
// form's lanes do) for the online softmax; the operand-rounded probabilities go to
// the simdgroup's `exchange` slot as the P V left operand, and once V is
// staged (over K) the output accumulates P V. The output is a cooperative
// tensor; per-row carries and inverses reach its rows through the slot's 16
// row floats. The form applies when 16 <= QT and every S fits over the tile
// (QT G F32 rows within a tile row's bytes).
constexpr bool prefill_tensors_fit(uint QT) {
    return QT >= 16 && QT * PREFILL_HEADS * 4 <= PREFILL_PITCH * 2;
}

template <uint QT, class History>
struct prefill_tensors {
    typedef typename History::Operand Operand;
    static constant constexpr uint ROWS = 16;
    static constant constexpr uint KEYS = PREFILL_KEYS;
    static constant constexpr uint W = ATTENTION_W;
    static constant constexpr uint WINDOW = W < PREFILL_WINDOW ? W : PREFILL_WINDOW;
    static constant constexpr int32_t PITCH = PREFILL_PITCH;
    static constant constexpr uint LANES = 32 / ROWS;
    static constant constexpr uint COLUMNS = KEYS / LANES;
    static constant constexpr uint THREADS = QT * PREFILL_HEADS * 4;
    // Floats of one simdgroup's exchange slot: P, then one value per row.
    static constant constexpr uint PUBLISHED = ROWS * KEYS * sizeof(Operand) / 4;
    static constant constexpr uint EXCHANGE = PUBLISHED + ROWS;
    typedef metal::extents<int32_t, W, ROWS> q_extents;
    typedef metal::extents<int32_t, W, KEYS> k_extents;
    typedef metal::extents<int32_t, WINDOW, KEYS> v_extents;
    typedef metal::extents<int32_t, KEYS, ROWS> s_extents;
    typedef metal::tensor<device Operand, q_extents, metal::tensor_inline> q_tensor;
    typedef metal::tensor<threadgroup Operand, k_extents, metal::tensor_inline> k_tensor;
    typedef metal::tensor<threadgroup Operand, v_extents, metal::tensor_inline> v_tensor;
    typedef metal::tensor<threadgroup float, s_extents, metal::tensor_inline> s_tensor;
    typedef metal::tensor<threadgroup Operand, s_extents, metal::tensor_inline> p_tensor;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, KEYS, W, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        metal::execution_simdgroup> score_op;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, WINDOW, KEYS, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroup> output_op;
    typedef typename output_op::template cooperative_tensor_row_reduction_destination_t<p_tensor, v_tensor, float>
        output_rows;

    // Each lane's row value (published by the row's first lane) as the
    // output's row tensor.
    static inline output_rows load_rows(threadgroup float *exchange, uint lane, float value) {
        threadgroup float *published = exchange + PUBLISHED;
        if (lane % LANES == 0)
            published[lane / LANES] = value;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        output_op op;
        output_rows rows = op.template get_row_reduction_destination_cooperative_tensor<p_tensor, v_tensor, float>();
        ATTENTION_UNROLL
        for (uint16_t i = 0; i < rows.get_capacity(); ++i)
            if (rows.is_valid_element(i))
                rows[i] = published[rows.get_multidimensional_index(i)[0]];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        return rows;
    }

    static inline void windows(History history, device const Scalar *query, device const Scalar *gate,
        device const int *visible, device const int *fresh, device Scalar *result, device const Operand *keys,
        device const Operand *values, device float *partials, device float *statistics, ulong M, ulong R,
        float scale, bool softplus, threadgroup Operand *staged, threadgroup const prefill_interval *intervals,
        threadgroup float *exchange, uint kv_head, uint partition, uint active, uint tiles_lo, uint tiles_hi,
        device const Operand *query_rows, uint head, ulong first_token, bool computes, uint owner,
        uint thread_index, uint lane) {
        constexpr uint H = SEISMIC_DIM_KV * SEISMIC_DIM_G;
        q_tensor q(const_cast<device Operand *>(query_rows), q_extents(),
            metal::array<int32_t, 2>{1, int32_t(H * W)});
        k_tensor k(staged, k_extents(), metal::array<int32_t, 2>{1, PITCH});
        threadgroup float *slot = reinterpret_cast<threadgroup float *>(staged) + owner * ROWS * KEYS;
        s_tensor s(slot, s_extents(), metal::array<int32_t, 2>{1, int32_t(KEYS)});
        threadgroup Operand *probabilities = reinterpret_cast<threadgroup Operand *>(exchange);
        p_tensor p(probabilities, s_extents(), metal::array<int32_t, 2>{1, int32_t(KEYS)});
        score_op score;
        output_op product;
        auto scores = score.template get_destination_cooperative_tensor<q_tensor, k_tensor, float>();
        auto output = product.template get_destination_cooperative_tensor<p_tensor, v_tensor, float>();
        const uint row = lane / LANES;
        const uint column0 = (lane % LANES) * COLUMNS;
        const ulong token = first_token + row;
        for (uint window = 0; window < W / WINDOW; ++window) {
            const uint window_first = window * WINDOW;
            v_tensor v(staged + window_first, v_extents(), metal::array<int32_t, 2>{1, PITCH});
            ATTENTION_UNROLL
            for (uint16_t i = 0; i < output.get_capacity(); ++i)
                if (output.is_valid_element(i))
                    output[i] = 0.0f;
            float maximum = -INFINITY;
            float denominator = 0.0f;

            uint tiles_before = 0;
            for (ulong index = 0; index <= R; ++index) {
                const prefill_interval interval = intervals[index];
                if (interval.hi <= interval.lo)
                    continue;
                const uint span_tiles = uint(interval.hi - interval.lo + int(KEYS) - 1) / KEYS;
                const uint span_first = tiles_before;
                tiles_before += span_tiles;
                if (span_first + span_tiles <= tiles_lo || span_first >= tiles_hi)
                    continue;
                const uint own_lo = metal::max(tiles_lo, span_first) - span_first;
                const uint own_hi = metal::min(tiles_hi, span_first + span_tiles) - span_first;
                const bool historical = index < R;
                const prefill_rows rows{visible, fresh, R, index, historical};
                for (uint own = own_lo; own < own_hi; ++own) {
                    const int first = interval.lo + int(own * KEYS);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (historical)
                        history.template stage_key<THREADS>(staged, first, interval.hi, kv_head, thread_index);
                    else
                        prefill_stage<THREADS>(staged, keys, first, interval.hi, kv_head, thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (computes)
                        score.run(q, k, scores);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (computes) {
                        scores.store(s);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        const bool common = first >= interval.common_lo
                            && first + int(KEYS) <= interval.common_hi;
                        // Rows' bounds are read only for a tile outside the
                        // common interval.
                        int row_lo = 0;
                        int row_hi = 0;
                        if (!common && token < M) {
                            const int2 bounds = rows.at(token);
                            row_lo = bounds.x;
                            row_hi = bounds.y;
                        }
                        float x[COLUMNS];
                        float tile_maximum = -INFINITY;
                        ATTENTION_UNROLL
                        for (uint j = 0; j < COLUMNS; ++j) {
                            float value = slot[row * KEYS + column0 + j] * scale;
                            if (!common) {
                                const int t = first + int(column0 + j);
                                if (!(t >= row_lo && t < row_hi))
                                    value = -INFINITY;
                            }
                            x[j] = value;
                            tile_maximum = metal::max(tile_maximum, value);
                        }
                        ATTENTION_UNROLL
                        for (ushort offset = 1; offset < LANES; offset <<= 1)
                            tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, offset));
                        const float next = metal::max(maximum, tile_maximum);
                        const bool seen = next > -INFINITY;
                        const float carry = seen ? metal::fast::exp2(maximum - next) : 1.0f;
                        // Probabilities enter the PV product as operands; the
                        // denominator sums them in F32.
                        float tile_sum = 0.0f;
                        ATTENTION_UNROLL
                        for (uint j = 0; j < COLUMNS; ++j) {
                            const float probability = seen ? metal::fast::exp2(x[j] - next) : 0.0f;
                            probabilities[row * KEYS + column0 + j] = Operand(probability);
                            tile_sum += probability;
                        }
                        ATTENTION_UNROLL
                        for (ushort offset = 1; offset < LANES; offset <<= 1)
                            tile_sum += simd_shuffle_xor(tile_sum, offset);
                        denominator = metal::fma(denominator, carry, tile_sum);
                        maximum = next;
                        // A tile that raises no row's maximum leaves the
                        // output as is.
                        if (!simd_all(carry == 1.0f)) {
                            output_rows carries = load_rows(exchange, lane, carry);
                            ATTENTION_UNROLL
                            for (uint16_t i = 0; i < output.get_capacity(); ++i)
                                if (output.is_valid_element(i))
                                    output[i] *= *carries.map_iterator(output.get_iterator(i));
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (historical)
                        history.template stage_value<THREADS>(staged, first, interval.hi, kv_head, thread_index);
                    else
                        prefill_stage<THREADS>(staged, values, first, interval.hi, kv_head, thread_index);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (computes)
                        product.run(p, v, output);
                }
            }
            if (!computes)
                continue;
            // A split tile stores (partial output, maximum, denominator) per
            // row and partition; an unsplit one its gated output.
            if (active > 1) {
                if (lane % LANES == 0 && token < M) {
                    const ulong slot_index = (ulong(partition) * M + token) * H + head;
                    statistics[slot_index * 2] = maximum;
                    statistics[slot_index * 2 + 1] = denominator;
                }
                ATTENTION_UNROLL
                for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                    if (!output.is_valid_element(i))
                        continue;
                    const auto index = output.get_multidimensional_index(i);
                    const ulong row_token = first_token + index[1];
                    if (row_token < M)
                        partials[((ulong(partition) * M + row_token) * H + head) * W + window_first + index[0]]
                            = output[i];
                }
                continue;
            }
            output_rows inverse = load_rows(exchange, lane, 1.0f / metal::max(denominator, 1e-30f));
            ATTENTION_UNROLL
            for (uint16_t i = 0; i < output.get_capacity(); ++i) {
                if (!output.is_valid_element(i))
                    continue;
                const auto index = output.get_multidimensional_index(i);
                const ulong row_token = first_token + index[1];
                const uint column = window_first + index[0];
                if (row_token < M)
                    result[(row_token * H + head) * W + column] = gate_output(query, gate, row_token, head, column,
                        output[i] * *inverse.map_iterator(output.get_iterator(i)), softplus);
            }
        }
    }
};

// The direct form (DIRECT): every simdgroup owns 16 rows (16 tokens of one
// query head) and nothing is staged. Per key tile a simdgroup forms S = Q K^T with the
// queries in registers and K a device tensor over the history rows (or the
// fresh rows' scratch), runs the online softmax on the cooperative scores,
// and accumulates P V with the probabilities as the product's cooperative
// left input and V a device tensor. The threadgroup shares only its run
// schedule, written once before the simdgroups start; they take no barrier
// after it and run independently.
//
// A tile's operand rows start at `start` <= first (`History::direct`); keys
// before `first`, past the tile and outside a row's interval are masked, and
// their values enter the product under a zero probability.
//
// The form relies on the cooperative layouts of one simdgroup: element i of
// the score destination and of the product's left input are the same (row,
// key), and a lane holds the same rows of the scores and of the output.
template <uint QT, class History>
struct prefill_direct {
    typedef typename History::Operand Operand;
    static constant constexpr uint ROWS = 16;
    // Keys per operation: the entry's key tiles (PREFILL_KEYS, the unit of
    // its partitions) run in 16-key steps, the fastest product shape.
    static constant constexpr uint KEYS = 16;
    static constant constexpr uint TILE = PREFILL_KEYS;
    static constant constexpr uint W = ATTENTION_W;
    static constant constexpr uint WINDOW = W < PREFILL_WINDOW ? W : PREFILL_WINDOW;
    typedef metal::extents<int32_t, W, ROWS> q_extents;
    typedef metal::extents<int32_t, W, KEYS> k_extents;
    typedef metal::extents<int32_t, WINDOW, KEYS> v_extents;
    typedef metal::extents<int32_t, KEYS, ROWS> s_extents;
    typedef metal::tensor<device Operand, q_extents, metal::tensor_inline> q_tensor;
    typedef metal::tensor<device Operand, k_extents, metal::tensor_inline> k_tensor;
    typedef metal::tensor<device Operand, v_extents, metal::tensor_inline> v_tensor;
    typedef metal::tensor<device Operand, s_extents, metal::tensor_inline> p_tensor;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, KEYS, W, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        metal::execution_simdgroup> score_op;
    typedef mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(ROWS, WINDOW, KEYS, false, false, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        metal::execution_simdgroup> output_op;
    // The cooperative layout of a 16-row simdgroup: a lane holds two rows,
    // element i in row slot (i / 4) % 2, of the scores (ELEMENTS a lane) and
    // of the output alike; a row is shared by the four lanes that differ in
    // lane bits 0 and 3. Every loop over a lane's elements takes these
    // constant counts: under the tensors' own `get_capacity()` the loops do
    // not unroll, which measured 20% of the launch.
    static constant constexpr uint SLOTS = 2;
    static constant constexpr uint QUERIES = ROWS * W / 32;
    static constant constexpr uint ELEMENTS = ROWS * KEYS / 32;
    static constant constexpr uint OUTPUTS = ROWS * WINDOW / 32;
    static constant constexpr float LAZY = 2.0f;
    static constexpr uint slot(uint i) { return (i >> 2) & 1u; }
    static inline float row_maximum(float value) {
        value = metal::max(value, simd_shuffle_xor(value, ushort(1)));
        return metal::max(value, simd_shuffle_xor(value, ushort(8)));
    }
    static inline float row_sum(float value) {
        value += simd_shuffle_xor(value, ushort(1));
        return value + simd_shuffle_xor(value, ushort(8));
    }

    // The threadgroup's schedule in threadgroup memory, after its intervals:
    // runs[0] holds (run count, active partitions), then the runs of its
    // partition's tiles [tiles_lo, tiles_hi) in span order. Per span, the
    // steps that lie in every row's interval and read their own operand
    // rows form one unmasked run; the steps before and after it are masked
    // runs. One thread writes it.
    static inline void schedule(History history, threadgroup const prefill_interval *intervals,
        threadgroup prefill_run *runs, ulong M, ulong R, uint kv_head, uint active, uint tiles_lo, uint tiles_hi) {
        uint count = 0;
        uint tiles_before = 0;
        for (uint index = 0; index <= uint(R); ++index) {
            const prefill_interval interval = intervals[index];
            if (interval.hi <= interval.lo)
                continue;
            const uint span_tiles = uint(interval.hi - interval.lo + int(TILE) - 1) / TILE;
            const uint span_first = tiles_before;
            tiles_before += span_tiles;
            if (span_first + span_tiles <= tiles_lo || span_first >= tiles_hi)
                continue;
            const int span_lo = interval.lo + int((metal::max(tiles_lo, span_first) - span_first) * TILE);
            const int span_hi = metal::min(interval.hi,
                interval.lo + int((metal::min(tiles_hi, span_first + span_tiles) - span_first) * TILE));
            // The keys every row sees lie in one slab (inside every row's
            // span); its whole steps from the first step boundary on are the
            // unmasked run.
            const int whole_lo = metal::min(span_hi,
                span_lo + (metal::max(interval.common_lo, span_lo) - span_lo + int(KEYS) - 1) / int(KEYS) * int(KEYS));
            int whole_hi = whole_lo;
            if (whole_lo < span_hi) {
                const int limit = index < uint(R) ? history.direct(whole_lo, kv_head).limit() : int(M) + int(KEYS);
                const int reach = metal::min(metal::min(interval.common_hi, span_hi), limit);
                whole_hi += metal::max(0, reach - whole_lo) / int(KEYS) * int(KEYS);
            }
            if (whole_lo > span_lo)
                runs[++count] = prefill_run{span_lo, whole_lo, index, 1u};
            if (whole_hi > whole_lo)
                runs[++count] = prefill_run{whole_lo, whole_hi, index, 0u};
            if (span_hi > whole_hi)
                runs[++count] = prefill_run{whole_hi, span_hi, index, 1u};
        }
        runs[0] = prefill_run{int(count), int(active), 0u, 0u};
    }

    static inline void windows(History history, device const Scalar *query, device const Scalar *gate,
        device const int *visible, device const int *fresh, device Scalar *result, device const Operand *keys,
        device const Operand *values, device float *partials, device float *statistics, ulong M, ulong R,
        float scale, bool softplus, threadgroup const prefill_interval *intervals,
        threadgroup const prefill_run *runs, uint kv_head, uint partition, device const Operand *query_rows,
        uint head, ulong first_token) {
        constexpr uint H = SEISMIC_DIM_KV * SEISMIC_DIM_G;
        constexpr uint KV = SEISMIC_DIM_KV;
        score_op score;
        output_op product;
        auto q = score.template get_left_input_cooperative_tensor<Operand, Operand, float>();
        ATTENTION_UNROLL
        for (uint16_t i = 0; i < QUERIES; ++i) {
            const auto index = q.get_multidimensional_index(i);
            q[i] = query_rows[ulong(index[1]) * H * W + index[0]];
        }
        auto scores = score.template get_destination_cooperative_tensor<q_tensor, k_tensor, float>();
        auto p = product.template get_left_input_cooperative_tensor<Operand, Operand, float>();
        auto output = product.template get_destination_cooperative_tensor<p_tensor, v_tensor, float>();
        const uint rows_total = uint(M);
        for (uint window = 0; window < W / WINDOW; ++window) {
            const uint window_first = window * WINDOW;
            ATTENTION_UNROLL
            for (uint16_t i = 0; i < OUTPUTS; ++i)
                output[i] = 0.0f;
            float maximum[SLOTS];
            float denominator[SLOTS];
            ATTENTION_UNROLL
            for (uint h = 0; h < SLOTS; ++h) {
                maximum[h] = -INFINITY;
                denominator[h] = 0.0f;
            }
            // One 16-key step over the operand rows at `key_rows` and
            // `value_rows`, whose scores `x` are scaled and masked.
            const auto absorb = [&](thread float (&x)[ELEMENTS], device const Operand *value_rows) {
                float tile_maximum[SLOTS];
                ATTENTION_UNROLL
                for (uint h = 0; h < SLOTS; ++h)
                    tile_maximum[h] = -INFINITY;
                ATTENTION_UNROLL
                for (uint16_t i = 0; i < ELEMENTS; ++i)
                    tile_maximum[slot(i)] = metal::max(tile_maximum[slot(i)], x[i]);
                // A row's reference maximum moves only when a tile exceeds
                // it by more than LAZY (so probabilities stay below 2^LAZY
                // and most tiles rescale nothing).
                float carry[SLOTS];
                bool rescale = false;
                ATTENTION_UNROLL
                for (uint h = 0; h < SLOTS; ++h) {
                    const float peak = row_maximum(tile_maximum[h]);
                    carry[h] = 1.0f;
                    if (peak > maximum[h] + LAZY) {
                        carry[h] = metal::fast::exp2(maximum[h] - peak);
                        maximum[h] = peak;
                        rescale = true;
                    }
                }
                // Probabilities enter the PV product as operands; each lane
                // sums its own in F32 (the row's lanes carry equal factors,
                // so their sums combine at the end).
                float tile_sum[SLOTS];
                ATTENTION_UNROLL
                for (uint h = 0; h < SLOTS; ++h)
                    tile_sum[h] = 0.0f;
                ATTENTION_UNROLL
                for (uint16_t i = 0; i < ELEMENTS; ++i) {
                    const float next = maximum[slot(i)];
                    const float probability = next > -INFINITY ? metal::fast::exp2(x[i] - next) : 0.0f;
                    p[i] = Operand(probability);
                    tile_sum[slot(i)] += probability;
                }
                ATTENTION_UNROLL
                for (uint h = 0; h < SLOTS; ++h)
                    denominator[h] = metal::fma(denominator[h], carry[h], tile_sum[h]);
                if (rescale) {
                    ATTENTION_UNROLL
                    for (uint16_t i = 0; i < OUTPUTS; ++i)
                        output[i] *= carry[slot(i)];
                }
                v_tensor v(const_cast<device Operand *>(value_rows) + window_first, v_extents(),
                    metal::array<int32_t, 2>{1, int32_t(KV * W)});
                product.run(p, v, output);
            };

            // The threadgroup's runs (`schedule`), so the loop over a run's
            // steps keeps nothing of the span walk in registers.
            const uint count = uint(runs[0].first);
            for (uint entry = 1; entry <= count; ++entry) {
                const prefill_run run = runs[entry];
                const bool historical = run.span < uint(R);
                // The operand rows that hold key `at`: its history slab, or
                // the fresh rows' scratch, which ends in KEYS rows of zeros.
                const auto rows_at = [&](int at) {
                    return historical ? history.direct(at, kv_head)
                        : direct_rows<Operand>{keys + kv_head * W, values + kv_head * W, 0, int(M) + int(KEYS)};
                };
                if (run.masked != 0) {
                    const prefill_interval interval = intervals[run.span];
                    const prefill_rows rows{visible, fresh, R, run.span, historical};
                    // A step takes the keys from `first` to the end of its
                    // tile, of their operand rows or of the run, whichever
                    // is first.
                    for (int first = run.first; first < run.end;) {
                        const direct_rows<Operand> operands = rows_at(first);
                        const int start = operands.start(first, KEYS);
                        const int next = metal::min(metal::min(first + int(KEYS), operands.limit()), run.end);
                        k_tensor k(const_cast<device Operand *>(operands.keys + operands.offset(start)), k_extents(),
                            metal::array<int32_t, 2>{1, int32_t(KV * W)});
                        score.run(q, k, scores);
                        const bool common = first >= interval.common_lo && next <= interval.common_hi;
                        float x[ELEMENTS];
                        ATTENTION_UNROLL
                        for (uint16_t i = 0; i < ELEMENTS; ++i) {
                            const auto at = scores.get_multidimensional_index(i);
                            const uint token = uint(first_token) + uint(at[1]);
                            int lo = first, hi = next;
                            if (!common) {
                                const int2 bounds = token < rows_total ? rows.at(token) : int2(0);
                                lo = metal::max(lo, bounds.x);
                                hi = metal::min(hi, bounds.y);
                            }
                            const int t = start + int(at[0]);
                            x[i] = t >= lo && t < hi ? scores[i] * scale : -INFINITY;
                        }
                        absorb(x, operands.values + operands.offset(start));
                        first = next;
                    }
                    continue;
                }
                const direct_rows<Operand> operands = rows_at(run.first);
                device const Operand *key_rows = operands.keys + operands.offset(run.first);
                device const Operand *value_rows = operands.values + operands.offset(run.first);
                for (uint steps = uint(run.end - run.first) / KEYS; steps > 0; --steps) {
                    k_tensor k(const_cast<device Operand *>(key_rows), k_extents(),
                        metal::array<int32_t, 2>{1, int32_t(KV * W)});
                    score.run(q, k, scores);
                    float x[ELEMENTS];
                    ATTENTION_UNROLL
                    for (uint16_t i = 0; i < ELEMENTS; ++i)
                        x[i] = scores[i] * scale;
                    absorb(x, value_rows);
                    key_rows += KEYS * KV * W;
                    value_rows += KEYS * KV * W;
                }
            }
            ATTENTION_UNROLL
            for (uint h = 0; h < SLOTS; ++h)
                denominator[h] = row_sum(denominator[h]);
            // A split tile stores (partial output, maximum, denominator) per
            // row and partition; an unsplit one its gated output.
            if (runs[0].end > 1) {
                // Every lane of a row holds its statistics and stores them.
                ATTENTION_UNROLL
                for (uint h = 0; h < SLOTS; ++h) {
                    const ulong token = first_token + scores.get_multidimensional_index(uint16_t(4 * h))[1];
                    if (token < M) {
                        const ulong slot_index = (ulong(partition) * M + token) * H + head;
                        statistics[slot_index * 2] = maximum[h];
                        statistics[slot_index * 2 + 1] = denominator[h];
                    }
                }
                ATTENTION_UNROLL
                for (uint16_t i = 0; i < OUTPUTS; ++i) {
                    const auto index = output.get_multidimensional_index(i);
                    const ulong row_token = first_token + index[1];
                    if (row_token < M)
                        partials[((ulong(partition) * M + row_token) * H + head) * W + window_first + index[0]]
                            = output[i];
                }
                continue;
            }
            ATTENTION_UNROLL
            for (uint16_t i = 0; i < OUTPUTS; ++i) {
                const auto index = output.get_multidimensional_index(i);
                const ulong row_token = first_token + index[1];
                const uint column = window_first + index[0];
                const float inverse = 1.0f / metal::max(denominator[slot(i)], 1e-30f);
                if (row_token < M)
                    result[(row_token * H + head) * W + column] = gate_output(query, gate, row_token, head, column,
                        output[i] * inverse, softplus);
            }
        }
    }
};

// The L2 kernel's exchange memory: the tensor form's slot per computing
// simdgroup (every element type is two bytes), when the form fits.
#define PREFILL_EXCHANGE(name, QT) \
    threadgroup float name[attention::prefill_tensors_fit(QT) ? (QT) * PREFILL_HEADS / 16 * (16 * PREFILL_KEYS / 2 + 16) : 1]
#else
#define PREFILL_EXCHANGE(name, QT) threadgroup float *name = nullptr
#endif

// The output windows of one simdgroup's rows over its partition's key tiles
// (L2's loop) on the fragment form. Every simdgroup owns rows, but the
// arithmetic stays guarded by `computes`: the guarded form measured 1.5x
// faster on Apple GPU family 10 (the staging and the fragment arithmetic are
// scheduled apart).
template <uint QT, class History>
inline void prefill_windows(History history, device const Scalar *query, device const Scalar *gate,
    device const int *visible, device const int *fresh, device Scalar *result,
    device const typename History::Operand *keys, device const typename History::Operand *values,
    device float *partials, device float *statistics, ulong M, ulong R, float scale, bool softplus,
    threadgroup typename History::Operand *staged, threadgroup const prefill_interval *intervals,
    uint kv_head, uint partition, uint active, uint tiles_lo, uint tiles_hi,
    device const typename History::Operand *query_rows, uint head, ulong first_token, bool computes,
    uint thread_index, thread prefill_fragments<QT, History> &state) {
    typedef prefill_fragments<QT, History> Form;
    constexpr uint W = ATTENTION_W;
    constexpr uint KEYS = PREFILL_KEYS;
    constexpr uint THREADS = QT * PREFILL_HEADS * (PREFILL_DIRECT != 0 ? 2 : 4);
    constexpr uint WINDOW = W < PREFILL_WINDOW ? W : PREFILL_WINDOW;
    for (uint window = 0; window < W / WINDOW; ++window) {
        const uint window_first = window * WINDOW;
        if (computes)
            Form::reset(state);

        uint tiles_before = 0;
        for (ulong index = 0; index <= R; ++index) {
            const prefill_interval interval = intervals[index];
            if (interval.hi <= interval.lo)
                continue;
            const uint span_tiles = uint(interval.hi - interval.lo + int(KEYS) - 1) / KEYS;
            const uint span_first = tiles_before;
            tiles_before += span_tiles;
            if (span_first + span_tiles <= tiles_lo || span_first >= tiles_hi)
                continue;
            const uint own_lo = metal::max(tiles_lo, span_first) - span_first;
            const uint own_hi = metal::min(tiles_hi, span_first + span_tiles) - span_first;
            const bool historical = index < R;
            const prefill_rows rows{visible, fresh, R, index, historical};
            for (uint own = own_lo; own < own_hi; ++own) {
                const int first = interval.lo + int(own * KEYS);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (historical)
                    history.template stage_key<THREADS>(staged, first, interval.hi, kv_head, thread_index);
                else
                    prefill_stage<THREADS>(staged, keys, first, interval.hi, kv_head, thread_index);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const bool common = first >= interval.common_lo
                    && first + int(KEYS) <= interval.common_hi;
                if (computes)
                    Form::scores(query_rows, staged, first, common, rows, first_token, M, scale, state);

                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (historical)
                    history.template stage_value<THREADS>(staged, first, interval.hi, kv_head, thread_index);
                else
                    prefill_stage<THREADS>(staged, values, first, interval.hi, kv_head, thread_index);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (computes)
                    Form::accumulate(staged, window_first, state);
            }
        }

        // Invalid rows keep running the later windows' barriers.
        if (computes)
            Form::store(query, gate, result, partials, statistics, first_token, M, head, partition, active,
                window_first, softplus, state);
    }
}

// The rows a simdgroup owns in a form of ROWS rows per simdgroup: simdgroup
// s owns rows tile * QT + (s % SPAN) * ROWS .. of query head kv * G + group *
// PREFILL_HEADS + s / SPAN (SPAN = QT / ROWS) when that head is in the group;
// the others only stage.
template <uint QT, uint ROWS>
struct prefill_owner {
    bool computes;
    uint owner;
    uint head;
    ulong first_token;
    prefill_owner(uint tile, uint kv_head, uint head_group, uint simd) {
        constexpr uint SPAN = QT / ROWS;
        const uint first_head = head_group * PREFILL_HEADS;
        computes = simd < QT * PREFILL_HEADS / ROWS && first_head + simd / SPAN < SEISMIC_DIM_G;
        owner = computes ? simd : 0;
        head = kv_head * SEISMIC_DIM_G + first_head + owner / SPAN;
        first_token = ulong(tile) * QT + (owner % SPAN) * ROWS;
    }
};

// L2's arguments shared by both forms' entries.
#define PREFILL_OWNED_PARAMETERS                                                                          \
    History history, device const Scalar *query, device const Scalar *gate, device const int *visible,     \
    device const int *fresh, device Scalar *result, device const typename History::Operand *queries,       \
    device const typename History::Operand *keys, device const typename History::Operand *values,          \
    device float *partials, device float *statistics, ulong M, ulong R, float scale, bool softplus,        \
    threadgroup typename History::Operand *staged, threadgroup const prefill_interval *intervals,          \
    threadgroup float *exchange, uint tile, uint kv_head, uint head_group, uint partition, uint active,    \
    uint tiles_lo, uint tiles_hi, uint thread_index, uint simd, uint lane
#define PREFILL_OWNED_ARGUMENTS                                                                           \
    history, query, gate, visible, fresh, result, queries, keys, values, partials, statistics, M, R, scale, \
    softplus, staged, intervals, exchange, tile, kv_head, head_group, partition, active, tiles_lo,         \
    tiles_hi, thread_index, simd, lane

template <uint QT, class History>
inline void prefill_owned_fragments(PREFILL_OWNED_PARAMETERS) {
    typedef prefill_fragments<QT, History> Form;
    const prefill_owner<QT, Form::ROWS> own(tile, kv_head, head_group, simd);
    Form state;
    Form::place(state, lane);
    prefill_windows<QT, History>(history, query, gate, visible, fresh, result, keys, values, partials,
        statistics, M, R, scale, softplus, staged, intervals, kv_head, partition, active, tiles_lo, tiles_hi,
        queries + (own.first_token * SEISMIC_DIM_KV * SEISMIC_DIM_G + own.head) * ATTENTION_W, own.head,
        own.first_token, own.computes, thread_index, state);
}

#if SEISMIC_HAS_TENSOR_OPS
template <uint QT, class History>
inline void prefill_owned_tensors(PREFILL_OWNED_PARAMETERS) {
    typedef prefill_tensors<QT, History> Form;
    const prefill_owner<QT, Form::ROWS> own(tile, kv_head, head_group, simd);
    Form::windows(history, query, gate, visible, fresh, result, keys, values, partials, statistics, M, R, scale,
        softplus, staged, intervals, exchange + own.owner * Form::EXCHANGE, kv_head, partition, active, tiles_lo,
        tiles_hi, queries + (own.first_token * SEISMIC_DIM_KV * SEISMIC_DIM_G + own.head) * ATTENTION_W, own.head,
        own.first_token, own.computes, own.owner, thread_index, lane);
}

// The direct form's simdgroups past the owners have nothing to do.
template <uint QT, class History>
inline void prefill_owned_direct(PREFILL_OWNED_PARAMETERS) {
    typedef prefill_direct<QT, History> Form;
    const prefill_owner<QT, Form::ROWS> own(tile, kv_head, head_group, simd);
    if (!own.computes)
        return;
    Form::windows(history, query, gate, visible, fresh, result, keys, values, partials, statistics, M, R, scale,
        softplus, intervals, reinterpret_cast<threadgroup const prefill_run *>(intervals + R + 1), kv_head,
        partition, queries + (own.first_token * SEISMIC_DIM_KV * SEISMIC_DIM_G + own.head) * ATTENTION_W, own.head,
        own.first_token);
}
#endif

// L2: threadgroup (QT-row tile, kv head and head group, key partition). A
// simdgroup owns ROWS rows of one query head of the group (8 in the fragment
// form, every simdgroup; 16 in the tensor form, the first half of the
// simdgroups); each staged K/V tile serves every head of the group, so the
// group's size trades K/V reuse against the threadgroup's size. The tile's key tiles (each span's union interval in
// PREFILL_KEYS steps, spans then fresh) split into consecutive runs of at
// least PREFILL_MIN_TILES over the partitions. Per key tile: K staged
// (history through the policy, fresh rows from scratch), scores = Q K^T with
// Q read from scratch (L1-resident; holding it in registers costs more
// occupancy than the loads), scaled into the exp2 domain in F32, the online
// softmax, then V staged (aliasing K) and the F32 output accumulated from
// activation-dtype probabilities (`prefill_windows`, on the fragment or the
// tensor-operation form). A head wider than PREFILL_WINDOW repeats this per
// output window. Query tiles dispatch last-first. A tile served by one
// partition stores its gated output directly; otherwise each partition
// stores (partial output, maximum, denominator) and the merge launch
// combines them. Operands (queries, staged tiles, probabilities) are the
// history policy's. `exchange` is PREFILL_EXCHANGE memory.
template <uint QT, class History>
inline void prefill_attend(History history, device const Scalar *query, device const Scalar *gate,
    device const int *visible, device const int *fresh, device Scalar *result,
    device const typename History::Operand *queries, device const typename History::Operand *keys,
    device const typename History::Operand *values, device float *partials,
    device float *statistics, device uint *counts, ulong M, ulong R, float scale, bool softplus,
    threadgroup uchar *shared, threadgroup float *exchange, uint3 group, uint3 groups, uint thread_index,
    uint simd, uint lane) {
    typedef typename History::Operand Operand;
    constexpr uint W = ATTENTION_W;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    constexpr uint KEYS = PREFILL_KEYS;
    constexpr uint WINDOW = W < PREFILL_WINDOW ? W : PREFILL_WINDOW;
    static_assert(W % WINDOW == 0, "output windows tile the head");
    static_assert(QT % 8 == 0 && QT <= 32, "query tiles are 8-row blocks within one simdgroup's lanes");
    // Query tiles dispatch last-first: in a causal chunk the last tiles see
    // the most keys, and starting them first shortens the grid's tail.
    const uint tile = groups.x - 1 - group.x;
    const uint kv_head = group.y / PREFILL_HEAD_GROUPS;
    const uint head_group = group.y % PREFILL_HEAD_GROUPS;
    const uint partition = group.z;
    threadgroup Operand *staged = reinterpret_cast<threadgroup Operand *>(shared);
    threadgroup prefill_interval *intervals = reinterpret_cast<threadgroup prefill_interval *>(
        shared + KEYS * PREFILL_PITCH * sizeof(Operand));

    if (simd == 0) {
        const ulong tile_row = ulong(tile) * QT + lane;
        const bool row_valid = lane < QT && tile_row < M;
        for (ulong index = 0; index <= R; ++index) {
            int lo = 0;
            int hi = 0;
            if (row_valid)
                form_span(visible, fresh, tile_row, R, index, lo, hi);
            const bool nonempty = row_valid && hi > lo;
            const int union_lo = simd_min(nonempty ? lo : INT_MAX);
            const int union_hi = simd_max(nonempty ? hi : INT_MIN);
            const int common_lo = simd_max(row_valid ? lo : INT_MIN);
            const int common_hi = simd_min(row_valid ? hi : INT_MAX);
            if (lane == 0)
                intervals[index] = prefill_interval{union_lo, union_hi, common_lo, common_hi};
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total_tiles = 0;
    for (ulong index = 0; index <= R; ++index) {
        const prefill_interval interval = intervals[index];
        if (interval.hi > interval.lo)
            total_tiles += uint(interval.hi - interval.lo + int(KEYS) - 1) / KEYS;
    }
    // The grid's key partitions: max(1, ceil(SPLIT_GROUPS / (tiles * KV))).
    const uint parts = groups.z;
    const uint per = metal::max(uint(PREFILL_MIN_TILES), (total_tiles + parts - 1) / parts);
    const uint active = metal::max(1u, (total_tiles + per - 1) / per);
    if (partition >= active)
        return;
    if (partition == 0 && group.y == 0 && thread_index == 0)
        counts[tile] = active;
    const uint tiles_lo = partition * per;
    const uint tiles_hi = metal::min(tiles_lo + per, total_tiles);

    // On tensor operations the direct form when the entry takes it, else the
    // tensor form where its rows fit; otherwise the fragment form. A DIRECT
    // threadgroup has half the simdgroups (16 rows each): on simdgroup
    // matrices each then takes its two 8-row blocks in turn.
    static_assert(PREFILL_DIRECT == 0 || QT % 16 == 0, "the direct form's simdgroups own 16 rows");
#if SEISMIC_HAS_TENSOR_OPS
    if constexpr (PREFILL_DIRECT != 0) {
        if (thread_index == 0)
            prefill_direct<QT, History>::schedule(history, intervals,
                reinterpret_cast<threadgroup prefill_run *>(intervals + R + 1), M, R, kv_head, active, tiles_lo,
                tiles_hi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefill_owned_direct<QT, History>(PREFILL_OWNED_ARGUMENTS);
    } else if constexpr (prefill_tensors_fit(QT)) {
        prefill_owned_tensors<QT, History>(PREFILL_OWNED_ARGUMENTS);
    } else {
        prefill_owned_fragments<QT, History>(PREFILL_OWNED_ARGUMENTS);
    }
#else
    if constexpr (PREFILL_DIRECT != 0) {
        const uint physical = simd;
        for (uint block = 0; block < 2; ++block) {
            const uint simd = 2 * physical + block;
            prefill_owned_fragments<QT, History>(PREFILL_OWNED_ARGUMENTS);
        }
    } else {
        prefill_owned_fragments<QT, History>(PREFILL_OWNED_ARGUMENTS);
    }
#endif
}

// L3: threadgroup (QT-row tile, query head), one thread per column. A tile
// that took several key partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
template <uint QT>
inline void prefill_merge(device const Scalar *query, device const Scalar *gate, device Scalar *result,
    device const float *partials, device const float *statistics, device const uint *counts,
    ulong M, uint tile, ulong head, uint column, bool softplus) {
    constexpr uint W = ATTENTION_W;
    constexpr uint H = SEISMIC_DIM_KV * SEISMIC_DIM_G;
    const uint count = counts[tile];
    if (count <= 1)
        return;
    for (ulong row = ulong(tile) * QT; row < metal::min(ulong(tile + 1) * QT, M); ++row) {
        const float attended = merge(partials, statistics, row * H + head, M * H, count, column);
        result[(row * H + head) * W + column] = gate_output(query, gate, row, head, column, attended, softplus);
    }
}

// ---------------------------------------------------------------------------
// Grouped-query matrix decode (`attention_decode*` with MATRIX = 1). The G
// query heads of a kv head are the rows of 8-row simdgroup-matrix blocks
// (ROWS = G rounded up to 8, padded with zero queries), so each staged K/V
// tile serves every head through fragment products instead of one cross-lane
// reduction per (head, key). Threadgroup (kv head, partition, row) and the
// partition bounds are `attention_decode`'s.
//
// A simdgroup's output covers every row block over one slice of WC = W /
// COLS columns, COLS the fewest slices keeping that to at most 128 columns
// (the register budget measured on M4). Simdgroups form TEAMS = SIMDS / COLS
// teams, one simdgroup per slice; a team scans a contiguous whole-tile
// sub-range of the partition. Each member stages its slice of every K/V tile
// into its own region (historical tiles through the history policy, fresh
// tiles prepared in place) and forms partial scores over its slice; with
// several slices the team (then the whole threadgroup) sums the partial
// scores in slice order through threadgroup memory, so every member holds the
// same scores and softmax state, and accumulates P.V into its own columns.
// Scores, the softmax state and the outputs are F32; queries and keys enter
// the score product as History::KeyOperand, probabilities and values the
// output product as F16. The teams' states merge in team order into the
// partition's partials and statistics, the layout `decode_output` merges.
// ---------------------------------------------------------------------------

// The matrix rows of a tile of `tokens` decode rows (token-major, head-minor,
// padded to whole 8-row blocks), and its column slices.
#define DECODE_MATRIX_ROWS(tokens) (((SEISMIC_DIM_G * (tokens) + 7) / 8) * 8)
#define DECODE_MATRIX_COLS(tokens)                                                                       \
    (DECODE_MATRIX_ROWS(tokens) / 8 * ATTENTION_W > 128 ? DECODE_MATRIX_ROWS(tokens) / 8 * ATTENTION_W / 128 \
                                                        : 1)

// Fresh rows [first, first + count) of the launch (keys when KEY, else
// values), columns [col0, col0 + WC), prepared into rows [0, count) of a
// simdgroup's tile region of pitch PITCH, the rest of its KEYS rows zero.
// The whole simdgroup calls it.
template <bool KEY, uint KEYS, uint WC, uint PITCH, class Operand>
inline void decode_matrix_fresh(threadgroup Operand *region, device const Scalar *key,
    device const Scalar *value, device const float *key_norm, device const float *value_norm,
    device const int *coordinates, device const int *rotary_components, device const float *rotary_frequencies,
    device const float *rotary_amplitudes, int first, uint count, uint kv_head, uint col0, float epsilon,
    uint lane) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    for (uint k = 0; k < KEYS; ++k) {
        float x[E];
        if (k < count) {
            const ulong token = ulong(first) + k;
            const ulong at = (token * SEISMIC_DIM_KV + kv_head) * W;
            if (KEY)
                head_rotary<ATTENTION_NORM>(key + at, key_norm, coordinates + token * 4, rotary_components,
                    rotary_frequencies, rotary_amplitudes, epsilon, lane, x);
            else
                head_norm<ATTENTION_VALUE_NORM>(value + at, value_norm, epsilon, lane, x);
        } else {
            ATTENTION_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] = 0.0f;
        }
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i) {
            const uint column = lane * E + i;
            if (column >= col0 && column < col0 + WC)
                region[k * PITCH + column - col0] = Operand(Scalar(x[i]));
        }
    }
}

// Threadgroup memory: the prepared queries [ROWS][W] (key operands); then the
// simdgroups' tile regions [SIMDS][KEYS][WC + 8] (2-byte operands) and, with
// several slices, the double-buffered partial scores [2][COLS][ROWS][KEYS]
// (F32), which the merged output [ROWS][W] (F32) aliases after the key loop;
// then the team states [TEAMS][ROWS][2] (F32). The contract's shared bytes
// are this sum.
template <uint SIMDS, uint KEYS, uint PARTS, uint TOKENS, class History>
inline void decode_matrix(History history, device const Scalar *query, device const Scalar *key,
    device const Scalar *value, device const float *query_norm, device const float *key_norm,
    device const float *value_norm, device const int *rotary_components, device const float *rotary_frequencies,
    device const float *rotary_amplitudes, device const int *coordinates, device const int *visible,
    device const int *fresh, device float *partials, device float *statistics, threadgroup uchar *shared,
    ulong R, ulong rows, float epsilon, float scale, uint span_min, uint kv_head, uint partition, ulong tile,
    uint thread_index, uint simd, uint lane) {
    typedef typename History::KeyOperand KeyOperand;
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    // Packed: one simdgroup per token owns its G = 8 heads over the whole
    // head width, so scores need no exchange; the four tokens share each
    // tile's K and V, decoded once into separate regions.
    constexpr bool PACKED = History::AFFINE && TOKENS == 4 && G == 8 && (W == 128 || W == 256)
        && SIMDS == TOKENS;
    constexpr uint ROWS = PACKED ? G : DECODE_MATRIX_ROWS(TOKENS);
    constexpr uint QUERY_ROWS = PACKED ? TOKENS * G : ROWS;
    constexpr uint RB = ROWS / 8;
    constexpr uint COLS = PACKED ? 1 : DECODE_MATRIX_COLS(TOKENS);
    constexpr uint TEAMS = PACKED ? 1 : SIMDS / COLS;
    constexpr bool DIRECT = History::AFFINE && TEAMS == 1;
    constexpr uint WC = W / COLS;
    constexpr uint PITCH = PACKED ? W : (DIRECT ? WC : WC + 8);
    constexpr uint DB = WC / 8;
    constexpr uint KB = KEYS / 8;
    constexpr uint TILE_BYTES = (PACKED ? 1 : SIMDS) * KEYS * PITCH * 2;
    constexpr uint EXCHANGE_BYTES = COLS > 1 ? 2 * SIMDS * ROWS * KEYS * 4 : 0;
    constexpr uint MERGED_BYTES = ROWS * W * 4;
    static_assert(sizeof(KeyOperand) == 2, "tile regions hold 2-byte operands");
    static_assert(KEYS % 8 == 0, "a key tile is whole 8-key fragments");
    static_assert(W % COLS == 0 && WC % 32 == 0, "column slices are whole code groups");
    static_assert(SIMDS % COLS == 0, "teams are whole sets of column slices");

    // The tile's rows [row0, row0 + TOKENS) below the launch's `rows`; its key
    // sequence is its spans' unions, a row's keys outside its own interval
    // masked.
    const ulong row0 = tile * TOKENS;
    const uint total = tile_total(visible, fresh, row0, TOKENS, rows, R);
    const uint span_keys = partition_span(total, span_min, PARTS);
    const uint partition_lo = partition * span_keys;
    if (partition_lo >= total)
        return;
    const uint partition_hi = metal::min(partition_lo + span_keys, total);
    const uint team = simd / COLS;
    const uint scan_team = PACKED ? 0 : team;
    const uint slice = simd % COLS;
    const uint col0 = slice * WC;

    threadgroup KeyOperand *queries = reinterpret_cast<threadgroup KeyOperand *>(shared);
    threadgroup uchar *middle = shared + (PACKED ? 0 : QUERY_ROWS * W * sizeof(KeyOperand));
    threadgroup KeyOperand *key_region = reinterpret_cast<threadgroup KeyOperand *>(middle)
        + (PACKED ? 0 : simd) * KEYS * PITCH;
    threadgroup half *value_region = reinterpret_cast<threadgroup half *>(key_region) + (PACKED ? KEYS * PITCH : 0);
    threadgroup float *exchange = reinterpret_cast<threadgroup float *>(DIRECT ? middle : middle + TILE_BYTES);
    threadgroup float *merged = reinterpret_cast<threadgroup float *>(middle);
    constexpr uint WORK_BYTES = DIRECT
        ? (TILE_BYTES > EXCHANGE_BYTES ? TILE_BYTES : EXCHANGE_BYTES)
        : (TILE_BYTES + EXCHANGE_BYTES > MERGED_BYTES ? TILE_BYTES + EXCHANGE_BYTES : MERGED_BYTES);
    threadgroup float *states = reinterpret_cast<threadgroup float *>(middle + WORK_BYTES);

    // The tile's queries (matrix row i is row row0 + i / G, head i % G),
    // rotary-prepared and rounded to the activation dtype (as the contract
    // publishes them); padding rows are zero.
    for (uint g = PACKED ? 0 : simd; g < ROWS; g += PACKED ? 1 : SIMDS) {
        float x[E];
        const uint global_g = PACKED ? team * G + g : g;
        const ulong row = row0 + global_g / G;
        if (global_g < TOKENS * G && row < rows) {
            head_rotary<ATTENTION_NORM>(query + (row * KV * G + kv_head * G + global_g % G) * ATTENTION_QUERY_STRIDE,
                query_norm, coordinates + row * 4, rotary_components, rotary_frequencies, rotary_amplitudes, epsilon,
                lane, x);
        } else {
            ATTENTION_UNROLL
            for (uint i = 0; i < E; ++i)
                x[i] = 0.0f;
        }
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            if (!PACKED || slice == 0)
                queries[global_g * W + lane * E + i] = KeyOperand(Scalar(x[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Packed tokens keep their query fragments in registers so the shared
    // query tile can be reused by the decoded K/V tile and score exchange.
    simdgroup_matrix<KeyOperand, 8, 8> prepared_query[DB];
    if (PACKED) {
        ATTENTION_UNROLL
        for (uint d = 0; d < DB; ++d)
            simdgroup_load(prepared_query[d], queries + team * G * W + col0 + d * 8, W);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_matrix<float, 8, 8> output[RB][DB];
    float maximum[RB];
    float denominator[RB];
    ATTENTION_UNROLL
    for (uint b = 0; b < RB; ++b) {
        ATTENTION_UNROLL
        for (uint d = 0; d < DB; ++d)
            output[b][d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        maximum[b] = -INFINITY;
        denominator[b] = 0.0f;
    }

    // Fragment coordinates: this lane holds row `fm`, columns `fn`, `fn + 1`.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;

    // Whole-tile team sub-ranges, so only a sub-range's and a span's last
    // tiles are partial; a team's members walk the same tiles.
    const uint sub = (((partition_hi - partition_lo + TEAMS - 1) / TEAMS) + KEYS - 1) / KEYS * KEYS;
    const uint first = metal::min(partition_lo + scan_team * sub, partition_hi);
    const uint last = metal::min(first + sub, partition_hi);
    uint parity = 0;

    // One KEYS-key tile of `count` keys from `token` (history or fresh rows)
    // of span `span`: scores, their exchange between a team's slices, the
    // online softmax and O += P V. Unless the tile lies in every row's
    // interval (`inside`), each row's keys outside its own interval are
    // masked. A tile that is not `live` (an exhausted team's round) only
    // takes the exchange barrier.
    // Stage a packed tile: each simdgroup decodes its stripe of the tile's
    // keys and values.
    auto stage_packed = [&](bool historical, int token, uint count) {
        constexpr uint STRIPE = PACKED ? KEYS / SIMDS : 1;
        typename History::template decode_tile<STRIPE, PACKED ? W : WC> stripe;
        const uint offset = simd * STRIPE;
        const uint available = count > offset ? metal::min(uint(STRIPE), count - offset) : 0;
        const int first = token + int(offset);
        threadgroup KeyOperand *stripe_key = key_region + offset * PITCH;
        threadgroup half *stripe_value = value_region + offset * PITCH;
        if (historical) {
            history.load_tile(stripe, first, first + int(available), kv_head, 0, lane);
            stripe.store_key_direct(stripe_key, lane);
            stripe.store_value_direct(stripe_value, lane);
        } else {
            decode_matrix_fresh<true, STRIPE, W, PITCH>(stripe_key, key, value, key_norm, value_norm, coordinates,
                rotary_components, rotary_frequencies, rotary_amplitudes, first, available, kv_head, 0, epsilon,
                lane);
            decode_matrix_fresh<false, STRIPE, W, PITCH>(stripe_value, key, value, key_norm, value_norm,
                coordinates, rotary_components, rotary_frequencies, rotary_amplitudes, first, available, kv_head,
                0, epsilon, lane);
        }
    };

    // Absorb a tile whose keys and values are in `keys` and `values` (a
    // packed tile was staged by `stage_packed`; other tiles are staged
    // here).
    auto absorb_tile = [&](bool live, bool historical, ulong span, bool inside, int token, uint count,
                           threadgroup KeyOperand *keys, threadgroup half *values) {
        typename History::template decode_tile<KEYS, WC> tile;
        simdgroup_matrix<float, 8, 8> scores[RB][KB];
        if (live) {
            if (!PACKED) {
                if (historical) {
                    history.load_tile(tile, token, token + int(count), kv_head, col0, lane);
                    if (DIRECT)
                        tile.store_key_direct(key_region, lane);
                    else
                        tile.store_key(key_region, lane);
                } else {
                    decode_matrix_fresh<true, KEYS, WC, PITCH>(key_region, key, value, key_norm, value_norm, coordinates,
                        rotary_components, rotary_frequencies, rotary_amplitudes, token, count, kv_head, col0,
                        epsilon, lane);
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            ATTENTION_UNROLL
            for (uint b = 0; b < RB; ++b) {
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j)
                    scores[b][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    simdgroup_matrix<KeyOperand, 8, 8> k;
                    simdgroup_load(k, keys + j * 8 * PITCH + (PACKED ? col0 : 0) + d * 8, PITCH,
                        ulong2(0, 0), true);
                    ATTENTION_UNROLL
                    for (uint b = 0; b < RB; ++b) {
                        if (PACKED)
                            simdgroup_multiply_accumulate(scores[b][j], prepared_query[d], k, scores[b][j]);
                        else {
                            simdgroup_matrix<KeyOperand, 8, 8> q;
                            simdgroup_load(q, queries + b * 8 * W + col0 + d * 8, W);
                            simdgroup_multiply_accumulate(scores[b][j], q, k, scores[b][j]);
                        }
                    }
                }
            }
        }
        if (DIRECT && !PACKED && live)
            threadgroup_barrier(mem_flags::mem_threadgroup);
        if (COLS > 1) {
            // Every member publishes its partial scores, then sums all
            // slices' in slice order, so the team's scores are identical.
            threadgroup float *published = exchange + (team * 2 + parity) * COLS * ROWS * KEYS;
            if (live) {
                ATTENTION_UNROLL
                for (uint b = 0; b < RB; ++b) {
                    ATTENTION_UNROLL
                    for (uint j = 0; j < KB; ++j)
                        simdgroup_store(scores[b][j], published + ((slice * RB + b) * KB + j) * 64, 8);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live) {
                ATTENTION_UNROLL
                for (uint b = 0; b < RB; ++b) {
                    ATTENTION_UNROLL
                    for (uint j = 0; j < KB; ++j) {
                        simdgroup_load(scores[b][j], published + (b * KB + j) * 64, 8);
                        for (uint other = 1; other < COLS; ++other) {
                            simdgroup_matrix<float, 8, 8> part;
                            simdgroup_load(part, published + ((other * RB + b) * KB + j) * 64, 8);
                            scores[b][j].thread_elements()[0] += part.thread_elements()[0];
                            scores[b][j].thread_elements()[1] += part.thread_elements()[1];
                        }
                    }
                }
            }
            parity ^= 1;
        }
        if (live) {
            // Online softmax per row block; key columns past `count`, and a
            // row's keys outside its own interval, are masked. Probabilities
            // enter the P.V product as F16; the denominator sums them in F32.
            simdgroup_matrix<half, 8, 8> probabilities[RB][KB];
            ATTENTION_UNROLL
            for (uint b = 0; b < RB; ++b) {
                // This lane's row of the block: its own interval of the span.
                int own_lo = int(0x80000000), own_hi = 0x7fffffff;
                if (!inside) {
                    const uint r = (PACKED ? team * G : 0) + b * 8 + fm;
                    const ulong row = row0 + r / G;
                    own_lo = own_hi = 0;
                    if (r < TOKENS * G && row < rows)
                        form_span(visible, fresh, row, R, span, own_lo, own_hi);
                }
                float tile_maximum = -INFINITY;
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    ATTENTION_UNROLL
                    for (uint e = 0; e < 2; ++e) {
                        float s = scores[b][j].thread_elements()[e] * scale;
                        const int position = token + int(j * 8 + fn + e);
                        if (j * 8 + fn + e >= count || position < own_lo || position >= own_hi)
                            s = -INFINITY;
                        scores[b][j].thread_elements()[e] = s;
                        tile_maximum = metal::max(tile_maximum, s);
                    }
                }
                tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
                tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
                const float next = metal::max(maximum[b], tile_maximum);
                const bool seen = next > -INFINITY;
                const float carry = seen ? metal::fast::exp2(maximum[b] - next) : 1.0f;
                float tile_sum = 0.0f;
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    ATTENTION_UNROLL
                    for (uint e = 0; e < 2; ++e) {
                        const float p = seen ? metal::fast::exp2(scores[b][j].thread_elements()[e] - next) : 0.0f;
                        probabilities[b][j].thread_elements()[e] = half(p);
                        tile_sum += p;
                    }
                }
                tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
                tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
                denominator[b] = metal::fma(denominator[b], carry, tile_sum);
                maximum[b] = next;
                ATTENTION_UNROLL
                for (uint d = 0; d < DB; ++d) {
                    output[b][d].thread_elements()[0] *= carry;
                    output[b][d].thread_elements()[1] *= carry;
                }
            }

            // The packed tile staged its values with its keys.
            if (!PACKED) {
                if (DIRECT && COLS > 1)
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                else
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                if (historical) {
                    if (DIRECT)
                        tile.store_value_direct(value_region, lane);
                    else
                        tile.store_value(value_region, lane);
                } else
                    decode_matrix_fresh<false, KEYS, WC, PITCH>(value_region, key, value, key_norm, value_norm, coordinates,
                        rotary_components, rotary_frequencies, rotary_amplitudes, token, count, kv_head, col0,
                        epsilon, lane);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    simdgroup_matrix<half, 8, 8> v;
                    simdgroup_load(v, values + j * 8 * PITCH + (PACKED ? col0 : 0) + d * 8, PITCH);
                    ATTENTION_UNROLL
                    for (uint b = 0; b < RB; ++b)
                        simdgroup_multiply_accumulate(output[b][d], probabilities[b][j], v, output[b][d]);
                }
            }
            if (!PACKED)
                simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    };

    // A packed tile is staged by the whole threadgroup, then absorbed; the
    // barrier after P.V frees the regions for the next tile.
    auto visit = [&](bool historical, ulong span, bool inside, int token, uint count) {
        if (PACKED) {
            stage_packed(historical, token, count);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        absorb_tile(true, historical, span, inside, token, count, key_region, value_region);
        if (PACKED)
            threadgroup_barrier(mem_flags::mem_threadgroup);
    };

    if (COLS == 1 || TEAMS == 1) {
        uint offset = 0;
        for (ulong span = 0; span <= R && offset < last; ++span) {
            const tile_interval interval = tile_span(visible, fresh, row0, TOKENS, rows, R, span);
            const uint length = uint(interval.hi - interval.lo);
            const uint begin = metal::max(first, offset);
            const uint end = metal::min(last, offset + length);
            for (uint position = begin; position < end; position += KEYS) {
                const int token = interval.lo + int(position - offset);
                const uint count = metal::min(uint(KEYS), end - position);
                visit(span < R, span, token >= interval.common_lo && token + int(count) <= interval.common_hi,
                    token, count);
            }
            offset += length;
        }
    } else {
        // Several teams exchange across slices at threadgroup barriers: every
        // team takes the same number of rounds (the most tiles of any team),
        // an exhausted team only taking the rounds' barriers, so they pair
        // up.
        uint rounds = 0;
        for (uint t = 0; t < TEAMS; ++t) {
            const uint team_first = metal::min(partition_lo + t * sub, partition_hi);
            rounds = metal::max(rounds, form_tiles(visible, fresh, row0, TOKENS, rows, R, team_first,
                metal::min(team_first + sub, partition_hi), KEYS));
        }
        // The walk: span `span` (the tile's union `interval`, `length` keys
        // at key offset `offset`), its part [position, end) of the team's
        // range still to tile.
        ulong span = 0;
        uint offset = 0;
        uint length = 0;
        tile_interval interval{0, 0, 0, 0};
        uint position = 0;
        uint end = 0;
        bool opened = false;
        for (uint round = 0; round < rounds; ++round) {
            while (position >= end && span <= R && (!opened || offset + length < last)) {
                if (opened) {
                    offset += length;
                    ++span;
                    if (span > R)
                        break;
                }
                opened = true;
                interval = tile_span(visible, fresh, row0, TOKENS, rows, R, span);
                length = uint(interval.hi - interval.lo);
                position = metal::max(first, offset);
                end = metal::min(last, offset + length);
            }
            const bool live = position < end;
            const int token = interval.lo + int(position - offset);
            const uint count = live ? metal::min(uint(KEYS), end - position) : 0;
            absorb_tile(live, span < R, span,
                token >= interval.common_lo && token + int(count) <= interval.common_hi, token, count, key_region,
                value_region);
            if (live)
                position += KEYS;
        }
    }

    // One team already has the partition result. Publish each matrix lane's
    // two columns directly, avoiding the shared F32 merge tile.
    if (DIRECT) {
        const uint live_rows = uint(metal::min(ulong(TOKENS), rows - row0)) * G;
        ATTENTION_UNROLL
        for (uint b = 0; b < RB; ++b) {
            const uint r = (PACKED ? team * G : 0) + b * 8 + fm;
            if (r >= live_rows)
                continue;
            const ulong slot = ((row0 + r / G) * KV + kv_head) * G * PARTS + (r % G) * PARTS + partition;
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                const uint column = col0 + d * 8 + fn;
                partials[slot * W + column] = output[b][d].thread_elements()[0];
                partials[slot * W + column + 1] = output[b][d].thread_elements()[1];
            }
            if (slice == 0 && fn == 0) {
                statistics[slot * 2] = maximum[b];
                statistics[slot * 2 + 1] = denominator[b];
            }
        }
        return;
    }

    // Merge: each row's partition maximum over the teams that saw keys; the
    // teams' outputs, rescaled to it, accumulate in team order into the
    // merged rows (each member its own columns), which alias the idle tile
    // regions. A team's members hold the same state; its first member
    // publishes it.
    if (slice == 0 && fn == 0) {
        ATTENTION_UNROLL
        for (uint b = 0; b < RB; ++b) {
            states[(team * ROWS + b * 8 + fm) * 2] = maximum[b];
            states[(team * ROWS + b * 8 + fm) * 2 + 1] = denominator[b];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ATTENTION_UNROLL
    for (uint b = 0; b < RB; ++b) {
        float partition_maximum = -INFINITY;
        for (uint t = 0; t < TEAMS; ++t) {
            if (states[(t * ROWS + b * 8 + fm) * 2 + 1] > 0.0f)
                partition_maximum = metal::max(partition_maximum, states[(t * ROWS + b * 8 + fm) * 2]);
        }
        const float weight = denominator[b] > 0.0f ? metal::fast::exp2(maximum[b] - partition_maximum) : 0.0f;
        ATTENTION_UNROLL
        for (uint d = 0; d < DB; ++d) {
            output[b][d].thread_elements()[0] *= weight;
            output[b][d].thread_elements()[1] *= weight;
        }
    }
    for (uint t = 0; t < TEAMS; ++t) {
        if (team == t) {
            ATTENTION_UNROLL
            for (uint b = 0; b < RB; ++b) {
                ATTENTION_UNROLL
                for (uint d = 0; d < DB; ++d) {
                    threadgroup float *at = merged + b * 8 * W + col0 + d * 8;
                    if (t > 0) {
                        simdgroup_matrix<float, 8, 8> sum;
                        simdgroup_load(sum, at, W);
                        output[b][d].thread_elements()[0] += sum.thread_elements()[0];
                        output[b][d].thread_elements()[1] += sum.thread_elements()[1];
                    }
                    simdgroup_store(output[b][d], at, W);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Matrix row i is row row0 + i / G, head i % G: slot ((row KV + kv) G +
    // head) PARTS + partition, rows past the batch skipped.
    const uint live_rows = uint(metal::min(ulong(TOKENS), rows - row0)) * G;
    for (uint item = thread_index; item < live_rows * W; item += SIMDS * 32) {
        const uint r = item / W;
        const uint column = item % W;
        const ulong slot = ((row0 + r / G) * KV + kv_head) * G * PARTS + (r % G) * PARTS + partition;
        partials[slot * W + column] = merged[r * W + column];
    }
    for (uint r = thread_index; r < live_rows; r += SIMDS * 32) {
        const ulong slot = ((row0 + r / G) * KV + kv_head) * G * PARTS + (r % G) * PARTS + partition;
        float partition_maximum = -INFINITY;
        for (uint t = 0; t < TEAMS; ++t) {
            if (states[(t * ROWS + r) * 2 + 1] > 0.0f)
                partition_maximum = metal::max(partition_maximum, states[(t * ROWS + r) * 2]);
        }
        float total_denominator = 0.0f;
        for (uint t = 0; t < TEAMS; ++t) {
            const float d = states[(t * ROWS + r) * 2 + 1];
            if (d > 0.0f)
                total_denominator = metal::fma(d,
                    metal::fast::exp2(states[(t * ROWS + r) * 2] - partition_maximum), total_denominator);
        }
        statistics[slot * 2] = partition_maximum;
        statistics[slot * 2 + 1] = total_denominator;
    }
}

} // namespace attention
