// Shared pieces of the gated attention entries (`qwen_attention_decode`,
// `qwen_attention_prefill` and their affine K8/V4 history forms
// `qwen_attention_{decode,prefill}_k8v4`): the per-head preparation (RMS norm
// and partial M-RoPE) held in one simdgroup's registers, a row's span walk,
// decode partition bounds, the online-softmax absorb (dense and corrected
// affine), the fixed-order merge of partial states, the decode partition
// publication and gated merge, the K/V append, and the affine codec (encode
// on append, per-lane code access). Each kernel keeps its own loop structure
// and calls these.
//
// Every dense operand is bound canonically (row-major, unit innermost stride),
// so rows are addressed from their logical offsets. Activations are addressed
// as the element's MSL scalar and converted with the compiler's conversions.

#include "common/element.h"

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

// One head row, RMS-normalized with `norm` and rotated on its first 2P
// columns (pair p by coordinate axis components[p] at frequencies[p]), into
// lane `lane`'s ATTENTION_E columns. The whole simdgroup calls it: the square
// sum is a simdgroup reduction and each rotated column's pair partner lives
// P / ATTENTION_E lanes away.
inline void prepare(device const Scalar *raw, device const float *norm,
    device const int *coordinates, device const int *components,
    device const float *frequencies, float epsilon, uint lane,
    thread float (&x)[ATTENTION_E]) {
    float squares = 0.0f;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i) {
        x[i] = float(raw[lane * ATTENTION_E + i]);
        squares = metal::fma(x[i], x[i], squares);
    }
    squares = simd_sum(squares);
    const float inverse = metal::rsqrt(squares / float(ATTENTION_W) + epsilon);
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i)
        x[i] = x[i] * inverse * norm[lane * ATTENTION_E + i];
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
        const uint pair = column % SEISMIC_DIM_P;
        float c;
        const float s = sincos(float(coordinates[components[pair]]) * frequencies[pair], c);
        x[i] = column < SEISMIC_DIM_P ? x[i] * c - partner[i] * s
                                      : x[i] * c + partner[i] * s;
    }
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

// Span `span` of `row`'s keys: visible history spans 0..R-1, then the fresh
// span R (rows of this batch).
inline void span(device const int *visible, device const int *fresh, ulong row, ulong spans, ulong span,
    thread int &lo, thread int &hi) {
    lo = span < spans ? visible[(row * spans + span) * 2] : fresh[row * 2];
    hi = span < spans ? visible[(row * spans + span) * 2 + 1] : fresh[row * 2 + 1];
}

// The total number of keys a row sees: its visible spans, then its fresh span.
inline uint visible_total(device const int *visible, device const int *fresh, ulong row, ulong spans) {
    uint total = 0;
    for (ulong index = 0; index <= spans; ++index) {
        int lo, hi;
        span(visible, fresh, row, spans, index, lo, hi);
        total += uint(metal::max(hi - lo, 0));
    }
    return total;
}

// Keys per decode partition for a row seeing `total` keys: at least `span`,
// and few enough that `parts` partitions cover the row. Never zero.
inline uint partition_span(uint total, uint span, uint parts) {
    return metal::max(span, (total + parts - 1) / parts);
}

// The online-softmax state of G query heads over one simdgroup's keys, in the
// exp2 domain, absorbing N keys at a time.
template <uint N>
inline void absorb(thread const float (&q)[SEISMIC_DIM_G][ATTENTION_E],
    thread const float (&k)[N][ATTENTION_E], thread const float (&v)[N][ATTENTION_E],
    thread float (&maximum)[SEISMIC_DIM_G], thread float (&denominator)[SEISMIC_DIM_G],
    thread float (&output)[SEISMIC_DIM_G][ATTENTION_E]) {
    ATTENTION_UNROLL
    for (uint g = 0; g < SEISMIC_DIM_G; ++g) {
        float score[N];
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j) {
            float partial = 0.0f;
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                partial = metal::fma(q[g][i], k[j][i], partial);
            score[j] = simd_sum(partial);
        }
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

// Publishes one decode partition: the SIMDS simdgroup states of G query heads
// (outputs relative to their own maxima) merge in simdgroup order into one
// partial per query head g at slot first + g * PARTS: the unnormalized output
// at partials[slot * W] relative to the partition maximum, and (maximum,
// denominator) at statistics[slot * 2]. `states` holds [SIMDS][G][2] floats,
// `columns` [SIMDS][W] floats of threadgroup memory.
template <uint SIMDS, uint PARTS>
inline void publish(thread const float (&maximum)[SEISMIC_DIM_G],
    thread const float (&denominator)[SEISMIC_DIM_G],
    thread const float (&output)[SEISMIC_DIM_G][ATTENTION_E], threadgroup float *states,
    threadgroup float *columns, device float *partials, device float *statistics, ulong first,
    uint simd, uint lane, uint thread_index) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint G = SEISMIC_DIM_G;
    if (lane == 0) {
        ATTENTION_UNROLL
        for (uint g = 0; g < G; ++g) {
            states[(simd * G + g) * 2] = maximum[g];
            states[(simd * G + g) * 2 + 1] = denominator[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        float partition_maximum = -INFINITY;
        for (uint s = 0; s < SIMDS; ++s)
            if (states[(s * G + g) * 2 + 1] > 0.0f)
                partition_maximum = metal::max(partition_maximum, states[(s * G + g) * 2]);
        const float weight = denominator[g] > 0.0f
            ? metal::fast::exp2(maximum[g] - partition_maximum) : 0.0f;
        ATTENTION_UNROLL
        for (uint i = 0; i < E; ++i)
            columns[simd * W + lane * E + i] = output[g][i] * weight;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong slot = first + g * PARTS;
        for (uint column = thread_index; column < W; column += SIMDS * 32) {
            float sum = 0.0f;
            for (uint s = 0; s < SIMDS; ++s)
                sum += columns[s * W + column];
            partials[slot * W + column] = sum;
        }
        if (thread_index == 0) {
            float total_denominator = 0.0f;
            for (uint s = 0; s < SIMDS; ++s) {
                const float d = states[(s * G + g) * 2 + 1];
                if (d > 0.0f)
                    total_denominator = metal::fma(d,
                        metal::fast::exp2(states[(s * G + g) * 2] - partition_maximum),
                        total_denominator);
            }
            statistics[slot * 2] = partition_maximum;
            statistics[slot * 2 + 1] = total_denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// The decode merge of one (query head, row) column over the row's `spans`
// visible spans: the row's non-empty partitions in partition order, then the
// sigmoid gate. A row that sees no key attends to zero.
template <uint SPAN, uint PARTS>
inline void decode_gate(device const Scalar *query_gate, device const int *visible,
    device const int *fresh, device Scalar *gated, device const float *partials,
    device const float *statistics, ulong spans, ulong head, ulong row, uint column) {
    constexpr uint W = ATTENTION_W;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    const uint total = visible_total(visible, fresh, row, spans);
    const uint span_keys = partition_span(total, SPAN, PARTS);
    const uint active = (total + span_keys - 1) / span_keys;
    const float attended = merge(partials, statistics, (row * KV * G + head) * PARTS, 1, active, column);
    const float gate = float(query_gate[(row * KV * G + head) * 2 * W + W + column]);
    gated[(row * KV * G + head) * W + column] = Scalar(attended / (1.0f + metal::exp(-gate)));
}

// ---------------------------------------------------------------------------
// Affine K8/V4 history (`qwen_attention_*_k8v4`). A (history row, kv head)
// vector is a code row of W * B / 32 u32 words (code i at bits B * (i % (32 /
// B)) of word i / (32 / B)) plus an F16 (scale, zero) pair; decoded value =
// code * scale + zero. A lane holding E columns of a vector owns E * B code
// bits: whole words when E * B >= 32, else a part of a word shared with its
// neighbours.
// ---------------------------------------------------------------------------

#define ATTENTION_KEY_BITS 8
#define ATTENTION_VALUE_BITS 4

template <uint B>
struct lane_codes {
    // Compile-time constants (MSL admits no non-constant-space static data).
    enum : uint {
        bits = ATTENTION_E * B,
        words = (bits + 31) / 32,
        row_words = ATTENTION_W * B / 32,
        levels = (1u << B) - 1u,
    };
    static_assert(bits % 32 == 0 || 32 % bits == 0,
        "a lane's codes are whole words or a power-of-two part of one");

    // This lane's codes of one vector's code row, shifted so its code i sits
    // at bits B * i of the (i * B / 32)-th word.
    static inline void load(device const uint *row, uint lane, thread uint (&w)[words]) {
        if (bits == 64) {
            const uint2 pair = *reinterpret_cast<device const uint2 *>(row + lane * 2);
            w[0] = pair.x;
            w[words - 1] = pair.y;
        } else if (bits >= 32) {
            ATTENTION_UNROLL
            for (uint j = 0; j < words; ++j)
                w[j] = row[lane * words + j];
        } else {
            w[0] = row[lane * bits / 32] >> ((lane * bits) % 32);
        }
    }

    // Code i of this lane's columns, as F32 (exact).
    static inline float code(thread const uint (&w)[words], uint i) {
        return float((w[i * B / 32] >> ((i * B) % 32)) & levels);
    }

    // Every code of this lane's columns, as F32 (exact). Whole words unpack
    // two codes at a time without a conversion instruction: a code c in the
    // mantissa of the half 1024 (0x6400 | c) is 1024 + c exactly, and one
    // half2 subtraction leaves c.
    static inline void unpack(thread const uint (&w)[words], thread float (&x)[ATTENTION_E]) {
        constexpr uint pair_mask = B == 8 ? 0x00FF00FFu : 0x000F000Fu;
        constexpr uint per_word = 32 / B;
        if (bits >= 32) {
            ATTENTION_UNROLL
            for (uint j = 0; j < words; ++j) {
                // Shift s of the word holds codes s and s + per_word / 2 in
                // its two halves.
                ATTENTION_UNROLL
                for (uint s = 0; s < per_word / 2; ++s) {
                    const half2 pair = as_type<half2>(((w[j] >> (s * B)) & pair_mask) | 0x64006400u)
                        - half2(1024.0h);
                    x[j * per_word + s] = float(pair.x);
                    x[j * per_word + s + per_word / 2] = float(pair.y);
                }
            }
        } else {
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                x[i] = code(w, i);
        }
    }
};

// Encodes one vector held by a simdgroup (lane `lane` owns columns [lane * E,
// lane * E + E)) with B-bit codes at `row` (its code row) and its coefficient
// pair at `coefficients`: zero = f16(min), scale = f16((max - min) / L),
// code = min(L, u32(fma(x - zero, 1 / scale, 0.5))), 0 when scale is 0.
template <uint B>
inline void encode(thread const float (&x)[ATTENTION_E], device uint *row,
    device half *coefficients, uint lane) {
    typedef lane_codes<B> codes;
    float low = x[0];
    float high = x[0];
    ATTENTION_UNROLL
    for (uint i = 1; i < ATTENTION_E; ++i) {
        low = metal::min(low, x[i]);
        high = metal::max(high, x[i]);
    }
    low = simd_min(low);
    high = simd_max(high);
    const half zero = half(low);
    const half scale = half((high - low) / float(codes::levels));
    const float inverse = float(scale) > 0.0f ? 1.0f / float(scale) : 0.0f;
    uint w[codes::words];
    ATTENTION_UNROLL
    for (uint j = 0; j < codes::words; ++j)
        w[j] = 0u;
    ATTENTION_UNROLL
    for (uint i = 0; i < ATTENTION_E; ++i) {
        const float t = metal::fma(x[i] - float(zero), inverse, 0.5f);
        const uint c = metal::min(uint(metal::max(t, 0.0f)), uint(codes::levels));
        w[i * B / 32] |= c << ((i * B) % 32);
    }
    if (codes::bits >= 32) {
        ATTENTION_UNROLL
        for (uint j = 0; j < codes::words; ++j)
            row[lane * codes::words + j] = w[j];
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        constexpr uint sharing = codes::bits >= 32 ? 1 : 32 / codes::bits;
        uint joined = w[0] << ((lane % sharing) * codes::bits);
        ATTENTION_UNROLL
        for (uint offset = 1; offset < sharing; offset *= 2)
            joined |= simd_shuffle_xor(joined, ushort(offset));
        if (lane % sharing == 0)
            row[lane / sharing] = joined;
    }
    if (lane == 0) {
        coefficients[0] = scale;
        coefficients[1] = zero;
    }
}

// The online-softmax state of G query heads absorbing N affine-coded keys and
// values. Scores are corrected, not decoded: scale * (q . code) + zero *
// qsum. The value product accumulates (p * scale) * code into `output` and
// sum(p * zero) into the per-head `bias`, both carried by the same factor,
// so output + bias is the attended sum.
template <uint N>
inline void absorb_affine(thread const float (&q)[SEISMIC_DIM_G][ATTENTION_E],
    thread const float (&qsum)[SEISMIC_DIM_G],
    thread const uint (&key)[N][lane_codes<ATTENTION_KEY_BITS>::words],
    thread const float2 (&key_coefficients)[N],
    thread const uint (&value)[N][lane_codes<ATTENTION_VALUE_BITS>::words],
    thread const float2 (&value_coefficients)[N],
    thread float (&maximum)[SEISMIC_DIM_G], thread float (&denominator)[SEISMIC_DIM_G],
    thread float (&output)[SEISMIC_DIM_G][ATTENTION_E], thread float (&bias)[SEISMIC_DIM_G]) {
    typedef lane_codes<ATTENTION_KEY_BITS> key_codes;
    typedef lane_codes<ATTENTION_VALUE_BITS> value_codes;
    constexpr uint G = SEISMIC_DIM_G;
    // Codes unpack one token at a time, so only one token's columns are live
    // as F32 next to the scores and the output.
    float score[G][N];
    ATTENTION_UNROLL
    for (uint j = 0; j < N; ++j) {
        float k[ATTENTION_E];
        key_codes::unpack(key[j], k);
        ATTENTION_UNROLL
        for (uint g = 0; g < G; ++g) {
            float partial = 0.0f;
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                partial = metal::fma(q[g][i], k[i], partial);
            score[g][j] = metal::fma(key_coefficients[j].x, simd_sum(partial),
                key_coefficients[j].y * qsum[g]);
        }
    }
    // Each value code's weight: its probability times the value scale.
    float weight[G][N];
    ATTENTION_UNROLL
    for (uint g = 0; g < G; ++g) {
        float next = maximum[g];
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j)
            next = metal::max(next, score[g][j]);
        const float carry = metal::fast::exp2(maximum[g] - next);
        float sum = 0.0f;
        float offset = bias[g] * carry;
        ATTENTION_UNROLL
        for (uint j = 0; j < N; ++j) {
            const float probability = metal::fast::exp2(score[g][j] - next);
            sum += probability;
            offset = metal::fma(probability, value_coefficients[j].y, offset);
            weight[g][j] = probability * value_coefficients[j].x;
        }
        denominator[g] = metal::fma(denominator[g], carry, sum);
        bias[g] = offset;
        maximum[g] = next;
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            output[g][i] *= carry;
    }
    ATTENTION_UNROLL
    for (uint j = 0; j < N; ++j) {
        float v[ATTENTION_E];
        value_codes::unpack(value[j], v);
        ATTENTION_UNROLL
        for (uint g = 0; g < G; ++g)
            ATTENTION_UNROLL
            for (uint i = 0; i < ATTENTION_E; ++i)
                output[g][i] = metal::fma(weight[g][j], v[i], output[g][i]);
    }
}

// ---------------------------------------------------------------------------
// Prefill (`qwen_attention_prefill`, `qwen_attention_prefill_k8v4`): the three
// launches' bodies, over a history policy that appends a row's key and value
// and stages history K/V tiles in the activation dtype.
// ---------------------------------------------------------------------------

// Keys per K/V tile staged in threadgroup memory.
#define PREFILL_KEYS 32
// A partition covers at least this many key tiles, so a query tile whose keys
// fit in a few partitions' worth runs unsplit and skips the merge.
#define PREFILL_MIN_TILES 16
// Row pitch of a staged tile, in elements.
#define PREFILL_PITCH (ATTENTION_W + 8)

// The interval union [lo, hi) of a tile's non-empty row intervals and the
// intersection [common_lo, common_hi) of all its row intervals, for one span.
struct prefill_interval {
    int lo;
    int hi;
    int common_lo;
    int common_hi;
};

// Copies key rows [first, first + PREFILL_KEYS) of one kv head (row-major
// [T, KV, W], 16-byte aligned) into `staged` (row pitch PREFILL_PITCH) in
// 16-byte pieces; rows at or past `end` are zero. The loop stays rolled: one
// piece in flight per thread keeps the staging registers off the
// accumulators' budget.
template <uint THREADS>
inline void prefill_stage(threadgroup Scalar *staged, device const Scalar *rows,
    int first, int end, uint kv_head, uint thread_index) {
    constexpr uint W = ATTENTION_W;
    constexpr uint PIECES = W / 8;
    ATTENTION_ROLLED
    for (uint item = thread_index; item < PREFILL_KEYS * PIECES; item += THREADS) {
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

// Dense history: [T, KV, W] activation planes.
struct dense_history {
    device Scalar *key;
    device Scalar *value;

    inline void append(int destination, uint kv_head, uint lane, thread const float (&k)[ATTENTION_E],
        thread const Scalar (&v)[ATTENTION_E]) const {
        attention::append(key, destination, kv_head, lane, k);
        attention::append(value, destination, kv_head, lane, v);
    }

    template <uint THREADS>
    inline void stage_key(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        prefill_stage<THREADS>(staged, key, first, end, kv_head, thread_index);
    }

    template <uint THREADS>
    inline void stage_value(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        prefill_stage<THREADS>(staged, value, first, end, kv_head, thread_index);
    }
};

// Affine K8/V4 history: code rows plus (scale, zero) pairs per (row, kv head).
// A staged tile holds the decoded values code * scale + zero rounded to the
// activation dtype, so the products are the dense ones.
struct affine_history {
    device uint *key_codes;
    device half *key_coefficients;
    device uint *value_codes;
    device half *value_coefficients;

    inline void append(int destination, uint kv_head, uint lane, thread const float (&k)[ATTENTION_E],
        thread const Scalar (&v)[ATTENTION_E]) const {
        const ulong vector = ulong(destination) * SEISMIC_DIM_KV + kv_head;
        float x[ATTENTION_E];
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            x[i] = float(Scalar(k[i]));
        encode<ATTENTION_KEY_BITS>(x, key_codes + vector * lane_codes<ATTENTION_KEY_BITS>::row_words,
            key_coefficients + vector * 2, lane);
        ATTENTION_UNROLL
        for (uint i = 0; i < ATTENTION_E; ++i)
            x[i] = float(v[i]);
        encode<ATTENTION_VALUE_BITS>(x,
            value_codes + vector * lane_codes<ATTENTION_VALUE_BITS>::row_words,
            value_coefficients + vector * 2, lane);
    }

    // Rows [first, first + PREFILL_KEYS) of one kv head decoded into `staged`,
    // one 16-byte code piece (128 / B codes) per item; rows at or past `end`
    // are zero.
    template <uint B, uint THREADS>
    static inline void stage(threadgroup Scalar *staged, device const uint *codes,
        device const half *coefficients, int first, int end, uint kv_head, uint thread_index) {
        constexpr uint W = ATTENTION_W;
        constexpr uint PER = 128 / B;
        constexpr uint PIECES = W / PER;
        constexpr uint MASK = (1u << B) - 1u;
        constexpr uint ROW_WORDS = lane_codes<B>::row_words;
        ATTENTION_ROLLED
        for (uint item = thread_index; item < PREFILL_KEYS * PIECES; item += THREADS) {
            const uint k = item / PIECES;
            const uint c = item % PIECES;
            const int t = first + int(k);
            threadgroup Scalar *to = staged + k * PREFILL_PITCH + c * PER;
            if (t < end) {
                const ulong vector = ulong(t) * SEISMIC_DIM_KV + kv_head;
                const uint4 words = *reinterpret_cast<device const uint4 *>(codes + vector * ROW_WORDS + c * 4);
                const float2 pair = float2(*reinterpret_cast<device const half2 *>(coefficients + vector * 2));
                // Element pairs of the piece, low element in the low half.
                uint packed[PER / 2];
                ATTENTION_UNROLL
                for (uint i = 0; i < PER; i += 2) {
                    const uint word = words[i * B / 32];
                    const uint shift = (i * B) % 32;
                    const Scalar lo = Scalar(metal::fma(float((word >> shift) & MASK), pair.x, pair.y));
                    const Scalar hi = Scalar(metal::fma(float((word >> (shift + B)) & MASK), pair.x, pair.y));
                    packed[i / 2] = uint(as_type<ushort>(lo)) | (uint(as_type<ushort>(hi)) << 16);
                }
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

    template <uint THREADS>
    inline void stage_key(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        stage<ATTENTION_KEY_BITS, THREADS>(staged, key_codes, key_coefficients, first, end, kv_head,
            thread_index);
    }

    template <uint THREADS>
    inline void stage_value(threadgroup Scalar *staged, int first, int end, uint kv_head,
        uint thread_index) const {
        stage<ATTENTION_VALUE_BITS, THREADS>(staged, value_codes, value_coefficients, first, end,
            kv_head, thread_index);
    }
};

// L1: one simdgroup per (row, query head or kv head), rows padded to whole
// QT tiles. Queries and keys are prepared in the activation dtype and go to
// scratch exactly as rounded (L2 applies the softmax scale to the F32
// scores); padding rows' queries are zero. The value is copied beside the key
// so L2 reads every fresh operand from aligned scratch, and the key and value
// are appended at the row's destination through the history policy.
template <uint QT, class History>
inline void prefill_prepare(History history, device const Scalar *query_gate,
    device const Scalar *key, device const Scalar *value, device const float *query_norm,
    device const float *key_norm, device const int *rotary_components,
    device const float *rotary_frequencies, device const int *coordinates,
    device const int *destinations, device Scalar *queries, device Scalar *keys,
    device Scalar *values, ulong M, float epsilon, uint group, uint simd, uint lane) {
    constexpr uint W = ATTENTION_W;
    constexpr uint E = ATTENTION_E;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    const ulong item = ulong(group) * 8 + simd;
    const ulong row = item / (KV * (G + 1));
    const ulong head = item % (KV * (G + 1));
    if (row >= (M + QT - 1) / QT * QT)
        return;
    if (row >= M) {
        if (head < KV * G)
            for (uint i = 0; i < E; ++i)
                queries[(row * KV * G + head) * W + lane * E + i] = Scalar(0.0f);
        return;
    }
    float x[E];
    if (head < KV * G) {
        prepare(query_gate + (row * KV * G + head) * 2 * W, query_norm,
            coordinates + row * 4, rotary_components, rotary_frequencies, epsilon, lane, x);
        const ulong at = (row * KV * G + head) * W + lane * E;
        for (uint i = 0; i < E; ++i)
            queries[at + i] = Scalar(x[i]);
        return;
    }
    const ulong kv_head = head - KV * G;
    const ulong source = (row * KV + kv_head) * W;
    prepare(key + source, key_norm, coordinates + row * 4, rotary_components,
        rotary_frequencies, epsilon, lane, x);
    Scalar v[E];
    for (uint i = 0; i < E; ++i) {
        keys[source + lane * E + i] = Scalar(x[i]);
        v[i] = value[source + lane * E + i];
        values[source + lane * E + i] = v[i];
    }
    const int destination = destinations[row];
    if (destination < 0)
        return;
    history.append(destination, uint(kv_head), lane, x, v);
}

// L2: threadgroup (QT-row tile, kv head, key partition). Simdgroup s owns 8
// rows: tokens tile * QT + (s % (QT / 8)) * 8 + [0, 8) of query head
// kv * G + s / (QT / 8). The tile's key tiles (each span's union interval in
// PREFILL_KEYS steps, spans then fresh) split into consecutive runs of at
// least PREFILL_MIN_TILES over the partitions. Per key tile: K staged
// (history through the policy, fresh rows from scratch), scores = Q K^T with
// Q's 8x8 fragments read from scratch (L1-resident; holding them in
// registers costs more occupancy than the loads), scaled into the exp2
// domain in F32, the online softmax, then V staged (aliasing K) and the F32
// output accumulated from activation-dtype probabilities. Every fragment
// array is fully unrolled so it stays in registers. Query tiles dispatch
// last-first. A tile served by one partition stores its gated output
// directly; otherwise each partition stores (partial output, maximum,
// denominator) and the merge launch combines them.
template <uint QT, class History>
inline void prefill_attend(History history, device const Scalar *query_gate,
    device const int *visible, device const int *fresh, device Scalar *gated,
    device const Scalar *queries, device const Scalar *keys, device const Scalar *values,
    device float *partials, device float *statistics, device uint *counts, ulong M, ulong R,
    float scale, threadgroup uchar *shared, uint3 group, uint3 groups, uint thread_index,
    uint simd, uint lane) {
    constexpr uint W = ATTENTION_W;
    constexpr uint KV = SEISMIC_DIM_KV;
    constexpr uint G = SEISMIC_DIM_G;
    constexpr uint KEYS = PREFILL_KEYS;
    constexpr uint BLOCKS = QT / 8;
    constexpr uint THREADS = QT * G * 4;
    constexpr uint PITCH = PREFILL_PITCH;
    constexpr uint DB = W / 8;
    constexpr uint KB = KEYS / 8;
    static_assert(QT % 8 == 0 && QT <= 32, "query tiles are 8-row blocks within one simdgroup's lanes");
    // Query tiles dispatch last-first: in a causal chunk the last tiles see
    // the most keys, and starting them first shortens the grid's tail.
    const uint tile = groups.x - 1 - group.x;
    const uint kv_head = group.y;
    const uint partition = group.z;
    threadgroup Scalar *staged = reinterpret_cast<threadgroup Scalar *>(shared);
    threadgroup prefill_interval *intervals = reinterpret_cast<threadgroup prefill_interval *>(
        shared + KEYS * PITCH * sizeof(Scalar));

    if (simd == 0) {
        const ulong tile_row = ulong(tile) * QT + lane;
        const bool row_valid = lane < QT && tile_row < M;
        for (ulong index = 0; index <= R; ++index) {
            int lo = 0;
            int hi = 0;
            if (row_valid)
                span(visible, fresh, tile_row, R, index, lo, hi);
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
    if (partition == 0 && kv_head == 0 && thread_index == 0)
        counts[tile] = active;
    const uint tiles_lo = partition * per;
    const uint tiles_hi = metal::min(tiles_lo + per, total_tiles);

    // Fragment coordinates: this lane holds row `fm`, columns `fn`, `fn + 1`.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + ((lane / 2) % 4);
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const uint head = kv_head * G + simd / BLOCKS;
    const ulong first_token = ulong(tile) * QT + (simd % BLOCKS) * 8;
    const ulong token = first_token + fm;
    const bool valid = token < M;
    device const Scalar *query_rows = queries + (first_token * KV * G + head) * W;

    simdgroup_matrix<float, 8, 8> output[DB];
    ATTENTION_UNROLL
    for (uint d = 0; d < DB; ++d)
        output[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
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
        device const int *bounds = historical ? visible + (token * R + index) * 2 : fresh + token * 2;
        for (uint own = own_lo; own < own_hi; ++own) {
            const int first = interval.lo + int(own * KEYS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (historical)
                history.template stage_key<THREADS>(staged, first, interval.hi, kv_head, thread_index);
            else
                prefill_stage<THREADS>(staged, keys, first, interval.hi, kv_head, thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_matrix<float, 8, 8> scores[KB];
            ATTENTION_UNROLL
            for (uint j = 0; j < KB; ++j)
                scores[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                simdgroup_matrix<Scalar, 8, 8> q;
                simdgroup_load(q, query_rows + d * 8, KV * G * W);
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    simdgroup_matrix<Scalar, 8, 8> k;
                    simdgroup_load(k, staged + j * 8 * PITCH + d * 8, PITCH, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(scores[j], q, k, scores[j]);
                }
            }

            // Rows' bounds are read only for a tile outside the common
            // interval, so they hold no registers across the key loop.
            const bool common = first >= interval.common_lo
                && first + int(KEYS) <= interval.common_hi;
            int row_lo = 0;
            int row_hi = 0;
            if (!common && valid) {
                row_lo = bounds[0];
                row_hi = bounds[1];
            }
            float tile_maximum = -INFINITY;
            ATTENTION_UNROLL
            for (uint j = 0; j < KB; ++j) {
                ATTENTION_UNROLL
                for (uint e = 0; e < 2; ++e) {
                    float s = scores[j].thread_elements()[e] * scale;
                    if (!common) {
                        const int t = first + int(j * 8 + fn + e);
                        if (!(t >= row_lo && t < row_hi))
                            s = -INFINITY;
                    }
                    scores[j].thread_elements()[e] = s;
                    tile_maximum = metal::max(tile_maximum, s);
                }
            }
            tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(1)));
            tile_maximum = metal::max(tile_maximum, simd_shuffle_xor(tile_maximum, ushort(8)));
            const float next = metal::max(maximum, tile_maximum);
            const bool seen = next > -INFINITY;
            const float carry = seen ? metal::fast::exp2(maximum - next) : 1.0f;
            // Probabilities enter the PV product in the activation dtype; the
            // denominator sums them in F32.
            simdgroup_matrix<Scalar, 8, 8> probabilities[KB];
            float tile_sum = 0.0f;
            ATTENTION_UNROLL
            for (uint j = 0; j < KB; ++j) {
                ATTENTION_UNROLL
                for (uint e = 0; e < 2; ++e) {
                    const float p = seen ? metal::fast::exp2(scores[j].thread_elements()[e] - next) : 0.0f;
                    probabilities[j].thread_elements()[e] = Scalar(p);
                    tile_sum += p;
                }
            }
            tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
            tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
            denominator = metal::fma(denominator, carry, tile_sum);
            maximum = next;
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                output[d].thread_elements()[0] *= carry;
                output[d].thread_elements()[1] *= carry;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (historical)
                history.template stage_value<THREADS>(staged, first, interval.hi, kv_head, thread_index);
            else
                prefill_stage<THREADS>(staged, values, first, interval.hi, kv_head, thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            ATTENTION_UNROLL
            for (uint d = 0; d < DB; ++d) {
                ATTENTION_UNROLL
                for (uint j = 0; j < KB; ++j) {
                    simdgroup_matrix<Scalar, 8, 8> v;
                    simdgroup_load(v, staged + j * 8 * PITCH + d * 8, PITCH);
                    simdgroup_multiply_accumulate(output[d], probabilities[j], v, output[d]);
                }
            }
        }
    }

    if (!valid)
        return;
    if (active > 1) {
        const ulong slot = (ulong(partition) * M + token) * (KV * G) + head;
        ATTENTION_UNROLL
        for (uint d = 0; d < DB; ++d)
            ATTENTION_UNROLL
            for (uint e = 0; e < 2; ++e)
                partials[slot * W + d * 8 + fn + e] = output[d].thread_elements()[e];
        if (fn == 0) {
            statistics[slot * 2] = maximum;
            statistics[slot * 2 + 1] = denominator;
        }
        return;
    }
    const float inverse = 1.0f / metal::max(denominator, 1e-30f);
    ATTENTION_UNROLL
    for (uint d = 0; d < DB; ++d) {
        ATTENTION_UNROLL
        for (uint e = 0; e < 2; ++e) {
            const uint column = d * 8 + fn + e;
            const float gate = float(query_gate[(token * KV * G + head) * 2 * W + W + column]);
            gated[(token * KV * G + head) * W + column] = Scalar(
                output[d].thread_elements()[e] * inverse / (1.0f + metal::exp(-gate)));
        }
    }
}

// L3: threadgroup (QT-row tile, query head), one thread per column. A tile
// that took several key partitions merges each of its rows' partitions in
// partition order and applies the gate; other tiles were stored by L2.
template <uint QT>
inline void prefill_merge(device const Scalar *query_gate, device Scalar *gated,
    device const float *partials, device const float *statistics, device const uint *counts,
    ulong M, uint tile, ulong head, uint column) {
    constexpr uint W = ATTENTION_W;
    constexpr uint H = SEISMIC_DIM_KV * SEISMIC_DIM_G;
    const uint count = counts[tile];
    if (count <= 1)
        return;
    for (ulong row = ulong(tile) * QT; row < metal::min(ulong(tile + 1) * QT, M); ++row) {
        const float attended = merge(partials, statistics, row * H + head, M * H, count, column);
        const float gate = float(query_gate[(row * H + head) * 2 * W + W + column]);
        gated[(row * H + head) * W + column] = Scalar(attended / (1.0f + metal::exp(-gate)));
    }
}

} // namespace attention
