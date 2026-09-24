// Shared device code of the CUDA gated-attention entries (`qwen_attention_
// decode`, `qwen_attention_prefill` and their affine K8/V4 history forms
// `qwen_attention_{decode,prefill}_k8v4`): the per-head q/k preparation
// (`norm_rotary`: RMS norm, weight, partial M-RoPE) computed by one warp whose
// lanes own contiguous dimensions, a row's span walk, partition bounds, the
// online-softmax absorb of N keys, the fixed-order merge of partial softmax
// states, the decode publication and gated merge, the history policies
// (dense planes; the affine codec: encode on append, per-lane code access),
// and the entry's tensor addressing. Each kernel keeps its own loop structure
// and calls these; the prefill bodies are in common/attention_prefill.cuh.
// Every activation tensor is canonical (unit innermost stride); rows and
// heads are addressed through their ABI strides.

#include "common/element.cuh"

namespace attention {

using element::u8;
using element::u16;
using element::u32;
using element::u64;
typedef element::Act Act;

constexpr int KV = static_cast<int>(SEISMIC_DIM_KV);
constexpr int G = static_cast<int>(SEISMIC_DIM_G);
constexpr int P = static_cast<int>(SEISMIC_DIM_P);
constexpr int W = static_cast<int>(2 * SEISMIC_DIM_P + SEISMIC_DIM_S);
// Dimensions owned by one lane when a warp holds a W-vector.
constexpr int DPL = W / 32;
static_assert(W % 32 == 0, "head width is a multiple of 32");
// log2(e): scores are kept in the exp2 domain.
constexpr float LOG2E = 1.4426950408889634f;

struct Inputs {
    // The launch's argument words, read by the ABI stride macros.
    const seismic_words_t *words;
    const u8 *query_gate;
    const u8 *key;
    const u8 *value;
    const float *query_norm;
    const float *key_norm;
    const int *components;
    const int *coordinates;
    const int *visible;
    const int *fresh;
    const int *destinations;
    const float *frequencies;
    float epsilon;
    float scale;
};

// The inputs every entry shares; its history planes are a separate policy
// (`DenseHistory`, `AffineHistory`).
#define ATTENTION_INPUTS()                                                                      \
    attention::Inputs {                                                                         \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE), SEISMIC_PTR(SEISMIC_BUFFER_KEY),                \
            SEISMIC_PTR(SEISMIC_BUFFER_VALUE),                                                  \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)),              \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),       \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)),             \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH)),                   \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),   \
            element::word_f32(SEISMIC_PARAM_EPSILON),                                           \
            element::word_f32(SEISMIC_PARAM_SCALE)                                              \
    }

// Element offsets (all canonical within a row).
#define ATTENTION_QUERY_AT(row, query_head)                                              \
    (static_cast<attention::u64>(row) * SEISMIC_QUERY_GATE_STRIDE_0 +                    \
     static_cast<attention::u64>(query_head) * 2 * attention::W)
#define ATTENTION_GATE_AT(row, query_head) (ATTENTION_QUERY_AT(row, query_head) + attention::W)
#define ATTENTION_KEY_AT(row, kv_head)                                                   \
    (static_cast<attention::u64>(row) * SEISMIC_KEY_STRIDE_0 +                           \
     static_cast<attention::u64>(kv_head) * attention::W)
#define ATTENTION_VALUE_AT(row, kv_head)                                                 \
    (static_cast<attention::u64>(row) * SEISMIC_VALUE_STRIDE_0 +                         \
     static_cast<attention::u64>(kv_head) * attention::W)
#define ATTENTION_VISIBLE(in, row, span, bound)                                          \
    ((in).visible[static_cast<attention::u64>(row) * SEISMIC_VISIBLE_STRIDE_0 +          \
                  static_cast<attention::u64>(span) * SEISMIC_VISIBLE_STRIDE_1 +         \
                  static_cast<attention::u64>(bound) * SEISMIC_VISIBLE_STRIDE_2])
#define ATTENTION_FRESH(in, row, bound)                                                  \
    ((in).fresh[static_cast<attention::u64>(row) * SEISMIC_FRESH_STRIDE_0 +              \
                static_cast<attention::u64>(bound) * SEISMIC_FRESH_STRIDE_1])
#define ATTENTION_DESTINATION(in, row)                                                   \
    ((in).destinations[static_cast<attention::u64>(row) * SEISMIC_DESTINATIONS_STRIDE_0])

// ---------------------------------------------------------------------------
// Span walk and partitions.

// Span `span` of `row`'s keys (0..R-1 visible history, R the fresh batch
// rows), as [lo, hi); an empty span has hi <= lo.
struct Span {
    int lo;
    int hi;
};
__device__ __forceinline__ Span span(const Inputs &in, u64 row, u64 span, u64 spans) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (span < spans) {
        return Span{ATTENTION_VISIBLE(in, row, span, 0), ATTENTION_VISIBLE(in, row, span, 1)};
    }
    return Span{ATTENTION_FRESH(in, row, 0), ATTENTION_FRESH(in, row, 1)};
}

// The total number of keys a row sees: its visible spans, then its fresh span.
__device__ __forceinline__ long long visible_total(const Inputs &in, u64 row, u64 spans) {
    long long total = 0;
    for (u64 index = 0; index <= spans; ++index) {
        const Span s = span(in, row, index, spans);
        total += s.hi > s.lo ? s.hi - s.lo : 0;
    }
    return total;
}

// Part `index` of [lo, hi) split into parts of `per` keys (the last ones
// shorter or empty), as [lo, hi) in the same key numbering.
struct Range {
    long long lo;
    long long hi;
};
__device__ __forceinline__ Range partition(Range whole, long long per, long long index) {
    const long long lo = min(whole.hi, whole.lo + per * index);
    return Range{lo, min(whole.hi, lo + per)};
}

// ---------------------------------------------------------------------------
// Online softmax.

// The online-softmax state of G query heads over one warp's keys, in the exp2
// domain; lane `lane` owns dimensions [lane * DPL, (lane + 1) * DPL).
struct State {
    float maximum[G];
    float denominator[G];
    float accumulator[G][DPL];
};

__device__ __forceinline__ void clear(State &state) {
#pragma unroll
    for (int h = 0; h < G; ++h) {
        state.maximum[h] = -__int_as_float(0x7f800000);
        state.denominator[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] = 0.0f;
    }
}

// The scores (exp2 domain when `q` is scaled into it) of N keys for G query
// heads: warp-reduced dot products.
template <int N>
__device__ __forceinline__ void scores(const float (&q)[G][DPL], const float (&k)[N][DPL],
                                       float (&score)[N][G]) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < DPL; ++d) dot = __fmaf_rn(q[h][d], k[t][d], dot);
            score[t][h] = seismic_warp_sum_f32(dot);
        }
    }
}

// Absorb `count` (1..N) keys with scores `score` (exp2 domain) and values `v`.
template <int N>
__device__ __forceinline__ void absorb(State &state, const float (&score)[N][G],
                                       const float (&v)[N][DPL], int count) {
#pragma unroll
    for (int h = 0; h < G; ++h) {
        float maximum = state.maximum[h];
#pragma unroll
        for (int t = 0; t < N; ++t)
            if (t < count) maximum = fmaxf(maximum, score[t][h]);
        // exp2(-inf) = 0 for the first keys of an empty state.
        const float carry = seismic_ex2_approx(state.maximum[h] - maximum);
        float denominator = state.denominator[h] * carry;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] *= carry;
#pragma unroll
        for (int t = 0; t < N; ++t) {
            if (t < count) {
                const float p = seismic_ex2_approx(score[t][h] - maximum);
                denominator += p;
#pragma unroll
                for (int d = 0; d < DPL; ++d)
                    state.accumulator[h][d] = __fmaf_rn(p, v[t][d], state.accumulator[h][d]);
            }
        }
        state.denominator[h] = denominator;
        state.maximum[h] = maximum;
    }
}

// The fixed-order merge of `count` partial softmax states for one column:
// state p has (maximum, denominator) at statistics[p * stride] and its output
// (relative to its own maximum) at values[p * pitch]. Empty states
// (denominator 0) are skipped. Returns the merged maximum; `denominator` and
// `accumulated` are the merged sums relative to it (0 when no state is
// non-empty).
__device__ __forceinline__ float merge(const float *statistics, u64 stride, const float *values,
                                       u64 pitch, int count, float &denominator,
                                       float &accumulated) {
    float maximum = -__int_as_float(0x7f800000);
    for (int p = 0; p < count; ++p)
        if (statistics[p * stride + 1] > 0.0f) maximum = fmaxf(maximum, statistics[p * stride]);
    denominator = 0.0f;
    accumulated = 0.0f;
    for (int p = 0; p < count; ++p) {
        const float l = statistics[p * stride + 1];
        if (l > 0.0f) {
            const float weight = seismic_ex2_approx(statistics[p * stride] - maximum);
            denominator = __fmaf_rn(l, weight, denominator);
            accumulated = __fmaf_rn(weight, values[p * pitch], accumulated);
        }
    }
    return maximum;
}

// Publishes one decode partition: the WARPS warp states of G query heads
// merge in warp order into one partial per (row, query head, partition)
// `part`: the unnormalized output at partials[slot * W] and (maximum,
// denominator) at statistics[slot * 2], slot = (row * KV * G + kv * G + h) *
// PARTS + part. `exchange` holds [WARPS][W] floats, `warp_stats` [WARPS][G][2]
// floats of shared memory.
template <int WARPS, int PARTS>
__device__ __forceinline__ void publish(const State &state, float *exchange, float *warp_stats,
                                        float *partials, float *statistics, u64 row, int kv,
                                        int part, int warp, int lane) {
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            warp_stats[(warp * G + h) * 2 + 0] = state.maximum[h];
            warp_stats[(warp * G + h) * 2 + 1] = state.denominator[h];
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h) {
        __syncthreads();
        element::f32_span_store(exchange + warp * W + lane * DPL, state.accumulator[h]);
        __syncthreads();
        const u64 at = (row * KV * G + kv * G + h) * PARTS + part;
        for (int column = threadIdx.x; column < W; column += WARPS * 32) {
            float denominator, accumulator;
            const float maximum = merge(warp_stats + h * 2, G * 2, exchange + column, W, WARPS,
                                        denominator, accumulator);
            partials[at * W + column] = accumulator;
            if (column == 0) {
                statistics[at * 2 + 0] = maximum;
                statistics[at * 2 + 1] = denominator;
            }
        }
    }
}

// The decode merge of one (query head, row) column: the row's PARTS
// partitions in partition order, then the sigmoid gate.
template <int PARTS>
__device__ __forceinline__ void decode_gate(const Inputs &in, const float *partials,
                                            const float *statistics, u8 *gated, int query_head,
                                            u64 row, int column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 first = (row * KV * G + query_head) * PARTS;
    float denominator, accumulator;
    merge(statistics + first * 2, 2, partials + first * W + column, W, PARTS, denominator,
          accumulator);
    const float gate = element::at<Act>(in.query_gate, ATTENTION_GATE_AT(row, query_head) + column);
    const float attended = accumulator / fmaxf(denominator, 1e-30f) / (1.0f + expf(-gate));
    element::put<Act>(gated,
                      row * SEISMIC_RESULT_0_STRIDE_0 + query_head * SEISMIC_RESULT_0_STRIDE_1 +
                          column * SEISMIC_RESULT_0_STRIDE_2,
                      attended);
}

// ---------------------------------------------------------------------------
// Per-head preparation and append.

// norm_rotary of one W-vector held by a warp, lane `lane` owning dimensions
// [lane * DPL, (lane + 1) * DPL): RMS over W, times the norm weight, then the
// partial rotation of the first 2P dimensions by the row's M-RoPE angles.
// `exchange` is W floats of shared memory private to the warp.
__device__ __forceinline__ void norm_rotary(float (&x)[DPL], const float *weight,
                                            u64 weight_stride, const Inputs &in, u64 row,
                                            float *exchange, int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    float squares = 0.0f;
#pragma unroll
    for (int d = 0; d < DPL; ++d) squares = __fmaf_rn(x[d], x[d], squares);
    squares = seismic_warp_sum_f32(squares);
    const float inverse = rsqrtf(squares / static_cast<float>(W) + in.epsilon);
    const int first = lane * DPL;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        x[d] = x[d] * inverse * weight[static_cast<u64>(first + d) * weight_stride];
    }
    if (first < 2 * P) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) exchange[first + d] = x[d];
    }
    __syncwarp();
    if (first < 2 * P) {
#pragma unroll
        for (int d = 0; d < DPL; ++d) {
            const int i = first + d;
            if (i < 2 * P) {
                const int pair = i % P;
                const int component =
                    in.components[static_cast<u64>(pair) * SEISMIC_ROTARY_COMPONENTS_STRIDE_0];
                const int coordinate =
                    in.coordinates[row * SEISMIC_COORDINATES_STRIDE_0 +
                                   static_cast<u64>(component) * SEISMIC_COORDINATES_STRIDE_1];
                const float angle =
                    static_cast<float>(coordinate) *
                    in.frequencies[static_cast<u64>(pair) * SEISMIC_ROTARY_FREQUENCIES_STRIDE_0];
                float sine, cosine;
                sincosf(angle, &sine, &cosine);
                x[d] = i < P ? x[d] * cosine - exchange[i + P] * sine
                             : x[d] * cosine + exchange[i - P] * sine;
            }
        }
    }
    __syncwarp();
}

// The prepared key of batch row `row`, kv head `kv_head`, rounded to the
// activation element exactly as history stores it.
__device__ __forceinline__ void prepared_key(const Inputs &in, u64 row, int kv_head,
                                             float (&k)[DPL], float *exchange, int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    element::span<Act, DPL, true>(in.key, ATTENTION_KEY_AT(row, kv_head) + lane * DPL, k);
    norm_rotary(k, in.key_norm, SEISMIC_KEY_NORM_STRIDE_0, in, row, exchange, lane);
#pragma unroll
    for (int d = 0; d < DPL; ++d) k[d] = Act::round(k[d]);
}

// Row `row`'s raw value for `kv_head`, lane `lane`'s dimensions.
__device__ __forceinline__ void fresh_value(const Inputs &in, u64 row, int kv_head,
                                            float (&v)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    element::span<Act, DPL, true>(in.value, ATTENTION_VALUE_AT(row, kv_head) + lane * DPL, v);
}

// ---------------------------------------------------------------------------
// History planes. Each policy appends a row's prepared key and raw value at
// its destination (one warp, the caller skips rows without one) and locates a
// (history row, kv head) vector in its planes. An entry source builds its
// policy from its own ABI (`HISTORY()`): a header may name only macros every
// including entry generates.

// Dense history: key and value planes [T, KV, W] of the activation element.
struct DenseHistory {
    static constexpr bool CODED = false;
    u8 *key;
    u8 *value;
    u64 key_row, key_head, value_row, value_head;

    // Element offsets of a (history row, kv head) vector.
    __device__ __forceinline__ u64 key_at(int token, int kv_head) const {
        return static_cast<u64>(token) * key_row + static_cast<u64>(kv_head) * key_head;
    }
    __device__ __forceinline__ u64 value_at(int token, int kv_head) const {
        return static_cast<u64>(token) * value_row + static_cast<u64>(kv_head) * value_head;
    }

    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        const u64 k_at = key_at(destination, kv_head) + lane * DPL;
        const u64 v_at = value_at(destination, kv_head) + lane * DPL;
#pragma unroll
        for (int d = 0; d < DPL; ++d) {
            element::put<Act>(key, k_at + d, k[d]);
            element::put<Act>(value, v_at + d, v[d]);
        }
    }
};

// Affine K8/V4 history (the `affine_k8_uniform_v4` codec,
// `qwen_attention_*_k8v4`). A (history row, kv head) vector is a code row of
// W * B / 32 u32 words (code i at bits B * (i % (32 / B)) of word
// i / (32 / B)) plus an F16 (scale, zero) pair; decoded value =
// code * scale + zero. Keys use B = 8, values B = 4.
constexpr int KEY_BITS = 8;
constexpr int VALUE_BITS = 4;

// A lane's share of one code row: its DPL codes are DPL * B bits, whole
// words or a power-of-two part of one word shared with its neighbours.
template <int B> struct LaneCodes {
    static constexpr int bits = DPL * B;
    static constexpr int words = (bits + 31) / 32;
    static constexpr int row_words = W * B / 32;
    static constexpr u32 levels = (1u << B) - 1u;
    static_assert(bits % 32 == 0 || 32 % bits == 0,
                  "a lane's codes are whole words or a power-of-two part of one");

    // This lane's codes of a code row, shifted so its code i sits at bits
    // B * i of word i * B / 32.
    __device__ static __forceinline__ void load(const u32 *row, int lane, u32 (&w)[words]) {
        if constexpr (bits == 64) {
            const uint2 pair = *reinterpret_cast<const uint2 *>(row + lane * 2);
            w[0] = pair.x;
            w[1] = pair.y;
        } else if constexpr (bits >= 32) {
#pragma unroll
            for (int j = 0; j < words; ++j) w[j] = row[lane * words + j];
        } else {
            w[0] = row[lane * bits / 32] >> ((lane * bits) % 32);
        }
    }

    // Code i of this lane's dimensions, as F32 (exact): the code in the
    // mantissa of 2^23, minus 2^23 (no conversion instruction).
    __device__ static __forceinline__ float code(const u32 (&w)[words], int i) {
        return __uint_as_float(0x4B000000u | ((w[i * B / 32] >> ((i * B) % 32)) & levels)) -
               8388608.0f;
    }
};

// Encodes one vector held by a warp (lane `lane` owns dimensions
// [lane * DPL, (lane + 1) * DPL)) as B-bit codes at `row` (its code row) and
// its (scale, zero) pair at `pair`: zero = f16(min), scale =
// f16((max - min) / L), code = min(L, u32(fma(x - zero, 1 / scale, 0.5))),
// 0 when scale is 0.
template <int B>
__device__ __forceinline__ void encode(const float (&x)[DPL], u32 *row, u32 *pair, int lane) {
    typedef LaneCodes<B> Codes;
    float low = x[0];
    float high = x[0];
#pragma unroll
    for (int d = 1; d < DPL; ++d) {
        low = fminf(low, x[d]);
        high = fmaxf(high, x[d]);
    }
    low = -seismic_warp_max_f32(-low);
    high = seismic_warp_max_f32(high);
    const u16 zero_bits = seismic_f32_to_f16(low);
    const u16 scale_bits = seismic_f32_to_f16((high - low) / static_cast<float>(Codes::levels));
    const float zero = seismic_f16_to_f32(zero_bits);
    const float scale = seismic_f16_to_f32(scale_bits);
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    u32 w[Codes::words];
#pragma unroll
    for (int j = 0; j < Codes::words; ++j) w[j] = 0u;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        const float t = __fmaf_rn(x[d] - zero, inverse, 0.5f);
        const u32 c = min(static_cast<u32>(fmaxf(t, 0.0f)), Codes::levels);
        w[d * B / 32] |= c << ((d * B) % 32);
    }
    if constexpr (Codes::bits >= 32) {
#pragma unroll
        for (int j = 0; j < Codes::words; ++j) row[lane * Codes::words + j] = w[j];
    } else {
        // 32 / bits neighbouring lanes share a word: join their parts.
        constexpr int sharing = 32 / Codes::bits;
        u32 joined = w[0] << ((lane % sharing) * Codes::bits);
#pragma unroll
        for (int offset = 1; offset < sharing; offset *= 2)
            joined |= seismic_shfl_xor_u32(joined, offset);
        if (lane % sharing == 0) row[lane / sharing] = joined;
    }
    if (lane == 0) *pair = static_cast<u32>(scale_bits) | (static_cast<u32>(zero_bits) << 16);
}

// The (scale, zero) pair of a vector, as F32.
__device__ __forceinline__ float2 coefficients(const u32 *pair) {
    return seismic_unpack_f16x2(*pair);
}

// Affine history: code planes [T, KV, W * B / 32] u32 and (scale, zero)
// planes [T, KV, 2] f16 per vector kind. Every plane is canonical: a vector's
// code row and its pair are contiguous.
struct AffineHistory {
    static constexpr bool CODED = true;
    u32 *key_codes;
    u32 *key_pairs;
    u32 *value_codes;
    u32 *value_pairs;
    u64 key_codes_row, key_codes_head, key_pairs_row, key_pairs_head;
    u64 value_codes_row, value_codes_head, value_pairs_row, value_pairs_head;

    // A vector's code row and (scale, zero) pair (as one u32).
    __device__ __forceinline__ u32 *key_row(int token, int kv_head) const {
        return key_codes + static_cast<u64>(token) * key_codes_row +
               static_cast<u64>(kv_head) * key_codes_head;
    }
    __device__ __forceinline__ u32 *key_pair(int token, int kv_head) const {
        return key_pairs + (static_cast<u64>(token) * key_pairs_row +
                            static_cast<u64>(kv_head) * key_pairs_head) / 2;
    }
    __device__ __forceinline__ u32 *value_row(int token, int kv_head) const {
        return value_codes + static_cast<u64>(token) * value_codes_row +
               static_cast<u64>(kv_head) * value_codes_head;
    }
    __device__ __forceinline__ u32 *value_pair(int token, int kv_head) const {
        return value_pairs + (static_cast<u64>(token) * value_pairs_row +
                              static_cast<u64>(kv_head) * value_pairs_head) / 2;
    }

    // The key is the prepared key rounded to the activation element (as `k`
    // holds it), the value the projected value.
    __device__ __forceinline__ void append(int destination, int kv_head, const float (&k)[DPL],
                                           const float (&v)[DPL], int lane) const {
        encode<KEY_BITS>(k, key_row(destination, kv_head), key_pair(destination, kv_head), lane);
        encode<VALUE_BITS>(v, value_row(destination, kv_head), value_pair(destination, kv_head),
                           lane);
    }
};

// Append row `row`'s prepared key `k` and raw value for `kv_head` at its
// destination through `history`, when it has one. One warp.
template <class History>
__device__ __forceinline__ void append(const Inputs &in, const History &history, u64 row,
                                       int kv_head, const float (&k)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int destination = ATTENTION_DESTINATION(in, row);
    if (destination < 0) return;
    float v[DPL];
    fresh_value(in, row, kv_head, v, lane);
    history.append(destination, kv_head, k, v, lane);
}

}  // namespace attention
