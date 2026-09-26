// Shared device code of the CUDA gated-attention entries (`gated_attention_
// decode`, `gated_attention_prefill` and their affine K8/V4 history forms
// `gated_attention_{decode,prefill}_k8v4`): the per-head q/k preparation
// (`norm_rotary`: RMS norm, weight, partial M-RoPE) computed by one warp whose
// lanes own contiguous dimensions, a row's span walk, partition bounds, the
// online-softmax absorb of N keys, the fixed-order merge of partial softmax
// states, the decode publication and gated merge, the history policies
// (dense planes; the affine codec: encode on append, per-lane code access),
// and the entry's tensor addressing. Each kernel keeps its own loop structure
// and calls these; the prefill bodies are in lib/attention/prefill.cuh.
// Every activation tensor is canonical (unit innermost stride); rows and
// heads are addressed through their ABI strides.

#include "../core/activation.cuh"

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

#include "history.cuh"

}  // namespace attention
