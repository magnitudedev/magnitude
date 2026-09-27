// Shared device code of the CUDA attention entries (`attention_decode`,
// `attention_prefill` and their affine K8/V4 history forms
// `attention_{decode,prefill}_k8v4`; contract in attention.seismic): the
// per-head q/k preparation (`norm_rotary`: optional RMS norm, amplitude-scaled
// rotary table) computed by one warp whose lanes own contiguous dimensions, a
// row's span walk, partition bounds, the online-softmax absorb of N keys, the
// fixed-order merge of partial softmax states, the decode publication and
// gated merge, the history policies (dense planes; the affine codec: encode on
// append, per-lane code access), and the entries' tensor addressing. Each
// kernel keeps its own loop structure and calls these; the prefill bodies are
// in lib/attention/prefill.cuh. Every activation tensor is canonical (unit
// innermost stride); rows and heads are addressed through their ABI strides.

#include "../core/activation.cuh"
#include <seismic/slab.cuh>

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

// The entry's form: queries with I interleaved gate columns, U separate gate
// values per query head, layers with (F = 1) or without fresh rows, optional
// q/k (N) and value (NV) norms.
constexpr int I = static_cast<int>(SEISMIC_DIM_I);
constexpr int U = static_cast<int>(SEISMIC_DIM_U);
constexpr bool FRESH = SEISMIC_DIM_F != 0;
constexpr bool NORM = SEISMIC_DIM_N != 0;
constexpr bool VALUE_NORM = SEISMIC_DIM_NV != 0;
static_assert(I == 0 || I == W, "interleaved gates are one per column");
static_assert(U == 0 || U == 1 || U == W, "separate gates are one per head or per column");
static_assert(I == 0 || U == 0, "a head's gates are interleaved or separate");

struct Inputs {
    // The launch's argument words, read by the ABI stride macros.
    const seismic_words_t *words;
    const u8 *query;
    const u8 *gate;
    const u8 *key;
    const u8 *value;
    const float *query_norm;
    const float *key_norm;
    const float *value_norm;
    const int *components;
    const int *coordinates;
    const int *visible;
    const int *fresh;
    const int *destinations;
    const float *frequencies;
    const float *amplitudes;
    float epsilon;
    float scale;
    bool softplus;
};

// The entry's inputs, and their tensor addressing. Keys and values are [F, M,
// KV * W]: axis 1 is the row.
#define ATTENTION_INPUTS()                                                                      \
    attention::Inputs {                                                                         \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_QUERY), SEISMIC_PTR(SEISMIC_BUFFER_GATE), \
            SEISMIC_PTR(SEISMIC_BUFFER_KEY), SEISMIC_PTR(SEISMIC_BUFFER_VALUE),                 \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)),              \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_VALUE_NORM)),            \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),       \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)),             \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH)),                   \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),   \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_AMPLITUDES)),    \
            element::word_f32(SEISMIC_PARAM_EPSILON),                                           \
            element::word_f32(SEISMIC_PARAM_SCALE),                                             \
            SEISMIC_PARAM_GATE_FUNCTION != 0                                                    \
    }
#define ATTENTION_QUERY_AT(row, query_head)                                              \
    (static_cast<attention::u64>(row) * SEISMIC_QUERY_STRIDE_0 +                         \
     static_cast<attention::u64>(query_head) * SEISMIC_QUERY_STRIDE_1)
#define ATTENTION_GATE_AT(row, query_head)                                               \
    (static_cast<attention::u64>(row) * SEISMIC_GATE_STRIDE_0 +                          \
     static_cast<attention::u64>(query_head) * SEISMIC_GATE_STRIDE_1)
#define ATTENTION_KEY_AT(row, kv_head)                                                   \
    (static_cast<attention::u64>(row) * SEISMIC_KEY_STRIDE_1 +                           \
     static_cast<attention::u64>(kv_head) * attention::W)
#define ATTENTION_VALUE_AT(row, kv_head)                                                 \
    (static_cast<attention::u64>(row) * SEISMIC_VALUE_STRIDE_1 +                         \
     static_cast<attention::u64>(kv_head) * attention::W)

// Control offsets, named alike in every attention entry's ABI.
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
    if (!FRESH) return Span{0, 0};
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

// The online-softmax state of H query heads over one warp's keys, in the exp2
// domain; lane `lane` owns dimensions [lane * DPL, (lane + 1) * DPL). A warp
// holds all G heads of its kv head (`State`), or one slice of them when the
// family's decode splits the group (so wide heads stay in registers).
template <int H>
struct Heads {
    float maximum[H];
    float denominator[H];
    float accumulator[H][DPL];
};
typedef Heads<G> State;

template <int H>
__device__ __forceinline__ void clear(Heads<H> &state) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
        state.maximum[h] = -__int_as_float(0x7f800000);
        state.denominator[h] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) state.accumulator[h][d] = 0.0f;
    }
}

// The scores (exp2 domain when `q` is scaled into it) of N keys for H query
// heads: warp-reduced dot products.
template <int N, int H>
__device__ __forceinline__ void scores(const float (&q)[H][DPL], const float (&k)[N][DPL],
                                       float (&score)[N][H]) {
#pragma unroll
    for (int t = 0; t < N; ++t) {
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < DPL; ++d) dot = __fmaf_rn(q[h][d], k[t][d], dot);
            score[t][h] = seismic_warp_sum_f32(dot);
        }
    }
}

// Absorb `count` (1..N) keys with scores `score` (exp2 domain) and values `v`.
template <int N, int H>
__device__ __forceinline__ void absorb(Heads<H> &state, const float (&score)[N][H],
                                       const float (&v)[N][DPL], int count) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
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

// Publishes one decode partition: the warp states merge in key-group order
// into one partial per (row, query head, partition) `part`: the unnormalized
// output at partials[slot * W] and (maximum, denominator) at
// statistics[slot * 2], slot = (row * KV * G + kv * G + head) * PARTS + part.
// Warp w holds the H = G / SLICES heads of slice w % SLICES over key group
// w / SLICES. `exchange` holds [WARPS][W] floats, `warp_stats` [WARPS][H][2]
// floats of shared memory.
template <int WARPS, int PARTS, int SLICES = 1, int H>
__device__ __forceinline__ void publish(const Heads<H> &state, float *exchange, float *warp_stats,
                                        float *partials, float *statistics, u64 row, int kv,
                                        int part, int warp, int lane) {
    static_assert(H * SLICES == G && WARPS % SLICES == 0, "slices split the group and the warps");
    constexpr int GROUPS = WARPS / SLICES;
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < H; ++h) {
            warp_stats[(warp * H + h) * 2 + 0] = state.maximum[h];
            warp_stats[(warp * H + h) * 2 + 1] = state.denominator[h];
        }
    }
#pragma unroll
    for (int h = 0; h < H; ++h) {
        __syncthreads();
        element::f32_span_store(exchange + warp * W + lane * DPL, state.accumulator[h]);
        __syncthreads();
        for (int item = threadIdx.x; item < SLICES * W; item += WARPS * 32) {
            const int slice = item / W;
            const int column = item % W;
            const u64 at = (row * KV * G + kv * G + slice * H + h) * PARTS + part;
            float denominator, accumulator;
            const float maximum = merge(warp_stats + (slice * H + h) * 2, SLICES * H * 2,
                                        exchange + slice * W + column, SLICES * W, GROUPS,
                                        denominator, accumulator);
            partials[at * W + column] = accumulator;
            if (column == 0) {
                statistics[at * 2 + 0] = maximum;
                statistics[at * 2 + 1] = denominator;
            }
        }
    }
}

// One attended column of (row, query head) times its output gate.
__device__ __forceinline__ float gated(const Inputs &in, u64 row, int query_head, int column,
                                       float attended) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    // Gate value column % count, interleaved after the queries or separate;
    // sigmoid or softplus.
    float gate;
    if constexpr (I > 0)
        gate = element::at<Act>(in.query, ATTENTION_QUERY_AT(row, query_head) + W + column);
    else if constexpr (U > 0)
        gate = element::at<Act>(in.gate, ATTENTION_GATE_AT(row, query_head) + column % (U > 0 ? U : 1));
    else
        return attended;
    return in.softplus ? attended * (fmaxf(gate, 0.0f) + logf(1.0f + expf(-fabsf(gate))))
                       : attended / (1.0f + expf(-gate));
}

// The decode merge of one (query head, row) column: the row's PARTS
// partitions in partition order, then the output gate.
template <int PARTS>
__device__ __forceinline__ void decode_gate(const Inputs &in, const float *partials,
                                            const float *statistics, u8 *result, int query_head,
                                            u64 row, int column) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const u64 first = (row * KV * G + query_head) * PARTS;
    float denominator, accumulator;
    merge(statistics + first * 2, 2, partials + first * W + column, W, PARTS, denominator,
          accumulator);
    const float attended = gated(in, row, query_head, column, accumulator / fmaxf(denominator, 1e-30f));
    element::put<Act>(result,
                      row * SEISMIC_RESULT_0_STRIDE_0 + query_head * SEISMIC_RESULT_0_STRIDE_1 +
                          column * SEISMIC_RESULT_0_STRIDE_2,
                      attended);
}

// ---------------------------------------------------------------------------
// Per-head preparation and append.

// The RMS norm of one W-vector held by a warp (lane `lane` owning dimensions
// [lane * DPL, (lane + 1) * DPL)) times the norm weight, when NORM; else the
// vector as is.
template <bool NORM>
__device__ __forceinline__ void head_norm(float (&x)[DPL], const float *weight, u64 weight_stride,
                                          float epsilon, int lane) {
    if constexpr (NORM) {
        float squares = 0.0f;
#pragma unroll
        for (int d = 0; d < DPL; ++d) squares = __fmaf_rn(x[d], x[d], squares);
        squares = seismic_warp_sum_f32(squares);
        const float inverse = rsqrtf(squares / static_cast<float>(W) + epsilon);
        const int first = lane * DPL;
#pragma unroll
        for (int d = 0; d < DPL; ++d) {
            x[d] = x[d] * inverse * weight[static_cast<u64>(first + d) * weight_stride];
        }
    }
}

// norm_rotary of one W-vector held by a warp: `head_norm`, then the rotation
// of the first 2P dimensions by the row's rotary table (pair p by coordinate
// axis components[p] at frequencies[p], its cosine and sine scaled by
// amplitudes[p]). `exchange` is W floats of shared memory private to the warp.
template <bool NORM>
__device__ __forceinline__ void norm_rotary(float (&x)[DPL], const float *weight,
                                            u64 weight_stride, const Inputs &in, u64 row,
                                            float *exchange, int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    head_norm<NORM>(x, weight, weight_stride, in.epsilon, lane);
    const int first = lane * DPL;
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
                const int pair = i % (P > 0 ? P : 1);
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
                const float amplitude =
                    in.amplitudes[static_cast<u64>(pair) * SEISMIC_ROTARY_AMPLITUDES_STRIDE_0];
                cosine *= amplitude;
                sine *= amplitude;
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
    norm_rotary<NORM>(k, in.key_norm, SEISMIC_KEY_NORM_STRIDE_1, in, row, exchange, lane);
#pragma unroll
    for (int d = 0; d < DPL; ++d) k[d] = Act::round(k[d]);
}

// The prepared query of batch row `row`, query head `query_head`, rounded to
// the activation element as the contract publishes it.
__device__ __forceinline__ void prepared_query(const Inputs &in, u64 row, int query_head,
                                               float (&q)[DPL], float *exchange, int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    element::span<Act, DPL, true>(in.query, ATTENTION_QUERY_AT(row, query_head) + lane * DPL, q);
    norm_rotary<NORM>(q, in.query_norm, SEISMIC_QUERY_NORM_STRIDE_1, in, row, exchange, lane);
#pragma unroll
    for (int d = 0; d < DPL; ++d) q[d] = Act::round(q[d]);
}

// Row `row`'s value for `kv_head`, lane `lane`'s dimensions: the raw value, or
// with a value norm the normalized value rounded to the activation.
__device__ __forceinline__ void fresh_value(const Inputs &in, u64 row, int kv_head,
                                            float (&v)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    element::span<Act, DPL, true>(in.value, ATTENTION_VALUE_AT(row, kv_head) + lane * DPL, v);
    if constexpr (VALUE_NORM) {
        head_norm<true>(v, in.value_norm, SEISMIC_VALUE_NORM_STRIDE_1, in.epsilon, lane);
#pragma unroll
        for (int d = 0; d < DPL; ++d) v[d] = Act::round(v[d]);
    }
}

#include "history.cuh"

}  // namespace attention
