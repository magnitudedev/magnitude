// Shared device code of the CUDA gated-attention entries: the per-head q/k
// preparation (`norm_rotary`: RMS norm, weight, partial M-RoPE) computed by
// one warp whose lanes own contiguous dimensions, and the entry's tensor
// addressing. Every activation tensor is canonical (unit innermost stride);
// rows and heads are addressed through their ABI strides.

#include "common/mixer.cuh"

namespace attn {

using mx::u8;
using mx::u16;
using mx::u32;
using mx::u64;

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
    u8 *history_key;
    u8 *history_value;
    const float *frequencies;
    float epsilon;
    float scale;
};

#define ATTN_INPUTS()                                                                           \
    attn::Inputs {                                                                              \
        &seismic_words_value, SEISMIC_PTR(SEISMIC_BUFFER_QUERY_GATE), SEISMIC_PTR(SEISMIC_BUFFER_KEY),                \
            SEISMIC_PTR(SEISMIC_BUFFER_VALUE),                                                  \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),            \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_KEY_NORM)),              \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_COMPONENTS)),       \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)),             \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_VISIBLE)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FRESH)),                   \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DESTINATIONS)),            \
            SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_KEY), SEISMIC_PTR(SEISMIC_BUFFER_HISTORY_VALUE), \
            reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ROTARY_FREQUENCIES)),   \
            mx::word_f32(SEISMIC_PARAM_EPSILON),                                                \
            mx::word_f32(SEISMIC_PARAM_SCALE)                                                   \
    }

// Element offsets (all canonical within a row).
#define ATTN_QUERY_AT(row, query_head)                                                   \
    (static_cast<attn::u64>(row) * SEISMIC_QUERY_GATE_STRIDE_0 +                         \
     static_cast<attn::u64>(query_head) * 2 * attn::W)
#define ATTN_GATE_AT(row, query_head) (ATTN_QUERY_AT(row, query_head) + attn::W)
#define ATTN_KEY_AT(row, kv_head)                                                        \
    (static_cast<attn::u64>(row) * SEISMIC_KEY_STRIDE_0 + static_cast<attn::u64>(kv_head) * attn::W)
#define ATTN_VALUE_AT(row, kv_head)                                                      \
    (static_cast<attn::u64>(row) * SEISMIC_VALUE_STRIDE_0 +                              \
     static_cast<attn::u64>(kv_head) * attn::W)
#define ATTN_HISTORY_KEY_AT(token, kv_head)                                              \
    (static_cast<attn::u64>(token) * SEISMIC_HISTORY_KEY_STRIDE_0 +                      \
     static_cast<attn::u64>(kv_head) * SEISMIC_HISTORY_KEY_STRIDE_1)
#define ATTN_HISTORY_VALUE_AT(token, kv_head)                                            \
    (static_cast<attn::u64>(token) * SEISMIC_HISTORY_VALUE_STRIDE_0 +                    \
     static_cast<attn::u64>(kv_head) * SEISMIC_HISTORY_VALUE_STRIDE_1)
#define ATTN_VISIBLE(in, row, span, bound)                                               \
    ((in).visible[static_cast<attn::u64>(row) * SEISMIC_VISIBLE_STRIDE_0 +               \
                  static_cast<attn::u64>(span) * SEISMIC_VISIBLE_STRIDE_1 +              \
                  static_cast<attn::u64>(bound) * SEISMIC_VISIBLE_STRIDE_2])
#define ATTN_FRESH(in, row, bound)                                                       \
    ((in).fresh[static_cast<attn::u64>(row) * SEISMIC_FRESH_STRIDE_0 +                   \
                static_cast<attn::u64>(bound) * SEISMIC_FRESH_STRIDE_1])
#define ATTN_DESTINATION(in, row) ((in).destinations[static_cast<attn::u64>(row) * SEISMIC_DESTINATIONS_STRIDE_0])

// The span of `row` with index `span` (0..R-1 visible history, R the fresh
// batch rows), as [lo, hi); an empty span has hi <= lo.
struct Span {
    int lo;
    int hi;
};
__device__ __forceinline__ Span row_span(const Inputs &in, u64 row, u64 span, u64 spans) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    if (span < spans) {
        return Span{ATTN_VISIBLE(in, row, span, 0), ATTN_VISIBLE(in, row, span, 1)};
    }
    return Span{ATTN_FRESH(in, row, 0), ATTN_FRESH(in, row, 1)};
}

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
    mx::act_span<DPL, true>(in.key, ATTN_KEY_AT(row, kv_head) + lane * DPL, k);
    norm_rotary(k, in.key_norm, SEISMIC_KEY_NORM_STRIDE_0, in, row, exchange, lane);
#pragma unroll
    for (int d = 0; d < DPL; ++d) k[d] = mx::act_round(k[d]);
}

// Append row `row`'s prepared key and raw value for `kv_head` at its
// destination, when it has one. One warp; `k` is the prepared key.
__device__ __forceinline__ void append(const Inputs &in, u64 row, int kv_head,
                                       const float (&k)[DPL], int lane) {
    [[maybe_unused]] const seismic_words_t &seismic_words_value = *in.words;
    const int destination = ATTN_DESTINATION(in, row);
    if (destination < 0) return;
    float v[DPL];
    mx::act_span<DPL, true>(in.value, ATTN_VALUE_AT(row, kv_head) + lane * DPL, v);
    const u64 key_at = ATTN_HISTORY_KEY_AT(destination, kv_head) + lane * DPL;
    const u64 value_at = ATTN_HISTORY_VALUE_AT(destination, kv_head) + lane * DPL;
#pragma unroll
    for (int d = 0; d < DPL; ++d) {
        mx::act_store(in.history_key, key_at + d, k[d]);
        mx::act_store(in.history_value, value_at + d, v[d]);
    }
}

}  // namespace attn
