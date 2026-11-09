// indexer_score on CUDA (contract and portable body in indexer.seismic), as
// Metal's (`metal/indexer_score.metal`):
//
// A block of 256 threads per (256 scores, row). A tile holding some of the
// row's listed rows first prepares the row's H query heads into shared
// memory, a warp per head (8 heads at a time): widened to F32,
// RMS-normalized with the query norm when NQ = 1, rotated at the row's
// coordinates (`rotary::layout`, exchanging through the head's own shared
// row) and rounded with the indexer's trained rounding, as `indexer_keys`
// prepares a key. Then thread t scores ordinal j = 256 x + t of the row's
// list: the j-th key-plane row of its spans (each thread walks the spans to
// it), streaming the key's columns once in fours into one accumulator per
// head, so each head's dot is the ascending FMA chain of the portable body;
// s = scale * sum_h relu(dot_h) * w_h, heads ascending. Every score past the
// list is -inf.
//
// Dynamic extents, strides and parameters read `seismic_words`.

#include "lib/core/activation.cuh"
#include "lib/core/head.cuh"
#include "lib/core/rotary.cuh"
#include "lib/core/rounding.cuh"
#include "lib/indexer/indexer.cuh"
#include <seismic/slab.cuh>

namespace score {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef element::Act Act;

constexpr int W = 2 * SEISMIC_DIM_P + SEISMIC_DIM_S;
constexpr int E = W / 32;
constexpr int H = SEISMIC_DIM_H;
static_assert(W % 32 == 0, "indexer head width must be a multiple of 32");

typedef ROTARY_ENTRY_LAYOUT(W) query_layout;

} // namespace score

extern "C" __global__ void indexer_score(SEISMIC_KERNEL_PARAMS) {
    using namespace score;
    extern __shared__ float prepared[];  // [H][W]
    const u64 row = blockIdx.y;
    const u64 n = SEISMIC_DIM_N;
    const u64 r = SEISMIC_DIM_R;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int *row_spans = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SPANS)) + row * SEISMIC_SPANS_STRIDE_0;
    const u64 span_stride = SEISMIC_SPANS_STRIDE_1;
    const u64 bound_stride = SEISMIC_SPANS_STRIDE_2;
    // The row's listed rows: those of its spans, at most N scored.
    u64 listed = 0;
    for (u64 span = 0; span < r; ++span)
        listed += indexer::span_length(row_spans[span * span_stride], row_spans[span * span_stride + bound_stride]);
    listed = listed < n ? listed : n;
    const u64 ordinal = static_cast<u64>(blockIdx.x) * 256 + threadIdx.x;
    float *out = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER)) + row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 out_stride = SEISMIC_RESULT_0_STRIDE_1;
    if (static_cast<u64>(blockIdx.x) * 256 >= listed) {
        if (ordinal < n) out[ordinal * out_stride] = -__int_as_float(0x7f800000);
        return;
    }
    // The row's query heads, prepared in F32.
    const float epsilon = element::word_f32(SEISMIC_PARAM_EPSILON);
    const rounding::Rounding trained =
        rounding::of(SEISMIC_PARAM_BLOCK, SEISMIC_PARAM_ELEMENT, SEISMIC_PARAM_ROUNDING_SCALE, SEISMIC_PARAM_AMAX_FLOOR);
    const rotary::table table = ROTARY_ENTRY_TABLE(
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_COORDINATES)) + row * SEISMIC_COORDINATES_STRIDE_0,
        SEISMIC_COORDINATES_STRIDE_1);
    const element::u8 *query = SEISMIC_PTR(SEISMIC_BUFFER_QUERY);
    for (int head = warp; head < H; head += 8) {
        float x[E];
#pragma unroll
        for (int i = 0; i < E; ++i)
            x[i] = element::at<Act>(query, row * SEISMIC_QUERY_STRIDE_0 + static_cast<u64>(head) * SEISMIC_QUERY_STRIDE_1 +
                                               static_cast<u64>(lane * E + i) * SEISMIC_QUERY_STRIDE_2);
        // A rounded query is rounded from its norm and rotation held in A.
        if constexpr (SEISMIC_DIM_NQ != 0) {
            head::rms_norm(x, reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_QUERY_NORM)),
                           SEISMIC_QUERY_NORM_STRIDE_1, epsilon, lane);
            rounding::hold_lanes<Act>(x, trained);
        }
        // The head's own shared row is its rotation's exchange until it holds
        // the prepared head.
        float *prepared_head = prepared + head * W;
        query_layout::rotate(x, table, prepared_head, lane);
        rounding::hold_lanes<Act>(x, trained);
        rounding::round_lanes<E>(x, trained);
#pragma unroll
        for (int i = 0; i < E; ++i) prepared_head[lane * E + i] = x[i];
    }
    __syncthreads();
    if (ordinal >= n) return;
    if (ordinal >= listed) {
        out[ordinal * out_stride] = -__int_as_float(0x7f800000);
        return;
    }
    // The ordinal's key-plane row (a listed ordinal lies within the spans).
    const int at = indexer::span_row(row_spans, r, span_stride, bound_stride, ordinal);
    const float *k = reinterpret_cast<const float *>(slab::row(reinterpret_cast<const u32 *>(SEISMIC_PTR(SEISMIC_BUFFER_KEYS)),
                                                               static_cast<u64>(at), SEISMIC_PARAM_SLAB_ROWS,
                                                               W * sizeof(float)));
    float dot[H];
#pragma unroll
    for (int head = 0; head < H; ++head) dot[head] = 0.0f;
    for (int column = 0; column < W; column += 4) {
        const float4 kx = *reinterpret_cast<const float4 *>(k + column);
#pragma unroll
        for (int head = 0; head < H; ++head) {
            const float4 q = *reinterpret_cast<const float4 *>(prepared + head * W + column);
            float d = dot[head];
            d = __fmaf_rn(q.x, kx.x, d);
            d = __fmaf_rn(q.y, kx.y, d);
            d = __fmaf_rn(q.z, kx.z, d);
            d = __fmaf_rn(q.w, kx.w, d);
            dot[head] = d;
        }
    }
    const element::u8 *head_weights = SEISMIC_PTR(SEISMIC_BUFFER_HEAD_WEIGHTS);
    float total = 0.0f;
#pragma unroll
    for (int head = 0; head < H; ++head) {
        const float weight =
            SEISMIC_DIM_HW != 0
                ? element::at<Act>(head_weights, row * SEISMIC_HEAD_WEIGHTS_STRIDE_0 +
                                                     static_cast<u64>(head) * SEISMIC_HEAD_WEIGHTS_STRIDE_1)
                : 1.0f;
        total = total + fmaxf(dot[head], 0.0f) * weight;
    }
    out[ordinal * out_stride] = element::word_f32(SEISMIC_PARAM_SCALE) * total;
}
