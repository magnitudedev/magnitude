// qwen_draft_rows: the draft head's input row; the CUDA form of
// `metal/qwen_draft_rows.metal`. Block (x, row) owns outputs 32x..32x+31 of
// one row: it decodes the successor token's embedding row (mma16 lane chunks
// or a dense table) and reads the conditioning row, RMS-normalizes both into
// the joined [2D] input rounded to A (F32 in shared memory), then each warp
// reduces four combine rows over the joined input.
#if defined(SEISMIC_TABLE_KIND_PACKED)
#define SJ_W0 SEISMIC_TABLE
#endif
#if defined(SEISMIC_COMBINE_KIND_PACKED)
#define SJ_W1 SEISMIC_COMBINE
#endif
#include "common/packets.cuh"

namespace {

constexpr unsigned DRAFT_WARPS = 8;
constexpr unsigned DRAFT_ROWS_PER_WARP = 4;

// Sum of one value per thread in a fixed order (warp sums, then the warps in
// index order); every thread receives the total.
__device__ __forceinline__ float draft_sum(float value, float *partials) {
    value = seismic_warp_sum_f32(value);
    if (threadIdx.x % 32 == 0)
        partials[threadIdx.x / 32] = value;
    __syncthreads();
    float total = 0.0f;
    for (unsigned warp = 0; warp < DRAFT_WARPS; ++warp)
        total += partials[warp];
    __syncthreads();
    return total;
}

} // namespace

extern "C" __global__ void qwen_draft_rows(SEISMIC_KERNEL_PARAMS) {
    using Conditioning = sj::Dense<SJ_DENSE_KIND(SEISMIC_CONDITIONING)>;
    using EmbeddingNorm = sj::Dense<SJ_DENSE_KIND(SEISMIC_EMBEDDING_NORM)>;
    using HiddenNorm = sj::Dense<SJ_DENSE_KIND(SEISMIC_HIDDEN_NORM)>;
    extern __shared__ float joined[];
    __shared__ float partials[DRAFT_WARPS];
    const sj::u64 D = SEISMIC_DIM_D;
    const sj::u64 row = blockIdx.y;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const sj::u64 token = (sj::u64)tokens[row * SEISMIC_TOKENS_STRIDE_0];

    // The embedding row rounded to A, and its sum of squares.
    float embedding_squares = 0.0f;
#if defined(SEISMIC_TABLE_KIND_PACKED)
    const auto table = SJ_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE));
    for (sj::u64 item = threadIdx.x; item < D / 64 * 4; item += blockDim.x) {
        const sj::u64 kb = item / 4;
        const sj::u32 t = (sj::u32)(item % 4);
        float values[16];
        sj::row_values16(table, token, kb, t, values);
        const sj::u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float value = sj::Act::round(values[4 * s + j]);
                joined[kb * 64 + 16 * s + offsets[j]] = value;
                embedding_squares = seismic_fma_rn(value, value, embedding_squares);
            }
    }
#else
    using Table = sj::Dense<SJ_DENSE_KIND(SEISMIC_TABLE)>;
    for (sj::u64 column = threadIdx.x; column < D; column += blockDim.x) {
        const float value = sj::Act::round(Table::load(
            SEISMIC_PTR(SEISMIC_BUFFER_TABLE), token * SEISMIC_TABLE_STRIDE_0 + column * SEISMIC_TABLE_STRIDE_1));
        joined[column] = value;
        embedding_squares = seismic_fma_rn(value, value, embedding_squares);
    }
#endif
    // The conditioning row (already A) and its sum of squares.
    float hidden_squares = 0.0f;
    for (sj::u64 column = threadIdx.x; column < D; column += blockDim.x) {
        const float value = Conditioning::load(SEISMIC_PTR(SEISMIC_BUFFER_CONDITIONING),
                                               row * SEISMIC_CONDITIONING_STRIDE_0
                                                   + column * SEISMIC_CONDITIONING_STRIDE_1);
        joined[D + column] = value;
        hidden_squares = seismic_fma_rn(value, value, hidden_squares);
    }
    const float embedding_inverse = rsqrtf(draft_sum(embedding_squares, partials) / (float)D + epsilon);
    const float hidden_inverse = rsqrtf(draft_sum(hidden_squares, partials) / (float)D + epsilon);
    // Both sums' barriers ordered every joined write before these updates.
    for (sj::u64 column = threadIdx.x; column < D; column += blockDim.x) {
        joined[column] = sj::Act::round(
            joined[column] * embedding_inverse
            * EmbeddingNorm::load(SEISMIC_PTR(SEISMIC_BUFFER_EMBEDDING_NORM), column * SEISMIC_EMBEDDING_NORM_STRIDE_0));
        joined[D + column] = sj::Act::round(
            joined[D + column] * hidden_inverse
            * HiddenNorm::load(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN_NORM), column * SEISMIC_HIDDEN_NORM_STRIDE_0));
    }
    __syncthreads();

    // Four combine rows per warp over the 2D joined inputs.
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
#if defined(SEISMIC_COMBINE_KIND_PACKED)
    const auto combine = SJ_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE));
#else
    using Combine = sj::Dense<SJ_DENSE_KIND(SEISMIC_COMBINE)>;
#endif
    for (unsigned r = 0; r < DRAFT_ROWS_PER_WARP; ++r) {
        const sj::u64 output = (sj::u64)blockIdx.x * DRAFT_WARPS * DRAFT_ROWS_PER_WARP
                               + warp * DRAFT_ROWS_PER_WARP + r;
        if (output >= D)
            break;
        float sum = 0.0f;
#if defined(SEISMIC_COMBINE_KIND_PACKED)
        for (sj::u64 item = lane; item < 2 * D / 64 * 4; item += 32) {
            const sj::u64 kb = item / 4;
            const sj::u32 t = (sj::u32)(item % 4);
            float values[16];
            sj::row_values16(combine, output, kb, t, values);
            const sj::u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sum = seismic_fma_rn(values[4 * s + j], joined[kb * 64 + 16 * s + offsets[j]], sum);
        }
#else
        for (sj::u64 column = lane; column < 2 * D; column += 32)
            sum = seismic_fma_rn(Combine::load(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE),
                                               output * SEISMIC_COMBINE_STRIDE_0 + column * SEISMIC_COMBINE_STRIDE_1),
                                 joined[column], sum);
#endif
        sum = seismic_warp_sum_f32(sum);
        if (lane == 0)
            result[row * SEISMIC_RESULT_0_STRIDE_0 + output * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}
