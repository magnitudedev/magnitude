// qwen_draft_rows: the draft head's input row; the CUDA form of
// `metal/qwen_draft_rows.metal`. Block (x, row) owns outputs 32x..32x+31 of
// one row: it decodes the successor token's embedding row (mma16 lane chunks
// or a dense table) and reads the conditioning row, RMS-normalizes both into
// the joined [2D] input rounded to A (F32 in shared memory), then each warp
// reduces four combine rows over the joined input.
#if defined(SEISMIC_TABLE_KIND_PACKED)
#define KERNEL_W0 SEISMIC_TABLE
#endif
#if defined(SEISMIC_COMBINE_KIND_PACKED)
#define KERNEL_W1 SEISMIC_COMBINE
#endif
#include "common/packets.cuh"
#include "common/reduce.cuh"

namespace {

using element::Act;
using element::u32;
using element::u64;

constexpr unsigned DRAFT_WARPS = 8;
constexpr unsigned DRAFT_ROWS_PER_WARP = 4;

} // namespace

extern "C" __global__ void qwen_draft_rows(SEISMIC_KERNEL_PARAMS) {
    using Conditioning = ELEMENT_OF(SEISMIC_CONDITIONING);
    using EmbeddingNorm = ELEMENT_OF(SEISMIC_EMBEDDING_NORM);
    using HiddenNorm = ELEMENT_OF(SEISMIC_HIDDEN_NORM);
    extern __shared__ float joined[];
    __shared__ float partials[DRAFT_WARPS];
    const u64 D = SEISMIC_DIM_D;
    const u64 row = blockIdx.y;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    // Column 0 of a (token, status) selection row; a failed selection is -1.
    const u64 token = (u64)max(tokens[row * SEISMIC_TOKENS_STRIDE_0], 0);

    // The embedding row rounded to A, and its sum of squares.
    float embedding_squares = 0.0f;
#if defined(SEISMIC_TABLE_KIND_PACKED)
    const auto table = KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE));
    for (u64 item = threadIdx.x; item < D / 64 * 4; item += blockDim.x) {
        const u64 kb = item / 4;
        const u32 t = (u32)(item % 4);
        float values[16];
        packets::row_values16(table, token, kb, t, values);
        const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float value = Act::round(values[4 * s + j]);
                joined[kb * 64 + 16 * s + offsets[j]] = value;
                embedding_squares = seismic_fma_rn(value, value, embedding_squares);
            }
    }
#else
    using Table = ELEMENT_OF(SEISMIC_TABLE);
    for (u64 column = threadIdx.x; column < D; column += blockDim.x) {
        const float value = Act::round(element::at<Table>(
            SEISMIC_PTR(SEISMIC_BUFFER_TABLE), token * SEISMIC_TABLE_STRIDE_0 + column * SEISMIC_TABLE_STRIDE_1));
        joined[column] = value;
        embedding_squares = seismic_fma_rn(value, value, embedding_squares);
    }
#endif
    // The conditioning row (already A) and its sum of squares.
    float hidden_squares = 0.0f;
    for (u64 column = threadIdx.x; column < D; column += blockDim.x) {
        const float value = element::at<Conditioning>(SEISMIC_PTR(SEISMIC_BUFFER_CONDITIONING),
                                                      row * SEISMIC_CONDITIONING_STRIDE_0
                                                          + column * SEISMIC_CONDITIONING_STRIDE_1);
        joined[D + column] = value;
        hidden_squares = seismic_fma_rn(value, value, hidden_squares);
    }
    // Fixed-order block sums (warp sums, then the DRAFT_WARPS warps in index
    // order); their barriers order every joined write before the updates.
    const float embedding_inverse = rsqrtf(reduce::group_sum(embedding_squares, partials) / (float)D + epsilon);
    const float hidden_inverse = rsqrtf(reduce::group_sum(hidden_squares, partials) / (float)D + epsilon);
    for (u64 column = threadIdx.x; column < D; column += blockDim.x) {
        joined[column] = Act::round(
            joined[column] * embedding_inverse
            * element::at<EmbeddingNorm>(SEISMIC_PTR(SEISMIC_BUFFER_EMBEDDING_NORM),
                                         column * SEISMIC_EMBEDDING_NORM_STRIDE_0));
        joined[D + column] = Act::round(
            joined[D + column] * hidden_inverse
            * element::at<HiddenNorm>(SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN_NORM), column * SEISMIC_HIDDEN_NORM_STRIDE_0));
    }
    __syncthreads();

    // Four combine rows per warp over the 2D joined inputs.
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
#if defined(SEISMIC_COMBINE_KIND_PACKED)
    const auto combine = KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE));
#else
    using Combine = ELEMENT_OF(SEISMIC_COMBINE);
#endif
    for (unsigned r = 0; r < DRAFT_ROWS_PER_WARP; ++r) {
        const u64 output = (u64)blockIdx.x * DRAFT_WARPS * DRAFT_ROWS_PER_WARP
                           + warp * DRAFT_ROWS_PER_WARP + r;
        if (output >= D)
            break;
        float sum = 0.0f;
#if defined(SEISMIC_COMBINE_KIND_PACKED)
        for (u64 item = lane; item < 2 * D / 64 * 4; item += 32) {
            const u64 kb = item / 4;
            const u32 t = (u32)(item % 4);
            float values[16];
            packets::row_values16(combine, output, kb, t, values);
            const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
            for (int s = 0; s < 4; ++s)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sum = seismic_fma_rn(values[4 * s + j], joined[kb * 64 + 16 * s + offsets[j]], sum);
        }
#else
        for (u64 column = lane; column < 2 * D; column += 32)
            sum = seismic_fma_rn(element::at<Combine>(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE),
                                                      output * SEISMIC_COMBINE_STRIDE_0 + column * SEISMIC_COMBINE_STRIDE_1),
                                 joined[column], sum);
#endif
        sum = seismic_warp_sum_f32(sum);
        if (lane == 0)
            result[row * SEISMIC_RESULT_0_STRIDE_0 + output * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}
