// Decode forms a joined input inside each output group. Prefill forms each
// row/stream once, then uses the shared tiled projection to reuse weights.
// A streamed table's slots are rows16 rows (<seismic/rows16.cuh>), a table
// held whole is mma16; both give every value the same bits.
#if defined(SEISMIC_TABLE_KIND_PACKED) && defined(SEISMIC_TABLE_LAYOUT_ROWS16)
#define ROWS16_T SEISMIC_TABLE
#else
#define KERNEL_W0 SEISMIC_TABLE
#endif
#define KERNEL_W1 SEISMIC_COMBINE
#include "lib/projection/projection.cuh"
#include "lib/core/gathered.cuh"
#include <seismic/rows16.cuh>

#if defined(ROWS16_T)
typedef rows16::T DraftTable;
#define DRAFT_TABLE_AT(pointer) ROWS16_T_AT(pointer)
#else
typedef packets::W0 DraftTable;
#define DRAFT_TABLE_AT(pointer) KERNEL_W0_AT(pointer)
#endif

namespace {

using element::Act;
using element::u32;
using element::u64;

constexpr unsigned DRAFT_WARPS = 8;
constexpr unsigned DRAFT_ROWS_PER_WARP = 4;

} // namespace

struct DraftInput {
    DraftTable table;
    const unsigned char *table_bytes, *conditioning, *embedding_norm, *hidden_norm;
    u64 table_row_stride, table_column_stride;
    u64 conditioning_row_stride, conditioning_column_stride;
    u64 embedding_norm_stride, hidden_norm_stride;
};

#define DRAFT_INPUT DraftInput{DRAFT_TABLE_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE)), \
    SEISMIC_PTR(SEISMIC_BUFFER_TABLE), SEISMIC_PTR(SEISMIC_BUFFER_CONDITIONING), \
    SEISMIC_PTR(SEISMIC_BUFFER_EMBEDDING_NORM), SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN_NORM), \
    SEISMIC_TABLE_STRIDE_0, SEISMIC_TABLE_STRIDE_1, SEISMIC_CONDITIONING_STRIDE_0, \
    SEISMIC_CONDITIONING_STRIDE_1, SEISMIC_EMBEDDING_NORM_STRIDE_0, SEISMIC_HIDDEN_NORM_STRIDE_0}

__device__ __forceinline__ void draft_join(const DraftInput &input, u64 token, u64 row, u64 stream,
                                          u64 D, float epsilon, float *joined, float *partials) {
    using Conditioning = ELEMENT_OF(SEISMIC_CONDITIONING);
    using EmbeddingNorm = ELEMENT_OF(SEISMIC_EMBEDDING_NORM);
    using HiddenNorm = ELEMENT_OF(SEISMIC_HIDDEN_NORM);
    // The embedding row rounded to A, and its sum of squares.
    float embedding_squares = 0.0f;
#if defined(SEISMIC_TABLE_KIND_PACKED)
    const auto table = input.table;
    for (u64 item = threadIdx.x; item < D / 64 * 4; item += blockDim.x) {
        const u64 kb = item / 4;
        const u32 t = (u32)(item % 4);
        const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
        float values[16];
#if defined(ROWS16_T)
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                values[4 * s + j] = table.value(token, kb * 64 + 16 * s + offsets[j]);
#else
        packets::row_values16(table, token, kb, t, values);
#endif
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
            input.table_bytes, token * input.table_row_stride + column * input.table_column_stride));
        joined[column] = value;
        embedding_squares = seismic_fma_rn(value, value, embedding_squares);
    }
#endif
    // The conditioning stream (activation or raw F32) and its sum of squares.
    float hidden_squares = 0.0f;
    for (u64 column = threadIdx.x; column < D; column += blockDim.x) {
        const float value = element::at<Conditioning>(input.conditioning,
                                                      row * input.conditioning_row_stride
                                                          + (stream * D + column) * input.conditioning_column_stride);
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
            * element::at<EmbeddingNorm>(input.embedding_norm,
                                         column * input.embedding_norm_stride));
        joined[D + column] = Act::round(
            joined[D + column] * hidden_inverse
            * element::at<HiddenNorm>(input.hidden_norm, (stream * D + column) * input.hidden_norm_stride));
    }
    __syncthreads();

}

extern "C" __global__ void draft_rows(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ float joined[];
    __shared__ float partials[DRAFT_WARPS];
    const u64 D = SEISMIC_DIM_D;
    const u64 row = blockIdx.y / SEISMIC_DIM_S;
    const u64 stream = blockIdx.y % SEISMIC_DIM_S;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const u64 token = gathered::row(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY)),
                                    SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                    tokens[row * SEISMIC_TOKENS_STRIDE_0]);

    draft_join(DRAFT_INPUT, token, row, stream, D, epsilon, joined, partials);
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
            result[row * SEISMIC_RESULT_0_STRIDE_0 + (stream * D + output) * SEISMIC_RESULT_0_STRIDE_1] = sum;
    }
}

extern "C" __global__ void draft_rows_join(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ float joined[];
    __shared__ float partials[DRAFT_WARPS];
    const u64 D = SEISMIC_DIM_D;
    const u64 row = blockIdx.y / SEISMIC_DIM_S;
    const u64 stream = blockIdx.y % SEISMIC_DIM_S;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const u64 token = gathered::row(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY)),
                                    SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                    tokens[row * SEISMIC_TOKENS_STRIDE_0]);

    draft_join(DRAFT_INPUT, token, row, stream, D, epsilon, joined, partials);
    auto *staged = SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED);
    for (u64 column = threadIdx.x; column < 2 * D; column += blockDim.x)
        element::put<Act>(staged, (row * SEISMIC_DIM_S + stream) * 2 * D + column, joined[column]);
}

template <class Shape>
__device__ __forceinline__ void draft_gemm(const unsigned char *staged, unsigned char *result,
                                          const packets::W1 &combine, u64 rows, u64 width) {
    extern __shared__ uint4 dynamic_shared[];
    const projection::Store<element::F32> out{result, width, 0};
    projection::gemm_run<Shape, false>(reinterpret_cast<projection::u8 *>(dynamic_shared),
        staged, 2 * width, nullptr, rows, 2 * width, projection::gemm_column<Shape>(), width,
        combine, projection::NoWeight{}, out);
}

extern "C" __global__ void draft_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    draft_gemm<projection::SmallGemm>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE)),
        SEISMIC_DIM_M * SEISMIC_DIM_S, SEISMIC_DIM_D);
}

extern "C" __global__ void draft_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    draft_gemm<projection::LargeGemm<2>>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED),
        SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER), KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE)),
        SEISMIC_DIM_M * SEISMIC_DIM_S, SEISMIC_DIM_D);
}
