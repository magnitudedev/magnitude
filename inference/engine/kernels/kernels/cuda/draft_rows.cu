// Decode forms a joined input inside each output group. Prefill forms each
// row/stream once, then uses the shared tiled projection to reuse weights.
// The table row is gathered (<seismic/gather.cuh>: a streamed table's slots
// are rows16 rows, a table held whole is mma16; both give every value the
// same bits).
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_T SEISMIC_TABLE
#define KERNEL_W1 SEISMIC_COMBINE
#include "lib/projection/projection.cuh"
#include "lib/core/gathered.cuh"
#include <seismic/gather.cuh>

namespace {

using activation::Act;
using element::u32;
using element::u64;

constexpr unsigned DRAFT_WARPS = 8;
constexpr unsigned DRAFT_ROWS_PER_WARP = 4;

} // namespace

struct DraftInput {
    packets::T table;
    const unsigned char *conditioning, *embedding_norm, *hidden_norm;
    u64 conditioning_row_stride, conditioning_column_stride;
    u64 embedding_norm_stride, hidden_norm_stride;
};

#define DRAFT_INPUT DraftInput{KERNEL_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE)), \
    SEISMIC_PTR(SEISMIC_BUFFER_CONDITIONING), SEISMIC_PTR(SEISMIC_BUFFER_EMBEDDING_NORM), \
    SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN_NORM), SEISMIC_CONDITIONING_STRIDE_0, \
    SEISMIC_CONDITIONING_STRIDE_1, SEISMIC_EMBEDDING_NORM_STRIDE_0, SEISMIC_HIDDEN_NORM_STRIDE_0}

template <bool WRITE_STAGED>
__device__ __forceinline__ void draft_join(const DraftInput &input, long long held, u64 row, u64 stream,
                                          u64 D, float epsilon, float *joined, float *partials,
                                          element::Out<Act> staged, u64 staged_row_base) {
    using Conditioning = ELEMENT_OF(SEISMIC_CONDITIONING);
    using EmbeddingNorm = ELEMENT_OF(SEISMIC_EMBEDDING_NORM);
    using HiddenNorm = ELEMENT_OF(SEISMIC_HIDDEN_NORM);
    const u64 vec_end = D & ~3ull;
    const bool d_aligned4 = (D & 3ull) == 0ull;
    // The embedding row rounded to A, and its sum of squares: zeros of a
    // row that names no table row.
    float embedding_squares = 0.0f;
    const u64 token = (u64)held;
    if (held < 0) {
        for (u64 col = (u64)threadIdx.x * 4ull; col < vec_end; col += (u64)blockDim.x * 4ull)
            *reinterpret_cast<float4 *>(joined + col) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (u64 col = vec_end + threadIdx.x; col < D; col += blockDim.x)
            joined[col] = 0.0f;
    } else {
        packets::gather_row(input.table, token, 0, D, [&](u64 column, float decoded) {
            const float value = Act::round(decoded);
            joined[column] = value;
            embedding_squares = seismic_fma_rn(value, value, embedding_squares);
        });
    }
    // The conditioning stream (activation or raw F32) cached in shared `joined[D..2D)`
    // so Pass 2 avoids re-reading DRAM while preserving exact F32 precision.
    float hidden_squares = 0.0f;
    const u64 cond_row_base = row * input.conditioning_row_stride;
    const u64 cond_stream_base = cond_row_base + stream * D * input.conditioning_column_stride;
    const bool cond_vec4 =
        input.conditioning_column_stride == 1ull &&
        ((reinterpret_cast<u64>(input.conditioning + cond_stream_base * Conditioning::bytes) &
          (4ull * Conditioning::bytes - 1ull)) == 0ull);
    if (cond_vec4) {
        for (u64 col = (u64)threadIdx.x * 4ull; col < vec_end; col += (u64)blockDim.x * 4ull) {
            float v[4];
            element::span<Conditioning, 4, true>(input.conditioning, cond_stream_base + col, v);
            if (d_aligned4) {
                element::f32_span_store<4>(joined + D + col, v);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    joined[D + col + i] = v[i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                hidden_squares = seismic_fma_rn(v[i], v[i], hidden_squares);
        }
        for (u64 col = vec_end + threadIdx.x; col < D; col += blockDim.x) {
            const float value = element::at<Conditioning>(input.conditioning, cond_stream_base + col);
            joined[D + col] = value;
            hidden_squares = seismic_fma_rn(value, value, hidden_squares);
        }
    } else {
        for (u64 col = threadIdx.x; col < D; col += blockDim.x) {
            const float value = element::at<Conditioning>(
                input.conditioning, cond_row_base + (stream * D + col) * input.conditioning_column_stride);
            joined[D + col] = value;
            hidden_squares = seismic_fma_rn(value, value, hidden_squares);
        }
    }
    // Fixed-order block sums (warp sums, then the DRAFT_WARPS warps in index
    // order); their barriers order every joined write before the updates.
    float squares[2] = {embedding_squares, hidden_squares};
    reduce::group_sum_n<2>(squares, partials);
    const float embedding_inverse = rsqrtf(squares[0] / (float)D + epsilon);
    const float hidden_inverse = rsqrtf(squares[1] / (float)D + epsilon);
    const u64 hn_stream_base = stream * D;
    const bool en_vec4 =
        input.embedding_norm_stride == 1ull &&
        ((reinterpret_cast<u64>(input.embedding_norm) & (4ull * EmbeddingNorm::bytes - 1ull)) == 0ull);
    const bool hn_vec4 =
        input.hidden_norm_stride == 1ull &&
        ((reinterpret_cast<u64>(input.hidden_norm + hn_stream_base * HiddenNorm::bytes) &
          (4ull * HiddenNorm::bytes - 1ull)) == 0ull);
    for (u64 col = (u64)threadIdx.x * 4ull; col < vec_end; col += (u64)blockDim.x * 4ull) {
        float emb[4], hid[4], en[4], hn[4];
        element::f32_span<4>(joined + col, emb);
        if (d_aligned4) {
            element::f32_span<4>(joined + D + col, hid);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                hid[i] = joined[D + col + i];
        }
        if (en_vec4) {
            element::span<EmbeddingNorm, 4, true>(input.embedding_norm, col, en);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                en[i] = element::at<EmbeddingNorm>(input.embedding_norm, (col + i) * input.embedding_norm_stride);
        }
        if (hn_vec4) {
            element::span<HiddenNorm, 4, true>(input.hidden_norm, hn_stream_base + col, hn);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                hn[i] = element::at<HiddenNorm>(input.hidden_norm, (hn_stream_base + col + i) * input.hidden_norm_stride);
        }
        float emb_out[4], hid_out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            emb_out[i] = emb[i] * embedding_inverse * en[i];
            hid_out[i] = hid[i] * hidden_inverse * hn[i];
            if constexpr (!WRITE_STAGED) {
                emb_out[i] = Act::round(emb_out[i]);
                hid_out[i] = Act::round(hid_out[i]);
            }
        }
        if constexpr (WRITE_STAGED) {
            staged.put_span<4>(staged_row_base + col, emb_out);
            staged.put_span<4>(staged_row_base + D + col, hid_out);
        } else {
            element::f32_span_store<4>(joined + col, emb_out);
            if (d_aligned4) {
                element::f32_span_store<4>(joined + D + col, hid_out);
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    joined[D + col + i] = hid_out[i];
            }
        }
    }
    for (u64 col = vec_end + threadIdx.x; col < D; col += blockDim.x) {
        const float emb_val = joined[col] * embedding_inverse
            * element::at<EmbeddingNorm>(input.embedding_norm, col * input.embedding_norm_stride);
        const float hid_val = joined[D + col] * hidden_inverse
            * element::at<HiddenNorm>(input.hidden_norm, (hn_stream_base + col) * input.hidden_norm_stride);
        if constexpr (WRITE_STAGED) {
            staged.put(staged_row_base + col, emb_val);
            staged.put(staged_row_base + D + col, hid_val);
        } else {
            joined[col] = Act::round(emb_val);
            joined[D + col] = Act::round(hid_val);
        }
    }
    if constexpr (!WRITE_STAGED)
        __syncthreads();
}

template <class Combine>
__device__ __forceinline__ void draft_combine_4(const Combine &combine, const float *joined, u64 base_output,
                                                unsigned active, u64 inputs, unsigned lane,
                                                float (&sums)[DRAFT_ROWS_PER_WARP]) {
    if constexpr (Combine::DENSE) {
        for (u64 column = lane; column < inputs; column += 32) {
            const float x = joined[column];
#pragma unroll
            for (unsigned r = 0; r < DRAFT_ROWS_PER_WARP; ++r) {
                if (r < active)
                    sums[r] = seismic_fma_rn(combine.value(base_output + r, column), x, sums[r]);
            }
        }
    } else {
        const u64 items = (inputs / 64) * 4;
        for (u64 item = lane; item < items; item += 32) {
            const u64 kb = item / 4;
            const u32 t = (u32)(item % 4);
            float x[16];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const u64 base_col = kb * 64 + 16 * s + 2 * t;
                const float2 lo = *reinterpret_cast<const float2 *>(joined + base_col);
                const float2 hi = *reinterpret_cast<const float2 *>(joined + base_col + 8);
                x[4 * s + 0] = lo.x;
                x[4 * s + 1] = lo.y;
                x[4 * s + 2] = hi.x;
                x[4 * s + 3] = hi.y;
            }
#pragma unroll
            for (unsigned r = 0; r < DRAFT_ROWS_PER_WARP; ++r) {
                if (r < active) {
                    float w[16];
                    packets::row_values16(combine, base_output + r, kb, t, w);
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        sums[r] = seismic_fma_rn(w[i], x[i], sums[r]);
                }
            }
        }
    }
}

extern "C" __global__ void draft_rows(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ float joined[];
    __shared__ float partials[2 * DRAFT_WARPS];
    const u64 D = SEISMIC_DIM_D;
    const u64 row = blockIdx.y / SEISMIC_DIM_S;
    const u64 stream = blockIdx.y % SEISMIC_DIM_S;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const long long held = gathered::row(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY)),
                                         SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                         tokens[row * SEISMIC_TOKENS_STRIDE_0]);

    draft_join<false>(DRAFT_INPUT, held, row, stream, D, epsilon, joined, partials, element::Out<Act>(nullptr), 0);
    // Four combine rows per warp over the 2D joined inputs: each shared
    // `joined` slice is loaded once into registers and dotted against all 4 rows.
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const u64 base_output = (u64)blockIdx.x * DRAFT_WARPS * DRAFT_ROWS_PER_WARP + warp * DRAFT_ROWS_PER_WARP;
    if (base_output >= D)
        return;
    const unsigned active = (D - base_output < DRAFT_ROWS_PER_WARP) ? (unsigned)(D - base_output) : DRAFT_ROWS_PER_WARP;
    const auto result = SEISMIC_RESULT_0_OUT;
    const auto combine = KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE));
    float sums[DRAFT_ROWS_PER_WARP] = {0.0f, 0.0f, 0.0f, 0.0f};
    draft_combine_4(combine, joined, base_output, active, 2 * D, lane, sums);
#pragma unroll
    for (unsigned r = 0; r < DRAFT_ROWS_PER_WARP; ++r) {
        if (r < active) {
            const float sum = seismic_warp_sum_f32(sums[r]);
            if (lane == 0)
                result.put(row * SEISMIC_RESULT_0_STRIDE_0 + (stream * D + base_output + r) * SEISMIC_RESULT_0_STRIDE_1,
                           sum);
        }
    }
}

extern "C" __global__ void draft_rows_join(SEISMIC_KERNEL_PARAMS) {
    extern __shared__ float joined[];
    __shared__ float partials[2 * DRAFT_WARPS];
    const u64 D = SEISMIC_DIM_D;
    const u64 row = blockIdx.y / SEISMIC_DIM_S;
    const u64 stream = blockIdx.y % SEISMIC_DIM_S;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const long long held = gathered::row(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY)),
                                         SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                         tokens[row * SEISMIC_TOKENS_STRIDE_0]);

    const element::Out<Act> staged = element::out_over_scratch<Act>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED));
    const u64 staged_row_base = (row * SEISMIC_DIM_S + stream) * 2 * D;
    draft_join<true>(DRAFT_INPUT, held, row, stream, D, epsilon, joined, partials, staged, staged_row_base);
}

template <class Shape>
__device__ __forceinline__ void draft_gemm(const unsigned char *staged, element::Out<element::F32> result,
                                          const packets::W1 &combine, u64 rows, u64 width) {
    extern __shared__ uint4 dynamic_shared[];
    const projection::Store<element::F32> out{result, width, 0};
    projection::gemm_run<Shape, false, false>(reinterpret_cast<projection::u8 *>(dynamic_shared),
        staged, 2 * width, nullptr, rows, 2 * width, projection::gemm_column<Shape>(), width,
        combine, projection::NoWeight{}, out);
}

extern "C" __global__ void draft_rows_gemm_small(SEISMIC_KERNEL_PARAMS) {
    draft_gemm<projection::SmallGemm>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED),
        SEISMIC_RESULT_0_OUT, KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE)),
        SEISMIC_DIM_M * SEISMIC_DIM_S, SEISMIC_DIM_D);
}

extern "C" __global__ void draft_rows_gemm(SEISMIC_KERNEL_PARAMS) {
    draft_gemm<projection::LargeGemm<2>>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_JOINED),
        SEISMIC_RESULT_0_OUT, KERNEL_W1_AT(SEISMIC_PTR(SEISMIC_BUFFER_COMBINE)),
        SEISMIC_DIM_M * SEISMIC_DIM_S, SEISMIC_DIM_D);
}
