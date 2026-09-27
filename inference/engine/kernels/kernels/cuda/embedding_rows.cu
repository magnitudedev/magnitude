// embedding_rows: one block per token row gathers and decodes its table
// row through the K1 decoders (mma16 lane chunks) or reads a dense table,
// times `scale`; with `normalize` the row's weightless RMS inverse is reduced
// first (a pass over the decoded row), then the row is decoded again and
// normalized. Results: the row rounded to A, and the same values as F32.
#if defined(SEISMIC_TABLE_KIND_PACKED)
#define KERNEL_W0 SEISMIC_TABLE
#endif
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include <seismic/packets.cuh>

using element::Act;
using element::u32;
using element::u64;
using element::u8;

#if defined(SEISMIC_TABLE_KIND_PACKED)
// The packed table row `token` times `scale`: `visit(column, value)` for
// every column, work item (k-block, column pair t) decoding 16 values.
template <class Table, class Visit>
__device__ __forceinline__ void embedding_row(const Table &table, u64 token, u64 width, float scale, Visit visit) {
    for (u64 item = threadIdx.x; item < width / 64 * 4; item += blockDim.x) {
        const u64 kb = item / 4;
        const u32 t = (u32)(item % 4);
        float values[16];
        packets::row_values16(table, token, kb, t, values);
        const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                visit(kb * 64 + 16 * s + offsets[j], values[4 * s + j] * scale);
    }
}
#else
// The dense table row at element `first` (column stride `stride`) times
// `scale`: `visit(column, value)` for every column.
template <class Visit>
__device__ __forceinline__ void embedding_row(const u8 *table, u64 first, u64 stride, u64 width, float scale,
                                              Visit visit) {
    using Table = ELEMENT_OF(SEISMIC_TABLE);
    for (u64 column = threadIdx.x; column < width; column += blockDim.x)
        visit(column, element::at<Table>(table, first + column * stride) * scale);
}
#endif

extern "C" __global__ void embedding_rows(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    u8 *embedded = SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER);
    float *wide = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_1_BUFFER));
    const u64 width = SEISMIC_DIM_D;
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const bool normalize = SEISMIC_PARAM_NORMALIZE != 0;
    // (token, status) rows; a failed selection (-1) embeds token 0.
    const u64 token = (u64)max(tokens[row * SEISMIC_TOKENS_STRIDE_0], 0);
    const u64 out0 = row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 out1 = row * SEISMIC_RESULT_1_STRIDE_0;
    const u64 stride1 = SEISMIC_RESULT_0_STRIDE_1, wide1 = SEISMIC_RESULT_1_STRIDE_1;
#if defined(SEISMIC_TABLE_KIND_PACKED)
    const auto table = KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE));
#define EMBEDDING_ROW(visit) embedding_row(table, token, width, scale, visit)
#else
    const u8 *table = SEISMIC_PTR(SEISMIC_BUFFER_TABLE);
    const u64 first = token * SEISMIC_TABLE_STRIDE_0, stride = SEISMIC_TABLE_STRIDE_1;
#define EMBEDDING_ROW(visit) embedding_row(table, first, stride, width, scale, visit)
#endif
    float inverse = 1.0f;
    if (normalize) {
        float squares = 0.0f;
        EMBEDDING_ROW([&](u64, float value) { squares = seismic_fma_rn(value, value, squares); });
        const float total = reduce::group_sum(squares, partials);
        inverse = rsqrtf(total / (float)width + __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON));
    }
    EMBEDDING_ROW([&](u64 column, float decoded) {
        const float value = Act::round(normalize ? decoded * inverse : decoded);
        element::put<Act>(embedded, out0 + column * stride1, value);
        wide[out1 + column * wide1] = value;
    });
#undef EMBEDDING_ROW
}
