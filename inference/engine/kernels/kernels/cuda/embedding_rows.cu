// embedding_rows: one block per token row gathers its table row
// (<seismic/gather.cuh>: both packed layouts give every value the same bits
// and visit columns in the same order), times `scale`; with `normalize` the
// row's weightless RMS inverse is reduced first (a pass over the gathered
// row), then the row is gathered again and normalized. Results: the row
// rounded to A, and the same values as F32.
#define KERNEL_A SEISMIC_ELEMENT_A
#define KERNEL_T SEISMIC_TABLE
#include "lib/core/activation.cuh"
#include "lib/core/gathered.cuh"
#include "lib/core/reduce.cuh"
#include <seismic/gather.cuh>

using activation::Act;
using element::u64;

extern "C" __global__ void embedding_rows(SEISMIC_KERNEL_PARAMS) {
    __shared__ float partials[32];
    const u64 row = blockIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    const int *tokens = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKENS));
    const auto embedded = SEISMIC_RESULT_0_OUT;
    const auto wide = SEISMIC_RESULT_1_OUT;
    const u64 width = SEISMIC_DIM_D;
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const bool normalize = SEISMIC_PARAM_NORMALIZE != 0;
    const long long held = gathered::row(reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_DIRECTORY)),
                                         SEISMIC_DIRECTORY_STRIDE_0, SEISMIC_DIM_R, SEISMIC_DIM_F,
                                         tokens[row * SEISMIC_TOKENS_STRIDE_0]);
    const u64 out0 = row * SEISMIC_RESULT_0_STRIDE_0;
    const u64 out1 = row * SEISMIC_RESULT_1_STRIDE_0;
    const u64 stride1 = SEISMIC_RESULT_0_STRIDE_1, wide1 = SEISMIC_RESULT_1_STRIDE_1;
    // A row that names no table row reads zeros (the whole block's).
    if (held < 0) {
        for (u64 column = threadIdx.x; column < width; column += blockDim.x) {
            embedded.put(out0 + column * stride1, 0.0f);
            wide.put(out1 + column * wide1, 0.0f);
        }
        return;
    }
    const u64 token = (u64)held;
    const auto table = KERNEL_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_TABLE));
    float inverse = 1.0f;
    if (normalize) {
        float squares = 0.0f;
        packets::gather_row(table, token, 0, width, [&](u64, float decoded) {
            const float value = decoded * scale;
            squares = seismic_fma_rn(value, value, squares);
        });
        const float total = reduce::group_sum(squares, partials);
        inverse = rsqrtf(total / (float)width + __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON));
    }
    packets::gather_row(table, token, 0, width, [&](u64 column, float decoded) {
        const float value = Act::round(normalize ? decoded * scale * inverse : decoded * scale);
        embedded.put(out0 + column * stride1, value);
        wide.put(out1 + column * wide1, value);
    });
}
