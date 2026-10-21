// corrector_rows: one threadgroup of 256 threads per row: values plus bias,
// then by form the identity (0), silu (1) or unit length (2, its squared
// length reduced over the threadgroup), plus the residual row when R = 1,
// published to Y.
#include <seismic/element.h>
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_ELEMENT_Y) output_element;

#define ROWS_THREADS 256u

kernel void corrector_rows(
    device const float *values [[buffer(SEISMIC_BUFFER_VALUES)]],
    device const float *bias [[buffer(SEISMIC_BUFFER_BIAS)]],
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device uchar *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[ROWS_THREADS / 32u];
    const ulong width = SEISMIC_DIM_N;
    const int form = int(SEISMIC_PARAM_FORM);
    device const float *line = values + ulong(row) * SEISMIC_VALUES_STRIDE_0;
    float scale = 1.0f;
    if (form == 2) {
        float partial = 0.0f;
        for (ulong column = thread_index; column < width; column += ROWS_THREADS) {
            const float biased = line[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
            partial = metal::fma(biased, biased, partial);
        }
        const float length = metal::sqrt(reduce::group_sum<ROWS_THREADS / 32u>(partial, partials, sg, lane));
        scale = 1.0f / metal::max(length, as_type<float>(uint(SEISMIC_PARAM_EPSILON)));
    }
    for (ulong column = thread_index; column < width; column += ROWS_THREADS) {
        const float biased = line[column * SEISMIC_VALUES_STRIDE_1] + bias[column * SEISMIC_BIAS_STRIDE_0];
        float out = biased;
        if (form == 1)
            out = biased / (1.0f + metal::exp(-biased));
        else if (form == 2)
            out = biased * scale;
        if (SEISMIC_DIM_R == 1)
            out += residual[ulong(row) * SEISMIC_RESIDUAL_STRIDE_1 + column * SEISMIC_RESIDUAL_STRIDE_2];
        element::put<output_element>(
            result, ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1, out);
    }
}
