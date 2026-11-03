// readout_scores_rows: scores[row, key] = scale * sum_d query[row, d] * keys[key, d],
// the products of A operands summed in F32. One simdgroup per key row: each
// lane sums its strided columns in order, then the simdgroup sums its lanes.
#include "lib/core/activation.h"

typedef element::Act activation;

kernel void readout_scores_rows(
    device const uchar *query [[buffer(SEISMIC_BUFFER_QUERY)]],
    device const uchar *keys [[buffer(SEISMIC_BUFFER_KEYS)]],
    device float *scores [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const ulong key = ulong(group.x) * 8ul + ulong(sg);
    const ulong row = ulong(group.y);
    // The key is uniform across a simdgroup: it leaves whole.
    if (key >= SEISMIC_DIM_N) return;
    float sum = 0.0f;
    for (ulong column = lane; column < SEISMIC_DIM_D; column += 32ul) {
        sum = fma(element::at<activation>(query, row * SEISMIC_QUERY_STRIDE_0 + column * SEISMIC_QUERY_STRIDE_1),
            element::at<activation>(keys, key * SEISMIC_KEYS_STRIDE_0 + column * SEISMIC_KEYS_STRIDE_1), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        scores[row * SEISMIC_RESULT_0_STRIDE_0 + key * SEISMIC_RESULT_0_STRIDE_1] =
            sum * as_type<float>(uint(SEISMIC_PARAM_SCALE));
    }
}
