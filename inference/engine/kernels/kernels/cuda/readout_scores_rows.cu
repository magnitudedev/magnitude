// readout_scores_rows: scores[row, key] = scale * sum_d query[row, d] * keys[key, d],
// the products of A operands summed in F32. One warp per key row: each lane
// sums its strided columns in order, then the warp sums its lanes. The CUDA
// form of `metal/readout_scores_rows.metal`.
#include "lib/core/activation.cuh"

extern "C" __global__ void readout_scores_rows(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const element::u8 *query = SEISMIC_PTR(SEISMIC_BUFFER_QUERY);
    const element::u8 *keys = SEISMIC_PTR(SEISMIC_BUFFER_KEYS);
    float *scores = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 key = (u64)blockIdx.x * 8 + threadIdx.x / 32;
    const u64 row = blockIdx.y;
    const unsigned lane = threadIdx.x % 32;
    // The key is uniform across a warp: it leaves whole.
    if (key >= SEISMIC_DIM_N)
        return;
    float sum = 0.0f;
    for (u64 column = lane; column < SEISMIC_DIM_D; column += 32)
        sum = seismic_fma_rn(element::at<element::Act>(query, row * SEISMIC_QUERY_STRIDE_0 + column * SEISMIC_QUERY_STRIDE_1),
                             element::at<element::Act>(keys, key * SEISMIC_KEYS_STRIDE_0 + column * SEISMIC_KEYS_STRIDE_1),
                             sum);
    sum = seismic_warp_sum_f32(sum);
    if (lane == 0)
        scores[row * SEISMIC_RESULT_0_STRIDE_0 + key * SEISMIC_RESULT_0_STRIDE_1] =
            sum * __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
}
