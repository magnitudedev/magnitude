#include <seismic/element.cuh>

typedef unsigned long long clustered_index;

extern "C" __global__ void place_logits_fill(SEISMIC_KERNEL_PARAMS) {
    const element::Out<element::F32> logits = SEISMIC_OUT_LOGITS;
    const clustered_index col = ((clustered_index)blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (col >= SEISMIC_DIM_V) return;
    const float neg_inf = __int_as_float(0xff800000);
    const float neg_inf4[4] = {neg_inf, neg_inf, neg_inf, neg_inf};
    const bool full_vec = SEISMIC_LOGITS_STRIDE_1 == 1 && col + 4u <= SEISMIC_DIM_V;
    const clustered_index limit = col + 4u < SEISMIC_DIM_V ? col + 4u : SEISMIC_DIM_V;
    for (clustered_index row = (clustered_index)blockIdx.y; row < SEISMIC_DIM_M; row += (clustered_index)gridDim.y) {
        const clustered_index row_base = row * SEISMIC_LOGITS_STRIDE_0;
        if (full_vec && (reinterpret_cast<clustered_index>(logits.stored(row_base)) & 15u) == 0u) {
            logits.put_span(row_base + col, neg_inf4);
        } else {
            for (clustered_index c = col; c < limit; ++c) {
                logits.put(row_base + c * SEISMIC_LOGITS_STRIDE_1, neg_inf);
            }
        }
    }
}

extern "C" __global__ void place_logits_rows(SEISMIC_KERNEL_PARAMS) {
    const float *scores = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCORES));
    const int *token_rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_TOKEN_ROWS));
    const element::Out<element::F32> logits = SEISMIC_OUT_LOGITS;
    const clustered_index col = (clustered_index)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= SEISMIC_DIM_S) return;
    for (clustered_index row = (clustered_index)blockIdx.y; row < SEISMIC_DIM_M; row += (clustered_index)gridDim.y) {
        const int token = token_rows[row * SEISMIC_TOKEN_ROWS_STRIDE_0 + col * SEISMIC_TOKEN_ROWS_STRIDE_1];
        if (token >= 0 && clustered_index(token) < SEISMIC_DIM_V)
            logits.put(row * SEISMIC_LOGITS_STRIDE_0 + clustered_index(token) * SEISMIC_LOGITS_STRIDE_1,
                       scores[row * SEISMIC_SCORES_STRIDE_0 + col * SEISMIC_SCORES_STRIDE_1]);
    }
}
