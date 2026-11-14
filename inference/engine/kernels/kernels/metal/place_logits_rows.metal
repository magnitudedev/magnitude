#include <seismic/element.h>
kernel void place_logits_fill(device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]], device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]], SEISMIC_OUT_LOGITS_PARAM(logits_cells), constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_V) return;
    const ulong row = index / SEISMIC_DIM_V, col = index % SEISMIC_DIM_V;
    element::Out<element::F32>(logits_cells).put(row * SEISMIC_LOGITS_STRIDE_0 + col * SEISMIC_LOGITS_STRIDE_1, -INFINITY);
}
kernel void place_logits_rows(device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]], device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]], SEISMIC_OUT_LOGITS_PARAM(logits_cells), constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]], uint raw_index [[thread_position_in_grid]]) {
    const ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_S) return;
    const ulong row = index / SEISMIC_DIM_S, col = index % SEISMIC_DIM_S;
    const int token = token_rows[row * SEISMIC_TOKEN_ROWS_STRIDE_0 + col * SEISMIC_TOKEN_ROWS_STRIDE_1];
    if (token >= 0 && ulong(token) < SEISMIC_DIM_V)
        element::Out<element::F32>(logits_cells).put(row * SEISMIC_LOGITS_STRIDE_0 + ulong(token) * SEISMIC_LOGITS_STRIDE_1, scores[row * SEISMIC_SCORES_STRIDE_0 + col * SEISMIC_SCORES_STRIDE_1]);
}
