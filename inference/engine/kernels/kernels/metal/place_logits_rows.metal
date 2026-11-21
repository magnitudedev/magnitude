#include <seismic/element.h>

kernel void place_logits_fill(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]],
    SEISMIC_OUT_LOGITS_PARAM(logits_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong col = ulong(position.x) * 4ul;
    const ulong row = ulong(position.y);
    if (row >= SEISMIC_DIM_M || col >= SEISMIC_DIM_V) return;
    const element::Out<element::F32> logits(logits_cells);
    const ulong row_base = row * SEISMIC_LOGITS_STRIDE_0;
    if (SEISMIC_LOGITS_STRIDE_1 == 1 && col + 4ul <= SEISMIC_DIM_V) {
        logits.put4(row_base + col, float4(-INFINITY));
    } else {
        const ulong limit = min(col + 4ul, ulong(SEISMIC_DIM_V));
        for (ulong c = col; c < limit; ++c)
            logits.put(row_base + c * SEISMIC_LOGITS_STRIDE_1, -INFINITY);
    }
}

kernel void place_logits_rows(
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const int *token_rows [[buffer(SEISMIC_BUFFER_TOKEN_ROWS)]],
    SEISMIC_OUT_LOGITS_PARAM(logits_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong col = ulong(position.x);
    const ulong row = ulong(position.y);
    if (row >= SEISMIC_DIM_M || col >= SEISMIC_DIM_S) return;
    const int token = token_rows[row * SEISMIC_TOKEN_ROWS_STRIDE_0 + col * SEISMIC_TOKEN_ROWS_STRIDE_1];
    if (token >= 0 && ulong(token) < SEISMIC_DIM_V)
        element::Out<element::F32>(logits_cells).put(
            row * SEISMIC_LOGITS_STRIDE_0 + ulong(token) * SEISMIC_LOGITS_STRIDE_1,
            scores[row * SEISMIC_SCORES_STRIDE_0 + col * SEISMIC_SCORES_STRIDE_1]);
}
