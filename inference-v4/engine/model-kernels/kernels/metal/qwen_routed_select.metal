inline ulong offset2(ulong row, ulong column, ulong stride0, ulong stride1) {
    return row * stride0 + column * stride1;
}
kernel void qwen_routed_select(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device int *routes [[buffer(SEISMIC_BUFFER_ROUTES)]],
    device float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[thread_position_in_grid]]) {
    ulong row = ulong(row_index);
    if (row >= SEISMIC_DIM_O) return;
    float maximum = -INFINITY;
    for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert) {
        maximum = metal::max(maximum, logits[offset2(row, expert,
            SEISMIC_LOGITS_STRIDE_0, SEISMIC_LOGITS_STRIDE_1)]);
    }
    float total = 0.0f;
    for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert) {
        total += metal::exp(logits[offset2(row, expert,
            SEISMIC_LOGITS_STRIDE_0, SEISMIC_LOGITS_STRIDE_1)] - maximum);
    }
    float selected_total = 0.0f;
    for (ulong rank = 0; rank < SEISMIC_DIM_K; ++rank) {
        float best = -INFINITY;
        int winner = -1;
        for (ulong expert = 0; expert < SEISMIC_DIM_E; ++expert) {
            bool already = false;
            for (ulong prior = 0; prior < rank; ++prior) {
                ulong prior_choice = SEISMIC_DIM_K - 1 - prior;
                already |= routes[offset2(row, prior_choice,
                    SEISMIC_ROUTES_STRIDE_0, SEISMIC_ROUTES_STRIDE_1)] == int(expert);
            }
            float probability = metal::exp(logits[offset2(row, expert,
                SEISMIC_LOGITS_STRIDE_0, SEISMIC_LOGITS_STRIDE_1)] - maximum) / total;
            if (!already && (probability > best
                    || (probability == best && int(expert) > winner))) {
                best = probability;
                winner = int(expert);
            }
        }
        ulong choice = SEISMIC_DIM_K - 1 - rank;
        routes[offset2(row, choice, SEISMIC_ROUTES_STRIDE_0,
            SEISMIC_ROUTES_STRIDE_1)] = winner;
        scores[offset2(row, choice, SEISMIC_SCORES_STRIDE_0,
            SEISMIC_SCORES_STRIDE_1)] = best;
        selected_total += best;
    }
    if (SEISMIC_PARAM_SELECTED != 0) {
        for (ulong choice = 0; choice < SEISMIC_DIM_K; ++choice) {
            ulong score_index = offset2(row, choice, SEISMIC_SCORES_STRIDE_0,
                SEISMIC_SCORES_STRIDE_1);
            scores[score_index] /= selected_total;
        }
    }
}
