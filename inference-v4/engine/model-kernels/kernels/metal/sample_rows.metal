inline uint2 multiply_high_low(uint left, uint right) {
    ulong product = ulong(left) * ulong(right);
    return uint2(uint(product >> 32), uint(product));
}

inline float sampled_score(float value, ulong token, device const uint *draws,
    constant ulong *seismic_words, ulong row) {
    if (draws[row * SEISMIC_DRAWS_STRIDE_0] != 1u) return value;
    uint4 counter(uint(token),
        draws[row * SEISMIC_DRAWS_STRIDE_0 + 3 * SEISMIC_DRAWS_STRIDE_1],
        draws[row * SEISMIC_DRAWS_STRIDE_0 + 4 * SEISMIC_DRAWS_STRIDE_1],
        draws[row * SEISMIC_DRAWS_STRIDE_0 + 5 * SEISMIC_DRAWS_STRIDE_1]);
    uint2 key(
        draws[row * SEISMIC_DRAWS_STRIDE_0 + SEISMIC_DRAWS_STRIDE_1],
        draws[row * SEISMIC_DRAWS_STRIDE_0 + 2 * SEISMIC_DRAWS_STRIDE_1]);
    for (uint round = 0; round < 10; ++round) {
        uint2 p0 = multiply_high_low(3528531795u, counter.x);
        uint2 p1 = multiply_high_low(3449720151u, counter.z);
        counter = uint4(p1.x ^ counter.y ^ key.x, p1.y,
            p0.x ^ counter.w ^ key.y, p0.y);
        key += uint2(2654435769u, 3144134277u);
    }
    float uniform = (float(counter.x >> 9) + 0.5f) * 0.00000011920928955078125f;
    return value - metal::log(-metal::log(uniform));
}

kernel void sample_rows(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],
    device int *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row_index [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    ulong row = ulong(row_index);
    if (row >= SEISMIC_DIM_M) return;
    uint simdgroups = 256u / simd_width;
    threadgroup float group_scores[32];
    threadgroup uint group_tokens[32];
    threadgroup uint group_bad[32];

    uint bad = 0u;
    float best_score = -INFINITY;
    uint best_token = 0xffffffffu;
    for (ulong token = ulong(tid); token < SEISMIC_DIM_V; token += 256ul) {
        float value = logits[row * SEISMIC_LOGITS_STRIDE_0
            + token * SEISMIC_LOGITS_STRIDE_1];
        bad |= metal::isnan(value) || value == INFINITY;
        uint word = mask[row * SEISMIC_MASK_STRIDE_0
            + (token / 32) * SEISMIC_MASK_STRIDE_1];
        if (((word >> uint(token % 32)) & 1u) == 0u
            || !(value > -INFINITY && value < INFINITY)) {
            continue;
        }
        float score = sampled_score(value, token, draws, seismic_words, row);
        if (best_token == 0xffffffffu || score > best_score
            || (score == best_score && token < ulong(best_token))) {
            best_score = score;
            best_token = uint(token);
        }
    }

    uint simd_bad = simd_max(bad);
    float simd_score = simd_max(best_score);
    uint simd_token = simd_min(best_score == simd_score ? best_token : 0xffffffffu);
    if (lane == 0) {
        group_bad[simdgroup] = simd_bad;
        group_scores[simdgroup] = simd_score;
        group_tokens[simdgroup] = simd_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        uint partial_bad = lane < simdgroups ? group_bad[lane] : 0u;
        float partial_score = lane < simdgroups ? group_scores[lane] : -INFINITY;
        uint partial_token = lane < simdgroups ? group_tokens[lane] : 0xffffffffu;
        uint row_bad = simd_max(partial_bad);
        float row_score = simd_max(partial_score);
        uint winner = simd_min(partial_score == row_score ? partial_token : 0xffffffffu);
        if (lane == 0) {
            int status = row_bad != 0u ? 2 : (winner == 0xffffffffu ? 1 : 0);
            result[row * SEISMIC_RESULT_STRIDE_0] = status == 0 ? int(winner) : -1;
            result[row * SEISMIC_RESULT_STRIDE_0 + SEISMIC_RESULT_STRIDE_1] = status;
        }
    }
}
