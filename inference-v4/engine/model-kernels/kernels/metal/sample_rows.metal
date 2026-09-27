// sample_rows (K5): Philox4x32-10 Gumbel-max selection per row in two
// launches. L1: each of PARTS partitions scans a contiguous vocabulary slice
// and keeps (score bits, token, invalid). L2: one simdgroup per row merges
// the partitions; the maximum score wins, ties go to the lowest token, and
// the winning score's bits are carried unchanged, so the result does not
// depend on PARTS.
//
// Counter (token, draws[3], draws[4], draws[5]), key (draws[1], draws[2]);
// draws[0] == 1 enables the Gumbel perturbation v - log(-log u) with
// u = ((x >> 9) + 0.5) * 2^-23. A token competes when its logit is finite
// and either its row is unconstrained (`constrained` 0) or its mask bit is set. Status 0 success, 1 no competing token, 2 some logit
// of the row is NaN or +inf.

constant constexpr uint sample_threads = 256;
constant constexpr uint no_token = 0xffffffffu;

inline uint2 sample_multiply(uint left, uint right) {
    ulong product = ulong(left) * ulong(right);
    return uint2(uint(product >> 32), uint(product));
}

inline float sample_score(float value, uint token, device const uint *draw, ulong stride) {
    if (draw[0] != 1u)
        return value;
    uint4 counter(token, draw[3 * stride], draw[4 * stride], draw[5 * stride]);
    uint2 key(draw[stride], draw[2 * stride]);
    for (uint round = 0; round < 10; ++round) {
        uint2 p0 = sample_multiply(3528531795u, counter.x);
        uint2 p1 = sample_multiply(3449720151u, counter.z);
        counter = uint4(p1.x ^ counter.y ^ key.x, p1.y, p0.x ^ counter.w ^ key.y, p0.y);
        key += uint2(2654435769u, 3144134277u);
    }
    float uniform = (float(counter.x >> 9) + 0.5f) * 0.00000011920928955078125f;
    return value - metal::log(-metal::log(uniform));
}

// The better of two (score, token) candidates: higher score, then lower token.
inline bool sample_better(float score, uint token, float best_score, uint best_token) {
    return best_token == no_token || (token != no_token
        && (score > best_score || (score == best_score && token < best_token)));
}

// Reduce (score, token, bad) over a simdgroup.
inline void sample_simd_reduce(thread float &score, thread uint &token, thread uint &bad) {
    bad = simd_max(bad);
    float best = simd_max(token == no_token ? -INFINITY : score);
    bool any = simd_any(token != no_token);
    uint winner = simd_min((token != no_token && score == best) ? token : no_token);
    score = best;
    token = any ? winner : no_token;
}

kernel void sample_rows_partition(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const uint *mask [[buffer(SEISMIC_BUFFER_MASK)]],
    device const int *constrained [[buffer(SEISMIC_BUFFER_CONSTRAINED)]],
    device const uint *draws [[buffer(SEISMIC_BUFFER_DRAWS)]],
    device uint *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[sample_threads / 32];
    threadgroup uint tokens[sample_threads / 32];
    threadgroup uint bads[sample_threads / 32];
    const uint part = group.x, row = group.y;
    const uint vocabulary = uint(SEISMIC_DIM_V);
    const uint span = (vocabulary + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    const uint begin = min(part * span, vocabulary), end = min(begin + span, vocabulary);
    device const float *values = logits + ulong(row) * SEISMIC_LOGITS_STRIDE_0;
    device const uint *words = mask + ulong(row) * SEISMIC_MASK_STRIDE_0;
    device const uint *draw = draws + ulong(row) * SEISMIC_DRAWS_STRIDE_0;
    // An unconstrained row admits every token; its mask row is never read.
    const bool masked = constrained[ulong(row) * SEISMIC_CONSTRAINED_STRIDE_0] != 0;
    float best = -INFINITY;
    uint best_token = no_token;
    uint bad = 0;
    for (uint token = begin + thread_index; token < end; token += sample_threads) {
        float value = values[ulong(token) * SEISMIC_LOGITS_STRIDE_1];
        bad |= uint(metal::isnan(value) || value == INFINITY);
        if (!(value > -INFINITY && value < INFINITY))
            continue;
        if (masked && ((words[ulong(token / 32u) * SEISMIC_MASK_STRIDE_1] >> (token % 32u)) & 1u) == 0u)
            continue;
        float score = sample_score(value, token, draw, SEISMIC_DRAWS_STRIDE_1);
        if (sample_better(score, token, best, best_token)) {
            best = score;
            best_token = token;
        }
    }
    sample_simd_reduce(best, best_token, bad);
    if (lane == 0) {
        scores[sg] = best;
        tokens[sg] = best_token;
        bads[sg] = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float score = lane < sample_threads / 32 ? scores[lane] : -INFINITY;
        uint token = lane < sample_threads / 32 ? tokens[lane] : no_token;
        uint flag = lane < sample_threads / 32 ? bads[lane] : 0u;
        sample_simd_reduce(score, token, flag);
        if (lane == 0) {
            device uint *out = partials + (ulong(row) * SEISMIC_TUNE_PARTS + part) * 3u;
            out[0] = as_type<uint>(score);
            out[1] = token;
            out[2] = flag;
        }
    }
}

kernel void sample_rows_merge(
    device const uint *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device int *result [[buffer(SEISMIC_BUFFER_RESULT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float best = -INFINITY;
    uint best_token = no_token;
    uint bad = 0;
    for (uint part = lane; part < SEISMIC_TUNE_PARTS; part += 32u) {
        device const uint *in = partials + (ulong(row) * SEISMIC_TUNE_PARTS + part) * 3u;
        float score = as_type<float>(in[0]);
        uint token = in[1];
        bad |= in[2];
        if (sample_better(score, token, best, best_token)) {
            best = score;
            best_token = token;
        }
    }
    sample_simd_reduce(best, best_token, bad);
    if (lane == 0) {
        int status = bad != 0u ? 2 : (best_token == no_token ? 1 : 0);
        result[ulong(row) * SEISMIC_RESULT_STRIDE_0] = status == 0 ? int(best_token) : -1;
        result[ulong(row) * SEISMIC_RESULT_STRIDE_0 + SEISMIC_RESULT_STRIDE_1] = status;
    }
}
