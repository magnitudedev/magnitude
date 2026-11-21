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

#include "lib/core/gumbel.h"

constant constexpr uint sample_threads = 256;
constant constexpr uint no_token = 0xffffffffu;

// The better of two (score, token) candidates: higher score, then lower token.
inline bool sample_better(float score, uint token, float best_score, uint best_token) {
    return token != no_token &&
        (best_token == no_token || score > best_score || (score == best_score && token < best_token));
}

template <bool NOISY>
inline void sample_consider(
    float value, uint token, bool masked,
    device const uint *words, ulong mask_stride,
    device const uint *draw, ulong draws_stride,
    thread float &best, thread uint &best_token, thread uint &bad) {
    bad |= uint(metal::isnan(value) || value == INFINITY);
    if (!(value > -INFINITY && value < INFINITY))
        return;
    if (masked && ((words[ulong(token >> 5u) * mask_stride] >> (token & 31u)) & 1u) == 0u)
        return;
    const float score = NOISY ? (value + gumbel::noise(token, draw, draws_stride)) : value;
    if (sample_better(score, token, best, best_token)) {
        best = score;
        best_token = token;
    }
}

template <bool NOISY>
inline void sample_scan_partition(
    device const float *values, ulong logits_stride,
    uint begin, uint end, uint thread_index, bool masked,
    device const uint *words, ulong mask_stride,
    device const uint *draw, ulong draws_stride,
    thread float &best, thread uint &best_token, thread uint &bad) {
    if (logits_stride == 1ul) {
        const uint vec_end = begin + ((end - begin) & ~3u);
        for (uint token = begin + thread_index * 4u; token < vec_end; token += sample_threads * 4u) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(values + token));
            sample_consider<NOISY>(v.x, token, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
            sample_consider<NOISY>(v.y, token + 1u, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
            sample_consider<NOISY>(v.z, token + 2u, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
            sample_consider<NOISY>(v.w, token + 3u, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
        }
        if (thread_index < end - vec_end) {
            const uint token = vec_end + thread_index;
            sample_consider<NOISY>(values[token], token, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
        }
    } else {
        for (uint token = begin + thread_index; token < end; token += sample_threads) {
            sample_consider<NOISY>(values[ulong(token) * logits_stride], token, masked, words, mask_stride, draw, draws_stride, best, best_token, bad);
        }
    }
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
    const bool noisy = draw[0] == 1u;
    float best = -INFINITY;
    uint best_token = no_token;
    uint bad = 0;
    if (noisy) {
        sample_scan_partition<true>(
            values, SEISMIC_LOGITS_STRIDE_1, begin, end, thread_index, masked,
            words, SEISMIC_MASK_STRIDE_1, draw, SEISMIC_DRAWS_STRIDE_1, best, best_token, bad);
    } else {
        sample_scan_partition<false>(
            values, SEISMIC_LOGITS_STRIDE_1, begin, end, thread_index, masked,
            words, SEISMIC_MASK_STRIDE_1, draw, SEISMIC_DRAWS_STRIDE_1, best, best_token, bad);
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
