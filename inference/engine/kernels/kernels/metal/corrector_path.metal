// corrector_path: one threadgroup of 256 threads (8 simdgroups) per slot walks
// its lattice greedily: for each of the first P mask rows, each simdgroup
// caches its strided slice of the step's reference row (`anchor` for row 0,
// the previous selection's `outgoing` row after it) in registers, scores its
// strided candidates with simdgroup-only reductions (zero threadgroup barriers
// per candidate), and one threadgroup barrier per step selects the first
// highest-scoring candidate.

#define PATH_THREADS 256u
#define PATH_WARPS (PATH_THREADS / 32u)

kernel void corrector_path(
    device const int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    device const float *incoming [[buffer(SEISMIC_BUFFER_INCOMING)]],
    device const float *outgoing [[buffer(SEISMIC_BUFFER_OUTGOING)]],
    device const float *anchor [[buffer(SEISMIC_BUFFER_ANCHOR)]],
    device int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint slot [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint OTHER_REG_CAP = 16u;
    constexpr ulong FULL_CHUNKS = (ulong(SEISMIC_DIM_F) + 31u) / 32u;
    constexpr uint OTHER_CHUNKS = FULL_CHUNKS < ulong(OTHER_REG_CAP) ? uint(FULL_CHUNKS) : OTHER_REG_CAP;
    constexpr uint OTHER_ARRAY_LEN = OTHER_CHUNKS > 0u ? OTHER_CHUNKS : 1u;

    threadgroup float best_scores[2][PATH_WARPS];
    threadgroup ulong best_cands[2][PATH_WARPS];

    const ulong count = SEISMIC_DIM_K, width = SEISMIC_DIM_F, steps = SEISMIC_DIM_P;
    const uint active_warps = count < ulong(PATH_WARPS) ? uint(count) : PATH_WARPS;

    ulong previous = 0;
    for (ulong step = 0; step < steps; ++step) {
        device const float *other;
        ulong other_stride;
        if (step == 0) {
            other = anchor + ulong(slot) * SEISMIC_ANCHOR_STRIDE_0;
            other_stride = SEISMIC_ANCHOR_STRIDE_1;
        } else {
            other = outgoing + ulong(slot) * SEISMIC_OUTGOING_STRIDE_0 + (step - 1) * SEISMIC_OUTGOING_STRIDE_1 +
                previous * SEISMIC_OUTGOING_STRIDE_2;
            other_stride = SEISMIC_OUTGOING_STRIDE_3;
        }

        float sg_best = -INFINITY;
        ulong sg_chosen = 0;
        if (sg < active_warps) {
            float other_reg[OTHER_ARRAY_LEN];
            _Pragma("clang loop unroll(full)")
            for (uint t = 0; t < OTHER_CHUNKS; ++t) {
                const ulong f = ulong(lane) + ulong(t) * 32u;
                other_reg[t] = f < width ? other[f * other_stride] : 0.0f;
            }
            device const float *step_incoming =
                incoming + ulong(slot) * SEISMIC_INCOMING_STRIDE_0 + step * SEISMIC_INCOMING_STRIDE_1;
            for (ulong candidate = ulong(sg); candidate < count; candidate += ulong(PATH_WARPS)) {
                device const float *row = step_incoming + candidate * SEISMIC_INCOMING_STRIDE_2;
                float partial = 0.0f;
                _Pragma("clang loop unroll(full)")
                for (uint t = 0; t < OTHER_CHUNKS; ++t) {
                    const ulong f = ulong(lane) + ulong(t) * 32u;
                    if (f < width)
                        partial = metal::fma(row[f * SEISMIC_INCOMING_STRIDE_3], other_reg[t], partial);
                }
                for (ulong f = ulong(lane) + ulong(OTHER_CHUNKS) * 32u; f < width; f += 32u)
                    partial = metal::fma(row[f * SEISMIC_INCOMING_STRIDE_3], other[f * other_stride], partial);
                const float score = simd_sum(partial);
                if (score > sg_best) {
                    sg_best = score;
                    sg_chosen = candidate;
                }
            }
            if (lane == 0u) {
                const uint buf = uint(step & 1u);
                best_scores[buf][sg] = sg_best;
                best_cands[buf][sg] = sg_chosen;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint buf = uint(step & 1u);
        float best = -INFINITY;
        ulong chosen = 0;
        for (uint w = 0; w < active_warps; ++w) {
            const float w_score = best_scores[buf][w];
            const ulong w_cand = best_cands[buf][w];
            if (w_score > best || (w_score == best && w_cand < chosen)) {
                best = w_score;
                chosen = w_cand;
            }
        }
        previous = chosen;
        if (thread_index == 0u) {
            const ulong at = step * SEISMIC_SELECTION_STRIDE_0 + ulong(slot) * SEISMIC_SELECTION_STRIDE_1;
            selection[at] = candidates[ulong(slot) * SEISMIC_CANDIDATES_STRIDE_0 + step * SEISMIC_CANDIDATES_STRIDE_1 +
                chosen * SEISMIC_CANDIDATES_STRIDE_2];
            selection[at + SEISMIC_SELECTION_STRIDE_2] = 0;
        }
    }
}
