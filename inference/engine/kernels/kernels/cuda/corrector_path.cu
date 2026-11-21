// corrector_path: one block of 256 threads (8 warps) per slot walks its
// lattice greedily: for each of the first P mask rows, each warp caches its
// strided slice of the step's reference row (`anchor` for row 0, the previous
// selection's `outgoing` row after it) in registers, scores its strided
// candidates with warp-only reductions (zero block barriers per candidate),
// and one block barrier per step selects the first highest-scoring candidate.

extern "C" __global__ void corrector_path(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned OTHER_REG_CAP = 16u;
    constexpr u64 FULL_CHUNKS = (static_cast<u64>(SEISMIC_DIM_F) + 31ull) / 32ull;
    constexpr unsigned OTHER_CHUNKS =
        FULL_CHUNKS < static_cast<u64>(OTHER_REG_CAP) ? static_cast<unsigned>(FULL_CHUNKS) : OTHER_REG_CAP;
    constexpr unsigned OTHER_ARRAY_LEN = OTHER_CHUNKS > 0u ? OTHER_CHUNKS : 1u;

    __shared__ float best_scores[2][32];
    __shared__ u64 best_cands[2][32];

    const int *candidates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const float *incoming = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INCOMING));
    const float *outgoing = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUTGOING));
    const float *anchor = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_ANCHOR));
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));

    const u64 slot = blockIdx.x;
    const u64 count = SEISMIC_DIM_K, factor = SEISMIC_DIM_F;
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u, warps = blockDim.x / 32u;
    const unsigned active_warps = count < static_cast<u64>(warps) ? static_cast<unsigned>(count) : warps;
    const float neg_inf = -__int_as_float(0x7f800000);

    u64 previous = 0;
    for (u64 step = 0; step < SEISMIC_DIM_P; ++step) {
        // Row 0 scores against the anchor, later rows against the previous
        // selection's outgoing row.
        const float *other = step == 0 ? anchor + slot * SEISMIC_ANCHOR_STRIDE_0
                                       : outgoing + slot * SEISMIC_OUTGOING_STRIDE_0 +
                                             (step - 1) * SEISMIC_OUTGOING_STRIDE_1 +
                                             previous * SEISMIC_OUTGOING_STRIDE_2;
        const u64 other_stride = step == 0 ? SEISMIC_ANCHOR_STRIDE_1 : SEISMIC_OUTGOING_STRIDE_3;

        float warp_best = neg_inf;
        u64 warp_chosen = 0;
        if (warp < active_warps) {
            float other_reg[OTHER_ARRAY_LEN];
#pragma unroll
            for (unsigned t = 0; t < OTHER_CHUNKS; ++t) {
                const u64 f = static_cast<u64>(lane) + static_cast<u64>(t) * 32u;
                other_reg[t] = f < factor ? other[f * other_stride] : 0.0f;
            }
            const float *step_incoming =
                incoming + slot * SEISMIC_INCOMING_STRIDE_0 + step * SEISMIC_INCOMING_STRIDE_1;
            for (u64 candidate = warp; candidate < count; candidate += warps) {
                const float *row = step_incoming + candidate * SEISMIC_INCOMING_STRIDE_2;
                float partial = 0.0f;
#pragma unroll
                for (unsigned t = 0; t < OTHER_CHUNKS; ++t) {
                    const u64 f = static_cast<u64>(lane) + static_cast<u64>(t) * 32u;
                    if (f < factor)
                        partial = seismic_fma_rn(row[f * SEISMIC_INCOMING_STRIDE_3], other_reg[t], partial);
                }
                for (u64 f = static_cast<u64>(lane) + static_cast<u64>(OTHER_CHUNKS) * 32u; f < factor; f += 32u)
                    partial = seismic_fma_rn(row[f * SEISMIC_INCOMING_STRIDE_3], other[f * other_stride], partial);
                const float score = seismic_warp_sum_f32(partial);
                if (score > warp_best) {
                    warp_best = score;
                    warp_chosen = candidate;
                }
            }
            if (lane == 0u) {
                const unsigned buf = static_cast<unsigned>(step & 1ull);
                best_scores[buf][warp] = warp_best;
                best_cands[buf][warp] = warp_chosen;
            }
        }

        __syncthreads();

        const unsigned buf = static_cast<unsigned>(step & 1ull);
        float best = neg_inf;
        u64 chosen = 0;
        for (unsigned w = 0; w < active_warps; ++w) {
            const float w_score = best_scores[buf][w];
            const u64 w_cand = best_cands[buf][w];
            if (w_score > best || (w_score == best && w_cand < chosen)) {
                best = w_score;
                chosen = w_cand;
            }
        }
        previous = chosen;
        if (threadIdx.x == 0) {
            const u64 at = step * SEISMIC_SELECTION_STRIDE_0 + slot * SEISMIC_SELECTION_STRIDE_1;
            selection[at] = candidates[slot * SEISMIC_CANDIDATES_STRIDE_0 + step * SEISMIC_CANDIDATES_STRIDE_1 +
                                       chosen * SEISMIC_CANDIDATES_STRIDE_2];
            selection[at + SEISMIC_SELECTION_STRIDE_2] = 0;
        }
    }
}
