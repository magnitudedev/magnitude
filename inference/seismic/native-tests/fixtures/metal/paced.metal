// Adds `x` into `state` after `PACE` rounds of work of the thread's own:
// 2^17 steps of a generator each, which no compiler folds away because what
// is written reads its last value. The steps never change that: a step
// flips the generator's low bit, their number is even, and only that bit's
// difference from the index's (none) is added. So `PACE` sets what a call
// costs (2 ms a round on an M4 Pro, 2026-10-10: far more than its
// submission) and separately what it writes: the sum at 0, the sum and 1.0
// above 5, and the sum and 0.5 (the default's) at every pace between.
kernel void paced(
    device float *state [[buffer(SEISMIC_BUFFER_STATE)]],
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        uint generated = index;
        const uint steps = uint(SEISMIC_TUNE_PACE) << 17;
        for (uint step = 0; step < steps; ++step) {
            generated = generated * 1664525u + 1013904223u;
        }
        const float bias = SEISMIC_TUNE_PACE == 0 ? 0.0f : (SEISMIC_TUNE_PACE > 5 ? 1.0f : 0.5f);
        const ulong at = index * SEISMIC_STATE_STRIDE_0;
        state[at] = state[at] + (x[index * SEISMIC_X_STRIDE_0] + bias + float((generated ^ index) & 1u));
    }
}
