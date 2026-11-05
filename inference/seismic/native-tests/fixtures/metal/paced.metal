// Adds `x` into `state` after `PACE` rounds of work of the thread's own:
// 2^16 dependent steps of a mixer each, `g = (g ^ (g >> 16)) * 0x45d9f3b`,
// whose whole last state is read: whether it is `0xFFFFFFFF`. No compiler
// shortens the chain. A step that is not affine has no closed form and no
// jump ahead (an affine generator's steps fold into a few multiply-adds: a
// round of 2^17 of them took 3.4 us on a GB10, 2026-11-03, within a round's
// noise), and every bit of the state is read (a compiler that reads only
// its low bit computes it). It is never `0xFFFFFFFF`, so 0 is added: each
// step is a bijection of the state, and the one index that reaches it after
// any pace's steps (on the CPU too, whose rounds are 2^12 steps) is past
// 2^30. So `PACE` sets what a call costs, far more than its submission (a
// pace 371 us on a GB10, a call of no pace 2 us, 2026-11-03), and
// separately what it writes: the sum at 0, the sum and 1.0 above 5, and the
// sum and 0.5 (the default's) at every pace between.
kernel void paced(
    device float *state [[buffer(SEISMIC_BUFFER_STATE)]],
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    if (index < SEISMIC_DIM_N) {
        uint generated = index;
        const uint steps = uint(SEISMIC_TUNE_PACE) << 16;
        for (uint step = 0; step < steps; ++step) {
            generated = (generated ^ (generated >> 16)) * 0x45d9f3bu;
        }
        const float bias = SEISMIC_TUNE_PACE == 0 ? 0.0f : (SEISMIC_TUNE_PACE > 5 ? 1.0f : 0.5f);
        const ulong at = index * SEISMIC_STATE_STRIDE_0;
        state[at] = state[at] + (x[index * SEISMIC_X_STRIDE_0] + bias + float(generated == 0xFFFFFFFFu));
    }
}
