// line_f32: S fused f32 multiply-adds a thread, in UNROLL independent
// chains seeded apart, in loop iterations of ROUNDS rounds of every chain;
// it writes `x` back (line.seismic).
kernel void line_f32(
    device const float *xs [[buffer(SEISMIC_BUFFER_X)]],
    device float *out [[buffer(SEISMIC_BUFFER_OUT)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint UNROLL = SEISMIC_TUNE_UNROLL;
    constexpr uint ROUNDS = 8;
    const ulong lane = ulong(index);
    if (lane >= SEISMIC_DIM_N)
        return;
    const float x = xs[lane * SEISMIC_X_STRIDE_0];
    float chain[UNROLL];
    for (uint k = 0; k < UNROLL; ++k)
        chain[k] = x + float(k);
    for (uint step = 0; step < SEISMIC_STATIC_S / (UNROLL * ROUNDS); ++step) {
#pragma unroll
        for (uint round = 0; round < ROUNDS; ++round)
#pragma unroll
            for (uint k = 0; k < UNROLL; ++k)
                chain[k] = fma(chain[k], x, x);
    }
    float total = 0.0f;
    for (uint k = 0; k < UNROLL; ++k)
        total += chain[k];
    out[lane * SEISMIC_OUT_STRIDE_0] = x + (as_type<uint>(total) == 0xFFFFFFFFu ? 1.0f : 0.0f);
}
