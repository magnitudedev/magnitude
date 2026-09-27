kernel void split_partial(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *staging [[threadgroup(0)]],
    uint part [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    const ulong n = SEISMIC_DIM_N;
    const ulong per = (n + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    const ulong begin = min(ulong(part) * per, n);
    const ulong end = min(begin + per, n);
    float total = 0.0f;
    for (ulong base = begin; base < end; base += SEISMIC_TUNE_WIDTH) {
        const ulong index = base + lane;
        staging[lane] = index < end ? x[index * SEISMIC_X_STRIDE_0] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const ulong count = min(ulong(SEISMIC_TUNE_WIDTH), end - base);
            for (ulong j = 0; j < count; ++j) {
                total += staging[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        partials[part] = total;
    }
}

kernel void split_merge(
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]]) {
    float total = 0.0f;
    for (uint part = 0; part < SEISMIC_TUNE_PARTS; ++part) {
        total += partials[part];
    }
    result[0] = total;
}
