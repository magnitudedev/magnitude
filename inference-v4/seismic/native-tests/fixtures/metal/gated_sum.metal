// One participant sums in element order after staging the whole vector in
// threadgroup memory.
kernel void gated_staged(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *staging [[threadgroup(0)]]) {
    const ulong n = SEISMIC_DIM_N;
    for (ulong i = 0; i < n; ++i) {
        staging[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (ulong i = 0; i < n; ++i) {
        total += staging[i];
    }
    result[0] = total;
}

// One participant sums in element order after mirroring the vector into
// scratch.
kernel void gated_mirrored(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *mirror [[buffer(SEISMIC_BUFFER_SCRATCH_MIRROR)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    const ulong n = SEISMIC_DIM_N;
    for (ulong i = 0; i < n; ++i) {
        mirror[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (ulong i = 0; i < n; ++i) {
        total += mirror[i];
    }
    result[0] = total;
}
