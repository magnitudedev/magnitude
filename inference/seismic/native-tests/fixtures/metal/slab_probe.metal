kernel void slab_probe(
    device const ulong *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]]) {
    device const float *first_slab = reinterpret_cast<device const float *>(x[0]);
    result[0] = first_slab[0];
}
