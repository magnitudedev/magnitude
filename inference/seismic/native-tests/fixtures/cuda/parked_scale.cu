// `parked_scale` (route.seismic): the kernel parks on the host's answer word,
// read by volatile loads under system-scope fences, then scales `x` by it.

extern "C" __global__ void parked_scale(SEISMIC_KERNEL_PARAMS) {
    const volatile int *answer = reinterpret_cast<const volatile int *>(SEISMIC_PTR(SEISMIC_BUFFER_ANSWER));
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    // Spins before the kernel stops waiting (see the Metal form).
    int seen = 0;
    for (unsigned spin = 0; spin < (1u << 24) && seen == 0; ++spin) {
        __threadfence_system();
        seen = answer[0];
    }
    for (unsigned long long index = 0; index < SEISMIC_DIM_N; ++index)
        result[index * SEISMIC_RESULT_0_STRIDE_0] = x[index * SEISMIC_X_STRIDE_0] * float(seen);
}
