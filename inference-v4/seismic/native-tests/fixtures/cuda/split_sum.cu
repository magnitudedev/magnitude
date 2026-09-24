extern "C" __global__ void split_partial(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *partials = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    extern __shared__ float staging[];
    const unsigned long long n = SEISMIC_DIM_N;
    const unsigned long long per = (n + SEISMIC_TUNE_PARTS - 1) / SEISMIC_TUNE_PARTS;
    const unsigned long long part = blockIdx.x;
    const unsigned long long lane = threadIdx.x;
    const unsigned long long begin = part * per < n ? part * per : n;
    const unsigned long long end = begin + per < n ? begin + per : n;
    float total = 0.0f;
    for (unsigned long long base = begin; base < end; base += SEISMIC_TUNE_WIDTH) {
        const unsigned long long index = base + lane;
        staging[lane] = index < end ? x[index * SEISMIC_X_STRIDE_0] : 0.0f;
        __syncthreads();
        if (lane == 0) {
            const unsigned long long remaining = end - base;
            const unsigned long long count =
                remaining < SEISMIC_TUNE_WIDTH ? remaining : SEISMIC_TUNE_WIDTH;
            for (unsigned long long j = 0; j < count; ++j) {
                total += staging[j];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        partials[part] = total;
    }
}

extern "C" __global__ void split_merge(SEISMIC_KERNEL_PARAMS) {
    const float *partials =
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    float total = 0.0f;
    for (unsigned part = 0; part < SEISMIC_TUNE_PARTS; ++part) {
        total += partials[part];
    }
    result[0] = total;
}
