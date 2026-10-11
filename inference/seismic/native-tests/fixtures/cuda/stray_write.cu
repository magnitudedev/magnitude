extern "C" __global__ void stray_write(SEISMIC_KERNEL_PARAMS) {
    float *state = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_STATE));
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < SEISMIC_DIM_N) {
        const bool stray = SEISMIC_TUNE_STRAY && index == SEISMIC_DIM_N - 1;
        state[index * SEISMIC_STATE_STRIDE_0] = stray ? 42.0f : x[index * SEISMIC_X_STRIDE_0];
    }
}
