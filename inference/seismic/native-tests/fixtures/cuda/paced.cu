// As `metal/paced.metal`: `PACE` rounds of 2^16 dependent mixer steps a
// thread, whose whole last state is read (it is never `0xFFFFFFFF`), then
// the sum with the pace's bias.
extern "C" __global__ void paced(SEISMIC_KERNEL_PARAMS) {
    float *state = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_STATE));
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const unsigned long long index =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < SEISMIC_DIM_N) {
        unsigned int generated = static_cast<unsigned int>(index);
        const unsigned int steps = static_cast<unsigned int>(SEISMIC_TUNE_PACE) << 16;
        for (unsigned int step = 0; step < steps; ++step) {
            generated = (generated ^ (generated >> 16)) * 0x45d9f3bu;
        }
        const float bias = SEISMIC_TUNE_PACE == 0 ? 0.0f : (SEISMIC_TUNE_PACE > 5 ? 1.0f : 0.5f);
        const float unchanged = static_cast<float>(generated == 0xFFFFFFFFu);
        const unsigned long long at = index * SEISMIC_STATE_STRIDE_0;
        state[at] = __fadd_rn(
            state[at],
            __fadd_rn(__fadd_rn(x[index * SEISMIC_X_STRIDE_0], bias), unchanged));
    }
}
