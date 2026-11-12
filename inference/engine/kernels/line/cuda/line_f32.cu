// line_f32: S fused f32 multiply-adds a thread, in UNROLL independent
// chains seeded apart, in loop iterations of ROUNDS rounds of every chain;
// it writes `x` back (line.seismic).
extern "C" __global__ void line_f32(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned UNROLL = SEISMIC_TUNE_UNROLL;
    constexpr unsigned ROUNDS = 8;
    const u64 lane = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= SEISMIC_DIM_N)
        return;
    const float x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X))[lane * SEISMIC_X_STRIDE_0];
    float chain[UNROLL];
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        chain[k] = x + (float)k;
    for (unsigned step = 0; step < SEISMIC_STATIC_S / (UNROLL * ROUNDS); ++step) {
#pragma unroll
        for (unsigned round = 0; round < ROUNDS; ++round)
#pragma unroll
            for (unsigned k = 0; k < UNROLL; ++k)
                chain[k] = fmaf(chain[k], x, x);
    }
    float total = 0.0f;
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        total += chain[k];
    reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT))[lane * SEISMIC_OUT_STRIDE_0] =
        x + (__float_as_uint(total) == 0xFFFFFFFFu ? 1.0f : 0.0f);
}
