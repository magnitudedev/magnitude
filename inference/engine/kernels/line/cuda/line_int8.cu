// line_int8: m16n8k32 s8 MMAs into s32, the int8 GEMM's product: S
// multiply-adds a lane (an MMA is 128 a lane), in UNROLL independent
// accumulators seeded apart, every operand byte twice each warp's first `x`
// truncated; it writes `x` back (line.seismic). A warp runs whole or not at
// all.
extern "C" __global__ void line_int8(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned UNROLL = SEISMIC_TUNE_UNROLL;
    constexpr unsigned MMAS = SEISMIC_STATIC_S / 128;
    const u64 lane = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 first = lane - threadIdx.x % 32;
    if (first >= SEISMIC_DIM_N)
        return;
    const float *xs = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const unsigned bytes = (unsigned)(int)(2.0f * xs[first * SEISMIC_X_STRIDE_0]) * 0x01010101u;
    int d[UNROLL][4];
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        d[k][0] = d[k][1] = d[k][2] = d[k][3] = (int)k;
    for (unsigned step = 0; step < MMAS / UNROLL; ++step) {
#pragma unroll
        for (unsigned k = 0; k < UNROLL; ++k)
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%0,%1,%2,%3};"
                : "+r"(d[k][0]), "+r"(d[k][1]), "+r"(d[k][2]), "+r"(d[k][3])
                : "r"(bytes), "r"(bytes), "r"(bytes), "r"(bytes), "r"(bytes), "r"(bytes));
    }
    unsigned total = 0;
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        total += (unsigned)((d[k][0] + d[k][1]) + (d[k][2] + d[k][3]));
    if (lane < SEISMIC_DIM_N) {
        const float x = xs[lane * SEISMIC_X_STRIDE_0];
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT))[lane * SEISMIC_OUT_STRIDE_0] =
            x + (total == 0xFFFFFFFFu ? 1.0f : 0.0f);
    }
}
