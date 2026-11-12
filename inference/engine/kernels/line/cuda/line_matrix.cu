// line_matrix: m16n8k16 bf16 MMAs into f32, the 16-bit path of the projection
// GEMMs: S multiply-adds a lane (an MMA is 64 a lane), in UNROLL independent
// accumulators seeded apart, the operands each warp's first `x` in bf16; it
// writes `x` back (line.seismic). A warp runs whole or not at all.
extern "C" __global__ void line_matrix(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned UNROLL = SEISMIC_TUNE_UNROLL;
    constexpr unsigned MMAS = SEISMIC_STATIC_S / 64;
    const u64 lane = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 first = lane - threadIdx.x % 32;
    if (first >= SEISMIC_DIM_N)
        return;
    const float *xs = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const unsigned half = __float_as_uint(xs[first * SEISMIC_X_STRIDE_0]) >> 16;
    const unsigned pair = half | (half << 16);
    float d[UNROLL][4];
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        d[k][0] = d[k][1] = d[k][2] = d[k][3] = (float)k;
    for (unsigned step = 0; step < MMAS / UNROLL; ++step) {
#pragma unroll
        for (unsigned k = 0; k < UNROLL; ++k)
            asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%0,%1,%2,%3};"
                : "+f"(d[k][0]), "+f"(d[k][1]), "+f"(d[k][2]), "+f"(d[k][3])
                : "r"(pair), "r"(pair), "r"(pair), "r"(pair), "r"(pair), "r"(pair));
    }
    float total = 0.0f;
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        total += (d[k][0] + d[k][1]) + (d[k][2] + d[k][3]);
    if (lane < SEISMIC_DIM_N) {
        const float x = xs[lane * SEISMIC_X_STRIDE_0];
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT))[lane * SEISMIC_OUT_STRIDE_0] =
            x + (__float_as_uint(total) == 0xFFFFFFFFu ? 1.0f : 0.0f);
    }
}
