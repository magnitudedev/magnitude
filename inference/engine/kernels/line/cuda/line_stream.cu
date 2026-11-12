// line_stream: a warp sums a row of C values, each lane UNROLL 16-byte
// loads in flight at once (C is a multiple of 32 lanes x 4 values x 16).
extern "C" __global__ void line_stream(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr unsigned UNROLL = SEISMIC_TUNE_UNROLL;
    const u64 row = (u64)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= SEISMIC_DIM_R)
        return;
    const unsigned lane = threadIdx.x % 32;
    const float4 *line = reinterpret_cast<const float4 *>(
        reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_INPUT)) + row * SEISMIC_INPUT_STRIDE_0);
    const u64 vectors = SEISMIC_DIM_C / 4;
    float partial[UNROLL];
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        partial[k] = 0.0f;
    for (u64 at = lane; at < vectors; at += 32 * UNROLL) {
        float4 loaded[UNROLL];
#pragma unroll
        for (unsigned k = 0; k < UNROLL; ++k)
            loaded[k] = __ldg(line + at + 32 * k);
#pragma unroll
        for (unsigned k = 0; k < UNROLL; ++k)
            partial[k] += (loaded[k].x + loaded[k].y) + (loaded[k].z + loaded[k].w);
    }
    float total = 0.0f;
#pragma unroll
    for (unsigned k = 0; k < UNROLL; ++k)
        total += partial[k];
    total = seismic_warp_sum_f32(total);
    if (lane == 0)
        reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_OUT))[row * SEISMIC_OUT_STRIDE_0] = total;
}
