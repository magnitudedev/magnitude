// One block streams `x` through STAGES shared-memory tiles of 512 floats
// (one 16-byte `cp.async` per thread per tile). Each thread publishes the
// chunk another thread copied, so the barrier's visibility is exercised too.
// Shared memory starts poisoned with NaN; a zero fill that did not happen
// leaves NaN past the tail and fails the fixture. `x` and the result are
// contiguous.

extern "C" __global__ void ptx_cp_async(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    extern __shared__ __align__(16) float stages[];
    __shared__ unsigned zero_fill_failed;
    const unsigned long long n = SEISMIC_DIM_N;
    const unsigned long long tiles = (n + 511) / 512;
    const unsigned thread = threadIdx.x;
    for (unsigned i = thread; i < SEISMIC_TUNE_STAGES * 512; i += 128) {
        stages[i] = __uint_as_float(0x7fc00000u);
    }
    if (thread == 0) {
        zero_fill_failed = 0;
    }
    __syncthreads();
    auto issue = [&](unsigned long long tile) {
        const unsigned long long first = tile * 512 + thread * 4;
        const unsigned long long remaining = first < n ? n - first : 0;
        const unsigned bytes = remaining >= 4 ? 16u : (unsigned)remaining * 4u;
        float *target = stages + (tile % SEISMIC_TUNE_STAGES) * 512 + thread * 4;
        seismic_cp_async_16_zfill(target, bytes > 0 ? x + first : x, bytes);
    };
    for (unsigned long long tile = 0; tile + 1 < SEISMIC_TUNE_STAGES; ++tile) {
        if (tile < tiles) {
            issue(tile);
        }
        seismic_cp_async_commit();
    }
    for (unsigned long long tile = 0; tile < tiles; ++tile) {
        const unsigned long long ahead = tile + SEISMIC_TUNE_STAGES - 1;
        if (ahead < tiles) {
            issue(ahead);
        }
        seismic_cp_async_commit();
        seismic_cp_async_wait<SEISMIC_TUNE_STAGES - 1>();
        __syncthreads();
        const unsigned source = 127 - thread;
        const float *chunk = stages + (tile % SEISMIC_TUNE_STAGES) * 512 + source * 4;
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned long long index = tile * 512 + source * 4 + j;
            if (index < n) {
                result[index] = chunk[j];
            } else if (__float_as_uint(chunk[j]) != 0u) {
                zero_fill_failed = 1;
            }
        }
        __syncthreads();
    }
    if (thread == 0 && zero_fill_failed != 0) {
        result[0] = __uint_as_float(0x7fc00000u);
    }
}
