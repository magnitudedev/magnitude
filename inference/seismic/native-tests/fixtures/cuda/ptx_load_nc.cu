// Each block copies 512 contiguous floats: one 16-byte non-coherent load per
// thread (L1-allocating or `no_allocate`), a two-word and a one-word load for
// a partial tail, after prefetching its successor block's chunk into L2.
#include <seismic/element.cuh>

extern "C" __global__ void ptx_load_nc(SEISMIC_KERNEL_PARAMS) {
    const float *x = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_X));
    const auto result = SEISMIC_RESULT_0_OUT;
    const unsigned long long n = SEISMIC_DIM_N;
    const unsigned long long first = (unsigned long long)blockIdx.x * 512 + threadIdx.x * 4;
    if (first + 512 < n) {
        seismic_prefetch_l2(x + first + 512);
    }
    if (first + 4 <= n) {
#if SEISMIC_TUNE_NO_ALLOCATE
        const uint4 words = seismic_ld_nc_na_v4(x + first);
#else
        const uint4 words = seismic_ld_nc_v4(x + first);
#endif
        result.put(first, __uint_as_float(words.x));
        result.put(first + 1, __uint_as_float(words.y));
        result.put(first + 2, __uint_as_float(words.z));
        result.put(first + 3, __uint_as_float(words.w));
        return;
    }
    unsigned long long index = first;
    if (index + 2 <= n) {
        const uint2 words = seismic_ld_nc_v2(x + index);
        result.put(index, __uint_as_float(words.x));
        result.put(index + 1, __uint_as_float(words.y));
        index += 2;
    }
    if (index < n) {
        result.put(index, __uint_as_float(seismic_ld_nc_u32(x + index)));
    }
}
