// Dense weight import over the [B, N, K] view; four elements per thread along
// K with a 3D grid (ceil_div(ceil_div(K, 4), 256), min(N, 65535), min(B, 65535))
// and grid-stride loops over y and z. The CUDA form of `metal/import_dense.metal`.
// The result is a record port (bank records import through it): matrix b is
// written from its own record's base.

#include "lib/core/activation.cuh"

typedef ELEMENT_OF(SEISMIC_ELEMENT_E) Source;
typedef ELEMENT_OF(SEISMIC_ELEMENT_U) Destination;

extern "C" __global__ void import_dense(SEISMIC_KERNEL_PARAMS) {
    typedef element::u64 u64;
    const unsigned char *source = reinterpret_cast<const unsigned char *>(SEISMIC_PTR(SEISMIC_BUFFER_SOURCE));
    const auto destination = SEISMIC_RESULT_0_OUT;
    const u64 k0 = (static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x) * 4u;
    if (k0 >= SEISMIC_DIM_K) return;

    const bool full_vec = (SEISMIC_SOURCE_STRIDE_2 == 1)
        && (SEISMIC_RESULT_0_STRIDE_2 == 1)
        && (k0 + 4u <= SEISMIC_DIM_K);
    const u64 count = (k0 + 4u <= SEISMIC_DIM_K) ? 4u : (SEISMIC_DIM_K - k0);

    for (u64 b = blockIdx.z; b < SEISMIC_DIM_B; b += gridDim.z) {
        const auto dst_region = destination.region(b);
        const u64 src_b = b * SEISMIC_SOURCE_STRIDE_0;
        for (u64 n = blockIdx.y; n < SEISMIC_DIM_N; n += gridDim.y) {
            const u64 src_base = src_b + n * SEISMIC_SOURCE_STRIDE_1;
            const u64 dst_base = n * SEISMIC_RESULT_0_STRIDE_1;
            if (full_vec) {
                const u64 src_offset = src_base + k0;
                const u64 dst_offset = dst_base + k0;
                const u64 src_addr = reinterpret_cast<u64>(source + src_offset * Source::bytes);
                const u64 dst_addr = reinterpret_cast<u64>(dst_region.stored(dst_offset));
                if (((src_addr & (4u * Source::bytes - 1u)) == 0)
                    && ((dst_addr & (4u * Destination::bytes - 1u)) == 0)) {
                    float values[4];
                    element::span<Source, 4, true>(source, src_offset, values);
                    dst_region.put_span<4>(dst_offset, values);
                    continue;
                }
            }
            for (u64 i = 0; i < count; ++i) {
                const u64 k = k0 + i;
                dst_region.put(dst_base + k * SEISMIC_RESULT_0_STRIDE_2,
                    element::at<Source>(source, src_base + k * SEISMIC_SOURCE_STRIDE_2));
            }
        }
    }
}
