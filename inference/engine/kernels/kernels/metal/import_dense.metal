#include <seismic/slab.h>
#include "lib/core/activation.h"

typedef ELEMENT_OF(SEISMIC_ELEMENT_E) Source;
typedef ELEMENT_OF(SEISMIC_ELEMENT_U) Destination;

// Four elements per thread along K of the [B, N, K] view, with a 3D grid
// (ceil_div(ceil_div(K, 4), 256), N, B). The result is a record port (bank
// records import through it): matrix b is written from its own record's base.
kernel void import_dense(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    SEISMIC_RESULT_0_PARAM(destination),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const ulong k0 = (ulong(group.x) * 256 + ulong(tid)) * 4;
    const ulong n = ulong(group.y);
    const ulong b = ulong(group.z);
    if (k0 >= SEISMIC_DIM_K || n >= SEISMIC_DIM_N || b >= SEISMIC_DIM_B) return;

    const ulong src_base = b * SEISMIC_SOURCE_STRIDE_0 + n * SEISMIC_SOURCE_STRIDE_1;
    const ulong dst_base = n * SEISMIC_RESULT_0_STRIDE_1;
    const element::Out<Destination> dst_region = slab::out_region<Destination>(destination, b);

    if (SEISMIC_SOURCE_STRIDE_2 == 1 && SEISMIC_RESULT_0_STRIDE_2 == 1 && k0 + 4 <= SEISMIC_DIM_K) {
        const float4 values = Source::load4(
            reinterpret_cast<device const Source::storage *>(source) + src_base + k0);
        dst_region.put4(dst_base + k0, values);
        return;
    }

    const ulong count = (k0 + 4 <= SEISMIC_DIM_K) ? 4 : (SEISMIC_DIM_K - k0);
    for (ulong i = 0; i < count; ++i) {
        const ulong k = k0 + i;
        dst_region.put(dst_base + k * SEISMIC_RESULT_0_STRIDE_2,
            element::at<Source>(source, src_base + k * SEISMIC_SOURCE_STRIDE_2));
    }
}
