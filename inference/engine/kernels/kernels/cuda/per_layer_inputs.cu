// per_layer_inputs: the per-layer input rows (PLE). Block (chunk, row), one
// thread per chunk column. The chunk's gathered values are decoded first into
// shared memory (packed rows: mma16 lane chunks of 16 values per work item,
// as `embedding_rows`; dense rows read in place), then each thread forms
//   (p * rms_inverse * norm + gathered * gathered_scale) * scale
// with p the scaled projection value and the chunk's square sum.
#if defined(SEISMIC_GATHERED_KIND_PACKED)
#define KERNEL_W0 SEISMIC_GATHERED
#endif
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include <seismic/packets.cuh>

using element::u32;
using element::u64;
using element::u8;

extern "C" __global__ void per_layer_inputs(SEISMIC_KERNEL_PARAMS) {
    using Norm = ELEMENT_OF(SEISMIC_NORM);
    constexpr u32 chunk = SEISMIC_DIM_P;
    __shared__ float partials[32];
    __shared__ float tokens[chunk];
    const u64 layer = blockIdx.x, row = blockIdx.y;
    const u32 i = threadIdx.x;
    const u64 first = layer * chunk;
#if defined(SEISMIC_GATHERED_KIND_PACKED)
    const auto table = KERNEL_W0_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATHERED));
    // Work item (k-block, column pair t) decodes 16 values of the chunk.
    for (u32 item = i; item < chunk / 64 * 4; item += blockDim.x) {
        const u32 kb = item / 4, t = item % 4;
        float values[16];
        packets::row_values16(table, row, first / 64 + kb, t, values);
        const u32 offsets[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
#pragma unroll
        for (int s = 0; s < 4; ++s)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tokens[kb * 64 + 16 * s + offsets[j]] = values[4 * s + j];
    }
#else
    using Table = ELEMENT_OF(SEISMIC_GATHERED);
    tokens[i] = element::at<Table>(SEISMIC_PTR(SEISMIC_BUFFER_GATHERED),
        row * SEISMIC_GATHERED_STRIDE_0 + (first + i) * SEISMIC_GATHERED_STRIDE_1);
#endif
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float gathered_scale = __uint_as_float((unsigned)SEISMIC_PARAM_GATHERED_SCALE);
    const float projected_scale = __uint_as_float((unsigned)SEISMIC_PARAM_PROJECTED_SCALE);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *projected = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTED));
    const float p = projected[row * SEISMIC_PROJECTED_STRIDE_0 + (first + i) * SEISMIC_PROJECTED_STRIDE_1]
        * projected_scale;
    // The reduction's barriers also publish the decoded tokens.
    const float total = reduce::group_sum(p * p, partials);
    const float inverse = rsqrtf(total / (float)chunk + epsilon);
    const float normalized =
        p * inverse * element::at<Norm>(SEISMIC_PTR(SEISMIC_BUFFER_NORM), (u64)i * SEISMIC_NORM_STRIDE_0);
    float *result = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    result[row * SEISMIC_RESULT_0_STRIDE_0 + (first + i) * SEISMIC_RESULT_0_STRIDE_1] =
        (normalized + tokens[i] * gathered_scale) * scale;
}
