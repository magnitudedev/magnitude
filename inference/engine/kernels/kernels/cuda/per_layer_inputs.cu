// per_layer_inputs: the per-layer input rows (PLE). Block (chunk, row), one
// thread per chunk column. The chunk's gathered values are decoded first into
// shared memory (<seismic/gather.cuh>, as `embedding_rows`), then each
// thread forms
//   (p * rms_inverse * norm + gathered * gathered_scale) * scale
// with p the scaled projection value and the chunk's square sum.
#define KERNEL_T SEISMIC_GATHERED
#include "lib/core/activation.cuh"
#include "lib/core/reduce.cuh"
#include <seismic/gather.cuh>

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
    packets::gather_row(KERNEL_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATHERED)), row, first, chunk,
                        [&](u64 column, float value) { tokens[column] = value; });
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
    SEISMIC_RESULT_0_OUT.put(row * SEISMIC_RESULT_0_STRIDE_0 + (first + i) * SEISMIC_RESULT_0_STRIDE_1,
                             (normalized + tokens[i] * gathered_scale) * scale);
}
