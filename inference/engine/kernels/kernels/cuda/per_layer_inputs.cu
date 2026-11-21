// per_layer_inputs: the per-layer input rows (PLE). Block (chunk, row) of
// min(P, 256) threads looping over the chunk: first the chunk's square sum
// for the RMS inverse, then each gathered value is decoded directly through
// `<seismic/gather.cuh>` and combined into
//   (p * rms_inverse * norm + gathered * gathered_scale) * scale
// with p the scaled projection value (no static shared tokens[chunk]).
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
    const u64 layer = blockIdx.x, row = blockIdx.y;
    const u64 first = layer * chunk;
    const float epsilon = __uint_as_float((unsigned)SEISMIC_PARAM_EPSILON);
    const float gathered_scale = __uint_as_float((unsigned)SEISMIC_PARAM_GATHERED_SCALE);
    const float projected_scale = __uint_as_float((unsigned)SEISMIC_PARAM_PROJECTED_SCALE);
    const float scale = __uint_as_float((unsigned)SEISMIC_PARAM_SCALE);
    const float *projected = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_PROJECTED));
    const u8 *norm = SEISMIC_PTR(SEISMIC_BUFFER_NORM);
    const u64 proj_row = row * SEISMIC_PROJECTED_STRIDE_0;
    const u64 out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    float squares = 0.0f;
    for (u32 i = threadIdx.x; i < chunk; i += blockDim.x) {
        const float p = projected[proj_row + (first + i) * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
        squares = seismic_fma_rn(p, p, squares);
    }
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(total / (float)chunk + epsilon);
    const auto out = SEISMIC_RESULT_0_OUT;
    packets::gather_row(KERNEL_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATHERED)), row, first, chunk,
                        [&](u64 i, float token) {
                            const u64 column = first + i;
                            const float p =
                                projected[proj_row + column * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
                            const float normalized =
                                p * inverse * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
                            out.put(out_row + column * SEISMIC_RESULT_0_STRIDE_1,
                                    (normalized + token * gathered_scale) * scale);
                        });
}
