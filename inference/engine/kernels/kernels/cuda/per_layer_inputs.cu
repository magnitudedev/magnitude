// per_layer_inputs: the per-layer input rows (PLE). Block (chunk, row) of
// min(P, 256) threads looping over the chunk: first the chunk's square sum
// for the RMS inverse (caching the scaled projection in shared memory when
// chunk <= 2048 so `projected` is read once from global memory), then each
// gathered value is decoded directly through `<seismic/gather.cuh>` and
// combined into
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
    constexpr u32 chunk4 = chunk & ~3u;
    constexpr bool CACHE_PROJ = chunk <= 2048u;
    __shared__ float partials[32];
    __shared__ __align__(16) float proj_scaled[CACHE_PROJ ? chunk : 1u];
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
    const float *proj_chunk = projected + proj_row + first * SEISMIC_PROJECTED_STRIDE_1;
    const bool proj_vec =
        SEISMIC_PROJECTED_STRIDE_1 == 1ull && (reinterpret_cast<u64>(proj_chunk) & 15ull) == 0ull;
    float squares = 0.0f;
    if (proj_vec) {
        const u32 step4 = blockDim.x * 4u;
        for (u32 i = threadIdx.x * 4u; i < chunk4; i += step4) {
            const float4 v = *reinterpret_cast<const float4 *>(proj_chunk + i);
            const float p0 = v.x * projected_scale;
            const float p1 = v.y * projected_scale;
            const float p2 = v.z * projected_scale;
            const float p3 = v.w * projected_scale;
            squares = seismic_fma_rn(p0, p0, squares);
            squares = seismic_fma_rn(p1, p1, squares);
            squares = seismic_fma_rn(p2, p2, squares);
            squares = seismic_fma_rn(p3, p3, squares);
            if constexpr (CACHE_PROJ) {
                *reinterpret_cast<float4 *>(proj_scaled + i) = make_float4(p0, p1, p2, p3);
            }
        }
        for (u32 i = chunk4 + threadIdx.x; i < chunk; i += blockDim.x) {
            const float p = proj_chunk[i] * projected_scale;
            squares = seismic_fma_rn(p, p, squares);
            if constexpr (CACHE_PROJ) {
                proj_scaled[i] = p;
            }
        }
    } else {
        for (u32 i = threadIdx.x; i < chunk; i += blockDim.x) {
            const float p = proj_chunk[static_cast<u64>(i) * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
            squares = seismic_fma_rn(p, p, squares);
            if constexpr (CACHE_PROJ) {
                proj_scaled[i] = p;
            }
        }
    }
    // `reduce::group_sum` synchronizes the block, publishing `proj_scaled` for Pass 2.
    const float total = reduce::group_sum(squares, partials);
    const float inverse = rsqrtf(total / (float)chunk + epsilon);
    const auto out = SEISMIC_RESULT_0_OUT;
    packets::gather_row(KERNEL_T_AT(SEISMIC_PTR(SEISMIC_BUFFER_GATHERED)), row, first, chunk,
                        [&](u64 i, float token) {
                            const u64 column = first + i;
                            float p;
                            if constexpr (CACHE_PROJ) {
                                p = proj_scaled[static_cast<u32>(i)];
                            } else {
                                (void)proj_scaled;
                                p = proj_chunk[i * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
                            }
                            const float normalized =
                                p * inverse * element::at<Norm>(norm, i * SEISMIC_NORM_STRIDE_0);
                            out.put(out_row + column * SEISMIC_RESULT_0_STRIDE_1,
                                    (normalized + token * gathered_scale) * scale);
                        });
}
