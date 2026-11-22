// per_layer_inputs: the per-layer input rows (PLE). Threadgroup (chunk,
// row) of min(P, 256) threads looping over the chunk: first the chunk's
// square sum for the RMS inverse (caching the scaled projection values in
// registers), then simdgroup-cooperative gather of the chunk's
// rows16/rows32/dense values (<seismic/gather.h>) in the same per-thread
// column order and
//   (p * rms_inverse * norm + gathered * gathered_scale) * scale
// with p the scaled projection value.
#define KERNEL_T SEISMIC_GATHERED
#include "lib/core/activation.h"
#include "lib/core/reduce.h"
#include <seismic/gather.h>

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

constant constexpr uint chunk = SEISMIC_DIM_P;
constant constexpr uint threads = chunk < 256u ? chunk : 256u;
constant constexpr uint simdgroups = threads / 32u;
constant constexpr uint items = (chunk + threads - 1u) / threads;
constant constexpr uint cached_items = items < 8u ? items : 8u;

kernel void per_layer_inputs(
    device const uchar *gathered [[buffer(SEISMIC_BUFFER_GATHERED)]],
    device const float *projected [[buffer(SEISMIC_BUFFER_PROJECTED)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[simdgroups];
    const uint tid = sg * 32u + lane;
    const ulong row = group.y;
    const uint width = uint(SEISMIC_DIM_L) * chunk;
    const uint first = group.x * chunk;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float gathered_scale = as_type<float>(uint(SEISMIC_PARAM_GATHERED_SCALE));
    const float projected_scale = as_type<float>(uint(SEISMIC_PARAM_PROJECTED_SCALE));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const ulong proj_row = row * SEISMIC_PROJECTED_STRIDE_0;
    const ulong out_row = row * SEISMIC_RESULT_0_STRIDE_0;
    float proj_scaled[cached_items];
    float squares = 0.0f;
    for (uint k = 0u; k < items; ++k) {
        const uint i = tid + k * threads;
        if (i < chunk) {
            const float p = projected[proj_row + ulong(first + i) * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
            if (k < cached_items)
                proj_scaled[k] = p;
            squares = metal::fma(p, p, squares);
        }
    }
    const float total = reduce::group_sum<simdgroups>(squares, partials, sg, lane);
    const float inverse = metal::rsqrt(total / float(chunk) + epsilon);
    const auto table = KERNEL_T_AT(gathered, width);
    const auto out = element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells);
    uint k = 0u;
    packets::gather_row(table, row, first, chunk, sg, simdgroups, lane, [&](uint i, float token) {
        const uint column = first + i;
        const float p = k < cached_items
            ? proj_scaled[cached_items == 1u ? 0u : k]
            : projected[proj_row + ulong(column) * SEISMIC_PROJECTED_STRIDE_1] * projected_scale;
        ++k;
        const float normalized = p * inverse * element::at<Norm>(norm, ulong(i) * SEISMIC_NORM_STRIDE_0);
        out.put(out_row + ulong(column) * SEISMIC_RESULT_0_STRIDE_1,
            (normalized + token * gathered_scale) * scale);
    });
}
