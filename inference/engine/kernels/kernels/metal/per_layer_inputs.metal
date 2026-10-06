// per_layer_inputs: the per-layer input rows (PLE). Threadgroup (chunk,
// row), one thread per chunk column: the scaled projection value, the
// chunk's square sum, then
//   (p * rms_inverse * norm + gathered * gathered_scale) * scale
// with the gathered value decoded from its packet (rows16 or dense).
#define KERNEL_W0 SEISMIC_GATHERED
#include "lib/core/activation.h"
#include "lib/core/reduce.h"
#include <seismic/packets.h>

typedef ELEMENT_OF(SEISMIC_NORM) Norm;

constant constexpr uint chunk = SEISMIC_DIM_P;

kernel void per_layer_inputs(
    device const uchar *gathered [[buffer(SEISMIC_BUFFER_GATHERED)]],
    device const float *projected [[buffer(SEISMIC_BUFFER_PROJECTED)]],
    device const uchar *norm [[buffer(SEISMIC_BUFFER_NORM)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint i [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[chunk / 32];
    const ulong row = group.y;
    const uint width = uint(SEISMIC_DIM_L) * chunk;
    const uint column = group.x * chunk + i;
    const float epsilon = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float gathered_scale = as_type<float>(uint(SEISMIC_PARAM_GATHERED_SCALE));
    const float projected_scale = as_type<float>(uint(SEISMIC_PARAM_PROJECTED_SCALE));
    const float scale = as_type<float>(uint(SEISMIC_PARAM_SCALE));
    const float p = projected[row * SEISMIC_PROJECTED_STRIDE_0 + ulong(column) * SEISMIC_PROJECTED_STRIDE_1]
        * projected_scale;
    const float total = reduce::group_sum<chunk / 32>(p * p, partials, sg, lane);
    const float inverse = metal::rsqrt(total / float(chunk) + epsilon);
    const packets::Rows16 layout = KERNEL_W0_LAYOUT(width);
    const typename packets::W0::packet packet =
        packets::Loader<packets::W0>::load(layout.base_of(gathered, row), layout.of(row), column / 32u, width);
    const float token = packets::value_at<packets::W0>(packet, column % 32u);
    const float normalized = p * inverse * element::at<Norm>(norm, ulong(i) * SEISMIC_NORM_STRIDE_0);
    result[row * SEISMIC_RESULT_0_STRIDE_0 + ulong(column) * SEISMIC_RESULT_0_STRIDE_1] =
        (normalized + token * gathered_scale) * scale;
}
