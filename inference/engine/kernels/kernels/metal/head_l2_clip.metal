// head_l2_clip: per-head optional channel scale, L2 norm clip, and optional
// position-dependent log-scaling `tau`. One 64-thread group per (head, row).
#include "lib/core/activation.h"
#include "lib/core/reduce.h"

typedef ELEMENT_OF(SEISMIC_X) InElem;
typedef ELEMENT_OF(SEISMIC_RESULT_0) OutElem;

kernel void head_l2_clip(
    device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device const float *scale [[buffer(SEISMIC_BUFFER_SCALE)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[2];
    const ulong head = group.x;
    const ulong row = group.y;
    const ulong W = SEISMIC_DIM_W;
    const bool has_scale = SEISMIC_DIM_SC > 0;
    const ulong scale_base = (SEISMIC_DIM_SC > 1 ? head : 0ul) * SEISMIC_SCALE_STRIDE_0;
    const float threshold = as_type<float>(uint(SEISMIC_PARAM_THRESHOLD));
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float log_floor = as_type<float>(uint(SEISMIC_PARAM_LOG_FLOOR));
    const float log_alpha = as_type<float>(uint(SEISMIC_PARAM_LOG_ALPHA));

    float tau = 1.0f;
    if (log_alpha != 0.0f) {
        const float pos = float(coordinates[row * SEISMIC_COORDINATES_STRIDE_0]);
        const float ratio = (pos + 1.0f) / log_floor;
        tau = 1.0f + log_alpha * metal::log(metal::max(1.0f, ratio));
    }

    const ulong x_base = row * SEISMIC_X_STRIDE_0 + head * SEISMIC_X_STRIDE_1;
    float sq = 0.0f;
    for (ulong c = thread_index; c < W; c += 64u) {
        float v = element::at<InElem>(x, x_base + c * SEISMIC_X_STRIDE_2);
        if (has_scale)
            v *= scale[scale_base + c * SEISMIC_SCALE_STRIDE_1];
        sq = metal::fma(v, v, sq);
    }
    const float total_sq = reduce::group_sum<2>(sq, partials, sg, lane);
    const float clip = metal::min(1.0f, threshold / (metal::sqrt(total_sq) + eps));
    const float factor = clip * tau;

    const element::Out<OutElem> result(result_cells);
    const ulong out_base = row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1;
    for (ulong c = thread_index; c < W; c += 64u) {
        float v = element::at<InElem>(x, x_base + c * SEISMIC_X_STRIDE_2);
        if (has_scale)
            v *= scale[scale_base + c * SEISMIC_SCALE_STRIDE_1];
        result.put(out_base + c * SEISMIC_RESULT_0_STRIDE_2, v * factor);
    }
}
