// rel_bias_project: `[M, H, D] A @ [E, D]^T W -> [M, H, E] f32`, with optional
// fused `head_l2_clip` when `threshold > 0.0` (rounding clipped R to `A` before
// the projection to preserve the exact intermediate dtype boundary).
// One 32-thread simdgroup per (head, row, 32-extent tile).
#include "lib/core/activation.h"

typedef ELEMENT_OF(SEISMIC_R) RElem;
typedef ELEMENT_OF(SEISMIC_WEIGHT) WElem;

kernel void rel_bias_project(
    device const uchar *r [[buffer(SEISMIC_BUFFER_R)]],
    device const uchar *weight [[buffer(SEISMIC_BUFFER_WEIGHT)]],
    device const float *scale [[buffer(SEISMIC_BUFFER_SCALE)]],
    device const int *coordinates [[buffer(SEISMIC_BUFFER_COORDINATES)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float r_sh[64];

    const ulong head = group.x;
    const ulong row = group.y;
    const ulong ext = ulong(group.z) * 32ul + ulong(thread_index);
    const ulong D = SEISMIC_DIM_D;
    const ulong E = SEISMIC_DIM_E;
    const bool has_scale = SEISMIC_DIM_SC > 0;
    const ulong scale_base = (SEISMIC_DIM_SC > 1 ? head : 0ul) * SEISMIC_SCALE_STRIDE_0;
    const float threshold = as_type<float>(uint(SEISMIC_PARAM_THRESHOLD));
    const float eps = as_type<float>(uint(SEISMIC_PARAM_EPSILON));
    const float log_floor = as_type<float>(uint(SEISMIC_PARAM_LOG_FLOOR));
    const float log_alpha = as_type<float>(uint(SEISMIC_PARAM_LOG_ALPHA));
    const bool do_clip = threshold > 0.0f;

    const ulong r_base = row * SEISMIC_R_STRIDE_0 + head * SEISMIC_R_STRIDE_1;
    const ulong w_base = ext * SEISMIC_WEIGHT_STRIDE_0;

    float factor = 1.0f;
    if (do_clip) {
        float tau = 1.0f;
        if (log_alpha != 0.0f) {
            const float pos = float(coordinates[row * SEISMIC_COORDINATES_STRIDE_0]);
            const float ratio = (pos + 1.0f) / log_floor;
            tau = 1.0f + log_alpha * metal::log(metal::max(1.0f, ratio));
        }
        float sq = 0.0f;
        for (ulong d = thread_index; d < D; d += 32ul) {
            float v = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
            if (has_scale)
                v *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
            sq = metal::fma(v, v, sq);
        }
        const float total_sq = simd_sum(sq);
        const float clip = metal::min(1.0f, threshold / (metal::sqrt(total_sq) + eps));
        factor = clip * tau;
    }

    if (D <= 64ul) {
        for (ulong d = thread_index; d < D; d += 32ul) {
            float v = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
            if (do_clip) {
                if (has_scale)
                    v *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
                v = RElem::round(v * factor);
            }
            r_sh[d] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ext >= E)
            return;
        float acc = 0.0f;
        for (ulong d = 0; d < D; ++d) {
            const float wv = element::at<WElem>(weight, w_base + d * SEISMIC_WEIGHT_STRIDE_1);
            acc = metal::fma(r_sh[d], wv, acc);
        }
        const element::Out<element::F32> result(result_cells);
        result.put(
            row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + ext * SEISMIC_RESULT_0_STRIDE_2,
            acc);
    } else {
        if (ext >= E)
            return;
        float acc = 0.0f;
        for (ulong d = 0; d < D; ++d) {
            float rv = element::at<RElem>(r, r_base + d * SEISMIC_R_STRIDE_2);
            if (do_clip) {
                if (has_scale)
                    rv *= scale[scale_base + d * SEISMIC_SCALE_STRIDE_1];
                rv = RElem::round(rv * factor);
            }
            const float wv = element::at<WElem>(weight, w_base + d * SEISMIC_WEIGHT_STRIDE_1);
            acc = metal::fma(rv, wv, acc);
        }
        const element::Out<element::F32> result(result_cells);
        result.put(
            row * SEISMIC_RESULT_0_STRIDE_0 + head * SEISMIC_RESULT_0_STRIDE_1
                + ext * SEISMIC_RESULT_0_STRIDE_2,
            acc);
    }
}
