// draft_path_step: one threadgroup of 256 threads per row caches
// joint[r] = predecessor[row, r] * hidden[row, r] once in dynamic threadgroup
// memory (streaming when R > JOINT_CAP), then assigns one simdgroup per
// candidate with simdgroup-only reductions and selects the first
// highest-scoring candidate.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.h"

typedef activation::Act activation_element;

kernel void draft_path_step(
    device const int *candidates [[buffer(SEISMIC_BUFFER_CANDIDATES)]],
    device const float *unary [[buffer(SEISMIC_BUFFER_UNARY)]],
    device const uchar *hidden [[buffer(SEISMIC_BUFFER_HIDDEN)]],
    device const uchar *predecessor [[buffer(SEISMIC_BUFFER_PREDECESSOR)]],
    device const uchar *successor [[buffer(SEISMIC_BUFFER_SUCCESSOR)]],
    device int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *joint [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    typedef typename activation_element::storage storage;
    constexpr ulong JOINT_CAP = 4096ul;
    constexpr ulong SIMDGROUPS = 8ul;

    threadgroup float sg_best[8];
    threadgroup ulong sg_chosen[8];

    const ulong count = SEISMIC_DIM_K;
    if (count == 0ul) {
        return;
    }
    const ulong rank = SEISMIC_DIM_R;
    const ulong row_u = ulong(row);
    const ulong pred_row = row_u * SEISMIC_PREDECESSOR_STRIDE_0;
    const ulong hid_row = row_u * SEISMIC_HIDDEN_STRIDE_0;
    const bool pred_unit = SEISMIC_PREDECESSOR_STRIDE_1 == 1ul;
    const bool hid_unit = SEISMIC_HIDDEN_STRIDE_1 == 1ul;
    const bool succ_unit = SEISMIC_SUCCESSOR_STRIDE_1 == 1ul;
    const ulong vec_end = rank & ~3ul;
    const bool cached = rank <= JOINT_CAP;

    device const storage *pred_storage = reinterpret_cast<device const storage *>(predecessor) + pred_row;
    device const storage *hid_storage = reinterpret_cast<device const storage *>(hidden) + hid_row;
    device const storage *succ_base = reinterpret_cast<device const storage *>(successor);

    if (cached) {
        if (pred_unit && hid_unit) {
            for (ulong r = ulong(thread_index) * 4ul; r < vec_end; r += 1024ul) {
                const float4 p = activation_element::load4(pred_storage + r);
                const float4 h = activation_element::load4(hid_storage + r);
                reinterpret_cast<threadgroup float4 *>(joint + r)[0] = p * h;
            }
            for (ulong r = vec_end + ulong(thread_index); r < rank; r += 256ul) {
                joint[r] = activation_element::load(pred_storage[r]) * activation_element::load(hid_storage[r]);
            }
        } else {
            for (ulong r = ulong(thread_index); r < rank; r += 256ul) {
                joint[r] =
                    element::at<activation_element>(predecessor, pred_row + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                    element::at<activation_element>(hidden, hid_row + r * SEISMIC_HIDDEN_STRIDE_1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float best = -INFINITY;
    ulong chosen = 0ul;

    for (ulong candidate = ulong(sg); candidate < count; candidate += SIMDGROUPS) {
        const ulong at = row_u * count + candidate;
        const ulong succ_row = at * SEISMIC_SUCCESSOR_STRIDE_0;
        float partial = 0.0f;

        if (cached) {
            if (succ_unit) {
                device const storage *succ_storage = succ_base + succ_row;
                float4 acc = float4(0.0f);
                for (ulong r = ulong(lane) * 4ul; r < vec_end; r += 128ul) {
                    const float4 j = reinterpret_cast<threadgroup const float4 *>(joint + r)[0];
                    const float4 s = activation_element::load4(succ_storage + r);
                    acc = metal::fma(j, s, acc);
                }
                partial = (acc.x + acc.y) + (acc.z + acc.w);
                for (ulong r = vec_end + ulong(lane); r < rank; r += 32ul) {
                    partial = metal::fma(joint[r], activation_element::load(succ_storage[r]), partial);
                }
            } else {
                for (ulong r = ulong(lane); r < rank; r += 32ul) {
                    partial = metal::fma(
                        joint[r],
                        element::at<activation_element>(successor, succ_row + r * SEISMIC_SUCCESSOR_STRIDE_1),
                        partial);
                }
            }
        } else {
            if (pred_unit && hid_unit && succ_unit) {
                device const storage *succ_storage = succ_base + succ_row;
                float4 acc = float4(0.0f);
                for (ulong r = ulong(lane) * 4ul; r < vec_end; r += 128ul) {
                    const float4 p = activation_element::load4(pred_storage + r);
                    const float4 h = activation_element::load4(hid_storage + r);
                    const float4 s = activation_element::load4(succ_storage + r);
                    acc = metal::fma(p * h, s, acc);
                }
                partial = (acc.x + acc.y) + (acc.z + acc.w);
                for (ulong r = vec_end + ulong(lane); r < rank; r += 32ul) {
                    const float j =
                        activation_element::load(pred_storage[r]) * activation_element::load(hid_storage[r]);
                    partial = metal::fma(j, activation_element::load(succ_storage[r]), partial);
                }
            } else {
                for (ulong r = ulong(lane); r < rank; r += 32ul) {
                    const float j =
                        element::at<activation_element>(predecessor, pred_row + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                        element::at<activation_element>(hidden, hid_row + r * SEISMIC_HIDDEN_STRIDE_1);
                    partial = metal::fma(
                        j,
                        element::at<activation_element>(successor, succ_row + r * SEISMIC_SUCCESSOR_STRIDE_1),
                        partial);
                }
            }
        }

        const float score = simd_sum(partial) +
            unary[row_u * SEISMIC_UNARY_STRIDE_0 + candidate * SEISMIC_UNARY_STRIDE_1];
        if (score > best) {
            best = score;
            chosen = candidate;
        }
    }

    if (lane == 0u && ulong(sg) < count) {
        sg_best[sg] = best;
        sg_chosen[sg] = chosen;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index == 0u) {
        const ulong active_sgs = count < SIMDGROUPS ? count : SIMDGROUPS;
        float best_score = -INFINITY;
        ulong best_candidate = 0ul;
        for (ulong w = 0ul; w < active_sgs; ++w) {
            const float score = sg_best[w];
            const ulong cand = sg_chosen[w];
            if (score > best_score || (score == best_score && score > -INFINITY && cand < best_candidate)) {
                best_score = score;
                best_candidate = cand;
            }
        }
        const ulong chosen_at = row_u * count + best_candidate;
        selection[row_u * SEISMIC_SELECTION_STRIDE_0] = candidates[chosen_at * SEISMIC_CANDIDATES_STRIDE_0];
        selection[row_u * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 0;
    }
}
