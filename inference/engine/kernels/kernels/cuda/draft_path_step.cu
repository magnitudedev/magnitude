// draft_path_step: one block of 256 threads per row caches
// joint[r] = predecessor[row, r] * hidden[row, r] once in dynamic shared
// memory (streaming when R > JOINT_CAP), then assigns one warp per candidate
// with warp-only reductions and selects the first highest-scoring candidate.
#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/core/activation.cuh"

extern "C" __global__ void draft_path_step(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    constexpr u64 JOINT_CAP = 4096;
    constexpr u64 VEC_ALIGN_MASK = u64(4u * activation::Act::bytes) - 1u;

    extern __shared__ uint4 dynamic_shared[];
    float *joint = reinterpret_cast<float *>(dynamic_shared);
    __shared__ float warp_best[32];
    __shared__ u64 warp_chosen[32];

    const u64 count = SEISMIC_DIM_K;
    if (count == 0) {
        return;
    }
    const u64 rank = SEISMIC_DIM_R;
    const u64 row = blockIdx.x;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned warps = blockDim.x >> 5;

    const int *candidates = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_CANDIDATES));
    const float *unary = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_UNARY));
    const element::u8 *hidden = SEISMIC_PTR(SEISMIC_BUFFER_HIDDEN);
    const element::u8 *predecessor = SEISMIC_PTR(SEISMIC_BUFFER_PREDECESSOR);
    const element::u8 *successor = SEISMIC_PTR(SEISMIC_BUFFER_SUCCESSOR);
    int *selection = reinterpret_cast<int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));

    const u64 pred_row = row * SEISMIC_PREDECESSOR_STRIDE_0;
    const u64 hid_row = row * SEISMIC_HIDDEN_STRIDE_0;
    const bool pred_unit = SEISMIC_PREDECESSOR_STRIDE_1 == 1;
    const bool hid_unit = SEISMIC_HIDDEN_STRIDE_1 == 1;
    const bool succ_unit = SEISMIC_SUCCESSOR_STRIDE_1 == 1;
    const bool pred_aligned =
        pred_unit && ((reinterpret_cast<u64>(predecessor + pred_row * activation::Act::bytes) & VEC_ALIGN_MASK) == 0);
    const bool hid_aligned =
        hid_unit && ((reinterpret_cast<u64>(hidden + hid_row * activation::Act::bytes) & VEC_ALIGN_MASK) == 0);
    const u64 vec_end = rank & ~u64(3);
    const bool cached = rank <= JOINT_CAP;

    if (cached) {
        if (pred_aligned && hid_aligned) {
            for (u64 r = u64(threadIdx.x) * 4; r < vec_end; r += u64(blockDim.x) * 4) {
                float p[4], h[4];
                element::span<activation::Act, 4, false>(predecessor, pred_row + r, p);
                element::span<activation::Act, 4, false>(hidden, hid_row + r, h);
                reinterpret_cast<float4 *>(joint + r)[0] =
                    make_float4(p[0] * h[0], p[1] * h[1], p[2] * h[2], p[3] * h[3]);
            }
            for (u64 r = vec_end + threadIdx.x; r < rank; r += blockDim.x) {
                joint[r] = element::at<activation::Act>(predecessor, pred_row + r) *
                           element::at<activation::Act>(hidden, hid_row + r);
            }
        } else {
            for (u64 r = threadIdx.x; r < rank; r += blockDim.x) {
                joint[r] = element::at<activation::Act>(predecessor, pred_row + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                           element::at<activation::Act>(hidden, hid_row + r * SEISMIC_HIDDEN_STRIDE_1);
            }
        }
        __syncthreads();
    }

    const float neg_inf = -__int_as_float(0x7f800000);
    float best = neg_inf;
    u64 chosen = 0;

    for (u64 candidate = warp; candidate < count; candidate += warps) {
        const u64 at = row * count + candidate;
        const u64 succ_row = at * SEISMIC_SUCCESSOR_STRIDE_0;
        const bool succ_aligned =
            succ_unit && ((reinterpret_cast<u64>(successor + succ_row * activation::Act::bytes) & VEC_ALIGN_MASK) == 0);
        float partial = 0.0f;

        if (cached) {
            if (succ_aligned) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                for (u64 r = u64(lane) * 4; r < vec_end; r += 128) {
                    const float4 j = reinterpret_cast<const float4 *>(joint + r)[0];
                    float s[4];
                    element::span<activation::Act, 4, false>(successor, succ_row + r, s);
                    acc0 = seismic_fma_rn(j.x, s[0], acc0);
                    acc1 = seismic_fma_rn(j.y, s[1], acc1);
                    acc2 = seismic_fma_rn(j.z, s[2], acc2);
                    acc3 = seismic_fma_rn(j.w, s[3], acc3);
                }
                partial = (acc0 + acc1) + (acc2 + acc3);
                for (u64 r = vec_end + lane; r < rank; r += 32) {
                    partial = seismic_fma_rn(
                        joint[r], element::at<activation::Act>(successor, succ_row + r), partial);
                }
            } else {
                for (u64 r = lane; r < rank; r += 32) {
                    partial = seismic_fma_rn(
                        joint[r],
                        element::at<activation::Act>(successor, succ_row + r * SEISMIC_SUCCESSOR_STRIDE_1),
                        partial);
                }
            }
        } else {
            if (pred_aligned && hid_aligned && succ_aligned) {
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                for (u64 r = u64(lane) * 4; r < vec_end; r += 128) {
                    float p[4], h[4], s[4];
                    element::span<activation::Act, 4, false>(predecessor, pred_row + r, p);
                    element::span<activation::Act, 4, false>(hidden, hid_row + r, h);
                    element::span<activation::Act, 4, false>(successor, succ_row + r, s);
                    acc0 = seismic_fma_rn(p[0] * h[0], s[0], acc0);
                    acc1 = seismic_fma_rn(p[1] * h[1], s[1], acc1);
                    acc2 = seismic_fma_rn(p[2] * h[2], s[2], acc2);
                    acc3 = seismic_fma_rn(p[3] * h[3], s[3], acc3);
                }
                partial = (acc0 + acc1) + (acc2 + acc3);
                for (u64 r = vec_end + lane; r < rank; r += 32) {
                    const float j = element::at<activation::Act>(predecessor, pred_row + r) *
                                    element::at<activation::Act>(hidden, hid_row + r);
                    partial = seismic_fma_rn(j, element::at<activation::Act>(successor, succ_row + r), partial);
                }
            } else {
                for (u64 r = lane; r < rank; r += 32) {
                    const float j =
                        element::at<activation::Act>(predecessor, pred_row + r * SEISMIC_PREDECESSOR_STRIDE_1) *
                        element::at<activation::Act>(hidden, hid_row + r * SEISMIC_HIDDEN_STRIDE_1);
                    partial = seismic_fma_rn(
                        j, element::at<activation::Act>(successor, succ_row + r * SEISMIC_SUCCESSOR_STRIDE_1), partial);
                }
            }
        }

        const float score = seismic_warp_sum_f32(partial) +
                            unary[row * SEISMIC_UNARY_STRIDE_0 + candidate * SEISMIC_UNARY_STRIDE_1];
        if (score > best) {
            best = score;
            chosen = candidate;
        }
    }

    if (lane == 0 && u64(warp) < count) {
        warp_best[warp] = best;
        warp_chosen[warp] = chosen;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const u64 active_warps = count < u64(warps) ? count : u64(warps);
        float best_score = neg_inf;
        u64 best_candidate = 0;
        for (u64 w = 0; w < active_warps; ++w) {
            const float score = warp_best[w];
            const u64 cand = warp_chosen[w];
            if (score > best_score || (score == best_score && score > neg_inf && cand < best_candidate)) {
                best_score = score;
                best_candidate = cand;
            }
        }
        const u64 chosen_at = row * count + best_candidate;
        selection[row * SEISMIC_SELECTION_STRIDE_0] = candidates[chosen_at * SEISMIC_CANDIDATES_STRIDE_0];
        selection[row * SEISMIC_SELECTION_STRIDE_0 + SEISMIC_SELECTION_STRIDE_1] = 0;
    }
}
