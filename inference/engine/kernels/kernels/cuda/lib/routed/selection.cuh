// Output selection of the `routed_select` contract, shared by `routed_select`,
// `routed_select_streamed` and `routed_hint_select`; the CUDA form of
// `metal/lib/routed/selection.h`. Projection of the router rows is the `dense`
// projection operator's (`lib/projection/operator.cuh`).
#include "../core/activation.cuh"
#include "../core/arrive.cuh"
#include "select.cuh"

// One row's selection by one warp (`select::rank_row`, its keys to `keys`).
template <class Logit, class Keys = selection::NoKeys>
__device__ __forceinline__ void select_row(unsigned long long row, Logit logit, const float *bias,
                                           const float *expert_scale, int *routes,
                                           const element::Out<element::F32> &weights,
                                           const seismic_words_t &seismic_words_value, Keys keys = Keys{}) {
    constexpr unsigned K = SEISMIC_DIM_K;
    const unsigned lane = threadIdx.x % 32;
    const select::Selected<K> selected = select::rank_row<static_cast<unsigned>(SEISMIC_DIM_E), K>(
        static_cast<int>(SEISMIC_PARAM_SCORE),
        [&](unsigned expert) { return logit(expert); },
        [&](int expert) { return bias[static_cast<unsigned long long>(expert) * SEISMIC_BIAS_STRIDE_0]; }, lane,
        keys);
    float denominator = 0.0f;
    for (unsigned slot = 0; slot < K; ++slot)
        denominator = seismic_add_rn(denominator, selected.chosen[slot]);
    if (lane == 0) {
        const int normalization = static_cast<int>(SEISMIC_PARAM_NORMALIZATION);
        const float scale = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_SCALE));
        const float epsilon = __uint_as_float(static_cast<unsigned>(SEISMIC_PARAM_NORMALIZATION_EPSILON));
        for (unsigned slot = 0; slot < K; ++slot) {
            float weight = seismic_mul_rn(select::normalized(normalization, selected.chosen[slot], denominator, epsilon), scale);
            if (SEISMIC_DIM_ES != 0)
                weight = seismic_mul_rn(weight,
                    expert_scale[static_cast<unsigned long long>(selected.winners[slot]) * SEISMIC_EXPERT_SCALE_STRIDE_0]);
            routes[row * SEISMIC_ROUTES_STRIDE_0 + slot * SEISMIC_ROUTES_STRIDE_1] = selected.winners[slot];
            weights.put(row * SEISMIC_WEIGHTS_STRIDE_0 + slot * SEISMIC_WEIGHTS_STRIDE_1, weight);
        }
    }
}

namespace routed {

// Vectorized 8-element row-major publication of RMS-normalized inputs shared
// by `routed_select` and `routed_select_streamed`, preserving exact `Act::round`
// and falling back cleanly on non-unit strides or unaligned addresses.
template <class Act, class Norm, class OutElem>
__device__ __forceinline__ void publish_ordered_norm_8(const float *row_ptr, const element::u8 *norm_ptr,
                                                       element::u64 c, float inv, element::Out<OutElem> pub_out,
                                                       element::u64 pub_idx) {
    const float4 x0 = *reinterpret_cast<const float4 *>(row_ptr + c);
    const float4 x1 = *reinterpret_cast<const float4 *>(row_ptr + c + 4ull);
    float w[8];
    element::span<Norm, 8, false>(norm_ptr, c, w);
    float out[8];
    out[0] = Act::round(x0.x * inv * w[0]);
    out[1] = Act::round(x0.y * inv * w[1]);
    out[2] = Act::round(x0.z * inv * w[2]);
    out[3] = Act::round(x0.w * inv * w[3]);
    out[4] = Act::round(x1.x * inv * w[4]);
    out[5] = Act::round(x1.y * inv * w[5]);
    out[6] = Act::round(x1.z * inv * w[6]);
    out[7] = Act::round(x1.w * inv * w[7]);
    pub_out.template put_span<8>(pub_idx, out);
}

template <class Act, class Norm, class OutElem, class NormRaw>
__device__ __forceinline__ void publish_ordered_norm_row(const NormRaw &norm_raw, unsigned row, float inv,
                                                         element::Out<OutElem> pub_out, element::u64 pub_s0,
                                                         element::u64 pub_s1) {
    const element::u64 k = norm_raw.width;
    const element::u64 pub_row = static_cast<element::u64>(row) * pub_s0;
    const float *row_ptr = norm_raw.x + static_cast<element::u64>(row) * norm_raw.stride;
    const bool vec = norm_raw.col_stride == 1ull && norm_raw.norm_stride == 1ull && pub_s1 == 1ull &&
                     (reinterpret_cast<element::u64>(row_ptr) & 15ull) == 0ull &&
                     (reinterpret_cast<element::u64>(norm_raw.norm) & 15ull) == 0ull &&
                     (reinterpret_cast<element::u64>(pub_out.stored(pub_row)) & 15ull) == 0ull;
    if (vec) {
        const element::u64 k8 = k & ~7ull;
        const element::u64 step = static_cast<element::u64>(blockDim.x) * 8ull;
        for (element::u64 c = static_cast<element::u64>(threadIdx.x) * 8ull; c < k8; c += step)
            publish_ordered_norm_8<Act, Norm>(row_ptr, norm_raw.norm, c, inv, pub_out, pub_row + c);
        for (element::u64 c = k8 + threadIdx.x; c < k; c += blockDim.x)
            pub_out.put(pub_row + c, norm_raw.ordered_at(row, c, inv));
    } else {
        for (element::u64 c = threadIdx.x; c < k; c += blockDim.x)
            pub_out.put(pub_row + c * pub_s1, norm_raw.ordered_at(row, c, inv));
    }
}

template <class Act, class Norm, class OutElem, class NormRaw>
__device__ __forceinline__ void publish_ordered_norm_rows(const NormRaw &norm_raw, unsigned m, const float *smem_inv,
                                                          element::Out<OutElem> pub_out, element::u64 pub_s0,
                                                          element::u64 pub_s1) {
    if (m == 0u)
        return;
    if (m == 1u) {
        publish_ordered_norm_row<Act, Norm>(norm_raw, 0u, smem_inv[0], pub_out, pub_s0, pub_s1);
        return;
    }
    const element::u64 k = norm_raw.width;
    const bool vec = norm_raw.col_stride == 1ull && norm_raw.norm_stride == 1ull && pub_s1 == 1ull &&
                     (reinterpret_cast<element::u64>(norm_raw.x) & 15ull) == 0ull &&
                     (norm_raw.stride & 3ull) == 0ull &&
                     (reinterpret_cast<element::u64>(norm_raw.norm) & 15ull) == 0ull &&
                     (reinterpret_cast<element::u64>(pub_out.stored(0)) & 15ull) == 0ull &&
                     ((pub_s0 * OutElem::bytes) & 15ull) == 0ull;
    if (vec) {
        const element::u64 k8 = k & ~7ull;
        const element::u64 vecs64 = k8 >> 3;
        const element::u64 total_vecs = static_cast<element::u64>(m) * vecs64;
        if (total_vecs <= 0xffffffffull) {
            const unsigned vecs = static_cast<unsigned>(vecs64);
            const unsigned total = static_cast<unsigned>(total_vecs);
            for (unsigned idx = threadIdx.x; idx < total; idx += blockDim.x) {
                const unsigned r = idx / vecs;
                const element::u64 c = static_cast<element::u64>(idx % vecs) * 8ull;
                const float *row_ptr = norm_raw.x + static_cast<element::u64>(r) * norm_raw.stride;
                publish_ordered_norm_8<Act, Norm>(row_ptr, norm_raw.norm, c, smem_inv[r], pub_out,
                                                  static_cast<element::u64>(r) * pub_s0 + c);
            }
        } else {
            const element::u64 step = static_cast<element::u64>(blockDim.x) * 8ull;
            for (unsigned r = 0u; r < m; ++r) {
                const float inv = smem_inv[r];
                const float *row_ptr = norm_raw.x + static_cast<element::u64>(r) * norm_raw.stride;
                const element::u64 pub_row = static_cast<element::u64>(r) * pub_s0;
                for (element::u64 c = static_cast<element::u64>(threadIdx.x) * 8ull; c < k8; c += step)
                    publish_ordered_norm_8<Act, Norm>(row_ptr, norm_raw.norm, c, inv, pub_out, pub_row + c);
            }
        }
        if (k8 < k) {
            for (unsigned r = 0u; r < m; ++r) {
                const float inv = smem_inv[r];
                const element::u64 pub_row = static_cast<element::u64>(r) * pub_s0;
                for (element::u64 c = k8 + threadIdx.x; c < k; c += blockDim.x)
                    pub_out.put(pub_row + c, norm_raw.ordered_at(r, c, inv));
            }
        }
    } else {
        const element::u64 total64 = static_cast<element::u64>(m) * k;
        if (total64 <= 0xffffffffull) {
            const unsigned k32 = static_cast<unsigned>(k);
            const unsigned total = static_cast<unsigned>(total64);
            for (unsigned idx = threadIdx.x; idx < total; idx += blockDim.x) {
                const unsigned r = idx / k32;
                const unsigned c = idx % k32;
                pub_out.put(static_cast<element::u64>(r) * pub_s0 + static_cast<element::u64>(c) * pub_s1,
                            norm_raw.ordered_at(r, c, smem_inv[r]));
            }
        } else {
            for (unsigned r = 0u; r < m; ++r) {
                const float inv = smem_inv[r];
                const element::u64 pub_row = static_cast<element::u64>(r) * pub_s0;
                for (element::u64 c = threadIdx.x; c < k; c += blockDim.x)
                    pub_out.put(pub_row + c * pub_s1, norm_raw.ordered_at(r, c, inv));
            }
        }
    }
}

} // namespace routed
