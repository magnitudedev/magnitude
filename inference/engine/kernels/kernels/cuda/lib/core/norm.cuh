// A row norm's arithmetic, the one every CUDA row norm computes (vision's
// rows, a pooled source's per-token LayerNorm): two-pass F32 statistics, the
// mean from the row's summed values and the inverse from its summed centred
// squares, then one expression per weight and bias form. The caller walks
// its row (from memory, or held across a warp) and reduces its partials; the
// counterpart of `vulkan/lib/core/norm.glsl`.
//
// This file is independent of any entry ABI.

#pragma once

namespace row_norm {

// The mean of a row of `width` values summing to `total`.
__device__ __forceinline__ float mean(float total, unsigned width) {
    return total / static_cast<float>(width);
}

// `squares` with one more centred value's square.
__device__ __forceinline__ float square(float centered, float squares) {
    return fmaf(centered, centered, squares);
}

// The inverse deviation of a row of `width` values whose centred squares sum
// to `squares` (the mean taken as 0: its root mean square).
__device__ __forceinline__ float inverse(float squares, unsigned width, float epsilon) {
    return rsqrtf(squares / static_cast<float>(width) + epsilon);
}

// A centred value normed: `centered * inverse [* weight] [+ bias]`, one
// expression per form, as the layer norm has always been written.
template <bool WEIGHT, bool BIAS>
__device__ __forceinline__ float apply(float centered, float inverse, float weight, float bias) {
    if constexpr (WEIGHT && BIAS)
        return centered * inverse * weight + bias;
    else if constexpr (WEIGHT)
        return centered * inverse * weight;
    else if constexpr (BIAS)
        return centered * inverse + bias;
    else
        return centered * inverse;
}

}  // namespace row_norm
