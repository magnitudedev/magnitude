// A row norm's arithmetic, the one every Metal row norm computes (vision's
// rows, a pooled source's per-token LayerNorm): two-pass F32 statistics, the
// mean from the row's summed values and the inverse from its summed centred
// squares, then one expression per weight and bias form. The caller walks
// its row (from memory, or held across a simdgroup) and reduces its
// partials. The counterpart of `cuda/lib/core/norm.cuh` and
// `vulkan/lib/core/norm.glsl`.
//
// This file is independent of any entry ABI.

#pragma once

namespace row_norm {

// The mean of a row of `width` values summing to `total`.
inline float mean(float total, uint width) {
    return total / float(width);
}

// `squares` with one more centred value's square.
inline float square(float centered, float squares) {
    return metal::fma(centered, centered, squares);
}

// The inverse deviation of a row of `width` values whose centred squares sum
// to `squares` (the mean taken as 0: its root mean square).
inline float inverse(float squares, uint width, float epsilon) {
    return metal::rsqrt(squares / float(width) + epsilon);
}

// A centred value normed: `centered * inverse [* weight] [+ bias]`, one
// expression per form (the compiler contracts a product and a sum only
// within an expression), as the layer norm has always been written.
template <bool WEIGHT, bool BIAS>
inline float apply(float centered, float inverse, float weight, float bias) {
    if (WEIGHT && BIAS)
        return centered * inverse * weight + bias;
    else if (WEIGHT)
        return centered * inverse * weight;
    else if (BIAS)
        return centered * inverse + bias;
    else
        return centered * inverse;
}

} // namespace row_norm
