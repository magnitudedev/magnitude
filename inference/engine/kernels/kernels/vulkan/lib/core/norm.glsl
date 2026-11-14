// A row norm's arithmetic, the one every Vulkan row norm computes (vision's
// rows, a pooled source's per-token LayerNorm): two-pass F32 statistics, the
// mean from the row's summed values and the inverse from its summed centred
// squares, then one expression per weight and bias form. The caller walks
// its row (from memory, or held across a logical subgroup) and reduces its
// partials. The counterpart of `cuda/lib/core/norm.cuh`.
//
// This file is independent of any entry ABI.

// The mean of a row of `width` values summing to `total`.
float norm_mean(float total, uint width) {
    return seismic_div_rn(total, float(width));
}

// `squares` with one more centred value's square.
float norm_square(float centered, float squares) {
    return seismic_fma_rn(centered, centered, squares);
}

// The inverse deviation of a row of `width` values whose centred squares sum
// to `squares` (the mean taken as 0: its root mean square).
float norm_inverse(float squares, uint width, float epsilon) {
    return inversesqrt(seismic_div_rn(squares, float(width)) + epsilon);
}

// A centred value normed: `centered * inverse [* weight] [+ bias]`, one
// expression per form, as the layer norm has always been written.
float norm_apply(float centered, float inverse, bool weighted, float weight, bool biased, float bias) {
    if (weighted && biased)
        return centered * inverse * weight + bias;
    else if (weighted)
        return centered * inverse * weight;
    else if (biased)
        return centered * inverse + bias;
    else
        return centered * inverse;
}
