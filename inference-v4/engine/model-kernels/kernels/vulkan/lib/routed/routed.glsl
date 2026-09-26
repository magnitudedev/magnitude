// Shared pieces of the routed (mixture-of-experts) Vulkan entries built on
// the projection family: expert row offsets in expert-stacked weights and the
// live-row count of a grouped block. The counterpart of `metal/lib/routed/routed.h`
// (the grouped-row prologue is `projection_grouped`).
#include "../projection/projection.glsl"

// First row of expert `expert` in an [E, N, K] rows16 tensor of N rows per
// expert.
uint64_t routed_expert_row(uint64_t expert, uint64_t rows) { return expert * rows; }

// Rows of the weight tensor at `base` starting at row `first`.
projection_weights routed_weights(uint64_t base, packets_rows16 geometry, uint64_t first, uint k) {
    return projection_weights(base + first * geometry.stride, geometry, k, 0ul);
}

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1. The
// expert GEMMs skip the products of padding rows (`live_rows`).
uint routed_block_rows(uint64_t order, uint64_t stride, uint rows) {
    uint low = 0u, high = rows;
    while (low < high) {
        const uint middle = (low + high) / 2u;
        if (element_i32_at(order + uint64_t(middle) * stride * 4ul) >= 0)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}
