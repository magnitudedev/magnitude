// Shared pieces of the routed (mixture-of-experts) CUDA entries built on the
// projection family (`routed_expand`, `_output`, `_experts`): the grouped
// blocks' row source. An expert's weights are matrix `expert` of its
// expert-stacked [E, N, K] tensor (`KERNEL_Wn_MATRIX`, dense or packed).

#include "../projection/projection.cuh"

namespace routed {

using element::u32;
using element::u64;
using element::u8;

// The live rows of a grouped block: its `order` entries are one expert's rows
// followed by -1 padding, so the count is the index of the first -1.
__device__ __forceinline__ u32 block_rows(const int *order, u64 stride, u32 rows) {
    u32 low = 0, high = rows;
    while (low < high) {
        const u32 middle = (low + high) / 2;
        if (order[middle * stride] >= 0)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

// The activation rows of a grouped block (a GEMM row source): tile row m
// reads activation row order[m]. The GEMM reads only the block's live rows.
struct GroupedRows {
    const u8 *x;
    u64 stride;
    const int *order;
    u64 order_stride;
    __device__ __forceinline__ const u8 *row(u64 m) const {
        return x + (u64)order[m * order_stride] * stride * 2;
    }
};

// Decode choices that share an expert (M <= 8 rows of K choices, a row's
// choices distinct experts). Choice c = m * K + k computes when it is its
// expert's first choice in that order, once for every choice of the expert:
// a GEMV row per choice, whose result does not depend on its neighbours, so
// each choice's bits are those of its own one-row GEMV. Routes are read from
// L2 (another launch wrote them).
__device__ __forceinline__ int route(const int *routes, u64 row_stride, u64 slot_stride, u32 slots, u32 choice) {
    return __ldcg(routes + (choice / slots) * row_stride + (choice % slots) * slot_stride);
}

// Whether `choice` is its expert's first choice.
__device__ __forceinline__ bool first_choice(const int *routes, u64 row_stride, u64 slot_stride, u32 slots,
                                             u32 choice) {
    const int expert = route(routes, row_stride, slot_stride, slots, choice);
    for (u32 c = 0; c < choice; ++c)
        if (route(routes, row_stride, slot_stride, slots, c) == expert)
            return false;
    return true;
}

// The choices of `choice`'s expert in order (at most one per row), into
// `chosen`; returns their count.
__device__ __forceinline__ u32 expert_choices(const int *routes, u64 row_stride, u64 slot_stride, u32 rows,
                                              u32 slots, u32 choice, u32 (&chosen)[8]) {
    const int expert = route(routes, row_stride, slot_stride, slots, choice);
    u32 count = 0;
    for (u32 c = choice; c < rows * slots; ++c)
        if (route(routes, row_stride, slot_stride, slots, c) == expert)
            chosen[count++] = c;
    return count;
}

// Activation rows read in place at element offsets: GEMV row r reads
// x[offsets[r] ..].
struct GatheredRows {
    const u8 *x;
    const u64 *offsets;
    static constexpr bool STAGED = false;
    static constexpr int FACTORS = 0;
    __device__ __forceinline__ void prepare_row(float *, u32, float *) const {}
    __device__ __forceinline__ float value(const float *, u32 m, u64 k) const {
        return projection::Act::round(element::at<projection::Act>(x, offsets[m] + k));
    }
    __device__ __forceinline__ u32 pair(const float *, u32 m, u64 k) const {
        return *reinterpret_cast<const u32 *>(x + (offsets[m] + k) * 2);
    }
};

} // namespace routed
