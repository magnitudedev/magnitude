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

} // namespace routed
