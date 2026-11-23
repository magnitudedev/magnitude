// The decode output projections of Qwen's fused routed feed-forward (M <=
// VERIFY_ROWS), shared by `routed_output` and `routed_output_project`: threadgroup (x, m)
// owns the TILE output channels of tile x for row m and runs K + 1
// simdgroups side by side: simdgroup k < K projects choice k through its
// expert's down rows, simdgroup K the shared
// expert's down rows. Each lane group (LANES lanes) of a simdgroup owns ROWS
// channels; its lanes own the packets sub, sub + LANES, ... of every
// channel's row (`routed::project_channels`).
//
// The projections meet in threadgroup memory; their join is
// `lib/routed/output_sum.h`'s. Gathered (`routed_output`'s GATHERED,
// `routed::output_project_gathered`), each expert's first choice simdgroup
// projects every choice of that expert in one weight pass, and row 0's shared
// expert simdgroup projects every row in one weight pass, publishing directly
// to `published`.

#define KERNEL_W0 SEISMIC_EXPERT_DOWN
#define KERNEL_W1 SEISMIC_SHARED_DOWN
// The includer defines ROUTED_SOURCE_ROWS, the sources' rows: the entry's
// XS (0 for a resident layer), or SOURCE_ROWS where the entry's layers
// stream.
#include "channels.h"

namespace routed {

template <uint ROWS, uint LANES> constexpr uint output_tile() {
    return (32u / LANES) * ROWS;
}

// The product packets each simdgroup stages at once (`project_channels`).
template <uint LANES> constexpr uint output_stage() {
    constexpr uint COLUMNS = uint(SEISMIC_DIM_F) > uint(SEISMIC_DIM_S) ? uint(SEISMIC_DIM_F) : uint(SEISMIC_DIM_S);
    return stage_packets<uint(SEISMIC_DIM_K) + 1u, LANES, (COLUMNS + 31u) / 32u, false>();
}

// Each simdgroup's region of the projections in threadgroup memory, in
// floats (`projection_region`).
template <uint ROWS, uint LANES> constexpr uint output_region() {
    return projection_region<output_tile<ROWS, LANES>(), output_stage<LANES>()>();
}

// Slot `slot`'s projection of row m, by one simdgroup: choice k = slot < K
// (its down rows of `expert_down`, `decode_source`, `expert_matrix`: a bank
// record of its table or a host tier record of `tier_down` read in place),
// or the shared expert (slot K), into `projected` (TILE channels from
// `first`; its `output_region()` floats, a stage of its product while it
// projects).
template <uint ROWS, uint LANES>
inline void output_project(device const uchar *expert_product, device const uchar *shared_product,
    device const int *routes, device const int *sources, device const ulong *expert_down,
    device const ulong *tier_down, device const uchar *shared_down, ulong m, uint first, uint slot, uint lane,
    threadgroup float *projected, constant ulong *seismic_words) {
    constexpr uint TILE = output_tile<ROWS, LANES>();
    constexpr uint K = uint(SEISMIC_DIM_K);
    const uint rows = uint(SEISMIC_DIM_H);
    if (slot < K) {
        const ulong k = slot;
        const slab::Source source = decode_source(sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
            SEISMIC_SOURCES_STRIDE_2, m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        if (source.none()) {
            // A choice reading no record (the CPU's, or a failed one): a zero
            // projection.
            for (uint channel = lane; channel < TILE; channel += 32)
                projected[channel] = 0.0f;
            return;
        }
        const auto in = activation(expert_product
                + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * Act::bytes,
            0, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = expert_matrix<packets::W0>(Experts{expert_down, tier_down}, source,
            KERNEL_W0_LAYOUT(SEISMIC_DIM_F), SEISMIC_DIM_F);
        // The down weights' operand rows (the product rows, or NVFP4
        // operands), which each simdgroup forms alone.
        const auto x = projection::operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4>::lone_row(
            in, uint(SEISMIC_DIM_F), lane);
        project_channels<packets::W0, ROWS, LANES, uint(SEISMIC_DIM_F), output_stage<LANES>()>(x, down, first, rows,
            uint(SEISMIC_DIM_F), projected, lane);
        return;
    }
    const auto in = activation(shared_product + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * Act::bytes, 0,
        SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
    const auto down = weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0, SEISMIC_DIM_S);
    const auto x = projection::operands<projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4>::lone_row(
        in, uint(SEISMIC_DIM_S), lane);
    project_channels<packets::W1, ROWS, LANES, uint(SEISMIC_DIM_S), output_stage<LANES>()>(x, down, first, rows,
        uint(SEISMIC_DIM_S), projected, lane);
}

// Gathered decode output projections: threadgroup `(group.x, u)` owns unique
// expert `u < worklist->count` (or the shared expert at `u == worklist->count`),
// and all `GATHERED_SUMMED_SIMDGROUPS` (4) simdgroups cooperatively stage the
// expert's rows once into `tg` and split the threadgroup's `4 * TILE` output
// channels (`sg_first = group_first + sg * TILE`). Each slot is published
// directly in A into `published`.
template <uint ROWS, uint LANES, bool ONE = false, typename Published>
inline void output_project_gathered(device const uchar *expert_product, device const uchar *shared_product,
    device const Worklist *worklist, device const ulong *expert_down, device const ulong *tier_down,
    device const uchar *shared_down, uint m_rows, uint u, uint group_first, uint tid, uint sg, uint lane,
    threadgroup GatheredSummedShared<LANES, ONE> &tg, Published published, constant ulong *seismic_words) {
    constexpr uint K = uint(SEISMIC_DIM_K);
    constexpr uint SLOTS = K + 1u;
    constexpr uint H = uint(SEISMIC_DIM_H);
    constexpr uint TILE = output_tile<ROWS, LANES>();
    const uint sg_first = group_first + sg * TILE;
    if constexpr (ONE) {
        constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
        project_gathered_expert<packets::W0, ROWS, LANES, SLOTS, uint(SEISMIC_DIM_F), H, NVFP4, true>(
            expert_product, SEISMIC_EXPERT_PRODUCT_STRIDE_0, SEISMIC_EXPERT_PRODUCT_STRIDE_1,
            SEISMIC_EXPERT_PRODUCT_STRIDE_2, worklist->entries[u], expert_down, tier_down,
            KERNEL_W0_LAYOUT(SEISMIC_DIM_F), sg_first, tid, lane, tg, published);
        return;
    } else {
        const uint multi_count = worklist->multi_count;
        if (u < multi_count) {
            constexpr bool NVFP4 = projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4;
            project_gathered_expert<packets::W0, ROWS, LANES, SLOTS, uint(SEISMIC_DIM_F), H, NVFP4, false>(
                expert_product, SEISMIC_EXPERT_PRODUCT_STRIDE_0, SEISMIC_EXPERT_PRODUCT_STRIDE_1,
                SEISMIC_EXPERT_PRODUCT_STRIDE_2, worklist->entries[worklist->capacity - 1u - u], expert_down,
                tier_down, KERNEL_W0_LAYOUT(SEISMIC_DIM_F), sg_first, tid, lane, tg, published);
            return;
        }
        if (u == multi_count && m_rows > 0) {
            if (tid < m_rows)
                tg.products[tid] = tid * SLOTS + K;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const auto down = weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0, SEISMIC_DIM_S);
            constexpr bool NVFP4 = projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4;
            for (uint base = 0u; base < m_rows; base += 4u) {
                const uint chunk = min(4u, m_rows - base);
                if (base > 0u)
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                const auto in = activation(shared_product + ulong(base) * SEISMIC_SHARED_PRODUCT_STRIDE_0 * Act::bytes,
                    SEISMIC_SHARED_PRODUCT_STRIDE_0, SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
                PROJECTION_FOR_ROWS_24(chunk,
                    (project_channels_published<packets::W1, ROWS, LANES, MAXM, uint(SEISMIC_DIM_S), H, NVFP4>(
                        in, down, chunk, tg.products + base, sg_first, H, uint(SEISMIC_DIM_S), tg.staged, published,
                        tid, lane)));
            }
        }
    }
}

} // namespace routed
