// The decode output projections of Qwen's fused routed feed-forward (M <=
// DECODE_ROWS), shared by `routed_output` and `routed_output_project`: threadgroup (x, m)
// owns the TILE output channels of tile x for row m and runs K + 1
// simdgroups side by side: simdgroup k < K projects choice k through its
// expert's down rows, simdgroup K the shared
// expert's down rows. Each lane group (LANES lanes) of a simdgroup owns ROWS
// channels; its lanes own the packets sub, sub + LANES, ... of every
// channel's row (`routed::project_channels`), each simdgroup staging its
// product row in threadgroup memory.
//
// The projections meet in threadgroup memory; their join is
// `lib/routed/output_sum.h`'s.

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
constexpr uint output_stage() {
    constexpr uint COLUMNS = uint(SEISMIC_DIM_F) > uint(SEISMIC_DIM_S) ? uint(SEISMIC_DIM_F) : uint(SEISMIC_DIM_S);
    return stage_packets<uint(SEISMIC_DIM_K) + 1u, (COLUMNS + 31u) / 32u>();
}

// Simdgroup `sg`'s projection: choice k = sg < K of row m (its down rows of
// `expert_down` as bound, `choice_record`, `choice_weights`: a bank record of
// its table or a host tier record of `tier_down` read in place, or a
// resident layer's expert of its stack), or the shared expert (sg = K), into
// `projected` (TILE channels from `first`), staging its product in `staged`
// (`output_stage() * 4` words).
template <uint ROWS, uint LANES, typename STACK>
inline void output_project(device const uchar *expert_product, device const uchar *shared_product,
    device const int *routes, device const int *sources, device const STACK *expert_down,
    device const ulong *tier_down, device const uchar *shared_down, ulong m, uint first, uint sg, uint lane,
    threadgroup uint4 *staged, threadgroup float *projected, constant ulong *seismic_words) {
    constexpr uint TILE = output_tile<ROWS, LANES>();
    constexpr uint K = uint(SEISMIC_DIM_K);
    const uint rows = uint(SEISMIC_DIM_H);
    if (sg < K) {
        const ulong k = sg;
        const int record = choice_record(expert_down, sources, ROUTED_SOURCE_ROWS, SEISMIC_SOURCES_STRIDE_1,
            SEISMIC_SOURCES_STRIDE_2, m, k, routes[m * SEISMIC_ROUTES_STRIDE_0 + k * SEISMIC_ROUTES_STRIDE_1]);
        if (record == -1) {
            // A choice reading no record (the CPU's, or a failed one): a zero
            // projection.
            for (uint channel = lane; channel < TILE; channel += 32)
                projected[channel] = 0.0f;
            return;
        }
        const auto in = activation(expert_product
                + (m * SEISMIC_EXPERT_PRODUCT_STRIDE_0 + k * SEISMIC_EXPERT_PRODUCT_STRIDE_1) * Act::bytes,
            0, SEISMIC_EXPERT_PRODUCT_STRIDE_2, uint(SEISMIC_DIM_F));
        const auto down = choice_weights<packets::W0>(expert_down, tier_down, record, KERNEL_W0_LAYOUT(SEISMIC_DIM_F),
            SEISMIC_DIM_H, SEISMIC_DIM_F);
        // The down weights' operand rows (the product rows, or NVFP4
        // operands), which each simdgroup forms alone.
        const auto x = projection::operands<projection::OperandRecipe<packets::W0_OPERANDS_NVFP4>::NVFP4>::lone_row(
            in, uint(SEISMIC_DIM_F), lane);
        project_channels<packets::W0, ROWS, LANES, uint(SEISMIC_DIM_F), output_stage()>(x, down, first, rows,
            uint(SEISMIC_DIM_F), staged, projected, lane);
        return;
    }
    const auto in = activation(shared_product + m * SEISMIC_SHARED_PRODUCT_STRIDE_0 * Act::bytes, 0,
        SEISMIC_SHARED_PRODUCT_STRIDE_1, uint(SEISMIC_DIM_S));
    const auto down = weights<packets::W1>(shared_down, KERNEL_W1_LAYOUT(SEISMIC_DIM_S), 0, SEISMIC_DIM_S);
    const auto x = projection::operands<projection::OperandRecipe<packets::W1_OPERANDS_NVFP4>::NVFP4>::lone_row(
        in, uint(SEISMIC_DIM_S), lane);
    project_channels<packets::W1, ROWS, LANES, uint(SEISMIC_DIM_S), output_stage()>(x, down, first, rows,
        uint(SEISMIC_DIM_S), staged, projected, lane);
}

} // namespace routed
