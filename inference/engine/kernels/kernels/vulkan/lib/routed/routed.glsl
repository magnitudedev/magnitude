// Shared pieces of the routed (mixture-of-experts) Vulkan entries built on
// the projection family: the record an expert's weights are read from and
// the live-row count of a grouped block. The counterpart of
// `metal/lib/routed/routed.h` (the grouped-row prologue is
// `projection_grouped`). An expert-stacked weight is a record port, bound as
// its records' address table, and an expert's weights are rows 0.. of its
// record's base (`slab_region(table, record)`).
#include <seismic/slab.glsl>
#include "../projection/projection.glsl"

// Rows of the weight tensor at `base` starting at row `first`.
projection_weights routed_weights(uint64_t base, packets_rows16 geometry, uint64_t first, uint k) {
    return projection_weights(base + first * geometry.stride, geometry, k, 0ul, 1ul);
}

// The weights of record `record` of the expert-stacked weight bound as the
// record table at `table`: its rows from its base.
projection_weights routed_record_weights(uint64_t table, uint64_t record, packets_rows16 geometry, uint k) {
    return routed_weights(slab_region(table, record), geometry, 0ul, k);
}

// Expert `expert`'s weights of an expert-stacked weight: record `slot` of
// the table at `records` where the layer's map holds it for the step
// (`slot` >= 0, the prefill entries' `EXPERTS_SLOT`), else its record by
// index in the table at `experts`.
projection_weights routed_expert_weights(uint64_t records, uint64_t experts, int slot, int expert,
    packets_rows16 geometry, uint k) {
    if (slot >= 0)
        return routed_record_weights(records, uint64_t(slot), geometry, k);
    return routed_record_weights(experts, uint64_t(expert), geometry, k);
}

// The record choice (m, k), of expert `expert`, of a decode expert entry
// reads its weights from (`decode_source`, routed.seismic): the expert
// itself without sources (`rows` 0, a resident layer), else the step's
// snapshot (`sources`, SOURCE_RECORD of its [SOURCE_ROWS, M, K] at strides
// `stride0`..`stride2`): a bank record >= 0, a host tier record t as
// -2 - t, or -1 for none (a choice the CPU computes, or a failed one), of
// which the device computes nothing.
int routed_decode_source(uint64_t sources, uint64_t rows, uint64_t stride0, uint64_t stride1, uint64_t stride2,
    uint64_t m, uint64_t k, int expert) {
    if (rows == 0ul)
        return expert;
    return element_i32_at(sources + (SEISMIC_CONST_SOURCE_RECORD * stride0 + m * stride1 + k * stride2) * 4ul);
}

// The weights a decode choice of source record `source` (`routed_decode_source`,
// not -1) reads of an expert-stacked weight: its bank record of the record
// table at `bank`, or its host tier record -2 - source of the record table at
// `tier`, read in place over the link.
projection_weights routed_source_weights(uint64_t bank, uint64_t tier, int source, packets_rows16 geometry, uint k) {
    return source >= 0 ? routed_record_weights(bank, uint64_t(source), geometry, k)
                       : routed_record_weights(tier, uint64_t(-2 - source), geometry, k);
}

// The gathered decode form's tables (`routed_gather`): ROUTED_GATHER_WORDS
// I32 a choice, from `table` + choice * ROUTED_GATHER_WORDS * 4: the count of
// the choices of the choice's expert where it is its expert's first choice
// in choice order (row-major), else 0; then those choices' rows (each its
// GEMV row's activation row, at most one a row); then their products'
// element offsets. A gathered GEMV reads each chosen expert's records once a
// step, for every row that names it: a host tier record read once per row
// would cost its link bytes again for each.
#define ROUTED_GATHER_WORDS (1u + 2u * uint(SEISMIC_CONST_DECODE_ROWS))

// Writes the tables of the `rows` x `slots` choices of `routes` (strides
// `routes0`, `routes1`) whose products lie at element strides `product0`,
// `product1`, an invocation a choice.
void routed_gather(uint64_t routes, uint64_t routes0, uint64_t routes1, uint rows, uint slots, uint64_t product0,
    uint64_t product1, uint64_t table) {
    for (uint choice = gl_LocalInvocationIndex; choice < rows * slots; choice += gl_WorkGroupSize.x) {
        const int expert = element_i32_at(routes + (uint64_t(choice / slots) * routes0
            + uint64_t(choice % slots) * routes1) * 4ul);
        const uint64_t at = table + uint64_t(choice) * ROUTED_GATHER_WORDS * 4ul;
        bool first = true;
        for (uint c = 0u; c < choice; ++c)
            first = first && element_i32_at(routes + (uint64_t(c / slots) * routes0
                + uint64_t(c % slots) * routes1) * 4ul) != expert;
        uint count = 0u;
        for (uint c = choice; first && c < rows * slots; ++c) {
            const uint64_t m = uint64_t(c / slots), k = uint64_t(c % slots);
            if (element_i32_at(routes + (m * routes0 + k * routes1) * 4ul) == expert) {
                element_i32_put(at + uint64_t(1u + count) * 4ul, int(m));
                element_i32_put(at + uint64_t(1u + uint(SEISMIC_CONST_DECODE_ROWS) + count) * 4ul,
                    int(m * product0 + k * product1));
                ++count;
            }
        }
        element_i32_put(at, int(count));
    }
}

// Zeroes features [first, first + count) of `features` of each of the epilogue's
// `rows` GEMV rows (`out_`'s outputs, `projection_out_row`), by the
// workgroup's invocations: a decode expansion's product rows of a choice
// with no record (`decode_source`: its product is zero).
void routed_zero_features(projection_epilogue out_, uint rows, uint first, uint count, uint features) {
    for (uint i = gl_LocalInvocationIndex; i < rows * count; i += gl_WorkGroupSize.x) {
        const uint n = first + i % count;
        if (n < features)
            element_put(out_.act, out_.y, projection_out_row(out_, i / count) + uint64_t(n) * out_.y1, 0.0);
    }
}

void routed_write_grid(uint64_t grid, uint x, uint y) {
    element_i32_put(grid, int(x));
    element_i32_put(grid + 4ul, int(y));
    element_i32_put(grid + 8ul, 1);
}

// The grids of an expert pass over `rows` grouped rows and `used` blocks of
// `tile_rows` rows (`routed_experts`): a stage workgroup per row, a quantize
// workgroup per block row, and `subtiles` row tiles of each used block by
// the expand's and down's column tiles.
void routed_expert_grids(uint rows, uint used, uint tile_rows, uint subtiles, uint expand_columns, uint down_columns,
    uint64_t stage, uint64_t expand, uint64_t quantize, uint64_t down) {
    routed_write_grid(stage, rows, 1u);
    routed_write_grid(expand, expand_columns, used * subtiles);
    routed_write_grid(quantize, used * tile_rows, 1u);
    routed_write_grid(down, down_columns, used * subtiles);
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
