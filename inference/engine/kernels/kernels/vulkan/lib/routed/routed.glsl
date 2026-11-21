// Shared pieces of the routed (mixture-of-experts) Vulkan entries built on
// the projection family: the record an expert's weights are read from and
// the live-row count of a grouped block. The counterpart of
// `metal/lib/routed/routed.h` (the grouped-row prologue is
// `projection_grouped`). An expert-stacked weight is a record port, bound as
// its records' address table, and an expert's weights are rows 0.. of its
// record's base (`slab_experts_matrix`, the record a `slab_source` names).
#include <seismic/slab.glsl>
#include "../projection/projection.glsl"

// Rows of the weight tensor at `base` starting at row `first`.
projection_weights routed_weights(uint64_t base, packets_rows16 geometry, uint64_t first, uint k) {
    return projection_weights(packets_row(base, geometry, first), geometry, k, 0ul, 1ul, 0u);
}

// The weights of record `record` of the expert-stacked weight bound as the
// record table at `table`: its rows from its base.
projection_weights routed_record_weights(uint64_t table, uint64_t record, packets_rows16 geometry, uint k) {
    return routed_weights(slab_region(table, record), geometry, 0ul, k);
}

// The source choice (m, k), of expert `expert`, of a decode expert entry
// reads its weights from (`decode_source`, routed.seismic): the expert
// itself without sources (`rows` 0, a resident layer), else the step's
// snapshot (`sources`, SOURCE_RECORD of its [SOURCE_ROWS, M, K] at strides
// `stride0`..`stride2`): a bank record, a host tier record (the secondary
// table, read in place over the link), or none (a choice the CPU computes,
// or a failed one), of which the device computes nothing.
slab_source routed_decode_source(uint64_t sources, uint64_t rows, uint64_t stride0, uint64_t stride1,
    uint64_t stride2, uint64_t m, uint64_t k, int expert) {
    if (rows == 0ul)
        return slab_source(expert);
    return slab_source(
        element_i32_at(sources + (SEISMIC_CONST_SOURCE_RECORD * stride0 + m * stride1 + k * stride2) * 4ul));
}

// The weights of an expert's matrix where `source` (not none) names it.
projection_weights routed_expert_matrix(slab_experts experts, slab_source source, packets_rows16 geometry, uint k) {
    return routed_weights(slab_experts_matrix(experts, source), geometry, 0ul, k);
}

// The gathered decode form's tables (`routed_gather`): ROUTED_GATHER_WORDS
// I32 a choice, from `table` + choice * ROUTED_GATHER_WORDS * 4: the count of
// the choices reading the choice's record where it is the first of them in
// choice order (row-major), else 0; then those choices' rows (each its GEMV
// row's activation row, at most one a row); then their products' element
// offsets. A choice reading no record (-1) is its own group. A gathered GEMV
// reads each chosen record once a step, for every row that reads it: a host
// tier record read once per row would cost its link bytes again for each.
#define ROUTED_GATHER_WORDS (1u + 2u * uint(SEISMIC_CONST_VERIFY_ROWS))

// Whether flat choice `choice`, reading `source`, runs in this pass: a
// two-pass entry defines ROUTED_CHOICE_ACTIVE(choice, source) before
// including this file; otherwise every choice runs.
#ifndef ROUTED_CHOICE_ACTIVE
#define ROUTED_CHOICE_ACTIVE(choice, source) true
#endif

// The source choice `choice` of `routes` reads (`routed_decode_source`).
slab_source routed_choice_source(uint64_t routes, uint64_t routes0, uint64_t routes1, uint64_t sources,
    uint64_t source_rows, uint64_t source0, uint64_t source1, uint64_t source2, uint slots, uint choice) {
    const uint64_t m = uint64_t(choice / slots), k = uint64_t(choice % slots);
    return routed_decode_source(sources, source_rows, source0, source1, source2, m, k,
        element_i32_at(routes + (m * routes0 + k * routes1) * 4ul));
}

// The expert choice `choice` of `routes` names.
int routed_choice_expert(uint64_t routes, uint64_t routes0, uint64_t routes1, uint slots, uint choice) {
    const uint64_t m = uint64_t(choice / slots), k = uint64_t(choice % slots);
    return element_i32_at(routes + (m * routes0 + k * routes1) * 4ul);
}

// Writes the tables of the `rows` x `slots` choices of `routes` (strides
// `routes0`, `routes1`; sources as `routed_decode_source`) whose products lie
// at element strides `product0`, `product1`, an invocation a choice. Choices
// group where they name one expert in one record, as CUDA's and Metal's
// gathers do. Only
// active choices (ROUTED_CHOICE_ACTIVE) group: an inactive one has count 0
// and joins no group, so it neither runs nor shadows an active choice of its
// record. The form runs at most VERIFY_ROWS rows, a row's choices distinct
// records, so a choice's count never reaches the bound; the bound (Metal's
// `routed::gather`'s) keeps the tables in bounds where routes break that
// invariant, since Vulkan has no trap.
void routed_gather(uint64_t routes, uint64_t routes0, uint64_t routes1, uint64_t sources, uint64_t source_rows,
    uint64_t source0, uint64_t source1, uint64_t source2, uint rows, uint slots, uint64_t product0,
    uint64_t product1, uint64_t table) {
    for (uint choice = gl_LocalInvocationIndex; choice < rows * slots; choice += gl_WorkGroupSize.x) {
        const slab_source source = routed_choice_source(routes, routes0, routes1, sources, source_rows, source0,
            source1, source2, slots, choice);
        const int record = source.code;
        const int expert = routed_choice_expert(routes, routes0, routes1, slots, choice);
        const uint64_t at = table + uint64_t(choice) * ROUTED_GATHER_WORDS * 4ul;
        bool first = ROUTED_CHOICE_ACTIVE(choice, source);
        for (uint c = 0u; c < choice && record != -1; ++c) {
            const slab_source other = routed_choice_source(routes, routes0, routes1, sources, source_rows, source0,
                source1, source2, slots, c);
            first = first && !(other.code == record
                && routed_choice_expert(routes, routes0, routes1, slots, c) == expert && ROUTED_CHOICE_ACTIVE(c, other));
        }
        uint count = 0u;
        for (uint c = choice; first && c < rows * slots && count < uint(SEISMIC_CONST_VERIFY_ROWS); ++c) {
            const uint64_t m = uint64_t(c / slots), k = uint64_t(c % slots);
            bool joins = c == choice;
            if (!joins && record != -1) {
                const slab_source other = routed_choice_source(routes, routes0, routes1, sources, source_rows,
                    source0, source1, source2, slots, c);
                joins = other.code == record && routed_choice_expert(routes, routes0, routes1, slots, c) == expert
                    && ROUTED_CHOICE_ACTIVE(c, other);
            }
            if (joins) {
                element_i32_put(at + uint64_t(1u + count) * 4ul, int(m));
                element_i32_put(at + uint64_t(1u + uint(SEISMIC_CONST_VERIFY_ROWS) + count) * 4ul,
                    int(m * product0 + k * product1));
                ++count;
            }
        }
        element_i32_put(at, int(count));
    }
}

void routed_gather(uint64_t routes, uint64_t routes0, uint64_t routes1, uint rows, uint slots, uint64_t product0,
    uint64_t product1, uint64_t table) {
    routed_gather(routes, routes0, routes1, 0ul, 0ul, 0ul, 0ul, 0ul, rows, slots, product0, product1, table);
}

// The GEMV rows of choice `choice` in the tables at `table`
// (`routed_gather`): their count (0 where it is not its expert's first
// choice), the table of their activation rows and that of their products'
// element offsets.
struct routed_gathered_rows {
    uint count;
    uint64_t rows;
    uint64_t outputs;
};

routed_gathered_rows routed_gathered(uint64_t table, uint choice) {
    const uint64_t at = table + uint64_t(choice) * ROUTED_GATHER_WORDS * 4ul;
    return routed_gathered_rows(uint(element_i32_at(at)), at + 4ul,
        at + (1ul + uint64_t(SEISMIC_CONST_VERIFY_ROWS)) * 4ul);
}

// Zeroes features [first, first + count) of `features` of each of the epilogue's
// `rows` GEMV rows (`out_`'s outputs, `projection_out_row`), by the
// workgroup's invocations: a decode expansion's product rows of a choice
// with no record (`decode_source`: its product is zero).
void routed_zero_features(projection_epilogue out_, uint rows, uint first, uint count, uint features) {
    for (uint i = gl_LocalInvocationIndex; i < rows * count; i += gl_WorkGroupSize.x) {
        const uint n = first + i % count;
        if (n < features)
            out_put(out_.y, projection_out_row(out_, i / count) + uint64_t(n) * out_.y1, 0.0);
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
    routed_write_grid(quantize, tile_rows, used);
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
