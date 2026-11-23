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

// Compact unique-expert worklist emitted once by a 1-workgroup `*_gather`
// launch for the gathered decode forms (`choice` and `summed`), enabling
// hardware indirect dispatch (`vkCmdDispatchIndirect`) over only the active
// unique `(expert, source)` groups.
// Layout: 16-byte header (`count` at +0), then `count` entries of
// `ROUTED_WORK_WORDS` 32-bit words each:
//   [0] expert, [1] source, [2] count, [3] pad,
//   [4 .. 4 + VERIFY_ROWS)              rows[r],
//   [4 + VERIFY_ROWS .. +2*VERIFY_ROWS) outputs[r].
#define ROUTED_WORK_WORDS (4u + 2u * uint(SEISMIC_CONST_VERIFY_ROWS))
#define ROUTED_WORK_BYTES (uint64_t(ROUTED_WORK_WORDS) * 4ul)

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

void routed_write_grid(uint64_t grid, uint x, uint y) {
    element_i32_put(grid, int(x));
    element_i32_put(grid + 4ul, int(y));
    element_i32_put(grid + 8ul, 1);
}

// Row/output encoding modes for `routed_build_worklist`:
//   0 (choice): rows[r] = m, outputs[r] = m * out0 + k * out1
//   1 (summed contiguous): rows[r] = m * slots + k, outputs[r] = (m << 16) | k
//   2 (summed strided):    rows[r] = m * out0 + k * out1, outputs[r] = (m << 16) | k
void routed_store_work_item(uint64_t entry_at, uint idx, uint m, uint k, uint slots, uint64_t out0, uint64_t out1,
    uint mode) {
    int row_val;
    int out_val;
    if (mode == 0u) {
        row_val = int(m);
        out_val = int(uint64_t(m) * out0 + uint64_t(k) * out1);
    } else {
        row_val = mode == 1u ? int(m * slots + k) : int(uint64_t(m) * out0 + uint64_t(k) * out1);
        out_val = int((m << 16u) | k);
    }
    element_i32_put(entry_at + uint64_t(4u + idx) * 4ul, row_val);
    element_i32_put(entry_at + uint64_t(4u + uint(SEISMIC_CONST_VERIFY_ROWS) + idx) * 4ul, out_val);
}

// Builds the compact unique-expert worklist at `worklist` and writes the
// indirect dispatch grid `(x_groups, unique_count + extra_y, 1)` to
// `gather_grid`. Requires 64-invocation workgroup with 4096 bytes of shared
// memory (`seismic_shared_i32[0..1023]`).
void routed_build_worklist(uint64_t routes, uint64_t routes0, uint64_t routes1, uint64_t sources, uint64_t source_rows,
    uint64_t source0, uint64_t source1, uint64_t source2, uint rows, uint slots, uint64_t out0, uint64_t out1,
    uint mode, uint x_groups, uint extra_y, uint64_t worklist, uint64_t gather_grid) {
    const uint total = rows * slots;
    const uint tid = gl_LocalInvocationIndex;
    if (total <= 256u) {
        for (uint c = tid; c < total; c += gl_WorkGroupSize.x) {
            const int ex = routed_choice_expert(routes, routes0, routes1, slots, c);
            const slab_source src = routed_choice_source(routes, routes0, routes1, sources, source_rows, source0,
                source1, source2, slots, c);
            seismic_shared_i32[c] = ex;
            seismic_shared_i32[256u + c] = src.code;
            seismic_shared_i32[768u + c] = ROUTED_CHOICE_ACTIVE(c, src) ? 1 : 0;
        }
        barrier();
        for (uint c = tid; c < total; c += gl_WorkGroupSize.x) {
            bool first = seismic_shared_i32[768u + c] != 0;
            const int ex = seismic_shared_i32[c];
            const int rec = seismic_shared_i32[256u + c];
            if (first && rec != -1) {
                for (uint j = 0u; j < c; ++j) {
                    if (seismic_shared_i32[768u + j] != 0 && seismic_shared_i32[256u + j] == rec
                        && seismic_shared_i32[j] == ex) {
                        first = false;
                        break;
                    }
                }
            }
            seismic_shared_i32[512u + c] = first ? 1 : 0;
        }
        barrier();
        if (tid == 0u) {
            uint unique_count = 0u;
            for (uint c = 0u; c < total; ++c) {
                const int flag = seismic_shared_i32[512u + c];
                seismic_shared_i32[512u + c] = flag != 0 ? int(unique_count) : -1;
                if (flag != 0)
                    ++unique_count;
            }
            element_i32_put(worklist, int(unique_count));
            routed_write_grid(gather_grid, x_groups, unique_count + extra_y);
        }
        barrier();
        for (uint c = tid; c < total; c += gl_WorkGroupSize.x) {
            const int slot_idx = seismic_shared_i32[512u + c];
            if (slot_idx < 0)
                continue;
            const int ex = seismic_shared_i32[c];
            const int rec = seismic_shared_i32[256u + c];
            const uint64_t at = worklist + 16ul + uint64_t(slot_idx) * ROUTED_WORK_BYTES;
            element_i32_put(at, ex);
            element_i32_put(at + 4ul, rec);
            element_i32_put(at + 12ul, 0);
            uint count = 0u;
            if (rec == -1) {
                routed_store_work_item(at, 0u, c / slots, c % slots, slots, out0, out1, mode);
                count = 1u;
            } else {
                for (uint j = c; j < total && count < uint(SEISMIC_CONST_VERIFY_ROWS); ++j) {
                    if (j == c || (seismic_shared_i32[768u + j] != 0 && seismic_shared_i32[256u + j] == rec
                            && seismic_shared_i32[j] == ex)) {
                        routed_store_work_item(at, count, j / slots, j % slots, slots, out0, out1, mode);
                        ++count;
                    }
                }
            }
            element_i32_put(at + 8ul, int(count));
        }
        return;
    }
    if (tid == 0u) {
        uint unique_count = 0u;
        for (uint c = 0u; c < total; ++c) {
            const slab_source src = routed_choice_source(routes, routes0, routes1, sources, source_rows, source0,
                source1, source2, slots, c);
            const int rec = src.code;
            const int ex = routed_choice_expert(routes, routes0, routes1, slots, c);
            bool first = ROUTED_CHOICE_ACTIVE(c, src);
            for (uint j = 0u; first && j < c && rec != -1; ++j) {
                const slab_source other = routed_choice_source(routes, routes0, routes1, sources, source_rows,
                    source0, source1, source2, slots, j);
                if (other.code == rec && routed_choice_expert(routes, routes0, routes1, slots, j) == ex
                    && ROUTED_CHOICE_ACTIVE(j, other))
                    first = false;
            }
            if (!first)
                continue;
            const uint64_t at = worklist + 16ul + uint64_t(unique_count) * ROUTED_WORK_BYTES;
            element_i32_put(at, ex);
            element_i32_put(at + 4ul, rec);
            element_i32_put(at + 12ul, 0);
            uint count = 0u;
            if (rec == -1) {
                routed_store_work_item(at, 0u, c / slots, c % slots, slots, out0, out1, mode);
                count = 1u;
            } else {
                for (uint j = c; j < total && count < uint(SEISMIC_CONST_VERIFY_ROWS); ++j) {
                    bool joins = j == c;
                    if (!joins) {
                        const slab_source other = routed_choice_source(routes, routes0, routes1, sources, source_rows,
                            source0, source1, source2, slots, j);
                        joins = other.code == rec && routed_choice_expert(routes, routes0, routes1, slots, j) == ex
                            && ROUTED_CHOICE_ACTIVE(j, other);
                    }
                    if (joins) {
                        routed_store_work_item(at, count, j / slots, j % slots, slots, out0, out1, mode);
                        ++count;
                    }
                }
            }
            element_i32_put(at + 8ul, int(count));
            ++unique_count;
        }
        element_i32_put(worklist, int(unique_count));
        routed_write_grid(gather_grid, x_groups, unique_count + extra_y);
    }
}

void routed_build_worklist(uint64_t routes, uint64_t routes0, uint64_t routes1, uint rows, uint slots, uint64_t out0,
    uint64_t out1, uint mode, uint x_groups, uint extra_y, uint64_t worklist, uint64_t gather_grid) {
    routed_build_worklist(routes, routes0, routes1, 0ul, 0ul, 0ul, 0ul, 0ul, rows, slots, out0, out1, mode, x_groups,
        extra_y, worklist, gather_grid);
}

uint routed_worklist_count(uint64_t worklist) {
    return uint(element_i32_at(worklist));
}

struct routed_work_entry {
    int expert;
    slab_source source;
    uint count;
    uint64_t rows;
    uint64_t outputs;
};

routed_work_entry routed_worklist_entry(uint64_t worklist, uint u) {
    const uint64_t at = worklist + 16ul + uint64_t(u) * ROUTED_WORK_BYTES;
    return routed_work_entry(
        element_i32_at(at),
        slab_source(element_i32_at(at + 4ul)),
        uint(element_i32_at(at + 8ul)),
        at + 16ul,
        at + (4ul + uint64_t(SEISMIC_CONST_VERIFY_ROWS)) * 4ul);
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
