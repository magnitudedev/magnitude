// Slot and version addressing of the recurrent mixer entries that bind the
// bank version contract (recurrent.seismic, short_conv.seismic,
// state_space.seismic): the slots `segments`, `stop`, `previous_bank`,
// `previous_tape`, `following_bank`, and bank arenas stored in slabs of
// SEISMIC_PARAM_SLAB_BANKS banks. The counterpart of
// `cuda/lib/recurrent/versions.cuh`, `metal/lib/recurrent/versions.h` and
// `cpu/lib/recurrent/versions.rs`.
#include <seismic/element.glsl>
#include <seismic/slab.glsl>

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct versions_slot {
    int lo;
    int hi;
    int stop;
    int source;
    int taped;
    int target;
};

versions_slot versions_slot_of(uint64_t slot) {
    const uint64_t segments = SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS);
    return versions_slot(element_i32_index(segments, slot * SEISMIC_SEGMENTS_STRIDE_0),
        element_i32_index(segments, slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_STOP), slot * SEISMIC_STOP_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_BANK), slot * SEISMIC_PREVIOUS_BANK_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_TAPE), slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0),
        element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_FOLLOWING_BANK), slot * SEISMIC_FOLLOWING_BANK_STRIDE_0));
}

// The index of the slot holding `row`, or B when no slot does: the slots
// partition a prefix of the rows in ascending order.
uint64_t versions_slot_index_of_row(int row) {
    const uint64_t segments = SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS);
    uint64_t low = 0ul;
    uint64_t high = SEISMIC_DIM_B;
    while (low < high) {
        const uint64_t middle = (low + high) / 2ul;
        if (element_i32_index(segments, middle * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1) <= row)
            low = middle + 1ul;
        else
            high = middle;
    }
    return low;
}

// The first row no slot covers.
int versions_covered_end() {
    const uint64_t count = SEISMIC_DIM_B;
    return count == 0ul ? 0
                        : element_i32_index(SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS),
                              (count - 1ul) * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1);
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
int versions_tape_rows(versions_slot slot) {
    return min(int(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Byte address of bank `index` of a slab-stored arena whose banks are `bytes`
// long. The quotient and remainder stay in 32 bits (`slab_row`'s u32 form):
// the NVIDIA driver (580) miscompiles the 64-bit division and remainder of one
// index when both are taken.
uint64_t versions_bank(uint64_t table, int index, uint64_t bytes) {
    return slab_row(table, uint(index), uint(SEISMIC_PARAM_SLAB_BANKS), bytes);
}
