// Slot and version addressing of the recurrent mixer entries that bind the
// bank version contract (recurrent.seismic, short_conv.seismic,
// state_space.seismic): the slots `segments`, `stop`, `previous_bank`,
// `previous_tape`, `following_bank`, and bank arenas stored in slabs of
// SEISMIC_PARAM_SLAB_BANKS banks. The counterpart of
// `cuda/lib/recurrent/versions.cuh`, `vulkan/lib/recurrent/versions.glsl` and
// `cpu/lib/recurrent/versions.rs`.

#include <seismic/slab.h>

namespace versions {

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct Slot {
    long lo;
    long hi;
    long stop;
    ulong source;
    long taped;
    ulong target;
};

// The slot arguments of the entry, in contract order.
struct Slots {
    device const int *segments;
    device const int *stop;
    device const int *previous_bank;
    device const int *previous_tape;
    device const int *following_bank;
};

inline Slot slot_of(Slots slots, ulong slot, constant ulong *seismic_words) {
    Slot result;
    result.lo = slots.segments[slot * SEISMIC_SEGMENTS_STRIDE_0];
    result.hi = slots.segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1];
    result.stop = slots.stop[slot * SEISMIC_STOP_STRIDE_0];
    result.source = ulong(slots.previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0]);
    result.taped = slots.previous_tape[slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0];
    result.target = ulong(slots.following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]);
    return result;
}

// The end of slot `slot`'s rows.
inline long slot_end(Slots slots, ulong slot, constant ulong *seismic_words) {
    return slots.segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1];
}

// The index of the slot holding `row`, or B when no slot does: the slots
// partition a prefix of the rows in ascending order.
inline ulong slot_index_of_row(Slots slots, long row, constant ulong *seismic_words) {
    ulong low = 0;
    ulong high = SEISMIC_DIM_B;
    while (low < high) {
        const ulong middle = (low + high) / 2;
        if (slot_end(slots, middle, seismic_words) <= row)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
inline long tape_rows(Slot slot, constant ulong *seismic_words) {
    return metal::min(long(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Bank `index` of a slab-stored arena whose banks are `stride` elements.
template <typename T>
inline device T *bank(device const ulong *table, ulong index, ulong stride, constant ulong *seismic_words) {
    return slab::row<T>(table, index, ulong(SEISMIC_PARAM_SLAB_BANKS), stride);
}

} // namespace versions
