// Slot and version addressing of the recurrent mixer entries that bind the
// bank version contract (recurrent.seismic, short_conv.seismic,
// state_space.seismic): the slots `segments`, `stop`, `previous_bank`,
// `previous_tape`, `following_bank`, and bank arenas stored in slabs of
// SEISMIC_PARAM_SLAB_BANKS banks. The counterpart of
// `metal/lib/recurrent/versions.h`, `vulkan/lib/recurrent/versions.glsl` and
// `cpu/lib/recurrent/versions.rs`.
#pragma once

#include <seismic/slab.cuh>

namespace versions {

typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned char u8;

// One slot's rows [lo, hi), its publication row count, the version it reads
// (bank `source` advanced by its first `taped` tape rows) and its successor.
struct Slot {
    int lo;
    int hi;
    int stop;
    int source;
    int taped;
    int target;
};

// The slot arguments of the entry, in contract order.
struct Slots {
    const int *segments;
    const int *stop;
    const int *previous_bank;
    const int *previous_tape;
    const int *following_bank;
};

#define VERSIONS_SLOTS()                                                                     \
    versions::Slots {                                                                        \
        reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SEGMENTS)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_STOP)),                 \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_BANK)),        \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_PREVIOUS_TAPE)),        \
            reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_FOLLOWING_BANK))        \
    }

__device__ __forceinline__ Slot slot_of(const Slots &slots, u64 slot, const seismic_words_t &seismic_words_value) {
    return Slot{slots.segments[slot * SEISMIC_SEGMENTS_STRIDE_0],
                slots.segments[slot * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1],
                slots.stop[slot * SEISMIC_STOP_STRIDE_0],
                slots.previous_bank[slot * SEISMIC_PREVIOUS_BANK_STRIDE_0],
                slots.previous_tape[slot * SEISMIC_PREVIOUS_TAPE_STRIDE_0],
                slots.following_bank[slot * SEISMIC_FOLLOWING_BANK_STRIDE_0]};
}

// The index of the slot holding `row`, or B when no slot does: the slots
// partition a prefix of the rows in ascending order.
__device__ __forceinline__ u64 slot_index_of_row(const Slots &slots, int row,
                                                 const seismic_words_t &seismic_words_value) {
    u64 low = 0;
    u64 high = SEISMIC_DIM_B;
    while (low < high) {
        const u64 middle = (low + high) / 2;
        if (slots.segments[middle * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1] <= row)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

// The first row no slot covers.
__device__ __forceinline__ int covered_end(const Slots &slots, const seismic_words_t &seismic_words_value) {
    const u64 count = SEISMIC_DIM_B;
    return count == 0 ? 0 : slots.segments[(count - 1) * SEISMIC_SEGMENTS_STRIDE_0 + SEISMIC_SEGMENTS_STRIDE_1];
}

// Rows the slot records in its successor's tape: those after the stop row, at
// most T.
__device__ __forceinline__ int tape_rows(const Slot &slot, const seismic_words_t &seismic_words_value) {
    return min(static_cast<int>(SEISMIC_DIM_T), slot.hi - slot.lo - slot.stop);
}

// Bank `index` of a slab-stored arena whose banks are `bytes` long.
__device__ __forceinline__ u8 *bank(const u32 *table, int index, u64 bytes,
                                    const seismic_words_t &seismic_words_value) {
    return slab::row(table, static_cast<u64>(index), SEISMIC_PARAM_SLAB_BANKS, bytes);
}

} // namespace versions
