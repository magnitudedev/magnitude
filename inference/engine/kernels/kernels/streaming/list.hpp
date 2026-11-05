// Expert streaming's host-written lists (`list_extent`: the sequence and
// count words, then each entry's from and to words, LIST_* in
// streaming_words.seismic), over the backend's shim (streaming/contract.hpp):
// their count, their entries' items and the grid a copy of them launches on,
// shared by `routed_copy`, `routed_move` (streaming/list_copy.hpp),
// `routed_convert` and `routed_list_grid`.
//
// A list lies in host-shared memory, where each read is a round trip over
// the link and small reads hardly overlap, so a group reads a list in as few
// reads as it can: its sequence and count by two threads at once, and its
// entries a chunk of LIST_CHUNK at a time, every thread a word of it, which
// the link carries as a few whole lines. A group's words
// (STREAMING_LIST_WORDS, laid out by streaming_words.seismic's LIST_GROUP_*)
// hold the item's pair, the chunk's first entry and count, then the chunk's
// words. The host writes every count within its list and every entry
// within the tensors its reader binds (`experts::Words::write_list`), so no
// reader bounds them. The pair is written and read only through
// `STREAMING_LIST_FROM` and `STREAMING_LIST_TO` (`streaming_list_from`,
// `streaming_list_to`).

#define STREAMING_LIST_WORDS (SEISMIC_CONST_LIST_GROUP_CHUNK + SEISMIC_CONST_LIST_CHUNK * SEISMIC_CONST_LIST_ENTRY_WORDS)
#define STREAMING_LIST_FROM STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_PAIR + SEISMIC_CONST_LIST_FROM)
#define STREAMING_LIST_TO STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_PAIR + SEISMIC_CONST_LIST_TO)

// The from and to words of the entry `streaming_list_pair` last read.
STREAMING_FN int streaming_list_from(STREAMING_GROUP_PARAM) {
    return STREAMING_LIST_FROM;
}

STREAMING_FN int streaming_list_to(STREAMING_GROUP_PARAM) {
    return STREAMING_LIST_TO;
}

// No chunk staged, by thread 0, which every group's first pair reads after
// its barrier.
STREAMING_FN void streaming_list_unstaged(STREAMING_GROUP_PARAM, u32 tid) {
    if (tid == 0u) {
        STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST) = 0;
        STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_STAGED) = 0;
    }
}

// The count of `list`'s entries (word stride `stride`), returned to every
// thread. Every thread calls it.
STREAMING_FN u32 streaming_list_entries(STREAMING_SYSTEM_READ list, u64 stride, STREAMING_GROUP_PARAM, u32 tid) {
    // The pair's words hold the count until the first pair.
    if (tid == 0u)
        STREAMING_LIST_TO = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride));
    streaming_barrier();
    const int entries = STREAMING_LIST_TO;
    streaming_list_unstaged(streaming_group, tid);
    return u32(entries);
}

// `streaming_list_entries` of a list written for step `step`, its sequence
// and count read by two threads at once; none for a list written for
// another step.
STREAMING_FN u32 streaming_step_entries(STREAMING_SYSTEM_READ list, u64 stride, int step, STREAMING_GROUP_PARAM,
    u32 tid) {
    // The pair's words hold the sequence and the count until the first pair.
    if (tid == 0u)
        STREAMING_LIST_FROM = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_SEQUENCE * stride));
    else if (tid == 1u)
        STREAMING_LIST_TO = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride));
    streaming_barrier();
    const int entries = STREAMING_LIST_FROM == step ? STREAMING_LIST_TO : 0;
    streaming_list_unstaged(streaming_group, tid);
    return u32(entries);
}

// The items of `entries` entries of `parts` parts each that a group visits:
// item i is part i % parts of entry i / parts, the group's items `first`,
// `first + step`, ... in order. For each, every thread calls
// `streaming_list_pair` with the item, then reads its entry's pair.
#define STREAMING_LIST_ITEMS(item, entries, parts, first, step)                                                         \
    for (u32 item = u32(first); item < u32((entries) * (parts)); item += u32(step))

// The from and to words of item `item`'s entry (of `parts` parts each, of
// `entries` entries), in the group's pair (`STREAMING_LIST_FROM`,
// `STREAMING_LIST_TO`), after a barrier. An entry past the chunk staged
// stages the chunk from it, at most LIST_CHUNK entries, every one of the
// group's `threads` threads reading words of it at once. Every thread
// calls it, with the same item, so every thread stages alike.
STREAMING_FN void streaming_list_pair(STREAMING_SYSTEM_READ list, u64 stride, u32 item, u32 parts, u32 entries,
    STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    const u32 entry = item / parts;
    streaming_barrier();
    const u32 first = u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST));
    const u32 staged = u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_STAGED));
    if (entry < first || entry >= first + staged) {
        const u32 count = min(entries - entry, u32(SEISMIC_CONST_LIST_CHUNK));
        const u32 words = count * u32(SEISMIC_CONST_LIST_ENTRY_WORDS);
        const u64 base = SEISMIC_CONST_LIST_ENTRIES + SEISMIC_CONST_LIST_ENTRY_WORDS * u64(entry);
        // Every thread has read the chunk's bounds before any rewrites them.
        streaming_barrier();
        for (u32 word = tid; word < words; word += threads)
            STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_CHUNK + int(word)) =
                streaming_load(STREAMING_AT(list, (base + u64(word)) * stride));
        if (tid == 0u) {
            STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST) = int(entry);
            STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_STAGED) = int(count);
        }
        streaming_barrier();
    }
    if (tid == 0u) {
        const int at = SEISMIC_CONST_LIST_GROUP_CHUNK
            + int((entry - u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST))) * u32(SEISMIC_CONST_LIST_ENTRY_WORDS));
        STREAMING_LIST_FROM = STREAMING_GROUP(at + int(SEISMIC_CONST_LIST_FROM));
        STREAMING_LIST_TO = STREAMING_GROUP(at + int(SEISMIC_CONST_LIST_TO));
    }
    streaming_barrier();
}

// A host-written list staged into device words `staged` (word stride
// `staged_stride`, the list's layout) by every thread of a group (`tid` of
// `threads`): its sequence, its count of step `step`'s entries (none for a
// list written for another step) and those entries' words. Every group of the launches after it reads the device's copy: a
// list read from host memory by every group costs a link round trip per
// group, reads that queue at the host behind each other (an RTX 3090,
// 2026-11-02: ~0.4 ms a 512-group copy). It fences first, so it reads what
// the host wrote before the park its launch follows.
STREAMING_FN void streaming_list_stage(STREAMING_SYSTEM_READ list, u64 stride, int step, STREAMING_DEVICE_WORDS staged,
    u64 staged_stride, u32 tid, u32 threads) {
    streaming_system_fence();
    const int sequence = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_SEQUENCE * stride));
    const int entries = sequence == step ? streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride)) : 0;
    if (tid == 0u) {
        streaming_device_store(STREAMING_AT(staged, SEISMIC_CONST_LIST_SEQUENCE * staged_stride), sequence);
        streaming_device_store(STREAMING_AT(staged, SEISMIC_CONST_LIST_COUNT * staged_stride), entries);
    }
    const u32 words = u32(entries) * u32(SEISMIC_CONST_LIST_ENTRY_WORDS);
    for (u32 word = tid; word < words; word += threads)
        streaming_device_store(STREAMING_AT(staged, (SEISMIC_CONST_LIST_ENTRIES + u64(word)) * staged_stride),
            streaming_load(STREAMING_AT(list, (SEISMIC_CONST_LIST_ENTRIES + u64(word)) * stride)));
}

// The launch grid over a list's entries (`routed_list_grid`), at `grid`
// (word stride `grid_stride`): its row of `grid[0]` blocks, which its host
// wrote once, while the step's list has entries; no row for a list of none
// or written for another step. One thread writes its rows, after a
// system-scope fence, so it reads what the host wrote before the park the
// launch follows.
STREAMING_FN void streaming_list_grid(STREAMING_SYSTEM_READ list, u64 stride, int step, STREAMING_DEVICE_WORDS grid,
    u64 grid_stride) {
    streaming_system_fence();
    const int entries = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_SEQUENCE * stride)) == step
        ? streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride))
        : 0;
    streaming_device_store(STREAMING_AT(grid, grid_stride), entries > 0 ? 1 : 0);
    streaming_device_store(STREAMING_AT(grid, 2 * grid_stride), 1);
}
