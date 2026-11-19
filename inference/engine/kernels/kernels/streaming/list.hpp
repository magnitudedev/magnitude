// Expert streaming's host-written lists (`list_extent`: the sequence and
// count words, then each entry's from and to words, LIST_* in
// streaming_words.seismic), over the backend's shim (streaming/contract.hpp):
// their count, their entries' items and the grid a copy of them launches on,
// shared by `routed_copy` (streaming/list_copy.hpp) and `routed_list_grid`.
//
// A list lies in host-shared memory, where each read is a round trip over
// the link and small reads hardly overlap, so a group reads a list in as few
// reads as it can: its sequence and count by two threads at once, and its
// entries a chunk of LIST_CHUNK at a time, every thread a word of it, which
// the link carries as a few whole lines. A group's words
// (STREAMING_LIST_WORDS, laid out by streaming_words.seismic's LIST_GROUP_*)
// hold the list's sequence and count, the chunk's first entry and count,
// then the chunk's words. Every thread reads an entry's
// from and to from the chunk (`streaming_list_from`, `streaming_list_to`).
// Group words are stored only between barriers in which no thread reads
// one, so every read the group steers by is the group's alike. The host
// writes every count within its list and every entry within the tensors
// its reader binds (`experts::Words::write_list`), so no reader bounds
// them.

#define STREAMING_LIST_WORDS (SEISMIC_CONST_LIST_GROUP_CHUNK + SEISMIC_CONST_LIST_CHUNK * SEISMIC_CONST_LIST_ENTRY_WORDS)
#define STREAMING_LIST_SEQUENCE STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_HEADER + SEISMIC_CONST_LIST_SEQUENCE)
#define STREAMING_LIST_COUNT STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_HEADER + SEISMIC_CONST_LIST_COUNT)

// Word `word` (SEISMIC_CONST_LIST_FROM or _TO) of entry `entry`, in the
// chunk `streaming_list_chunk` staged.
STREAMING_FN int streaming_list_word(u32 entry, int word, STREAMING_GROUP_PARAM) {
    return STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_CHUNK
        + int((entry - u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST))) * u32(SEISMIC_CONST_LIST_ENTRY_WORDS))
        + word);
}

// The from and to words of entry `entry`, after `streaming_list_chunk` of an
// item of it.
STREAMING_FN int streaming_list_from(u32 entry, STREAMING_GROUP_PARAM) {
    return streaming_list_word(entry, int(SEISMIC_CONST_LIST_FROM), streaming_group);
}

STREAMING_FN int streaming_list_to(u32 entry, STREAMING_GROUP_PARAM) {
    return streaming_list_word(entry, int(SEISMIC_CONST_LIST_TO), streaming_group);
}

// The count of the entries of `list` (word stride `stride`) written for
// step `step`, returned to every thread, its sequence and count read by two
// threads at once; none for a list written for another step. No chunk is
// staged after it: thread 0 clears the chunk's bounds before the barrier,
// which every group's first `streaming_list_chunk` reads after it. Every
// thread calls it.
STREAMING_FN u32 streaming_step_entries(STREAMING_SYSTEM_READ list, u64 stride, int step, STREAMING_GROUP_PARAM,
    u32 tid) {
    if (tid == 0u) {
        STREAMING_LIST_SEQUENCE = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_SEQUENCE * stride));
        STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST) = 0;
        STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_STAGED) = 0;
    } else if (tid == 1u)
        STREAMING_LIST_COUNT = streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride));
    streaming_barrier();
    return u32(STREAMING_LIST_SEQUENCE == step ? STREAMING_LIST_COUNT : 0);
}

// The items of `entries` entries of `parts` parts each that a group visits:
// item i is part i % parts of entry i / parts, the group's items `first`,
// `first + step`, ... in order. For each, every thread calls
// `streaming_list_chunk` with the item, then reads its entry's words.
#define STREAMING_LIST_ITEMS(item, entries, parts, first, step)                                                         \
    for (u32 item = u32(first); item < u32((entries) * (parts)); item += u32(step))

// Item `item`'s entry (of `parts` parts each, of `entries` entries) in the
// group's chunk: an entry past the chunk staged stages the chunk from it,
// at most LIST_CHUNK entries, every one of the group's `threads` threads
// reading words of it at once, between two barriers (every thread has read
// the chunk before any rewrites it, and its words before any reads them).
// Every thread calls it, with the same item, so every thread stages alike.
STREAMING_FN void streaming_list_chunk(STREAMING_SYSTEM_READ list, u64 stride, u32 item, u32 parts, u32 entries,
    STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    const u32 entry = item / parts;
    const u32 first = u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_FIRST));
    const u32 staged = u32(STREAMING_GROUP(SEISMIC_CONST_LIST_GROUP_STAGED));
    if (entry < first || entry >= first + staged) {
        const u32 count = min(entries - entry, u32(SEISMIC_CONST_LIST_CHUNK));
        const u32 words = count * u32(SEISMIC_CONST_LIST_ENTRY_WORDS);
        const u64 base = SEISMIC_CONST_LIST_ENTRIES + SEISMIC_CONST_LIST_ENTRY_WORDS * u64(entry);
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
