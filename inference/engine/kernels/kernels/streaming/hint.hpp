// Gate-ahead hints (`hint` and `hint_selected` in routed.seismic), over the
// backend's shim (streaming/contract.hpp): a layer ahead's choices for the
// decode rows, ranked from its router logits as its routing ranks them (the
// fused form's `rank_row`, lib/routed/route; the general form's,
// lib/routed/select), and each one not named before published to a ring the
// host reads in rank order (every row's first choice, then every row's
// second, and so on), as the choice it was first hinted as
// (`streaming_ring_entry`, its weight the probability its ranking gave it),
// so the ring is best first and its readers read
// ahead in its order (they skip what the layer holds), as a seqlock
// (`streaming_ring_open`, `streaming_ring_close`). A hint never changes a
// bank, a slot map or an output.

// The group words the hints take: each row's K choices in rank order
// (`rows` x K), then the keep flag of each choice in publication order, then
// each choice's weight's f32 bits, as its choice (its ring word the
// magnitude's, `streaming_ring_entry`).
#define STREAMING_HINT_WORDS(rows) (3 * (rows) * SEISMIC_DIM_K)
#define STREAMING_HINT_CHOSEN(row) ((row) * SEISMIC_DIM_K)
#define STREAMING_HINT_KEEP(rows, order) ((rows) * SEISMIC_DIM_K + (order))
#define STREAMING_HINT_WEIGHT(rows, row) (2 * (rows) * SEISMIC_DIM_K + (row) * SEISMIC_DIM_K)

// The group word of the choice published `order`-th: rank `order / rows` of
// row `order % rows`.
STREAMING_FN u32 streaming_hint_choice(u32 rows, u32 order) {
    return (order % rows) * u32(SEISMIC_DIM_K) + order / rows;
}

// A hint or front ring's seqlock (streaming_words.seismic's ring), by one
// thread: the sequence 0 (none, which the host skips) and a system-scope
// fence; then its entries; then the count and the step's guesses' kinds, a
// fence, the step's sequence and a fence. A host read that finds the same
// nonzero sequence before and after its loads read one step's whole list.
STREAMING_FN void streaming_ring_open(STREAMING_SYSTEM_WORDS ring, u64 ring0) {
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_SEQUENCE * ring0), 0);
    streaming_system_fence();
}

// Entry `entry` of the ring: a choice of the pass, its expert, its rank (0
// a row's best), its row and its weight's word (HINT_EXPERT, HINT_RANK,
// HINT_ROW, HINT_WEIGHT). `weight` is the weight's f32 bits; the word is its
// magnitude's (streaming_words.seismic), the sign cleared here, the one
// place every ring's weight is written, so -0.0 publishes as +0 and no word
// is negative.
STREAMING_FN void streaming_ring_entry(STREAMING_SYSTEM_WORDS ring, u64 ring0, int entry, int expert, int rank, int row,
    int weight) {
    const u64 at = SEISMIC_CONST_HINT_ENTRY(u64(entry));
    streaming_store(STREAMING_AT(ring, (at + SEISMIC_CONST_HINT_EXPERT) * ring0), expert);
    streaming_store(STREAMING_AT(ring, (at + SEISMIC_CONST_HINT_RANK) * ring0), rank);
    streaming_store(STREAMING_AT(ring, (at + SEISMIC_CONST_HINT_ROW) * ring0), row);
    streaming_store(STREAMING_AT(ring, (at + SEISMIC_CONST_HINT_WEIGHT) * ring0), weight & 0x7fffffff);
}

// The close, of step `step` (its words of stride `step0`): its guesses'
// kinds and its low word, the one place a ring's header takes them.
STREAMING_FN void streaming_ring_close(STREAMING_SYSTEM_WORDS ring, u64 ring0, int count,
    STREAMING_DEVICE_READ step, u64 step0) {
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_COUNT * ring0), count);
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_GUESSES * ring0),
        streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_GUESSES * step0)));
    streaming_system_fence();
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_SEQUENCE * ring0),
        streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step0)));
    streaming_system_fence();
}

// The group that publishes a distance ahead's hints, once every row's
// choices and their weights' words are in its group words
// (`STREAMING_HINT_CHOSEN`, `STREAMING_HINT_WEIGHT`, written by the
// includer's ranking, each row by one simdgroup): each choice's keep flag,
// then thread 0 publishes for step `step` (its words of stride `step0`).
// Every thread calls it.
STREAMING_FN void streaming_hint(u32 rows, STREAMING_SYSTEM_WORDS ring, u64 ring0,
    STREAMING_DEVICE_READ step, u64 step0, STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    streaming_barrier();
    const u32 total = rows * u32(SEISMIC_DIM_K);
    for (u32 o = tid; o < total; o += threads) {
        const int expert = STREAMING_GROUP(streaming_hint_choice(rows, o));
        bool kept = expert >= 0;
        for (u32 earlier = 0; earlier < o && kept; ++earlier)
            kept = STREAMING_GROUP(streaming_hint_choice(rows, earlier)) != expert;
        STREAMING_GROUP(STREAMING_HINT_KEEP(rows, o)) = kept ? 1 : 0;
    }
    streaming_barrier();
    if (tid != 0)
        return;
    streaming_ring_open(ring, ring0);
    int count = 0;
    for (u32 o = 0; o < total; ++o)
        if (STREAMING_GROUP(STREAMING_HINT_KEEP(rows, o)) != 0) {
            streaming_ring_entry(ring, ring0, count, STREAMING_GROUP(streaming_hint_choice(rows, o)), int(o / rows),
                int(o % rows), STREAMING_GROUP(STREAMING_HINT_WEIGHT(rows, o % rows) + o / rows));
            ++count;
        }
    streaming_ring_close(ring, ring0, count, step, step0);
}
