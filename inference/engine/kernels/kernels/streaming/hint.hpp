// Gate-ahead hints (`hint` and `hint_selected` in routed.seismic), over the
// backend's shim (streaming/contract.hpp): a layer ahead's choices for the
// decode rows, ranked from its router logits as its routing ranks them (the
// fused form's `rank_row`, lib/routed/route; the general form's,
// lib/routed/select), and each one not named before published to a ring the
// host reads in rank order (every row's first choice, then every row's
// second, and so on), with the rank of the choice it was hinted as
// (HINT_EXPERT, HINT_RANK), so the ring is best first and its readers read
// ahead in its order (they skip what the layer holds),
// as a seqlock: the sequence 0 (none, which the host skips), a system-scope
// fence, the experts and the count, a fence, then the step's sequence. A
// host read that finds the same nonzero sequence before and after its loads
// read one step's whole list. A hint never changes a bank, a slot map or an
// output.

// The group words the hints take: each row's K choices in rank order
// (`rows` x K), then the keep flag of each choice in publication order.
#define STREAMING_HINT_WORDS(rows) (2 * (rows) * SEISMIC_DIM_K)
#define STREAMING_HINT_CHOSEN(row) ((row) * SEISMIC_DIM_K)
#define STREAMING_HINT_KEEP(rows, order) ((rows) * SEISMIC_DIM_K + (order))

// The group word of the choice published `order`-th: rank `order / rows` of
// row `order % rows`.
STREAMING_FN u32 streaming_hint_choice(u32 rows, u32 order) {
    return (order % rows) * u32(SEISMIC_DIM_K) + order / rows;
}

// The group that publishes a distance ahead's hints, once every row's
// choices are in its group words (`STREAMING_HINT_CHOSEN`, written by the
// includer's ranking, each row by one simdgroup): each choice's keep flag,
// then thread 0 publishes. Every thread calls it.
STREAMING_FN void streaming_hint(u32 rows, STREAMING_SYSTEM_WORDS ring, u64 ring0,
    int sequence, STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
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
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_SEQUENCE * ring0), 0);
    streaming_system_fence();
    int count = 0;
    for (u32 o = 0; o < total; ++o)
        if (STREAMING_GROUP(STREAMING_HINT_KEEP(rows, o)) != 0) {
            const u64 entry = SEISMIC_CONST_RING_EXPERTS + u64(SEISMIC_CONST_HINT_ENTRY_WORDS) * u64(count);
            streaming_store(STREAMING_AT(ring, (entry + SEISMIC_CONST_HINT_EXPERT) * ring0),
                STREAMING_GROUP(streaming_hint_choice(rows, o)));
            streaming_store(STREAMING_AT(ring, (entry + SEISMIC_CONST_HINT_RANK) * ring0), int(o / rows));
            ++count;
        }
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_COUNT * ring0), count);
    streaming_system_fence();
    streaming_store(STREAMING_AT(ring, SEISMIC_CONST_RING_SEQUENCE * ring0), sequence);
    streaming_system_fence();
}
