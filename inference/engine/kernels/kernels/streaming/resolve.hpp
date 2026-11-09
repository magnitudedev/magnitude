// Expert streaming's decode resolve and assignment (`resolve` and `assign`
// in routed.seismic), shared by every streamed route, over the backend's
// shim (streaming/contract.hpp): after its selection, the group that holds
// every row's routes resolves each expert's source once, on its first
// choice, into the step's snapshot (`sources`, which every later entry of
// the layer reads), every choice of the expert taking it:
// - the bank record its layer's map holds for the step (`streaming_held`
//   over the group's map `map` [GM, MAP_WORDS], the layer's expert e at row
//   `home + e`, its home `policy[POLICY_HOME]`);
// - else the host tier record its tier map names (`tier`, `tiered` 0
//   without a tier) as -2 - record;
// - else none (-1, absent).
// Beside it each thread writes its choices and their sources as resolved
// into the step's routing record (`routing`'s row q mod FLIGHT, sources
// from word RING_EXPERTS + `width`), and thread 0 its count and sequence:
// host-shared stores, since the host reads the record at the step's park
// (the records its entries may still read, `decode::parked_reads`), so the
// publication barrier orders them before the request.
//
// Across a link (`tiered`), at a layer with a CPU site (`sited`, as its
// graph encodes it: a layer without one reads every miss in place), the
// group then assigns the layer's missed experts (tier-held or absent)
// between the GPU in place and the CPU by the layer's policy lines, the one
// cost function's (streaming_words.seismic's assignment). The group first
// takes the policy words into group memory, each word one thread's load, so
// the host-shared words cross the link once a layer; each thread marks its
// choices' rows (their expert's, on its first choice); thread 0 then
// assigns. Every choice of a CPU expert is sourced
// none for the GPU (SOURCE_RECORD -1) and its tier record for the CPU
// (SOURCE_SITE t, or -2 while absent: the tier wait names it).
//
// On an absent choice thread 0 then requests the step's slot of the layer's
// file table (`file`, its slot the routing record's, q mod `flight`), after
// every thread's stores (`streaming_publish_barrier`): each choice (an
// absent expert as itself, any other as -1 - expert), the count, then the
// request (`streaming_request_slot`). It does not wait.
//
// A whole group's route (`whole`) sources each choice from its home record,
// `policy[POLICY_HOME]` + expert: every expert is at home, nothing misses,
// and it writes neither the routing record nor the grids, which nothing
// reads of a whole layer.

// The group words a resolve of `choices` choices takes: the missing flag;
// then each choice's expert, its source as resolved, its CPU site (-1: the
// GPU's) and its rows (its expert's, on the expert's first choice; 0 on any
// other); the policy words.
#define STREAMING_RESOLVE_WORDS(choices) (1 + 4 * (choices) + SEISMIC_CONST_POLICY_WORDS)
#define STREAMING_RESOLVE_EXPERT(choice) (1 + (choice))
#define STREAMING_RESOLVE_SOURCE(choices, choice) (1 + (choices) + (choice))
#define STREAMING_RESOLVE_SITE(choices, choice) (1 + 2 * (choices) + (choice))
#define STREAMING_RESOLVE_ROWS(choices, choice) (1 + 3 * (choices) + (choice))
#define STREAMING_RESOLVE_POLICY(choices, word) (1 + 4 * (choices) + (word))

// A choice's word of a file request (its entry's FILE_CHOICE): an absent expert as
// itself, any other as -1 - expert.
STREAMING_FN int streaming_request_choice(int expert, bool absent) {
    return absent ? expert : -1 - expert;
}

// Choice c's expert's first choice, in choice order.
STREAMING_FN u32 streaming_first_choice(u32 c, STREAMING_GROUP_PARAM) {
    const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
    for (u32 earlier = 0; earlier < c; ++earlier)
        if (STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(earlier)) == expert)
            return earlier;
    return c;
}

// A policy word from group memory, as a u64 line.
STREAMING_FN u64 streaming_line(u32 choices, int word, STREAMING_GROUP_PARAM) {
    return u64(STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, word)));
}

// The group's loads for the assignment, by every thread: the policy words,
// then each thread's choices' rows.
STREAMING_FN void streaming_assign_loads(u32 choices, STREAMING_SYSTEM_READ policy, STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    for (u32 word = tid; word < u32(SEISMIC_CONST_POLICY_WORDS); word += threads)
        STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, word)) =
            streaming_load(STREAMING_AT(policy, word * SEISMIC_POLICY_STRIDE_0));
    for (u32 c = tid; c < choices; c += threads) {
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        int rows = 0;
        for (u32 other = 0; other < choices; ++other)
            rows += STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(other)) == expert ? 1 : 0;
        STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)) = streaming_first_choice(c, streaming_group) == c ? rows : 0;
    }
}

// The assignment, by thread 0, over the group's resolved sources and rows
// (streaming_words.seismic's): the GPU's queue from its distinct hit
// experts' resident lines, the CPU's from zero; each distinct missed
// expert, by rows (most first, from `most`, the step's rows), ties in
// choice order, to the CPU where it ends there no later than in place, else
// to the GPU in place. Its site is written on its first choice; the snapshot
// gives every choice its first's. While POLICY_PRICED is clear it assigns
// nothing: every miss stays the GPU's in place.
STREAMING_FN void streaming_assign(u32 choices, u32 most, STREAMING_GROUP_PARAM) {
    if (STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, SEISMIC_CONST_POLICY_PRICED)) == 0)
        return;
    const u64 hit = streaming_line(choices, SEISMIC_CONST_POLICY_HIT_NS, streaming_group);
    const u64 hit_row = streaming_line(choices, SEISMIC_CONST_POLICY_HIT_ROW_NS, streaming_group);
    const u64 link = streaming_line(choices, SEISMIC_CONST_POLICY_LINK_NS, streaming_group);
    const u64 link_row = streaming_line(choices, SEISMIC_CONST_POLICY_LINK_ROW_NS, streaming_group);
    u64 gpu = 0;
    for (u32 c = 0; c < choices; ++c) {
        const u64 rows = u64(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)));
        if (rows != 0 && STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) >= 0)
            gpu += max(hit, rows * hit_row);
    }
    u64 cpu = 0;
    bool served = false;
    for (u32 wanted = most; wanted > 0; --wanted)
        for (u32 c = 0; c < choices; ++c) {
            const int source = STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c));
            if (source >= 0 || u32(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c))) != wanted)
                continue;
            const u64 rows = u64(wanted);
            const u64 one = streaming_line(choices, SEISMIC_CONST_POLICY_CPU_NS, streaming_group);
            const u64 each = streaming_line(choices, SEISMIC_CONST_POLICY_CPU_ROW_NS, streaming_group);
            const u64 handshake =
                served ? u64(0) : streaming_line(choices, SEISMIC_CONST_POLICY_HANDSHAKE_NS, streaming_group);
            const u64 on_cpu = cpu + one + (rows - 1) * each + handshake;
            const u64 in_place = gpu + max(link, rows * link_row);
            if (on_cpu <= in_place) {
                cpu = on_cpu;
                served = true;
                STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) = source <= -2 ? -2 - source : -2;
            } else
                gpu = in_place;
        }
}

// Choice c of `choices` is the route of `routes` at row c / K, slot c % K
// (`routes0`, `routes1` its strides); `step` the step pair (stride
// `step0`). Every thread of the group calls it, after the routes are the
// group's own stores.
STREAMING_FN void streaming_resolve(STREAMING_DEVICE_READ routes, u64 routes0, u64 routes1, u32 choices,
    STREAMING_DEVICE_READ step, u64 step0, bool whole, STREAMING_SYSTEM_READ map, u64 map0, u64 map1,
    STREAMING_SYSTEM_READ tier, u64 tier0, u32 tiered, u32 sited,
    STREAMING_DEVICE_WORDS sources, u64 sources0, u64 sources1, u64 sources2,
    STREAMING_SYSTEM_WORDS routing, u64 routing0, u64 routing1, u32 flight, u32 width,
    STREAMING_DEVICE_WORDS grids, u64 grids0, u64 grids1, STREAMING_SYSTEM_WORDS file, u64 file0, u64 file1,
    STREAMING_SYSTEM_READ policy, STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    const u32 K = u32(SEISMIC_DIM_K);
    if (whole) {
        streaming_device_barrier();
        for (u32 c = tid; c < choices; c += threads) {
            const int expert =
                streaming_device_load(STREAMING_AT(routes, u64(c / K) * routes0 + u64(c % K) * routes1));
            const int home = streaming_load(STREAMING_AT(policy, SEISMIC_CONST_POLICY_HOME * SEISMIC_POLICY_STRIDE_0));
            const u64 at = u64(c / K) * sources1 + u64(c % K) * sources2;
            streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_RECORD * sources0 + at), home + expert);
            streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at), -1);
        }
        return;
    }
    const u64 q = streaming_step(step, step0);
    const int low = streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step0));
    const int home = streaming_load(STREAMING_AT(policy, SEISMIC_CONST_POLICY_HOME * SEISMIC_POLICY_STRIDE_0));
    STREAMING_SYSTEM_WORDS record = STREAMING_AT(routing, (q % u64(flight)) * routing0);
    if (tid == 0)
        STREAMING_GROUP(0) = 0;
    streaming_device_barrier();
    for (u32 c = tid; c < choices; c += threads)
        STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c)) =
            streaming_device_load(STREAMING_AT(routes, u64(c / K) * routes0 + u64(c % K) * routes1));
    streaming_barrier();
    // Each expert's source, read once, on its first choice: the host maps
    // and unmaps while a step runs, so two reads of an expert's words may
    // differ, and every choice of an expert takes its expert's one source.
    for (u32 c = tid; c < choices; c += threads) {
        if (streaming_first_choice(c, streaming_group) != c)
            continue;
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        int source = streaming_held(STREAMING_AT(map, u64(home + expert) * map0), map1, q);
        if (source < 0 && tiered != 0) {
            // Acquire: the host maps a tier record with release once its
            // bytes are whole, which the step's entries then read.
            const int held = streaming_load_acquire(STREAMING_AT(tier, u64(expert) * tier0));
            source = held >= 0 ? -2 - held : -1;
        }
        STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) = source;
    }
    streaming_barrier();
    for (u32 c = tid; c < choices; c += threads) {
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        const u32 first = streaming_first_choice(c, streaming_group);
        const int source = STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, first));
        if (source == -1)
            STREAMING_GROUP(0) = 1;
        if (first != c)
            STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) = source;
        STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) = -1;
        streaming_store(STREAMING_AT(record, (SEISMIC_CONST_RING_EXPERTS + c) * routing1), expert);
        streaming_store(STREAMING_AT(record, (SEISMIC_CONST_RING_EXPERTS + width + c) * routing1), source);
    }
    if (tid == 0) {
        streaming_store(STREAMING_AT(record, SEISMIC_CONST_RING_COUNT * routing1), int(choices));
        streaming_store(STREAMING_AT(record, SEISMIC_CONST_RING_SEQUENCE * routing1), low);
    }
    streaming_barrier();
    const bool assigns = tiered != 0 && sited != 0;
    if (assigns)
        streaming_assign_loads(choices, policy, streaming_group, tid, threads);
    streaming_barrier();
    if (assigns && tid == 0)
        streaming_assign(choices, choices / K, streaming_group);
    streaming_barrier();
    // The snapshot: each choice's source for the GPU and its CPU site, its
    // expert's first choice's.
    for (u32 c = tid; c < choices; c += threads) {
        const u64 at = u64(c / K) * sources1 + u64(c % K) * sources2;
        const int site = STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, streaming_first_choice(c, streaming_group)));
        streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_RECORD * sources0 + at),
            site == -1 ? STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) : -1);
        streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at), site);
    }
    // Every thread's stores, published before the request.
    streaming_publish_barrier();
    if (tid != 0)
        return;
    const bool missing = STREAMING_GROUP(0) != 0;
    STREAMING_DEVICE_WORDS wait = STREAMING_AT(grids, SEISMIC_CONST_WAIT_GRID * grids0);
    streaming_device_store(wait, missing ? 1 : 0);
    streaming_device_store(STREAMING_AT(wait, grids1), 1);
    streaming_device_store(STREAMING_AT(wait, 2 * grids1), 1);
    if (!missing)
        return;
    STREAMING_SYSTEM_WORDS slot = STREAMING_AT(file, (q % u64(flight)) * file0);
    for (u32 c = 0; c < choices; ++c)
        streaming_store(STREAMING_AT(slot,
                            (SEISMIC_CONST_FILE_ENTRIES + SEISMIC_CONST_FILE_ENTRY_WORDS * c + SEISMIC_CONST_FILE_CHOICE)
                                * file1),
            streaming_request_choice(STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c)),
                STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) == -1));
    streaming_store(STREAMING_AT(slot, SEISMIC_CONST_FILE_COUNT * file1), int(choices));
    streaming_request_slot(slot, file1, q);
}
