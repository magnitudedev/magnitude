// Expert streaming's decode resolve and assignment (`resolve` and `assign`
// in routed.seismic), shared by every streamed route, over the backend's
// shim (streaming/contract.hpp) and the rings' writer (streaming/hint.hpp,
// which the includer includes first): after its selection, the group that holds
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
// into the step's routing record (`routing`'s row q mod FLIGHT, choice c's
// source at ROUTING_SOURCE(`width`, c)), and thread 0 its count, guesses' kinds and
// sequence:
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
// request (`streaming_request_slot`). It does not wait. Thread 0 writes the
// landing launch's grid (`land_grid`, `routed_land`'s): `land_groups`
// threadgroups where the step requested, else none.
//
// Front rows (the step's STEP_FRONT bits) compute only the choices of an
// expert the step holds or an exact row chooses (`streaming_computes`),
// whose bytes are on the device or read anyway: any other keeps
// source -1 in the snapshot alone (no record, so it contributes nothing;
// the routing record keeps its source as resolved), and is no miss of the
// step (no wait, no request word, no assignment). Thread 0 publishes every
// front row's choice, with its weight, to the layer's front ring (`front`)
// once the request is out (`streaming_publish_front`); the host reads the
// absent ones ahead for the step the front row's token is exact in.
//
// A whole group's route (`whole`) sources each choice from its home record,
// `policy[POLICY_HOME]` + expert, every one held, front rows' too: every
// expert is at home, nothing misses, and its landing grid is empty; it
// writes its routing record, each choice's source its home record, so the
// host reads every layer's routing of every step.
//
// Every route also writes each row's selection key of every expert into the
// step's slot of the layer's key records (`keys`, streaming_words.seismic;
// `streaming_record_key`, from its selecting lanes, before the resolve),
// which placement ranks every expert by.

// The group words a resolve of `choices` choices takes: the missing flag;
// then each choice's expert, its source as resolved, its CPU site (-1: the
// GPU's), its rows (its expert's exact rows, on the expert's standing choice; 0 on any
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

// Step q's slot of a table of a slot for each step in flight (`table`, its
// slots `table0` words apart): its row q mod `flight`.
STREAMING_FN STREAMING_SYSTEM_WORDS streaming_slot_of(STREAMING_SYSTEM_WORDS table, u64 table0, u64 q, u32 flight) {
    return STREAMING_AT(table, streaming_modulo(q, u64(flight)) * table0);
}

// Row `row`'s selection key of expert `expert`, as its f32 bits `key`, into
// the step's key record `record` (its rows and experts `keys1`, `keys2`
// words apart). Relaxed store: the host reads keys only after step completion.
STREAMING_FN void streaming_record_key(STREAMING_SYSTEM_WORDS record, u64 keys1, u64 keys2, u32 row, u32 expert,
    int key) {
    streaming_store_relaxed(STREAMING_AT(record, u64(row) * keys1 + u64(expert) * keys2), key);
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

// Whether row `row` is a front row of `front_rows` (STEP_FRONT's bits).
STREAMING_FN bool streaming_front_row(int front_rows, u32 row) {
    return ((u32(front_rows) >> row) & 1u) != 0u;
}

// Whether choice c, its expert of source `source` as resolved (-1 absent),
// is computed: any choice of an exact row (a row not of `front_rows`); a
// front row's where the step holds its expert or an exact row chooses it.
STREAMING_FN bool streaming_computes(u32 c, u32 choices, int source, int front_rows, STREAMING_GROUP_PARAM) {
    const u32 K = u32(SEISMIC_DIM_K);
    if (!streaming_front_row(front_rows, c / K) || source != -1)
        return true;
    const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
    for (u32 other = 0; other < choices; ++other)
        if (!streaming_front_row(front_rows, other / K) && STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(other)) == expert)
            return true;
    return false;
}

// The routing record's head, by one thread: the count of its `choices`, the
// step's guesses' kinds, then its sequence (the step's low word).
STREAMING_FN void streaming_record_head(STREAMING_SYSTEM_WORDS record, u64 routing1, u32 choices,
    STREAMING_DEVICE_READ step, u64 step0) {
    streaming_store(STREAMING_AT(record, SEISMIC_CONST_RING_COUNT * routing1), int(choices));
    streaming_store(STREAMING_AT(record, SEISMIC_CONST_RING_GUESSES * routing1),
        streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_GUESSES * step0)));
    streaming_store(STREAMING_AT(record, SEISMIC_CONST_RING_SEQUENCE * routing1),
        streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step0)));
}

// The landing launch's grid (`routed_land`'s, of stride `land_grid0`), by
// one thread: a row of `groups` threadgroups (none: it dispatches nothing).
STREAMING_FN void streaming_land_grid(STREAMING_DEVICE_WORDS land_grid, u64 land_grid0, int groups) {
    streaming_device_store(STREAMING_AT(land_grid, 0), groups);
    streaming_device_store(STREAMING_AT(land_grid, land_grid0), 1);
    streaming_device_store(STREAMING_AT(land_grid, 2 * land_grid0), 1);
}

// Whether choice c stands for its expert in the assignment: the first of the
// choices of rows not of `front_rows` that name it (a front row's never does).
STREAMING_FN bool streaming_stands(u32 c, int front_rows, STREAMING_GROUP_PARAM) {
    const u32 K = u32(SEISMIC_DIM_K);
    if (streaming_front_row(front_rows, c / K))
        return false;
    const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
    for (u32 earlier = 0; earlier < c; ++earlier)
        if (!streaming_front_row(front_rows, earlier / K) && STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(earlier)) == expert)
            return false;
    return true;
}

// The choice that stands for choice c's expert in the assignment: the first
// of the exact rows' choices of it, else (an expert only front rows name,
// which the assignment never sites) its first.
STREAMING_FN u32 streaming_standing(u32 c, u32 choices, int front_rows, STREAMING_GROUP_PARAM) {
    const u32 K = u32(SEISMIC_DIM_K);
    const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
    for (u32 candidate = 0; candidate < choices; ++candidate)
        if (!streaming_front_row(front_rows, candidate / K) && STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(candidate)) == expert)
            return candidate;
    return streaming_first_choice(c, streaming_group);
}

// The layer's front ring, in parallel across the group once the request is out:
// every choice of the front rows, in rank order (rank r of a row is its choice
// K - 1 - r, as routing ranks them), each as its expert, rank, row and weight's
// word (its magnitude's bits of `weights`, `streaming_ring_entry`; each front
// row is its own source, so an expert two front rows chose is named by both),
// as a seqlock (streaming/hint.hpp). The host's interest ledger publishes every
// one as an interest of the step its token is exact in: it reads the absent
// ones that pay, and the bank's ranking holds the ones it wants until that step
// has logged its own read of the layer.
#ifndef STREAMING_WEIGHTS_READ
#define STREAMING_WEIGHTS_READ STREAMING_DEVICE_READ
#define STREAMING_WEIGHT_WORD(weights, index) streaming_device_load(STREAMING_AT(weights, index))
#endif

STREAMING_FN void streaming_publish_front(u32 choices, int front_rows, STREAMING_DEVICE_READ step, u64 step0,
    STREAMING_SYSTEM_WORDS ring, u64 ring0, STREAMING_WEIGHTS_READ weights, u64 weights0, u64 weights1,
    STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    const u32 K = u32(SEISMIC_DIM_K);
    const u32 rows = choices / K;
    if (tid == 0)
        streaming_ring_open(ring, ring0);
    streaming_barrier();
    u32 front_count = 0;
    for (u32 r = 0; r < rows; ++r)
        if (streaming_front_row(front_rows, r))
            ++front_count;
    const u32 total = front_count * K;
    if (front_count > 0) {
        for (u32 idx = tid; idx < total; idx += threads) {
            const u32 rank = idx / front_count;
            const u32 target = idx % front_count;
            u32 row = 0;
            u32 seen = 0;
            for (u32 r = 0; r < rows; ++r) {
                if (streaming_front_row(front_rows, r)) {
                    if (seen == target) {
                        row = r;
                        break;
                    }
                    ++seen;
                }
            }
            const u32 choice = K - 1 - rank;
            streaming_ring_entry(ring, ring0, int(idx), STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(row * K + choice)),
                int(rank), int(row), STREAMING_WEIGHT_WORD(weights, u64(row) * weights0 + u64(choice) * weights1));
        }
    }
    streaming_publish_barrier();
    if (tid == 0)
        streaming_ring_close(ring, ring0, int(total), step, step0);
}

// The assignment, by thread 0, over the group's resolved sources and rows
// (streaming_words.seismic's): the GPU's queue from its distinct hit
// experts' resident lines, the CPU's from zero; each distinct missed
// expert, by rows (most first, from `max_miss_rows` down to 1), ties in
// choice order, to the CPU where it ends there no later than in place, else
// to the GPU in place, a read in place no sooner than the link is free of
// the step's moves (POLICY_LINK_BUSY_NS). Its site is written on its standing
// choice (the first of the exact rows'); the snapshot gives every choice its
// standing one's. Caller guards on POLICY_PRICED != 0.
STREAMING_FN void streaming_assign(u32 choices, STREAMING_GROUP_PARAM) {
    const u64 hit = streaming_line(choices, SEISMIC_CONST_POLICY_HIT_NS, streaming_group);
    const u64 hit_row = streaming_line(choices, SEISMIC_CONST_POLICY_HIT_ROW_NS, streaming_group);
    const u64 link = streaming_line(choices, SEISMIC_CONST_POLICY_LINK_NS, streaming_group);
    const u64 link_row = streaming_line(choices, SEISMIC_CONST_POLICY_LINK_ROW_NS, streaming_group);
    const u64 busy = streaming_line(choices, SEISMIC_CONST_POLICY_LINK_BUSY_NS, streaming_group);
    const u64 one = streaming_line(choices, SEISMIC_CONST_POLICY_CPU_NS, streaming_group);
    const u64 each = streaming_line(choices, SEISMIC_CONST_POLICY_CPU_ROW_NS, streaming_group);
    const u64 handshake_ns = streaming_line(choices, SEISMIC_CONST_POLICY_HANDSHAKE_NS, streaming_group);
    u64 gpu = 0;
    u32 max_miss_rows = 0;
    u32 min_miss_c = choices;
    u32 max_miss_c = 0;
    for (u32 c = 0; c < choices; ++c) {
        const u32 rows = u32(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)));
        if (rows == 0)
            continue;
        if (STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) >= 0) {
            gpu += max(hit, u64(rows) * hit_row);
        } else {
            if (rows > max_miss_rows)
                max_miss_rows = rows;
            if (c < min_miss_c)
                min_miss_c = c;
            if (c > max_miss_c)
                max_miss_c = c;
        }
    }
    if (max_miss_rows == 0)
        return;
    u64 cpu = 0;
    bool served = false;
    for (u32 wanted = max_miss_rows; wanted > 0; --wanted) {
        const u64 rows = u64(wanted);
        const u64 cpu_work = one + (rows - 1) * each;
        const u64 link_work = max(link, rows * link_row);
        for (u32 c = min_miss_c; c <= max_miss_c; ++c) {
            const int source = STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c));
            if (source >= 0 || u32(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c))) != wanted)
                continue;
            const u64 on_cpu = cpu + cpu_work + (served ? u64(0) : handshake_ns);
            const u64 in_place = max(gpu, busy) + link_work;
            if (on_cpu <= in_place) {
                cpu = on_cpu;
                served = true;
                STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) = source <= -2 ? -2 - source : -2;
            } else {
                gpu = in_place;
            }
        }
    }
}

// Choice c of `choices` is the route of `routes` at row c / K, slot c % K
// (`routes0`, `routes1` its strides), weighed by `weights` (its routing
// weights, read as their bits, of strides `weights0`, `weights1`); `step`
// the step pair (stride `step0`); `land_grid` (stride `land_grid0`) the
// landing launch's grid, of `land_groups` threadgroups while it lands.
// Every thread of the group calls it, after the routes and weights are the
// group's own stores. Where the backend's group barrier does not make one
// thread's device stores visible to another (Vulkan's plain references),
// the includer defines STREAMING_ROUTES_GROUPED and its selection also
// stores each choice's expert at the group word STREAMING_RESOLVE_EXPERT(c),
// which the resolve then reads in place of `routes`.
STREAMING_FN void streaming_resolve(STREAMING_DEVICE_READ routes, u64 routes0, u64 routes1, u32 choices,
    STREAMING_WEIGHTS_READ weights, u64 weights0, u64 weights1,
    STREAMING_DEVICE_READ step, u64 step0, bool whole, STREAMING_SYSTEM_READ map, u64 map0, u64 map1,
    STREAMING_SYSTEM_READ tier, u64 tier0, u32 tiered, u32 sited,
    STREAMING_DEVICE_WORDS sources, u64 sources0, u64 sources1, u64 sources2,
    STREAMING_SYSTEM_WORDS routing, u64 routing0, u64 routing1, u32 flight, u32 width,
    STREAMING_SYSTEM_WORDS file, u64 file0, u64 file1,
    STREAMING_SYSTEM_READ policy, STREAMING_SYSTEM_WORDS front, u64 front0,
    STREAMING_DEVICE_WORDS land_grid, u64 land_grid0, int land_groups, STREAMING_GROUP_PARAM, u32 tid, u32 threads) {
    const u32 K = u32(SEISMIC_DIM_K);
    const int front_rows = streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_FRONT * step0));
    if (tid == 0)
        STREAMING_GROUP(0) = 0;
    // Load the entire policy table once into group memory (one coalesced
    // system read across the first POLICY_WORDS threads, overlapped with the
    // route loads below).
    for (u32 word = tid; word < u32(SEISMIC_CONST_POLICY_WORDS); word += threads)
        STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, word)) =
            streaming_load(STREAMING_AT(policy, word * SEISMIC_POLICY_STRIDE_0));
#ifdef STREAMING_ROUTES_GROUPED
    streaming_barrier();
#else
    streaming_device_barrier();
    for (u32 c = tid; c < choices; c += threads)
        STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c)) =
            streaming_device_load(STREAMING_AT(routes, u64(c / K) * routes0 + u64(c % K) * routes1));
    streaming_barrier();
#endif
    const int home = STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, SEISMIC_CONST_POLICY_HOME));
    const bool is_whole = whole || STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, SEISMIC_CONST_POLICY_WHOLE)) != 0;
    const bool priced = STREAMING_GROUP(STREAMING_RESOLVE_POLICY(choices, SEISMIC_CONST_POLICY_PRICED)) != 0;
    if (!whole)
        streaming_barrier();
    const u64 q = streaming_step(step, step0);
    STREAMING_SYSTEM_WORDS record = streaming_slot_of(routing, routing0, q, flight);
    if (is_whole) {
        for (u32 c = tid; c < choices; c += threads) {
            const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
            const u64 at = u64(c / K) * sources1 + u64(c % K) * sources2;
            const bool computes = front_rows == 0 || streaming_computes(c, choices, home + expert, front_rows, streaming_group);
            streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_RECORD * sources0 + at),
                computes ? home + expert : -1);
            streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at), -1);
            streaming_store(STREAMING_AT(record, SEISMIC_CONST_ROUTING_EXPERT(c) * routing1), expert);
            streaming_store(STREAMING_AT(record, SEISMIC_CONST_ROUTING_SOURCE(width, c) * routing1), home + expert);
        }
        if (tid == 0) {
            streaming_record_head(record, routing1, choices, step, step0);
            streaming_land_grid(land_grid, land_grid0, 0);
        }
        return;
    }
    // Each expert's source, read once, on its first choice: the host maps
    // and unmaps while a step runs, so two reads of an expert's words may
    // differ, and every choice of an expert takes its expert's one source.
    // Batch all relaxed system loads (MAP_UNTIL, MAP_STAGE, tier[expert])
    // across every unique expert owned by this thread, execute one system
    // acquire fence, then issue the dependent MAP_FROM and MAP_RECORD loads
    // for potential VRAM hits (q < until). STREAMING_RESOLVE_ROWS and
    // STREAMING_RESOLVE_SITE are unused until the broadcast pass below and
    // temporarily hold each choice's first index and (q < until) flag.
    bool loaded = false;
    for (u32 c = tid; c < choices; c += threads) {
        const u32 first = streaming_first_choice(c, streaming_group);
        STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)) = int(first);
        if (first != c)
            continue;
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        STREAMING_SYSTEM_READ map_row = STREAMING_AT(map, u64(home + expert) * map0);
        const u64 until = streaming_load_u64(STREAMING_AT(map_row, SEISMIC_CONST_MAP_UNTIL * map1));
        const u32 staged = u32(streaming_load(STREAMING_AT(map_row, SEISMIC_CONST_MAP_STAGE * map1)));
        const int tier_held = tiered != 0 ? streaming_load(STREAMING_AT(tier, u64(expert) * tier0)) : -1;
        loaded = true;
        int fallback = -1;
        const u32 slot = staged & 0xFFFFu;
        if (slot != 0u && (staged >> 16) == u32(q & 0xFFFFu))
            fallback = int(slot - 1u);
        else if (tiered != 0 && tier_held >= 0)
            fallback = -2 - tier_held;
        STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) = fallback;
        STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) = q < until ? 1 : 0;
    }
    if (loaded)
        streaming_acquire_fence();
    for (u32 c = tid; c < choices; c += threads) {
        if (u32(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c))) != c)
            continue;
        if (STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) == 0)
            continue;
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        STREAMING_SYSTEM_READ map_row = STREAMING_AT(map, u64(home + expert) * map0);
        const u64 from = streaming_load_u64(STREAMING_AT(map_row, SEISMIC_CONST_MAP_FROM * map1));
        const int rec = streaming_load(STREAMING_AT(map_row, SEISMIC_CONST_MAP_RECORD * map1));
        if (q >= from && rec >= 0)
            STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) = rec;
    }
    streaming_barrier();
    const bool assigns = tiered != 0 && sited != 0 && priced;
    for (u32 c = tid; c < choices; c += threads) {
        const int expert = STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c));
        const u32 first = u32(STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)));
        const int source = STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, first));
        if (source == -1 && (front_rows == 0 || streaming_computes(c, choices, source, front_rows, streaming_group)))
            STREAMING_GROUP(0) = 1;
        if (first != c)
            STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c)) = source;
        STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, c)) = -1;
        if (assigns) {
            const bool stands = front_rows == 0 ? (first == c) : streaming_stands(c, front_rows, streaming_group);
            int rows = 0;
            if (stands) {
                for (u32 other = c; other < choices; ++other)
                    if ((front_rows == 0 || !streaming_front_row(front_rows, other / K))
                        && STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(other)) == expert)
                        ++rows;
            }
            STREAMING_GROUP(STREAMING_RESOLVE_ROWS(choices, c)) = rows;
        }
        streaming_store(STREAMING_AT(record, SEISMIC_CONST_ROUTING_EXPERT(c) * routing1), expert);
        streaming_store(STREAMING_AT(record, SEISMIC_CONST_ROUTING_SOURCE(width, c) * routing1), source);
    }
    if (tid == 0)
        streaming_record_head(record, routing1, choices, step, step0);
    streaming_barrier();
    if (assigns) {
        if (tid == 0)
            streaming_assign(choices, streaming_group);
        streaming_barrier();
    }
    // The snapshot and (when absent choices exist) the file request entries,
    // written in parallel across all threads before the publication barrier.
    const int missing = STREAMING_GROUP(0);
    STREAMING_SYSTEM_WORDS file_slot = STREAMING_AT(file, streaming_modulo(q, u64(flight)) * file0);
    for (u32 c = tid; c < choices; c += threads) {
        const u64 at = u64(c / K) * sources1 + u64(c % K) * sources2;
        const int source = STREAMING_GROUP(STREAMING_RESOLVE_SOURCE(choices, c));
        const bool computes = front_rows == 0 || streaming_computes(c, choices, source, front_rows, streaming_group);
        int site = -1;
        if (assigns && computes) {
            const u32 stand = front_rows == 0
                ? streaming_first_choice(c, streaming_group)
                : streaming_standing(c, choices, front_rows, streaming_group);
            site = STREAMING_GROUP(STREAMING_RESOLVE_SITE(choices, stand));
        }
        streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_RECORD * sources0 + at),
            computes && site == -1 ? source : -1);
        streaming_device_store(STREAMING_AT(sources, SEISMIC_CONST_SOURCE_SITE * sources0 + at), site);
        if (missing != 0) {
            streaming_store(STREAMING_AT(file_slot,
                                (SEISMIC_CONST_FILE_ENTRY(c) + SEISMIC_CONST_FILE_CHOICE) * file1),
                streaming_request_choice(STREAMING_GROUP(STREAMING_RESOLVE_EXPERT(c)), source == -1 && computes));
        }
    }
    // Every thread's stores (routing, sources, file entries), published before the request.
    streaming_publish_barrier();
    if (tid == 0) {
        streaming_land_grid(land_grid, land_grid0, missing != 0 ? land_groups : 0);
        if (missing != 0) {
            streaming_store(STREAMING_AT(file_slot, SEISMIC_CONST_FILE_COUNT * file1), int(choices));
            streaming_request_slot(file_slot, file1, q);
        }
    }
    if (front_rows != 0)
        streaming_publish_front(choices, front_rows, step, step0, front, front0, weights, weights0, weights1,
            streaming_group, tid, threads);
}
