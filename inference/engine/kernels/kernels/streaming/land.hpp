// Expert streaming's landing launch (`routed_land`, `routed_round` in
// routed.seismic; streaming_words.seismic's landing list), on every
// backend, over the backend's shim (streaming/contract.hpp) and its window
// library's conversion: the one statement of how a launch converts records
// as their reads land.
//
// The host appends each record to the launch's landing list the moment its
// read lands and publishes the count with one 64-bit release store of the
// head (the step's low word, the count). The launch's threadgroups loop:
// thread 0 claims work (an item of an entry) through the launch's device
// counter, so claims follow landing order; awaits the claim's entry, by
// the head's count, and by the exchange's rule on the launch's slot (the
// layer's file slot, or the round's lane), reading the head once more once
// the rule holds; then the whole threadgroup converts the item through the
// window library, and the threadgroup that places an entry's last item
// stores its DONE. A threadgroup stops once its claim's entry is past the
// count the head holds when the rule holds (every entry appended, or the
// step failed), or once its wait's host is gone (`streaming_host_gone`).
// Past the last read byte only the last record's conversion remains.
//
// One poller. However many threadgroups wait, one reads the host's words
// at a time: whichever waiting threadgroup finds no poller takes the
// launch's poll by a compare-and-swap of its LEAD counter, polls the head
// and the slot (`streaming_land_poll`), publishes what it hears (HEARD,
// device memory), and hands the poll back as it stops waiting; the others
// spin on HEARD in device memory, and take the poll whenever it is free.
// So a spinning wait's patience is one poller's spins, the cost of a spin
// the pulse's patience is stated in (streaming_words.seismic: a spin of
// many pollers contending for a discrete device's bus is many times
// slower), and the waits add one reader's traffic to the link, however
// many threadgroups convert. No threadgroup is chosen ahead: the poll is
// held only by a running threadgroup, which frees it before it does
// anything else, so no wait depends on another threadgroup being resident.
//
// Starvation. Claims follow landing order, so a parked threadgroup always
// holds the earliest unlanded entry's item and every claim below it is a
// landed entry's, which a running threadgroup converts without waiting.
// Every park waits on the host (a round's arrival is every threadgroup's
// own, `routed_round`): a threadgroup that waits beside the poller waits
// only on a running threadgroup, which took the poll while it ran and frees
// it as it stops, and takes the poll itself once it is free, so the lack
// of a cross-threadgroup forward-progress guarantee cannot deadlock the
// launch; a threadgroup not yet running holds no claim and no poll. It
// relies, per backend, on:
// - Metal: `volatile coherent(system)` loads under system-scope fences see
//   host stores while the kernel runs (`metal/lib/streaming/control.h`);
//   the pulse ends a park whose host is gone (Metal never ends a dead
//   process's kernel); and no GPU watchdog ends a spinning compute kernel
//   on macOS beyond what the existing parks already rely on.
// - CUDA: volatile and acquire loads of mapped host memory under
//   `__threadfence_system` see host stores while the kernel runs
//   (`cuda/lib/streaming/control.cuh`); the pulse; and no display watchdog
//   on the compute hosts.
// - Vulkan: running work sees no host write, so the park before the call
//   opens once the slot is answered, every entry appended: the first poll
//   reads the head once and hears it final, nothing overlaps, and the
//   argument is trivial.
// Correctness never depends on how many threadgroups are co-resident.
//
// The launch's `sync` scratch `counters` (zero when it starts, restored to
// zero by it): the claims (LAND_CLAIMS), the arrivals (LAND_ARRIVALS), the
// poll (LAND_LEAD: 1 while a threadgroup polls the host, its every holder
// clearing it as it stops waiting), what the poller heard (LAND_HEARD: 2 x
// the head's count, plus 1 once that count is final), and from LAND_ITEMS
// each entry's items placed. The last threadgroup to arrive restores every
// one but the poll, which no waiting threadgroup holds by then, so a
// launch that stopped part way (an expired wait, a failed step) leaves
// none set.
//
// The includer's conversion, every thread calling each hook:
// - `STREAMING_LAND_PARAM` and `STREAMING_LAND_ARG`: its conversion
//   context, a parameter and its argument, defined before including this
//   file, which declares the hooks below; the includer defines them after
//   it (its backend's `lib/streaming/land`);
// - `u32 streaming_land_items(STREAMING_LAND_PARAM, u32 tensor)`: the items
//   of a record of routed tensor `tensor` (STREAMING_LAND_GATE, _UP,
//   _DOWN), none for an up-only record's gate: the window library's items
//   (`record_items`), one owner;
// - `streaming_land_load`, `streaming_land_stage`,
//   `streaming_land_place(STREAMING_LAND_PARAM, u32 tensor, int from, int
//   to, u32 item)`: item `item` of entry (from, to)'s record of `tensor`,
//   converted in three steps: begun loading (a backend that holds a
//   window's bytes in registers loads them, others nothing), staged into
//   its window (others nothing), and placed (beginning and ending with a
//   barrier, the window free after it). Between the stage and the place of a claim, the next claim is
//   loaded where its entry has landed, so its loads are in flight while the
//   claim places; no landed item's conversion waits on an unlanded one.

#define STREAMING_LAND_GATE 0u
#define STREAMING_LAND_UP 1u
#define STREAMING_LAND_DOWN 2u

// The counters' words.
#define STREAMING_LAND_CLAIMS SEISMIC_CONST_LAND_CLAIMS
#define STREAMING_LAND_ARRIVALS SEISMIC_CONST_LAND_ARRIVALS
#define STREAMING_LAND_LEAD SEISMIC_CONST_LAND_LEAD
#define STREAMING_LAND_HEARD SEISMIC_CONST_LAND_HEARD
#define STREAMING_LAND_ITEMS SEISMIC_CONST_LAND_ITEMS

// The group words (LAND_GROUP_WORDS): the claim in hand (-1: stop), its
// entry's from and to, and whether its entry is known landed.
#define STREAMING_LAND_CLAIM 0
#define STREAMING_LAND_FROM 1
#define STREAMING_LAND_TO 2
#define STREAMING_LAND_LANDED 3

STREAMING_FN u32 streaming_land_items(STREAMING_LAND_PARAM, u32 tensor);
STREAMING_FN void streaming_land_load(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item);
STREAMING_FN void streaming_land_stage(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item);
STREAMING_FN void streaming_land_place(STREAMING_LAND_PARAM, u32 tensor, int from, int to, u32 item);

// The count of entries the landing list at `landing` (word stride
// `stride`) holds for step `q`: its head's, read by one 64-bit acquire load,
// where its low word is q's, else none.
STREAMING_FN u32 streaming_land_count(STREAMING_SYSTEM_WORDS landing, u64 stride, u64 q) {
    const u64 head = streaming_load_u64_acquire(STREAMING_AT(landing, SEISMIC_CONST_LANDING_HEAD * stride));
    return u32(head) == u32(q) ? u32(head >> 32) : 0u;
}

// The poller's wait for entry `entry` of step `q`'s landing list at
// `landing` (stride `landing_stride`), by the exchange's rule on the
// launch's slot at `slot` (stride `slot_stride`): the head's count once it
// holds the entry; else, once the slot is answered or failed, the count the
// head then holds where it holds the entry; -1 to stop: the entry is past
// that count, the slot was never requested for q (nothing lands), or a
// spinning wait's host is gone (`streaming_host_gone`, as
// `streaming_await_slot`'s). Where parks are submission boundaries, the
// park before the call opened once the slot was answered, so the poll looks
// once. Each look's count, and whether it is final, it publishes to HEARD
// at `counters` for the threadgroups that wait beside it.
STREAMING_FN int streaming_land_poll(STREAMING_SYSTEM_WORDS landing, u64 landing_stride,
    STREAMING_SYSTEM_WORDS slot, u64 slot_stride, u64 q, u32 entry, STREAMING_DEVICE_WORDS counters) {
    const u64 failed = q | (u64(u32(SEISMIC_CONST_SLOT_FAILED)) << 32);
    const bool requested =
        streaming_load_u64_acquire(STREAMING_AT(slot, SEISMIC_CONST_SLOT_REQUEST * slot_stride)) == q;
#if SEISMIC_PARKS_IN_KERNELS
    int pulse = 0;
    u32 still = 0u;
#endif
    int published = -1;
    for (u32 spins = 0u;; ++spins) {
        u32 count = streaming_land_count(landing, landing_stride, q);
        bool ended = !requested;
        if (requested && count <= entry) {
            const u64 answer = streaming_load_u64_acquire(STREAMING_AT(slot, SEISMIC_CONST_SLOT_ANSWER * slot_stride));
            if (answer == q || answer == failed) {
                count = streaming_land_count(landing, landing_stride, q);
                ended = true;
            } else {
#if SEISMIC_PARKS_IN_KERNELS
                ended = streaming_host_gone(slot, slot_stride, q, spins, pulse, still);
#else
                ended = true;
#endif
            }
        }
        // Published as it changes, so a still spin costs what the pulse's
        // patience was measured at; HEARD only grows, as a poller that took
        // the poll after another may have looked before the other's last
        // publication.
        const int heard = int(2u * count) + (ended ? 1 : 0);
        if (heard != published) {
            published = heard;
            int held = streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_HEARD), 0);
            while (held < heard) {
                const int seen = streaming_device_cas(STREAMING_AT(counters, STREAMING_LAND_HEARD), held, heard);
                held = seen == held ? heard : seen;
            }
        }
        if (count > entry)
            return int(count);
        if (ended)
            return -1;
    }
}

// Thread 0's wait for entry `entry` of step `q`'s landing list at
// `landing` (stride `landing_stride`), on the launch's slot at `slot`
// (stride `slot_stride`), as the launch's one poller's
// (`streaming_land_poll`): the count it heard once that holds the entry
// (its words read after an acquire load of the head that holds it); -1
// once what it heard is final and does not, or the head does not hold what
// was heard (an invariant violation: the slot's SLOT_VIOLATED stored). A waiting threadgroup that
// finds the poll free takes it (LEAD at `counters`), polls, and frees it.
STREAMING_FN int streaming_land_await(STREAMING_SYSTEM_WORDS landing, u64 landing_stride,
    STREAMING_SYSTEM_WORDS slot, u64 slot_stride, u64 q, u32 entry, STREAMING_DEVICE_WORDS counters) {
    for (;;) {
        const int heard = streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_HEARD), 0);
        const u32 count = u32(heard) >> 1;
        if (count > entry) {
            // The head holds what the poller heard (it only grows within a
            // step): this acquire of it orders the entry's words after it.
            if (streaming_land_count(landing, landing_stride, q) > entry)
                return int(count);
            // A head that does not is no state the protocol reaches: the
            // step is stored violated in a word of its own, which the host
            // panics on, and the thread stops.
            streaming_store(STREAMING_AT(slot, SEISMIC_CONST_SLOT_VIOLATED * slot_stride),
                int(u32(q) | u32(SEISMIC_CONST_VIOLATED_STAMP)));
            streaming_system_fence();
            return -1;
        }
        if ((heard & 1) != 0)
            return -1;
        if (streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_LEAD), 0) == 0
            && streaming_device_cas(STREAMING_AT(counters, STREAMING_LAND_LEAD), 0, 1) == 0) {
            const int polled = streaming_land_poll(landing, landing_stride, slot, slot_stride, q, entry, counters);
            streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_LEAD), -1);
            return polled;
        }
    }
}

// The tensor and item a claim's position `within` its entry's items names:
// the gate's items, then the up's, then the down's.
STREAMING_FN u32 streaming_land_tensor_of(u32 within, u32 gate, u32 up) {
    return within < gate ? STREAMING_LAND_GATE : within < gate + up ? STREAMING_LAND_UP : STREAMING_LAND_DOWN;
}
STREAMING_FN u32 streaming_land_item_of(u32 within, u32 gate, u32 up, u32 tensor) {
    return within - (tensor == STREAMING_LAND_GATE ? 0u : tensor == STREAMING_LAND_UP ? gate : gate + up);
}

// Entry `entry`'s from and to, read from its landed words into the group's.
STREAMING_FN void streaming_land_entry(STREAMING_SYSTEM_WORDS landing, u64 landing_stride, u32 entry,
    STREAMING_GROUP_PARAM) {
    const u64 words = SEISMIC_CONST_LANDING_ENTRIES + SEISMIC_CONST_LANDING_ENTRY_WORDS * u64(entry);
    STREAMING_GROUP(STREAMING_LAND_FROM) =
        streaming_load(STREAMING_AT(landing, (words + SEISMIC_CONST_LANDING_FROM) * landing_stride));
    STREAMING_GROUP(STREAMING_LAND_TO) =
        streaming_load(STREAMING_AT(landing, (words + SEISMIC_CONST_LANDING_TO) * landing_stride));
}

// The landing launch's loop, by every thread `tid` of threadgroup `group`
// words (STREAMING_GROUP_PARAM), one of `groups`: step `q`'s landing list at
// `landing` (stride `landing_stride`) of at most `capacity` entries,
// awaited by the rule on the slot at `slot` (stride `slot_stride`), each
// entry's items converted through the includer's hooks, the launch's
// counters at `counters`. It returns once the threadgroup has stopped and
// arrived. A launch that lands nothing (no record converts: where the host
// tier places) does not call it: an entry of no items has no claims.
STREAMING_FN void streaming_land(STREAMING_LAND_PARAM, STREAMING_SYSTEM_WORDS landing, u64 landing_stride,
    STREAMING_SYSTEM_WORDS slot, u64 slot_stride, u64 q, STREAMING_DEVICE_WORDS counters, u32 capacity, u32 groups,
    STREAMING_GROUP_PARAM, u32 tid) {
    const u32 gate = streaming_land_items(STREAMING_LAND_ARG, STREAMING_LAND_GATE);
    const u32 up = streaming_land_items(STREAMING_LAND_ARG, STREAMING_LAND_UP);
    const u32 per_entry = gate + up + streaming_land_items(STREAMING_LAND_ARG, STREAMING_LAND_DOWN);
    // Thread 0's own: the entries it has seen landed, and the entry whose
    // from and to the group's words hold (-1: none).
    u32 known = 0u;
    int read = -1;
    if (tid == 0u) {
        STREAMING_GROUP(STREAMING_LAND_CLAIM) = streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_CLAIMS), 1);
        STREAMING_GROUP(STREAMING_LAND_LANDED) = 0;
    }
    streaming_barrier();
    for (;;) {
        // A claim taken ahead whose entry had not landed: awaited now, then
        // loaded.
        if (STREAMING_GROUP(STREAMING_LAND_LANDED) == 0) {
            if (tid == 0u) {
                const u32 entry = u32(STREAMING_GROUP(STREAMING_LAND_CLAIM)) / per_entry;
                if (entry >= known) {
                    const int count =
                        streaming_land_await(landing, landing_stride, slot, slot_stride, q, entry, counters);
                    if (count < 0)
                        STREAMING_GROUP(STREAMING_LAND_CLAIM) = -1;
                    else
                        known = u32(count);
                }
                if (STREAMING_GROUP(STREAMING_LAND_CLAIM) >= 0 && int(entry) != read) {
                    read = int(entry);
                    streaming_land_entry(landing, landing_stride, entry, streaming_group);
                }
            }
            streaming_barrier();
            if (STREAMING_GROUP(STREAMING_LAND_CLAIM) < 0)
                break;
        }
        // The claim in hand: its entry landed, its item begun loading.
        const u32 claim = u32(STREAMING_GROUP(STREAMING_LAND_CLAIM));
        const int from = STREAMING_GROUP(STREAMING_LAND_FROM);
        const int to = STREAMING_GROUP(STREAMING_LAND_TO);
        const u32 entry = claim / per_entry;
        const u32 within = claim % per_entry;
        const u32 tensor = streaming_land_tensor_of(within, gate, up);
        const u32 item = streaming_land_item_of(within, gate, up, tensor);
        if (STREAMING_GROUP(STREAMING_LAND_LANDED) == 0)
            streaming_land_load(STREAMING_LAND_ARG, tensor, from, to, item);
        streaming_land_stage(STREAMING_LAND_ARG, tensor, from, to, item);
        // Every thread has read the claim's group words before thread 0
        // takes the next.
        streaming_barrier();
        // The next claim, taken now and loaded at once where its entry is
        // known landed or the head shows it (one look, no wait).
        if (tid == 0u) {
            const int next = streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_CLAIMS), 1);
            const u32 next_entry = u32(next) / per_entry;
            if (next_entry >= known)
                known = max(known, streaming_land_count(landing, landing_stride, q));
            const bool landed = next_entry < known;
            if (landed && int(next_entry) != read) {
                read = int(next_entry);
                streaming_land_entry(landing, landing_stride, next_entry, streaming_group);
            }
            STREAMING_GROUP(STREAMING_LAND_CLAIM) = next;
            STREAMING_GROUP(STREAMING_LAND_LANDED) = landed ? 1 : 0;
        }
        streaming_barrier();
        if (STREAMING_GROUP(STREAMING_LAND_LANDED) != 0) {
            const u32 next = u32(STREAMING_GROUP(STREAMING_LAND_CLAIM)) % per_entry;
            const u32 next_tensor = streaming_land_tensor_of(next, gate, up);
            streaming_land_load(STREAMING_LAND_ARG, next_tensor, STREAMING_GROUP(STREAMING_LAND_FROM),
                STREAMING_GROUP(STREAMING_LAND_TO), streaming_land_item_of(next, gate, up, next_tensor));
        }
        streaming_land_place(STREAMING_LAND_ARG, tensor, from, to, item);
        // The item's stores, every thread's, before thread 0's count of it;
        // the entry's last item's count, before its DONE.
        streaming_publish_barrier();
        if (tid == 0u) {
            streaming_system_fence();
            const int landed = streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_ITEMS + int(entry)), 1);
            if (landed + 1 == int(per_entry)) {
                streaming_system_fence();
                streaming_store(STREAMING_AT(landing, (SEISMIC_CONST_LANDING_ENTRIES
                    + SEISMIC_CONST_LANDING_ENTRY_WORDS * u64(entry) + SEISMIC_CONST_LANDING_DONE) * landing_stride),
                    int(u32(q)));
            }
        }
    }
    // The arrival: the last restores every counter.
    if (tid == 0u) {
        streaming_system_fence();
        if (streaming_device_add(STREAMING_AT(counters, STREAMING_LAND_ARRIVALS), 1) + 1 == int(groups)) {
            streaming_system_fence();
            streaming_device_store(STREAMING_AT(counters, STREAMING_LAND_CLAIMS), 0);
            streaming_device_store(STREAMING_AT(counters, STREAMING_LAND_ARRIVALS), 0);
            streaming_device_store(STREAMING_AT(counters, STREAMING_LAND_HEARD), 0);
            for (u32 entry = 0u; entry < capacity; ++entry)
                streaming_device_store(STREAMING_AT(counters, STREAMING_LAND_ITEMS + int(entry)), 0);
        }
    }
}
