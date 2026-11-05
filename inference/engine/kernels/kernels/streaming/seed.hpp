// Expert streaming, a prompt group's seeding commit (`routed_seed_commit` in
// routed.seismic), over the backend's shim (streaming/contract.hpp): unless
// the list was written for another step or the step's request failed (its
// last round's lane of the layer's scratch table does not hold the step's
// answer), each listed (expert, record) pair's record is committed with the
// step's low word, after a system-scope fence: its seed copy (`routed_copy`,
// before it) has written it. The device writes no map: the host maps a
// committed seed. One group reads the host-written list's count into its
// memory once. The includer includes streaming/list.hpp and
// streaming/round.hpp first.

// The group words a seeding takes: its count.
#define STREAMING_SEED_WORDS 1

// Every thread of the one group calls it: `step` the step pair (stride
// `step0`), `scratch` the layer's scratch table.
STREAMING_FN void streaming_seed_commit(STREAMING_SYSTEM_READ list, STREAMING_DEVICE_READ step, u64 step0,
    STREAMING_SYSTEM_READ scratch, STREAMING_SYSTEM_WORDS commits, STREAMING_GROUP_PARAM, u32 tid,
    u32 threads STREAMING_WORDS_PARAM) {
    const u64 stride = SEISMIC_LIST_STRIDE_0;
    const u64 q = streaming_step(step, step0);
    const int sequence = streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step0));
    if (tid == 0) {
        const u64 last = streaming_scratch_lane(q, u64(SEISMIC_DIM_LANES) - 1 STREAMING_WORDS_ARG);
        const bool answered = streaming_load_u64_acquire(STREAMING_AT(scratch,
                                  last + SEISMIC_CONST_SLOT_ANSWER * SEISMIC_SCRATCH_STRIDE_1))
            == q;
        STREAMING_GROUP(0) = answered && streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_SEQUENCE * stride)) == sequence
            ? streaming_load(STREAMING_AT(list, SEISMIC_CONST_LIST_COUNT * stride))
            : 0;
    }
    // The copies before the commit, each thread's commits after them.
    streaming_system_fence();
    streaming_device_barrier();
    const int entries = STREAMING_GROUP(0);
    for (int entry = int(tid); entry < entries; entry += int(threads)) {
        const u64 at = SEISMIC_CONST_LIST_ENTRIES + SEISMIC_CONST_LIST_ENTRY_WORDS * u64(entry);
        const int record = streaming_load(STREAMING_AT(list, (at + SEISMIC_CONST_LIST_TO) * stride));
        streaming_store(STREAMING_AT(commits, u64(record) * SEISMIC_COMMITS_STRIDE_0), sequence);
    }
    streaming_system_fence();
}
