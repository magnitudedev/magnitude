// Expert streaming, a scratch window's round grids (`routed_scratch_grids`
// in routed.seismic), over the backend's shim (streaming/contract.hpp):
// after the window's round 0 landing launch, one thread reads the count of
// records the window's rounds convert from its round-0 lane's answer (none
// where the host failed it) and writes each round's park grid,
// `routed_round`'s: LAND_GROUPS threadgroups while the round has records, or
// the round before has and the window fill is not followed
// (SCRATCH_FOLLOWED), else none. A round-0 lane still unanswered here had
// round 0's wait end on its still pulse (`streaming_await_slot`), its count
// unknown: every round of the window launches and none converts, so the
// device arrives at every round the host serves, in order, as it does once
// the count is known. The window's later launches read them in order. The
// includer includes streaming/round.hpp first.

// One thread calls it, for step `q`, reading window `window[0]`'s round-0
// lane of its slot of the layer's scratch table `scratch`.
STREAMING_FN void streaming_scratch_grids(u64 q, STREAMING_SYSTEM_READ scratch, STREAMING_DEVICE_READ window,
    STREAMING_DEVICE_WORDS grids STREAMING_WORDS_PARAM) {
    const u64 w = u64(u32(streaming_device_load(window)));
    STREAMING_SYSTEM_READ first = STREAMING_AT(scratch,
        streaming_scratch_lane(q, u64(SEISMIC_CONST_SCRATCH_ROUND) + w * SEISMIC_DIM_RW STREAMING_WORDS_ARG));
    const u64 answer = streaming_load_u64_acquire(STREAMING_AT(first, SEISMIC_CONST_SLOT_ANSWER * SEISMIC_SCRATCH_STRIDE_1));
    const bool unanswered = answer != q && answer != (q | (u64(u32(SEISMIC_CONST_SLOT_FAILED)) << 32));
    const int count = answer == q ? streaming_load(STREAMING_AT(first, SEISMIC_CONST_SCRATCH_STAGED * SEISMIC_SCRATCH_STRIDE_1)) : 0;
    const bool after =
        answer != q || streaming_load(STREAMING_AT(first, SEISMIC_CONST_SCRATCH_FOLLOWED * SEISMIC_SCRATCH_STRIDE_1)) == 0;
    const int per_round = int(SEISMIC_PARAM_ROUND_RECORDS);
    for (u32 round = 0u; round < u32(SEISMIC_DIM_RW); ++round) {
        const int left = count - int(round) * per_round;
        const int before = round > 0u ? int(round) - 1 : 0;
        const bool parks = unanswered || left > 0 || (after && count > before * per_round);
        const u64 at = u64(round) * SEISMIC_GRIDS_STRIDE_0;
        streaming_device_store(STREAMING_AT(grids, at), parks ? int(SEISMIC_CONST_LAND_GROUPS) : 0);
        streaming_device_store(STREAMING_AT(grids, at + SEISMIC_GRIDS_STRIDE_1), 1);
        streaming_device_store(STREAMING_AT(grids, at + 2 * SEISMIC_GRIDS_STRIDE_1), 1);
    }
}
