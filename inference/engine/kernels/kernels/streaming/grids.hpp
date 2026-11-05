// Expert streaming, a scratch request's round grids (`routed_scratch_grids`
// in routed.seismic), over the backend's shim (streaming/contract.hpp):
// after round 0's wait, one thread reads the count of records the request's
// rounds convert from its answer (none where the host failed it) and writes
// each round's park grid and its converts' grids over records of F and of
// H rows (`rows_groups`, `columns_groups` groups a row of records). The
// step's later launches read them in order. The includer includes
// streaming/round.hpp first.

// Round `round`'s grid `grid` of `grids`: (x, y, 1).
STREAMING_FN void streaming_write_grid(STREAMING_DEVICE_WORDS grids, u32 round, int grid, int x, int y
    STREAMING_WORDS_PARAM) {
    const u64 at = u64(round) * SEISMIC_GRIDS_STRIDE_0 + u64(grid) * SEISMIC_GRIDS_STRIDE_1;
    streaming_device_store(STREAMING_AT(grids, at), x);
    streaming_device_store(STREAMING_AT(grids, at + SEISMIC_GRIDS_STRIDE_2), y);
    streaming_device_store(STREAMING_AT(grids, at + 2 * SEISMIC_GRIDS_STRIDE_2), 1);
}

// One thread calls it, for step `q`, its request the SCRATCH_ASK lane of
// its slot of the layer's scratch table `scratch`.
STREAMING_FN void streaming_scratch_grids(u64 q, STREAMING_SYSTEM_READ scratch, STREAMING_DEVICE_WORDS grids
    STREAMING_WORDS_PARAM) {
    STREAMING_SYSTEM_READ ask =
        STREAMING_AT(scratch, streaming_scratch_lane(q, u64(SEISMIC_CONST_SCRATCH_ASK) STREAMING_WORDS_ARG));
    const int count =
        streaming_load_u64_acquire(STREAMING_AT(ask, SEISMIC_CONST_SLOT_ANSWER * SEISMIC_SCRATCH_STRIDE_1)) == q
        ? streaming_load(STREAMING_AT(ask, SEISMIC_CONST_SCRATCH_STAGED * SEISMIC_SCRATCH_STRIDE_1))
        : 0;
    const int per_round = int(SEISMIC_PARAM_ROUND_RECORDS);
    for (u32 round = 0u; round < u32(SEISMIC_DIM_R); ++round) {
        const int left = count - int(round) * per_round;
        const int records = left < 0 ? 0 : left < per_round ? left : per_round;
        streaming_write_grid(grids, round, int(SEISMIC_CONST_PARK_GRID), records < 1 ? records : 1, 1
            STREAMING_WORDS_ARG);
        streaming_write_grid(grids, round, int(SEISMIC_CONST_ROWS_GRID), int(SEISMIC_PARAM_ROWS_GROUPS), records
            STREAMING_WORDS_ARG);
        streaming_write_grid(grids, round, int(SEISMIC_CONST_COLUMNS_GRID), int(SEISMIC_PARAM_COLUMNS_GROUPS), records
            STREAMING_WORDS_ARG);
    }
}
