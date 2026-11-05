// Expert streaming, a merge set's pass's map resolved into device memory
// (`routed_pass_map` in routed.seismic), over the backend's shim
// (streaming/contract.hpp): the step word `step` copied into `pass_step`,
// and each of `experts` map rows the record the map holds for the step
// (`streaming_held`, the map protocol's one read) written to its row of
// `resolved`, mapped from the step on, or -1. Every thread of the group
// (`tid` of `threads`) resolves its experts. The pass's expert entries read
// these device words in place of the host-shared ones, so no GEMM block
// reads a host-shared word.
STREAMING_FN void streaming_pass_map(STREAMING_SYSTEM_READ map, u64 row_stride, u64 word_stride, u64 experts,
    STREAMING_DEVICE_READ step, u64 step_stride, STREAMING_DEVICE_WORDS resolved, u64 resolved_row,
    u64 resolved_word, STREAMING_DEVICE_WORDS pass_step, u64 pass_step_stride, u32 tid, u32 threads) {
    const int low = streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_LOW * step_stride));
    const int high = streaming_device_load(STREAMING_AT(step, SEISMIC_CONST_STEP_HIGH * step_stride));
    const u64 q = (u64(u32(high)) << 32) | u64(u32(low));
    if (tid == 0u) {
        streaming_device_store(STREAMING_AT(pass_step, SEISMIC_CONST_STEP_LOW * pass_step_stride), low);
        streaming_device_store(STREAMING_AT(pass_step, SEISMIC_CONST_STEP_HIGH * pass_step_stride), high);
    }
    for (u32 expert = tid; expert < u32(experts); expert += threads) {
        const u64 row = u64(expert) * resolved_row;
        streaming_device_store(STREAMING_AT(resolved, row + SEISMIC_CONST_MAP_FROM * resolved_word), low);
        streaming_device_store(STREAMING_AT(resolved, row + (SEISMIC_CONST_MAP_FROM + 1) * resolved_word), high);
        streaming_device_store(STREAMING_AT(resolved, row + SEISMIC_CONST_MAP_UNTIL * resolved_word), -1);
        streaming_device_store(STREAMING_AT(resolved, row + (SEISMIC_CONST_MAP_UNTIL + 1) * resolved_word), -1);
        streaming_device_store(STREAMING_AT(resolved, row + SEISMIC_CONST_MAP_RECORD * resolved_word),
            streaming_held(STREAMING_AT(map, u64(expert) * row_stride), word_stride, q));
    }
}
