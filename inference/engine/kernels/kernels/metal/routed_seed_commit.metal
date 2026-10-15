// Expert streaming, a scratch fill's seeding (`routed_seed_commit` in
// routed.seismic): after `routed_copy` placed the converted experts into
// their bank slots, the listed slots are stamped and, unless the step failed,
// their map entries published under a system-scope fence before their
// commits, as `routed_bank_fill` commits a decode step's conversions.

#include "lib/streaming/control.h"

kernel void routed_seed_commit(
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM const int *control [[buffer(SEISMIC_BUFFER_CONTROL)]],
    SYSTEM int *slot_map [[buffer(SEISMIC_BUFFER_SLOT_MAP)]],
    SYSTEM int *stamps [[buffer(SEISMIC_BUFFER_STAMPS)]],
    SYSTEM int *commits [[buffer(SEISMIC_BUFFER_COMMITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const int sequence = step[0];
    streaming::system_fence();
    if (list[0] != sequence || control[streaming::FAILURE] == sequence)
        return;
    const int count = min(list[1], int(SEISMIC_DIM_E));
    for (int entry = int(tid); entry < count; entry += int(threads)) {
        const int expert = list[2 + 2 * entry], slot = list[3 + 2 * entry];
        if (expert < 0 || ulong(expert) >= SEISMIC_DIM_X || slot < 0 || ulong(slot) >= SEISMIC_DIM_BANK)
            continue;
        stamps[ulong(slot) * SEISMIC_STAMPS_STRIDE_0] = sequence;
        slot_map[ulong(expert) * SEISMIC_SLOT_MAP_STRIDE_0] = slot;
    }
    streaming::system_fence();
    threadgroup_barrier(mem_flags::mem_device);
    for (int entry = int(tid); entry < count; entry += int(threads)) {
        const int expert = list[2 + 2 * entry], slot = list[3 + 2 * entry];
        if (expert < 0 || ulong(expert) >= SEISMIC_DIM_X || slot < 0 || ulong(slot) >= SEISMIC_DIM_BANK)
            continue;
        commits[ulong(slot) * SEISMIC_COMMITS_STRIDE_0] = sequence;
    }
    streaming::system_fence();
}
