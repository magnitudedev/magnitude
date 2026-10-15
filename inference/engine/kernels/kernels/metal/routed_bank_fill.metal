// Expert streaming: a streamed layer's absent experts (routed.seismic,
// `routed_bank_fill`), between its route and the resident `routed_gate_up`.
// Threadgroup row y is choice y of the decode rows. A hit binds its slot and
// returns. A choice the route requested waits for the host's
// acknowledgement (one thread, under system-scope fences, watching the
// heartbeat), binds the slot the host placed it in and, if it is the first
// requested choice of that expert with staging records, its threadgroups
// convert the gate, up and down records from staging into the
// slot by windows (lib/streaming/window.h), each conversion its own repack
// binding (lib/streaming/bind_*.h). Threadgroup 0 stamps the slot and, once
// the request is acknowledged and has not failed, commits it.

#include "lib/streaming/control.h"

#include "lib/streaming/bind_gate.h"
#include "lib/streaming/bind_up.h"
#include "lib/streaming/bind_down.h"
#include "lib/streaming/window.h"

#define STAGED_BYTES volatile coherent(system) device const uchar

kernel void routed_bank_fill(
    device const int *slots [[buffer(SEISMIC_BUFFER_SLOTS)]],
    SYSTEM int *stamps [[buffer(SEISMIC_BUFFER_STAMPS)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *control [[buffer(SEISMIC_BUFFER_CONTROL)]],
    SYSTEM const int *placement [[buffer(SEISMIC_BUFFER_PLACEMENT)]],
    SYSTEM const int *heartbeat [[buffer(SEISMIC_BUFFER_HEARTBEAT)]],
    STAGED_BYTES *staged_gate [[buffer(SEISMIC_BUFFER_STAGED_GATE)]],
    STAGED_BYTES *staged_up [[buffer(SEISMIC_BUFFER_STAGED_UP)]],
    STAGED_BYTES *staged_down [[buffer(SEISMIC_BUFFER_STAGED_DOWN)]],
    device uchar *expert_gate [[buffer(SEISMIC_BUFFER_EXPERT_GATE)]],
    device uchar *expert_up [[buffer(SEISMIC_BUFFER_EXPERT_UP)]],
    device uchar *expert_down [[buffer(SEISMIC_BUFFER_EXPERT_DOWN)]],
    SYSTEM int *slot_map [[buffer(SEISMIC_BUFFER_SLOT_MAP)]],
    SYSTEM int *commits [[buffer(SEISMIC_BUFFER_COMMITS)]],
    device int *bound [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 groups [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup int words[5];
    const uint threads = simdgroups * 32;
    const uint choice = group.y;
    const ulong m = choice / SEISMIC_DIM_K, k = choice % SEISMIC_DIM_K;
    device int *bound_word = bound + m * SEISMIC_RESULT_0_STRIDE_0 + k * SEISMIC_RESULT_0_STRIDE_1;
    const int resolved = slots[m * SEISMIC_SLOTS_STRIDE_0 + k * SEISMIC_SLOTS_STRIDE_1];
    if (resolved >= 0) {
        if (group.x == 0 && tid == 0)
            *bound_word = resolved;
        return;
    }
    const int sequence = step[0];
    const ulong column = ulong(choice) * SEISMIC_PLACEMENT_STRIDE_1;
    if (tid == 0) {
        const bool acknowledged = streaming::await_word(placement + 4 * SEISMIC_PLACEMENT_STRIDE_0 + column,
            control + streaming::FAILURE, heartbeat, sequence);
        words[0] = acknowledged ? 1 : 0;
        // The host's placement of this choice: its slot and its gate, up
        // and down staging records, published before the acknowledgement.
        if (acknowledged)
            for (uint row = 0; row < 4; ++row)
                words[1 + row] = placement[ulong(row) * SEISMIC_PLACEMENT_STRIDE_0 + column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // After a failure slot 0 keeps later addresses in bounds; the host
    // discards the step's results.
    const int slot = words[0] != 0 ? words[1] : 0;
    if (group.x == 0 && tid == 0)
        *bound_word = slot;
    if (words[0] == 0 || words[2] < 0)
        return;
    // The request union assigns one slot and record triple per expert; only
    // its first requested choice converts it.
    const int expert = control[streaming::EXPERTS + choice];
    for (uint prior = 0; prior < choice; ++prior) {
        const ulong at = ulong(prior) * SEISMIC_PLACEMENT_STRIDE_1;
        const int prior_slot = slots[(prior / SEISMIC_DIM_K) * SEISMIC_SLOTS_STRIDE_0 +
            (prior % SEISMIC_DIM_K) * SEISMIC_SLOTS_STRIDE_1];
        if (prior_slot < 0 && control[streaming::EXPERTS + prior] == expert && placement[at] == slot &&
            placement[SEISMIC_PLACEMENT_STRIDE_0 + at] >= 0)
            return;
    }
    const ulong records[3] = {ulong(words[2]), ulong(words[3]), ulong(words[4])};
    const ulong gate_items = streaming::record_items<GateBinding>(seismic_words);
    const ulong up_items = streaming::record_items<UpBinding>(seismic_words);
    const ulong down_items = streaming::record_items<DownBinding>(seismic_words);
    for (ulong item = group.x; item < gate_items; item += groups.x)
        streaming::convert_item<GateBinding>(staged_gate, expert_gate, seismic_words, records[0], ulong(slot), item,
            shared, tid, threads, sg, simdgroups, lane);
    for (ulong item = group.x; item < up_items; item += groups.x)
        streaming::convert_item<UpBinding>(staged_up, expert_up, seismic_words, records[1], ulong(slot), item,
            shared, tid, threads, sg, simdgroups, lane);
    for (ulong item = group.x; item < down_items; item += groups.x)
        streaming::convert_item<DownBinding>(staged_down, expert_down, seismic_words, records[2], ulong(slot),
            item, shared, tid, threads, sg, simdgroups, lane);
    if (group.x != 0 || tid != 0)
        return;
    stamps[ulong(slot) * SEISMIC_STAMPS_STRIDE_0] = sequence;
    // Commit only a conversion of an acknowledged, successful request.
    if (!streaming::await_word(control + streaming::ACKNOWLEDGED, control + streaming::FAILURE, heartbeat,
            sequence) ||
        control[streaming::FAILURE] == sequence || expert < 0 || ulong(expert) >= SEISMIC_DIM_X)
        return;
    slot_map[ulong(expert) * SEISMIC_SLOT_MAP_STRIDE_0] = slot;
    streaming::system_fence();
    commits[ulong(slot) * SEISMIC_COMMITS_STRIDE_0] = sequence;
    streaming::system_fence();
}
