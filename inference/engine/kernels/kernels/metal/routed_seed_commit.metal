// Expert streaming, a prompt group's seeding (`routed_seed_commit` in
// routed.seismic): one threadgroup's seeding (streaming/seed.hpp).

#include "lib/streaming/control.h"
#include "../streaming/list.hpp"
#include "../streaming/round.hpp"
#include "../streaming/seed.hpp"

kernel void routed_seed_commit(
    SYSTEM const int *list [[buffer(SEISMIC_BUFFER_LIST)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM const int *scratch [[buffer(SEISMIC_BUFFER_SCRATCH)]],
    SYSTEM int *commits [[buffer(SEISMIC_BUFFER_COMMITS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    // The count.
    threadgroup int words[STREAMING_SEED_WORDS];
    streaming_seed_commit(list, step, SEISMIC_STEP_STRIDE_0, scratch, commits, words, tid, threads, seismic_words);
}
