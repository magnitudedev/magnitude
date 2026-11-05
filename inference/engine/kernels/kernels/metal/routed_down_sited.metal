// The CPU's join of a streamed layer's general decode combine
// (`routed_down_sited` in routed.seismic): thread (n, m) sums channel n of
// row m (`lib/routed/down_sum.h`) over the published projections of
// `routed_down_project`. A threadgroup of a step with a CPU choice spins on
// the step's slot of the site table (thread 0, `streaming_await_slot`); then each thread
// adds the CPU's answer, read through system-coherent loads
// (`lib/routed/sited.h`), after the slot-order sum.

#include "lib/streaming/control.h"
#include "../streaming/site.hpp"
#include "lib/routed/routed.h"
#include "lib/routed/down_sum.h"
#include "lib/routed/sited.h"

kernel void routed_down_sited(
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const uchar *published [[buffer(SEISMIC_BUFFER_PUBLISHED)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM const int *site [[buffer(SEISMIC_BUFFER_SITE)]],
    SYSTEM const float *site_output [[buffer(SEISMIC_BUFFER_SITE_OUTPUT)]],
    device uchar *value [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
    threadgroup int ready;
    const ulong m = group.y;
    const ulong q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    const ulong slot = q % ulong(SEISMIC_DIM_FLIGHT);
    const bool answered = routed::site_answered(sources, site + slot * SEISMIC_SITE_STRIDE_0, q, &ready, tid, seismic_words);
    const uint n = group.x * threads.x + tid;
    if (n >= uint(SEISMIC_DIM_H))
        return;
    const routed::Answer answer = routed::answer_at(site_output + slot * SEISMIC_SITE_OUTPUT_STRIDE_0, answered, m, n,
        seismic_words);
    routed::down_publish(base, weights, value, m, n, routed::PublishedProjections{published, seismic_words, m, n}, answer,
        seismic_words);
}
