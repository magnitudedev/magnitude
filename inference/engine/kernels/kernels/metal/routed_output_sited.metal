// The CPU's join of a streamed layer of Qwen's fused decode output
// (`routed_output_sited` in routed.seismic): thread (n, m) joins channel n of
// row m (`lib/routed/output_sum.h`) over the published projections of
// `routed_output_project`, waiting as `routed_down_sited` waits.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/streaming/control.h"
#include "../streaming/site.hpp"
#include "lib/routed/routed.h"
#include "lib/routed/output_sum.h"
#include "lib/routed/sited.h"

kernel void routed_output_sited(
    device const float *residual [[buffer(SEISMIC_BUFFER_RESIDUAL)]],
    device const float *inject [[buffer(SEISMIC_BUFFER_INJECT)]],
    device const uchar *published [[buffer(SEISMIC_BUFFER_PUBLISHED)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const float *scores [[buffer(SEISMIC_BUFFER_SCORES)]],
    device const float *coefficient [[buffer(SEISMIC_BUFFER_COEFFICIENT)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *site [[buffer(SEISMIC_BUFFER_SITE)]],
    SYSTEM const float *site_output [[buffer(SEISMIC_BUFFER_SITE_OUTPUT)]],
    SEISMIC_RESULT_0_PARAM(value_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
    threadgroup int ready;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);
    const ulong m = group.y;
    const ulong q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
    const ulong slot = q % ulong(SEISMIC_DIM_FLIGHT);
    const bool answered = routed::site_answered(sources, site + slot * SEISMIC_SITE_STRIDE_0, q, &ready, tid, seismic_words);
    const uint first = group.x * threads.x;
    const uint c = tid * 4u;
    const uint n = first + c;
    if (c >= threads.x || n >= uint(SEISMIC_DIM_H))
        return;
    SYSTEM const float *slot_output = site_output + slot * SEISMIC_SITE_OUTPUT_STRIDE_0;
    if (c + 4u <= threads.x && n + 4u <= uint(SEISMIC_DIM_H)) {
        const routed::Answer4 answer = routed::answer_at4(slot_output, answered, m, n, seismic_words);
        routed::output_publish4(residual, inject, scores, coefficient, value, m, n,
            routed::PublishedProjections{published, seismic_words, m, n}, answer, seismic_words);
    } else {
        const uint end = metal::min(threads.x, uint(SEISMIC_DIM_H) - first);
        for (uint i = c; i < end; ++i) {
            const uint ch = first + i;
            const routed::Answer answer = routed::answer_at(slot_output, answered, m, ch, seismic_words);
            routed::output_publish(residual, inject, scores, coefficient, value, m, ch,
                routed::PublishedProjections{published, seismic_words, m, ch}, answer, seismic_words);
        }
    }
}
