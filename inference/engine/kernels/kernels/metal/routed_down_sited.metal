// The CPU's join of a streamed layer's general decode combine
// (`routed_down_sited` in routed.seismic): thread (n, m) sums channel n of
// row m (`lib/routed/down_sum.h`) over the published projections of
// `routed_down_project`. A threadgroup of a step with a CPU choice spins on
// the step's slot of the site table (thread 0, `streaming_await_slot`); then each thread
// adds the CPU's answer, read through system-coherent loads
// (`lib/routed/sited.h`), after the slot-order sum.

#define KERNEL_A SEISMIC_ELEMENT_A
#include "lib/streaming/control.h"
#include "../streaming/site.hpp"
#include "lib/routed/routed.h"
#include "lib/routed/down_sum.h"
#include "lib/routed/sited.h"

namespace {

// Four adjacent channels n .. n + 3 of row m's published projections, in A.
struct PublishedProjections4 {
    device const uchar *published;
    constant ulong *seismic_words;
    ulong m;
    uint n;
    float4 at4(uint k) const thread {
        device const typename routed::Act::storage *row =
            reinterpret_cast<device const typename routed::Act::storage *>(published)
            + m * SEISMIC_PUBLISHED_STRIDE_0 + ulong(k) * SEISMIC_PUBLISHED_STRIDE_1;
        if (SEISMIC_PUBLISHED_STRIDE_2 == 1)
            return routed::Act::load4(row + ulong(n));
        return float4(
            routed::Act::load(row[ulong(n + 0u) * SEISMIC_PUBLISHED_STRIDE_2]),
            routed::Act::load(row[ulong(n + 1u) * SEISMIC_PUBLISHED_STRIDE_2]),
            routed::Act::load(row[ulong(n + 2u) * SEISMIC_PUBLISHED_STRIDE_2]),
            routed::Act::load(row[ulong(n + 3u) * SEISMIC_PUBLISHED_STRIDE_2]));
    }
};

// Four adjacent channels n .. n + 3 of row m of the slot's answer `output`
// (`site_output`) where the CPU answered the step, else none.
inline routed::Answer4 answer_at4(SYSTEM const float *output, bool answered, ulong m, uint n,
    constant ulong *seismic_words) {
    if (!answered)
        return routed::Answer4{false, float4(0.0f)};
    streaming_system_fence();
    SYSTEM const float *row = output + m * SEISMIC_SITE_OUTPUT_STRIDE_1;
    if (SEISMIC_SITE_OUTPUT_STRIDE_2 == 1 && (reinterpret_cast<ulong>(row + ulong(n)) & 15ul) == 0ul)
        return routed::Answer4{true, *reinterpret_cast<SYSTEM const float4 *>(row + ulong(n))};
    return routed::Answer4{true, float4(
        row[ulong(n + 0u) * SEISMIC_SITE_OUTPUT_STRIDE_2],
        row[ulong(n + 1u) * SEISMIC_SITE_OUTPUT_STRIDE_2],
        row[ulong(n + 2u) * SEISMIC_SITE_OUTPUT_STRIDE_2],
        row[ulong(n + 3u) * SEISMIC_SITE_OUTPUT_STRIDE_2])};
}

} // namespace

kernel void routed_down_sited(
    device const float *base [[buffer(SEISMIC_BUFFER_BASE)]],
    device const uchar *published [[buffer(SEISMIC_BUFFER_PUBLISHED)]],
    device const int *sources [[buffer(SEISMIC_BUFFER_SOURCES)]],
    device const float *weights [[buffer(SEISMIC_BUFFER_WEIGHTS)]],
    device const int *step [[buffer(SEISMIC_BUFFER_STEP)]],
    SYSTEM int *site [[buffer(SEISMIC_BUFFER_SITE)]],
    SYSTEM const float *site_output [[buffer(SEISMIC_BUFFER_SITE_OUTPUT)]],
    SEISMIC_RESULT_0_PARAM(value_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
    constexpr uint H = uint(SEISMIC_DIM_H);
    threadgroup int ready;
    threadgroup ulong slot_shared;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT_0)> value(value_cells);
    const ulong m = group.y;
    ulong q = 0ul;
    ulong slot = 0ul;
    if (tid == 0u) {
        q = streaming_step(step, SEISMIC_STEP_STRIDE_0);
        slot = q % ulong(SEISMIC_DIM_FLIGHT);
        slot_shared = slot;
    }
    const bool answered = routed::site_answered(sources, site + slot * SEISMIC_SITE_STRIDE_0, q, &ready, tid, seismic_words);
    slot = slot_shared;
    const uint c = tid * 4u;
    if (c >= threads.x)
        return;
    const uint n = group.x * threads.x + c;
    if (n >= H)
        return;
    SYSTEM const float *slot_output = site_output + slot * SEISMIC_SITE_OUTPUT_STRIDE_0;
    if (c + 4u <= threads.x && n + 4u <= H) {
        const routed::Answer4 answer = answer_at4(slot_output, answered, m, n, seismic_words);
        routed::down_publish4(base, weights, value, m, n, PublishedProjections4{published, seismic_words, m, n},
            answer, seismic_words);
    } else {
        for (uint i = 0u; i < 4u && c + i < threads.x && n + i < H; ++i) {
            const routed::Answer answer = routed::answer_at(slot_output, answered, m, n + i, seismic_words);
            routed::down_publish(base, weights, value, m, n + i,
                routed::PublishedProjections{published, seismic_words, m, n + i}, answer, seismic_words);
        }
    }
}
