// token_confidence in two launches. token_confidence_partition: each of
// CONFIDENCE_PARTS threadgroups of a selection reduces its contiguous slice of
// the selection's logits row to (maximum, sum of exp(logit - maximum)).
// token_confidence_merge: one thread per selection combines its slices and
// publishes the probability of the selected token (0 when its selection
// failed).
#include "lib/core/reduce.h"

#define CONFIDENCE_THREADS 256u
#define CONFIDENCE_PARTS 64u

kernel void token_confidence_partition(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float shared[CONFIDENCE_THREADS / 32u];
    const uint part = group.x, row = group.y;
    const ulong vocabulary = SEISMIC_DIM_V;
    const ulong span = (vocabulary + CONFIDENCE_PARTS - 1) / CONFIDENCE_PARTS;
    const ulong begin = metal::min(ulong(part) * span, vocabulary), end = metal::min(begin + span, vocabulary);
    device const float *line = logits + ulong(rows[ulong(row) * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    float local = -INFINITY;
    for (ulong i = begin + thread_index; i < end; i += CONFIDENCE_THREADS)
        local = metal::max(local, line[i * SEISMIC_LOGITS_STRIDE_1]);
    const float maximum = reduce::group_max<CONFIDENCE_THREADS / 32u>(local, shared, sg, lane);
    float partial = 0.0f;
    for (ulong i = begin + thread_index; i < end; i += CONFIDENCE_THREADS)
        partial += metal::exp(line[i * SEISMIC_LOGITS_STRIDE_1] - maximum);
    const float total = reduce::group_sum<CONFIDENCE_THREADS / 32u>(partial, shared, sg, lane);
    if (thread_index == 0) {
        device float *out = partials + (ulong(row) * CONFIDENCE_PARTS + part) * 2u;
        out[0] = maximum;
        out[1] = total;
    }
}

kernel void token_confidence_merge(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    device float *confidence [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= SEISMIC_DIM_M) return;
    device const float *in = partials + ulong(row) * CONFIDENCE_PARTS * 2u;
    float maximum = -INFINITY;
    for (uint part = 0; part < CONFIDENCE_PARTS; ++part)
        maximum = metal::max(maximum, in[part * 2u]);
    float total = 0.0f;
    for (uint part = 0; part < CONFIDENCE_PARTS; ++part)
        if (in[part * 2u + 1u] > 0.0f)
            total += in[part * 2u + 1u] * metal::exp(in[part * 2u] - maximum);
    device const float *line = logits + ulong(rows[ulong(row) * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    device const int *selected = selection + ulong(row) * SEISMIC_SELECTION_STRIDE_0;
    confidence[ulong(row) * SEISMIC_RESULT_0_STRIDE_0] = selected[SEISMIC_SELECTION_STRIDE_1] == 0
        ? metal::exp(line[ulong(selected[0]) * SEISMIC_LOGITS_STRIDE_1] - maximum) / total
        : 0.0f;
}
