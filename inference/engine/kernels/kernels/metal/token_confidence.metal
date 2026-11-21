// token_confidence in two launches. token_confidence_partition: each of
// CONFIDENCE_PARTS threadgroups of a selection reduces its contiguous slice of
// the selection's logits row in a single pass to (maximum, sum of exp(logit -
// maximum)). token_confidence_merge: one simdgroup per selection combines its
// slices and publishes the probability of the selected token (0 when its
// selection failed).
#include <seismic/element.h>
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
    threadgroup reduce::OnlineSoftmax shared[CONFIDENCE_THREADS / 32u];
    const uint part = group.x, row = group.y;
    const ulong vocabulary = SEISMIC_DIM_V;
    const ulong span = (vocabulary + CONFIDENCE_PARTS - 1) / CONFIDENCE_PARTS;
    const ulong begin = metal::min(ulong(part) * span, vocabulary), end = metal::min(begin + span, vocabulary);
    device const float *line = logits + ulong(rows[ulong(row) * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    const ulong count = end - begin;
    if (SEISMIC_LOGITS_STRIDE_1 == 1ul) {
        device const float *slice = line + begin;
        const ulong count4 = count & ~3ul;
        for (ulong i = ulong(thread_index) * 4ul; i < count4; i += ulong(CONFIDENCE_THREADS) * 4ul) {
            const float4 v = float4(*reinterpret_cast<device const packed_float4 *>(slice + i));
            state.add(v.x);
            state.add(v.y);
            state.add(v.z);
            state.add(v.w);
        }
        for (ulong i = count4 + thread_index; i < count; i += CONFIDENCE_THREADS)
            state.add(slice[i]);
    } else {
        for (ulong i = begin + thread_index; i < end; i += CONFIDENCE_THREADS)
            state.add(line[i * SEISMIC_LOGITS_STRIDE_1]);
    }
    const reduce::OnlineSoftmax result =
        reduce::group_online_softmax<CONFIDENCE_THREADS / 32u>(state, shared, sg, lane);
    if (thread_index == 0) {
        device float *out = partials + (ulong(row) * CONFIDENCE_PARTS + part) * 2u;
        out[0] = result.max;
        out[1] = result.sum;
    }
}

kernel void token_confidence_merge(
    device const float *logits [[buffer(SEISMIC_BUFFER_LOGITS)]],
    device const int *rows [[buffer(SEISMIC_BUFFER_ROWS)]],
    device const int *selection [[buffer(SEISMIC_BUFFER_SELECTION)]],
    device const float *partials [[buffer(SEISMIC_BUFFER_SCRATCH_PARTIALS)]],
    SEISMIC_OUT_CONFIDENCE_PARAM(confidence_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const float *in = partials + ulong(row) * CONFIDENCE_PARTS * 2u;
    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    for (uint part = lane; part < CONFIDENCE_PARTS; part += 32u)
        state = reduce::OnlineSoftmax::merge(state, reduce::OnlineSoftmax{in[part * 2u], in[part * 2u + 1u]});
    state = reduce::warp_online_softmax(state);
    if (lane == 0) {
        device const float *line = logits + ulong(rows[ulong(row) * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
        device const int *selected = selection + ulong(row) * SEISMIC_SELECTION_STRIDE_0;
        const element::Out<element::F32> confidence(confidence_cells);
        confidence.put(ulong(row) * SEISMIC_CONFIDENCE_STRIDE_0,
            selected[SEISMIC_SELECTION_STRIDE_1] == 0 && state.sum != 0.0f
                ? metal::exp(line[ulong(selected[0]) * SEISMIC_LOGITS_STRIDE_1] - state.max) / state.sum
                : 0.0f);
    }
}
