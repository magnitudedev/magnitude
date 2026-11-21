// token_confidence in two launches, as `metal/token_confidence.metal`:
// token_confidence_partition reduces each of CONFIDENCE_PARTS contiguous
// slices of a selection's logits row in a single pass to (maximum, sum of
// exp(logit - maximum)); token_confidence_merge, one warp per selection,
// combines them and publishes the probability of the selected token (0 when
// its selection failed).
#include <seismic/element.cuh>
#include "lib/core/reduce.cuh"

#define CONFIDENCE_PARTS 64ull

extern "C" __global__ void token_confidence_partition(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ reduce::OnlineSoftmax shared[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const int *rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    float *partials = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const u64 part = blockIdx.x, row = blockIdx.y;
    const u64 vocabulary = SEISMIC_DIM_V;
    const u64 span = (vocabulary + CONFIDENCE_PARTS - 1) / CONFIDENCE_PARTS;
    const u64 begin = min(part * span, vocabulary), end = min(begin + span, vocabulary);
    const float *line = logits + u64(rows[row * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    const u64 count = end - begin;
    const float *slice = line + begin;
    if (SEISMIC_LOGITS_STRIDE_1 == 1ull && (reinterpret_cast<u64>(slice) & 3ull) == 0ull) {
        const u64 misalign = (reinterpret_cast<u64>(slice) & 15ull) >> 2;
        const u64 head = misalign == 0ull ? 0ull : min(4ull - misalign, count);
        if (threadIdx.x < head)
            state.add(slice[threadIdx.x]);
        const float *aligned = slice + head;
        const u64 vec_count = (count - head) & ~3ull;
        const u64 step4 = static_cast<u64>(blockDim.x) * 4ull;
        for (u64 i = static_cast<u64>(threadIdx.x) * 4ull; i < vec_count; i += step4) {
            const float4 v = *reinterpret_cast<const float4 *>(aligned + i);
            state.add(v.x);
            state.add(v.y);
            state.add(v.z);
            state.add(v.w);
        }
        for (u64 i = head + vec_count + threadIdx.x; i < count; i += blockDim.x)
            state.add(slice[i]);
    } else {
        for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x)
            state.add(line[i * SEISMIC_LOGITS_STRIDE_1]);
    }
    const reduce::OnlineSoftmax result = reduce::group_online_softmax(state, shared);
    if (threadIdx.x == 0) {
        partials[(row * CONFIDENCE_PARTS + part) * 2] = result.max;
        partials[(row * CONFIDENCE_PARTS + part) * 2 + 1] = result.sum;
    }
}

extern "C" __global__ void token_confidence_merge(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const int *rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    const int *selection = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const float *partials = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const auto confidence = SEISMIC_OUT_CONFIDENCE;
    const u64 row = blockIdx.x;
    const float *in = partials + row * CONFIDENCE_PARTS * 2;
    reduce::OnlineSoftmax state = reduce::OnlineSoftmax::empty();
    for (u64 part = threadIdx.x; part < CONFIDENCE_PARTS; part += 32ull)
        state = reduce::OnlineSoftmax::merge(state, reduce::OnlineSoftmax{in[part * 2], in[part * 2 + 1]});
    state = reduce::warp_online_softmax(state);
    if (threadIdx.x == 0) {
        const float *line = logits + u64(rows[row * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
        const int *selected = selection + row * SEISMIC_SELECTION_STRIDE_0;
        confidence.put(row * SEISMIC_CONFIDENCE_STRIDE_0,
            selected[SEISMIC_SELECTION_STRIDE_1] == 0 && state.sum != 0.0f
                ? expf(line[u64(selected[0]) * SEISMIC_LOGITS_STRIDE_1] - state.max) / state.sum
                : 0.0f);
    }
}
