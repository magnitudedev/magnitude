// token_confidence in two launches, as `metal/token_confidence.metal`:
// token_confidence_partition reduces each of CONFIDENCE_PARTS contiguous
// slices of a selection's logits row to (maximum, sum of exp(logit -
// maximum)); token_confidence_merge, one thread per selection, combines them
// and publishes the probability of the selected token (0 when its selection
// failed).
#include "lib/core/reduce.cuh"

#define CONFIDENCE_PARTS 64ull

extern "C" __global__ void token_confidence_partition(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    __shared__ float shared[32];
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const int *rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    float *partials = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    const u64 part = blockIdx.x, row = blockIdx.y;
    const u64 vocabulary = SEISMIC_DIM_V;
    const u64 span = (vocabulary + CONFIDENCE_PARTS - 1) / CONFIDENCE_PARTS;
    const u64 begin = min(part * span, vocabulary), end = min(begin + span, vocabulary);
    const float *line = logits + u64(rows[row * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    float local = -__int_as_float(0x7f800000);
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x)
        local = fmaxf(local, line[i * SEISMIC_LOGITS_STRIDE_1]);
    const float maximum = reduce::group_max(local, shared);
    float partial = 0.0f;
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x)
        partial += expf(line[i * SEISMIC_LOGITS_STRIDE_1] - maximum);
    const float total = reduce::group_sum(partial, shared);
    if (threadIdx.x == 0) {
        partials[(row * CONFIDENCE_PARTS + part) * 2] = maximum;
        partials[(row * CONFIDENCE_PARTS + part) * 2 + 1] = total;
    }
}

extern "C" __global__ void token_confidence_merge(SEISMIC_KERNEL_PARAMS) {
    typedef unsigned long long u64;
    const float *logits = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_LOGITS));
    const int *rows = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_ROWS));
    const int *selection = reinterpret_cast<const int *>(SEISMIC_PTR(SEISMIC_BUFFER_SELECTION));
    const float *partials = reinterpret_cast<const float *>(SEISMIC_PTR(SEISMIC_BUFFER_SCRATCH_PARTIALS));
    float *confidence = reinterpret_cast<float *>(SEISMIC_PTR(SEISMIC_RESULT_0_BUFFER));
    const u64 row = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= SEISMIC_DIM_M)
        return;
    const float *in = partials + row * CONFIDENCE_PARTS * 2;
    float maximum = -__int_as_float(0x7f800000);
    for (u64 part = 0; part < CONFIDENCE_PARTS; ++part)
        maximum = fmaxf(maximum, in[part * 2]);
    float total = 0.0f;
    for (u64 part = 0; part < CONFIDENCE_PARTS; ++part)
        if (in[part * 2 + 1] > 0.0f)
            total += in[part * 2 + 1] * expf(in[part * 2] - maximum);
    const float *line = logits + u64(rows[row * SEISMIC_ROWS_STRIDE_0]) * SEISMIC_LOGITS_STRIDE_0;
    const int *selected = selection + row * SEISMIC_SELECTION_STRIDE_0;
    confidence[row * SEISMIC_RESULT_0_STRIDE_0] = selected[SEISMIC_SELECTION_STRIDE_1] == 0
        ? expf(line[u64(selected[0]) * SEISMIC_LOGITS_STRIDE_1] - maximum) / total
        : 0.0f;
}
