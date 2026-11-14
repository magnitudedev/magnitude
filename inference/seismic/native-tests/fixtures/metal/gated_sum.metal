#include <seismic/element.h>

// One participant sums in element order after staging the whole vector in
// threadgroup memory.
kernel void gated_staged(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    threadgroup float *staging [[threadgroup(0)]]) {
    const ulong n = SEISMIC_DIM_N;
    for (ulong i = 0; i < n; ++i) {
        staging[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (ulong i = 0; i < n; ++i) {
        total += staging[i];
    }
    element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells).put(0, total);
}

// One participant sums in element order after mirroring the vector into
// scratch.
kernel void gated_mirrored(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *mirror [[buffer(SEISMIC_BUFFER_SCRATCH_MIRROR)]],
    SEISMIC_RESULT_0_PARAM(result_cells),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]]) {
    const ulong n = SEISMIC_DIM_N;
    for (ulong i = 0; i < n; ++i) {
        mirror[i] = x[i * SEISMIC_X_STRIDE_0];
    }
    float total = 0.0f;
    for (ulong i = 0; i < n; ++i) {
        total += mirror[i];
    }
    element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result_cells).put(0, total);
}
