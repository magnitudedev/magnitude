// Group r writes result row r through the owner's handle with operation r;
// each thread writes spans of eight elements of the row. The result is
// canonical (contiguous rows of N, N % 8 == 0). A value moves from `x` to the
// handle as loaded, without arithmetic, so a NaN payload reaches the owner
// unchanged.
#include <seismic/element.h>

typedef ELEMENT_OF(SEISMIC_RESULT_0) PublishElement;

kernel void owner_publish(
    device const float *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(result),
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const element::Out<PublishElement> out(result);
    for (ulong first = 8ul * lane; first < SEISMIC_DIM_N; first += 8ul * lanes) {
        float v[8];
        for (uint i = 0; i < 8; ++i) {
            v[i] = x[ulong(row) * SEISMIC_X_STRIDE_0 + (first + i) * SEISMIC_X_STRIDE_1];
        }
        const ulong at = ulong(row) * SEISMIC_RESULT_0_STRIDE_0 + first;
        switch (row) {
        case 0: // put
            for (uint i = 0; i < 8; ++i) {
                out.put(at + i, v[i]);
            }
            break;
        case 1: // put2 at even indices
            for (uint i = 0; i < 8; i += 2) {
                out.put2(at + i, v[i], v[i + 1]);
            }
            break;
        case 2: // put2 at odd indices
            out.put(at, v[0]);
            for (uint i = 1; i < 7; i += 2) {
                out.put2(at + i, v[i], v[i + 1]);
            }
            out.put(at + 7, v[7]);
            break;
        case 3: // put4 at four-element alignment
            out.put4(at, float4(v[0], v[1], v[2], v[3]));
            out.put4(at + 4, float4(v[4], v[5], v[6], v[7]));
            break;
        case 4: // put4 at an odd element
            out.put(at, v[0]);
            out.put4(at + 1, float4(v[1], v[2], v[3], v[4]));
            out.put(at + 5, v[5]);
            out.put(at + 6, v[6]);
            out.put(at + 7, v[7]);
            break;
        default: // put8 as one 16-byte store
            out.put8(at, float4(v[0], v[2], v[4], v[6]), float4(v[1], v[3], v[5], v[7]));
            break;
        }
    }
}
