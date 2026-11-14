#include <seismic/element.h>

kernel void slab_probe(
    device const ulong *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_RESULT_0_PARAM(result)) {
    device const float *first_slab = reinterpret_cast<device const float *>(x[0]);
    element::Out<ELEMENT_OF(SEISMIC_RESULT_0)>(result).put(0, first_slab[0]);
}
