#include <seismic/element.h>

// The first element of `x`, of whichever dense element `E` binds, as F32.
kernel void read_first(
    device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_OUT_RESULT_PARAM(result_cells)) {
    typedef ELEMENT_OF(SEISMIC_ELEMENT_E) E;
    const element::Out<ELEMENT_OF(SEISMIC_RESULT)> result(result_cells);
    result.put(0, E::load(*reinterpret_cast<device const E::storage *>(x)));
}
