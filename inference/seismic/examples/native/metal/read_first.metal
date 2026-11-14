#include <seismic/element.h>

kernel void read_first(
    device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    SEISMIC_OUT_RESULT_PARAM(result_cells)) {
    const element::Out<ELEMENT_OF(SEISMIC_RESULT)> result(result_cells);
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
    result.put(0, float(*reinterpret_cast<device const half *>(x)));
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_Q8G32)
    const uchar code = x[SEISMIC_ELEMENT_E_PLANE_0_OFFSET];
    const float scale = *reinterpret_cast<device const float *>(
        x + SEISMIC_ELEMENT_E_PLANE_1_OFFSET);
    result.put(0, float(code) * scale);
#else
#error "read_first fixture supports only f16 and q8g32"
#endif
}
