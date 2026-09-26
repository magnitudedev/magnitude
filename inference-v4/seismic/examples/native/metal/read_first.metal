kernel void read_first(
    device const uchar *x [[buffer(SEISMIC_BUFFER_X)]],
    device float *result [[buffer(SEISMIC_BUFFER_RESULT)]]) {
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
    result[0] = float(*reinterpret_cast<device const half *>(x));
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_Q8G32)
    const uchar code = x[SEISMIC_ELEMENT_E_PLANE_0_OFFSET];
    const float scale = *reinterpret_cast<device const float *>(
        x + SEISMIC_ELEMENT_E_PLANE_1_OFFSET);
    result[0] = float(code) * scale;
#else
#error "read_first fixture supports only f16 and q8g32"
#endif
}
