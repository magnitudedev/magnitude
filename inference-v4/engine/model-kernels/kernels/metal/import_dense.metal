inline float import_load(device const uchar *source, ulong index) {
#if defined(SEISMIC_ELEMENT_E_REPRESENTATION_F32)
    return reinterpret_cast<device const float *>(source)[index];
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_F16)
    return float(reinterpret_cast<device const half *>(source)[index]);
#elif defined(SEISMIC_ELEMENT_E_REPRESENTATION_BF16)
    return as_type<float>(uint(reinterpret_cast<device const ushort *>(source)[index]) << 16);
#else
#error "import_dense source must be f32, f16, or bf16"
#endif
}

inline void import_store(device uchar *destination, ulong index, float value) {
#if defined(SEISMIC_ELEMENT_U_REPRESENTATION_F32)
    reinterpret_cast<device float *>(destination)[index] = value;
#elif defined(SEISMIC_ELEMENT_U_REPRESENTATION_F16)
    reinterpret_cast<device half *>(destination)[index] = half(value);
#elif defined(SEISMIC_ELEMENT_U_REPRESENTATION_BF16)
    uint bits = as_type<uint>(value);
    uint rounding = 0x7fffu + ((bits >> 16) & 1u);
    reinterpret_cast<device ushort *>(destination)[index] = ushort((bits + rounding) >> 16);
#else
#error "import_dense destination must be f32, f16, or bf16"
#endif
}

kernel void import_dense(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device uchar *destination [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_N) return;
    import_store(destination, index, import_load(source, index));
}
