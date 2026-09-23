inline float vision_feature_load(device const uchar *base, ulong logical) {
#if defined(SEISMIC_ELEMENT_A_REPRESENTATION_F32)
    return *reinterpret_cast<device const float *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE);
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_F16)
    return float(*reinterpret_cast<device const half *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE));
#elif defined(SEISMIC_ELEMENT_A_REPRESENTATION_BF16)
    return as_type<float>(uint(*reinterpret_cast<device const ushort *>(base + logical * SEISMIC_ELEMENT_A_PACKET_SIZE)) << 16);
#else
#error "vision features require dense activations"
#endif
}

kernel void qwen_vision_feature_output(
    device const uchar *source [[buffer(SEISMIC_BUFFER_SOURCE)]],
    device float *result [[buffer(SEISMIC_RESULT_0_BUFFER)]],
    constant ulong *seismic_words [[buffer(SEISMIC_BUFFER_WORDS)]],
    uint raw_index [[thread_position_in_grid]]) {
    ulong index = ulong(raw_index);
    if (index >= SEISMIC_DIM_M * SEISMIC_DIM_D) return;
    ulong row = index / SEISMIC_DIM_D;
    ulong column = index % SEISMIC_DIM_D;
    ulong source_index = row * SEISMIC_SOURCE_STRIDE_0 + column * SEISMIC_SOURCE_STRIDE_1;
    ulong result_index = row * SEISMIC_RESULT_0_STRIDE_0 + column * SEISMIC_RESULT_0_STRIDE_1;
    result[result_index] = vision_feature_load(source, source_index);
}
