// A complete row reduction. Hooks see native elements; no partial result escapes.
template<typename T, uint WIDTH, uint THREADS, uint CHUNKS, typename Body, typename Weight>
inline void magnitude_rms(const thread Body& body, Weight weight,
                          float eps, uint row, uint tid, ushort lane, ushort group,
                          threadgroup float* partial) {
    float features[CHUNKS * 4];
    float squares = 0.0f;
    #pragma clang loop unroll(full)
    for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
        uint column = chunk * THREADS * 4 + tid * 4;
        if (column + 4 <= WIDTH) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                float value = float(body.input(size_t(row) * WIDTH + column + i));
                features[chunk * 4 + i] = value;
                squares += value * value;
            }
        }
    }
    if (group == 0) partial[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = simd_sum(squares);
    if (lane == 0) partial[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        float total = simd_sum(partial[lane]);
        if (lane == 0) partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = metal::precise::rsqrt(partial[0] / float(WIDTH) + eps);
    #pragma clang loop unroll(full)
    for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
        uint column = chunk * THREADS * 4 + tid * 4;
        if (column + 4 <= WIDTH) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                T original = T(features[chunk * 4 + i]);
                T result = T(float(T(features[chunk * 4 + i] * inverse)) * float(weight[column + i]));
                body.output(size_t(row) * WIDTH + column + i, result, original);
            }
        }
    }
}
