// A complete row reduction. Hooks see native elements; no partial result escapes.
template<typename Row, typename Weight>
inline void magnitude_rms(const thread Row& body, Weight weight,
                          float eps, threadgroup float* partial) {
    using T = typename Row::Value;
    constexpr uint WIDTH = Row::width, THREADS = Row::threads;
    constexpr uint CHUNKS = (WIDTH + THREADS * 4 - 1) / (THREADS * 4);
    uint tid = body.thread_index;
    ushort lane = body.lane, group = body.simd_group;
    float features[CHUNKS * 4];
    float squares = 0.0f;
    #pragma clang loop unroll(full)
    for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
        uint column = chunk * THREADS * 4 + tid * 4;
        if (column + 4 <= WIDTH) {
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) {
                float value = float(body.load(column + i));
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
                body.store(column + i, result, original);
            }
        }
    }
}
