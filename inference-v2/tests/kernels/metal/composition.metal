inline float row_dot(const device float* x, const device float* w, uint K, ushort lane) {
    float sum = 0.0f;
    for (uint k = lane; k < K; k += 32) sum += x[k] * w[k];
    return simd_sum(sum);
}
template<typename Body>
inline typename Body::Result row_fold(const device float* x, uint K, ushort lane,
                                     const thread Body& body) {
    typename Body::State state = body.initial();
    for (uint k = lane; k < K; k += 32) {
        float sample = x[k];
        body.step(state, k, sample);
    }
    return body.finish(state);
}
inline float dot_step(float state, float x, float weight) { return state + x * weight; }
inline float dot_finish(float state) { return simd_sum(state); }
inline float bump(float x) { return x + 2.0f; }
inline float pair_subtract(float current, float partner) { return current - partner; }

// A second row algorithm: no RMS-specific core interface or planner rule.
template<uint WIDTH, uint THREADS, typename Body>
inline void row_transform(const thread Body& body, uint row, uint tid) {
    for (uint column = tid; column < WIDTH; column += THREADS) {
        size_t index = size_t(row) * WIDTH + column;
        float original = body.input(index);
        body.output(index, original * 2.0f + 1.0f, original);
    }
}
